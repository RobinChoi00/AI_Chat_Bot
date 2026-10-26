"""Exercise support continuity and delivery evidence without external services."""
from __future__ import annotations

import uuid

import pytest

from tests.test_evaluation_sales_dialogues import isolated_sales_api
import sales_agent as agent
from sales_order_support import is_email_status_question


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *a, **kw: None)
    monkeypatch.setattr(agent, 'fetch_stock_many', lambda *a, **kw: {})


def turn(message, prefs, **kwargs):
    reply = agent.respond(message, prefs=prefs, **kwargs)
    for key, value in (reply.prefs_patch or {}).items():
        prefs[key] = {**prefs.get(key, {}), **value} if isinstance(value, dict) else value
    return reply


@pytest.mark.parametrize('opening', [
    'I already ordered a Maestro 4D, order 123456. Where is it?',
    'I bought Maestro last week. Order 481920 is still not here. Check its progress.',
    "My chair hasn't arrived. Can you check my order?",
    'My order number is 481921. It has not shipped yet.',
])
def test_existing_purchase_routes_to_sales_and_preserves_cancellation_followup(opening):
    prefs = {}
    first = turn(opening, prefs)
    assert first.intent == 'order_status'
    assert first.handoff
    assert 'ext. 2' in first.reply
    assert 'service@' not in first.reply
    assert not first.products
    second = turn('My order email is buyer@example.test.', prefs)
    assert second.intent == 'order_status'
    assert not second.lead_capture
    last = turn('Could you cancel the order instead?', prefs)
    assert last.intent == 'cancel_refund'
    assert 'have not cancelled' in last.reply
    assert not last.lead_capture


def test_return_followup_uses_existing_order_and_never_claims_approval():
    prefs = {}
    first = turn('My Paragon was delivered yesterday. I need to return this order.', prefs)
    assert first.intent == 'cancel_refund'
    followup = turn('What do you need from me to start that return?', prefs)
    assert followup.intent == 'cancel_refund'
    assert 'checkout email' in followup.reply
    assert 'have not' in followup.reply


@pytest.mark.parametrize('message', [
    "Don't cancel my order. Where is it?",
    'I do not want to cancel my order; I need its tracking status.',
])
def test_negated_cancellation_does_not_start_cancellation(message):
    reply = agent.respond(message, prefs={'sales_support_topic': 'order_status'})
    assert reply.intent == 'order_status'
    assert reply.handoff_reason != 'cancel_refund'


@pytest.mark.parametrize('message', [
    'My remote is broken. Where is my order?',
    'I need a replacement remote for my order.',
])
def test_concrete_technical_fault_keeps_warranty_route(message):
    reply = agent.respond(message, prefs={'sales_support_topic': 'order_status'})
    assert reply.intent in {'warranty_redirect', 'parts_technician'}
    assert reply.handoff


def test_new_shopping_task_owns_its_contact_email_instead_of_old_order_context():
    prefs = {}
    turn('Where is my order?', prefs)
    turn('Show me the price of the Maestro 4D.', prefs)
    assert not prefs['sales_support_topic']
    turn('', prefs, payload='lead:save_pick')
    captured = turn('new-pick@example.test', prefs)
    assert captured.lead_capture['email'] == 'new-pick@example.test'


def test_return_motion_spec_is_not_a_return_order_request():
    reply = agent.respond('Does this chair return to the upright position?', prefs={
        'sales_support_topic': 'order_status', 'pending_primary': 'Maestro 4D',
    })
    assert reply.intent != 'cancel_refund'


def test_stopping_shopping_search_preserves_fit_facts_without_cancelling_order():
    prefs = {}
    turn('Help me find a chair under $3500.', prefs)
    reply = turn('Never mind the chair suggestions; stop that search.', prefs)
    assert reply.intent == 'recommendation_cancel'
    assert not reply.handoff
    assert not prefs['awaiting_recommend']
    assert prefs['recommend_prefs']['budget_max_usd'] == '3500'
    assert 'stopped the chair search' in reply.reply
    assert 'order' not in reply.reply.lower()


def test_pending_showroom_can_answer_policy_and_cancel_only_local_form():
    prefs = {}
    turn('', prefs, payload='cta:showroom:window:weekday_am')
    policy = turn('Before sharing an email, what are your return rules?', prefs)
    assert policy.intent == 'prepurchase_policy'
    assert '30 days' in policy.reply
    assert not policy.lead_capture
    assert not prefs['awaiting_showroom_email']
    cancelled = turn('Please cancel that showroom request.', prefs)
    assert cancelled.intent == 'showroom_request_cancel'
    assert 'have not cancelled a booked appointment' in cancelled.reply
    assert not prefs['showroom_request_pending']
    assert prefs['pending_showroom_window'] == ''


def test_email_request_is_not_an_email_delivery_status_question():
    assert not is_email_status_question('Could you send me an email about this pick?')
    assert is_email_status_question('Did you send the email, or only save the request?')
    reply = agent.respond('Was it actually sent?', prefs={'followup_request_saved': True})
    assert reply.intent == 'email_status'
    assert 'do not have confirmed' in reply.reply


@pytest.mark.parametrize('question,new_request', [
    ('¿Ya enviaste el correo?', 'Envíame el correo, por favor.'),
    ('이메일 보냈나요?', '이메일 발송해 주세요.'),
])
def test_native_status_question_is_distinct_from_new_email_request(question, new_request):
    assert is_email_status_question(question)
    assert not is_email_status_question(new_request)


@pytest.mark.parametrize('message,prefs', [
    ('쇼룸 예약 취소하지 마세요.', {'conversation_language': 'ko', 'showroom_request_pending': True}),
    ('No quiero cancelar mi pedido.', {'conversation_language': 'es', 'sales_support_topic': 'order_status'}),
])
def test_negated_multilingual_action_does_not_clear_or_cancel(message, prefs):
    from sales_order_support import maybe_sales_support_reply
    reply = maybe_sales_support_reply(message, prefs, domain='osakiusa.com')
    assert reply is None or reply.intent not in {'showroom_request_cancel', 'cancel_refund'}


@pytest.fixture(params=['public', 'tidio'])
def customer(isolated_sales_api, monkeypatch, request):
    import sales_router, sales_tidio_router
    router = sales_router if request.param == 'public' else sales_tidio_router
    if request.param == 'tidio':
        isolated_sales_api.app.include_router(sales_tidio_router.router)
        monkeypatch.setenv('TIDIO_ENABLED', '1')
        monkeypatch.setenv('TIDIO_DOMAIN', 'osakiusa.com')
        monkeypatch.delenv('TIDIO_TURN_SECRET', raising=False)
    router.limiter.reset()
    scheduled = []
    monkeypatch.setattr(router, 'schedule_lead_delivery', lambda *a, **kw: scheduled.append(kw))
    session = 'support-' + uuid.uuid4().hex
    assert isolated_sales_api.post(f'/api/v1/warranty/session/{session}/consent', json={'domain': 'osakiusa.com'}).status_code == 200

    def chat(message='', payload=None):
        result = isolated_sales_api.post('/api/v1/sales/chat' if request.param == 'public' else '/api/v1/sales/tidio/turn', json={
            'session_id': session, 'message': message, 'payload': payload, 'domain': 'osakiusa.com',
        })
        assert result.status_code == 200, result.text
        return result.json()
    chat.router = router
    return session, chat, scheduled


def save_followup(chat):
    chat('Price of the Maestro 4D')
    chat(payload='lead:save_pick')
    return chat('shopper@example.test')


@pytest.mark.parametrize('contact', [
    'The email on that purchase is synthetic.buyer@example.test.',
    'For checkout I used synthetic.buyer@example.test.',
    'The checkout email was synthetic.buyer@example.test.',
    'The email on that purchase is cancel.buyer@example.test.',
])
def test_checkout_contact_followup_stays_with_known_order_without_creating_lead(customer, contact):
    import warranty_models
    from sales_models import SalesLead
    session, chat, scheduled = customer
    first = chat('I bought the Maestro last week. Order 481920 is still not here. Can you check its progress?')
    assert first['intent'] == 'order_status'
    result = chat(contact)
    assert result['intent'] == 'order_status'
    assert result['handoff']
    assert result['handoff_reason'] == 'order_status'
    assert 'Thanks for the checkout email' in result['reply']
    assert 'does not by itself verify' in result['reply']
    assert 'have not looked up live tracking' in result['reply']
    assert not result.get('products')
    assert not scheduled
    with warranty_models.warranty_db_session() as db:
        assert db.query(SalesLead).filter_by(session_id=session).count() == 0


def test_checkout_email_does_not_imply_order_context_or_marketing_consent(customer):
    import warranty_models
    from sales_models import SalesLead
    session, chat, scheduled = customer
    result = chat('The email on that purchase is synthetic.buyer@example.test.')
    assert result['intent'] != 'order_status'
    assert not scheduled
    with warranty_models.warranty_db_session() as db:
        assert db.query(SalesLead).filter_by(session_id=session).count() == 0


def test_explicit_new_pick_email_form_owns_contact_even_if_order_topic_is_legacy():
    prefs = {'sales_support_topic': 'order_status', 'awaiting_email_for_pick': True,
             'pending_primary': 'Maestro 4D', 'pending_pick_summary': 'Chosen pick'}
    reply = agent.respond('The checkout email was new-pick@example.test.', prefs=prefs)
    assert reply.intent == 'lead_capture'
    assert reply.lead_capture['email'] == 'new-pick@example.test'


@pytest.mark.parametrize('status,required', [
    ('pending', 'notification to sales is pending'),
    ('sent', 'accepted by the email service for the sales inbox'),
    ('failed', 'notification to sales failed'),
])
def test_public_api_uses_latest_scoped_lead_evidence(customer, status, required):
    import warranty_models
    from sales_models import SalesLead
    session, chat, scheduled = customer
    saved = save_followup(chat)
    assert 'saved' in saved['reply']
    assert len(scheduled) == 1
    with warranty_models.warranty_db_session() as db:
        lead = db.query(SalesLead).filter_by(session_id=session).one()
        lead.forwarded = status
        db.add(SalesLead(session_id=session, domain='titanchair.com', email='other@example.test', forwarded='sent'))
        db.add(SalesLead(session_id='another-session', domain='osakiusa.com', email='other@example.test', forwarded='sent'))
    result = chat('Was an email actually sent, or did you just save my request?')
    assert result['intent'] == 'email_status'
    assert required in result['reply']
    if status == 'sent':
        assert 'does not confirm inbox arrival' in result['reply']
        assert 'delivery of a copy to you' in result['reply']
    assert len(scheduled) == 1


def test_newer_pending_lead_and_new_unsaved_attempt_do_not_reuse_old_sent(customer):
    import warranty_models
    from sales_models import SalesLead
    session, chat, _ = customer
    save_followup(chat)
    with warranty_models.warranty_db_session() as db:
        old = db.query(SalesLead).filter_by(session_id=session).one()
        old.forwarded = 'sent'
        db.add(SalesLead(session_id=session, domain='osakiusa.com', email='latest@example.test', forwarded='pending'))
    pending = chat('Did you send the email?')
    assert 'notification to sales is pending' in pending['reply']
    chat(payload='lead:save_pick')
    unsaved = chat('Did you send the email?')
    assert 'has not been submitted yet' in unsaved['reply']


@pytest.mark.parametrize('question,required', [
    ('¿Ya enviaste el correo?', 'aceptó'),
    ('이메일 보냈나요?', '접수했습니다'),
])
def test_native_email_status_question_uses_actual_delivery_evidence(customer, question, required):
    import warranty_models
    from sales_models import SalesLead
    session, chat, _ = customer
    save_followup(chat)
    with warranty_models.warranty_db_session() as db:
        db.query(SalesLead).filter_by(session_id=session).one().forwarded = 'sent'
    result = chat(question)
    assert result['intent'] == 'email_status'
    assert required in result['reply']


@pytest.mark.parametrize('failure', ['none', 'raise'])
def test_capture_failure_never_claims_saved_or_schedules_a_notification(customer, monkeypatch, failure):
    session, chat, scheduled = customer
    def capture(**kw):
        if failure == 'raise':
            raise RuntimeError('synthetic persistence failure')
        return None
    monkeypatch.setattr(chat.router, 'capture_lead', capture)
    result = save_followup(chat)
    assert 'could not be saved' in result['reply']
    assert 'Thanks' not in result['reply']
    assert not scheduled
    later = chat('Did you send the email?')
    assert 'could not be saved' in later['reply']


def test_failed_new_capture_cannot_report_older_success(customer, monkeypatch):
    import warranty_models
    from sales_models import SalesLead
    session, chat, scheduled = customer
    save_followup(chat)
    with warranty_models.warranty_db_session() as db:
        db.query(SalesLead).filter_by(session_id=session).one().forwarded = 'sent'
    monkeypatch.setattr(chat.router, 'capture_lead', lambda **kw: None)
    chat(payload='lead:save_pick')
    chat('second@example.test')
    response = chat('Did you send the email?')
    assert 'could not be saved' in response['reply']
    assert 'accepted by the email service' not in response['reply']
    assert len(scheduled) == 1


def test_closing_submitted_showroom_form_does_not_withdraw_saved_request(customer):
    import warranty_models
    from sales_models import SalesLead
    session, chat, scheduled = customer
    chat(payload='cta:showroom:window:saturday')
    saved = chat('showroom@example.test')
    assert 'Saved a **visit request**' in saved['reply']
    cancelled = chat('Please cancel my showroom appointment.')
    assert cancelled['intent'] == 'showroom_request_cancel'
    assert 'have not cancelled a booked appointment' in cancelled['reply']
    with warranty_models.warranty_db_session() as db:
        rows = db.query(SalesLead).filter_by(session_id=session).all()
        assert len(rows) == 1
        assert rows[0].reason == 'showroom_visit'
        assert rows[0].forwarded == 'pending'
    assert len(scheduled) == 1


def test_scheduling_failure_keeps_saved_request_pending_without_delivery_claim(customer, monkeypatch):
    import warranty_models
    from sales_models import SalesLead
    session, chat, _ = customer
    def unavailable(*a, **kw):
        raise RuntimeError('synthetic scheduler failure')
    monkeypatch.setattr(chat.router, 'schedule_lead_delivery', unavailable)
    saved = save_followup(chat)
    assert 'saved' in saved['reply']
    assert 'sent' not in saved['reply'].lower()
    status = chat('Did you send the email?')
    assert 'notification to sales is pending' in status['reply']
    with warranty_models.warranty_db_session() as db:
        assert db.query(SalesLead).filter_by(session_id=session).one().forwarded == 'pending'
