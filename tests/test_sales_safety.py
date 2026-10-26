"""Physical hazards interrupt both customer channels before payload routing."""
import pytest

from tests.test_evaluation_sales_dialogues import isolated_sales_api


@pytest.mark.parametrize('endpoint', ['/api/v1/sales/chat', '/api/v1/sales/tidio/turn'])
@pytest.mark.parametrize('report,language,stop', [
    ('My Maestro chair is smoking. Please cancel the order.', 'en', 'Stop using the chair'),
    ('Se ven llamas debajo de mi silla.', 'es', 'Deje de usar la silla'),
    ('의자에서 연기가 나요. 주문을 취소해 주세요.', 'ko', '의자 사용을 즉시 중단'),
])
def test_report_interrupts_payload_and_stays_stopped_after_acknowledgement(isolated_sales_api, monkeypatch, endpoint, report, language, stop):
    import sales_tidio_router
    import warranty_router
    from sales_models import SalesLead, get_or_create_session, get_session_collected, merge_session_collected
    from warranty_models import warranty_db_session

    warranty_router.limiter.reset()
    monkeypatch.setenv('TIDIO_ENABLED', '1')
    monkeypatch.setenv('TIDIO_TURN_SECRET', 'synthetic-safety-secret')
    client, sid = isolated_sales_api, 'synthetic-sales-safety'
    client.app.include_router(sales_tidio_router.router)
    assert client.post(f'/api/v1/warranty/session/{sid}/consent', json={'domain': 'osakiusa.com'}).status_code == 200
    get_or_create_session(sid, domain='osakiusa.com')
    merge_session_collected(sid, {'awaiting_email_for_pick': True, 'awaiting_showroom_window': True,
                                 'awaiting_recommend': True, 'pending_primary': 'Maestro',
                                 'pending_compare_pair': ['Maestro', 'Paragon']})
    for message, payload in [(report, 'menu'), ('Thanks, I will stop using it.', 'lead:save_pick'), ('', 'start_over')]:
        response = client.post(endpoint, headers={'X-Tidio-Turn-Secret': 'synthetic-safety-secret'},
                               json={'session_id': sid, 'domain': 'osakiusa.com', 'message': message, 'payload': payload})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['handoff'] is True
        assert body['handoff_reason'] == 'safety_hazard'
        assert stop in body['reply']
        assert 'ext. 3' in body['reply']
        assert not body.get('products')
        assert not body['quick_replies']
        if endpoint.endswith('/tidio/turn'):
            assert body['next_action'] == 'warranty_redirect'
        saved = get_session_collected(sid)
        assert saved['sales_safety_active'] is True
        assert saved['sales_safety_report'] == report
        assert saved['sales_safety_language'] == language
        assert saved['awaiting_recommend'] is False
        assert saved['awaiting_email_for_pick'] is False
        assert saved['awaiting_showroom_window'] is False
        assert saved['pending_primary'] == ''
        assert saved['pending_compare_pair'] == []
    with warranty_db_session() as db:
        assert db.query(SalesLead).count() == 0


@pytest.mark.parametrize('message', [
    'There is no smoke, no fire, and no sparks. Compare the Maestro and Paragon.',
    '배송 일정을 다음 주로 연기해 주세요.',
    'The manual says "stop if there is smoke". There is no smoke.',
    'I want a chair for lower back pain.',
    'I am shopping for a chair because I have burning back pain.',
    'Will a chair cure my sciatica?',
    'No smoke or sparks. I want a chair for lower back pain.',
    'Do you sell fire-resistant chairs for lower back pain?',
    'The manual says "stop if there is smoke". I want a chair for lower back pain.',
    'smoke.buyer@example.test',
    'The manual link is https://example.test/smoke-warning.pdf',
])
def test_non_reports_and_ordinary_clinical_questions_do_not_become_fire_incidents(message):
    from sales_agent import respond
    from sales_safety import physical_hazard_report

    assert not physical_hazard_report(message)
    result = respond(message)
    assert result.handoff_reason != 'safety_hazard'


@pytest.mark.parametrize('message', [
    'Touching the frame gives me a mild shock.',
    'The chair burned my hand.',
    '쇠 부분을 만지니 찌릿해요.',
    'The chair hurts my back.',
])
def test_physical_reports_and_clearly_chair_caused_injury_interrupt_shopping(message):
    from sales_agent import respond

    result = respond(message, payload='recommend')
    assert result.handoff_reason == 'safety_hazard'
    assert not result.products


def test_contact_metadata_is_not_a_report_but_cannot_hide_an_actual_report():
    from sales_agent import respond

    address = 'smoke.buyer@example.test'
    contact = respond(address, prefs={'awaiting_email_for_pick': True, 'pending_pick_summary': 'Synthetic catalog pick'})
    assert contact.handoff_reason != 'safety_hazard'
    assert contact.lead_capture['email'] == address
    incident = respond('My chair is smoking. Contact me at '+address)
    assert incident.handoff_reason == 'safety_hazard'
    assert address in incident.prefs_patch['sales_safety_report']
    assert 'order cancellation' not in incident.reply
    cancel = respond('My chair is smoking. Please cancel my order.')
    assert 'No order cancellation has been confirmed.' in cancel.reply
