"""Truthful Sales outcome/cohort and delivery regressions, entirely offline."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import json
import smtplib
import sys

import pytest
import pytz
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import customer_journey as journey
import sales_agent
import sales_email_transport as transport
import sales_leads
import sales_metrics
import sales_models as sm
import sales_outcomes
import sales_router
import warranty_models as wm

NOW = pytz.timezone('America/Chicago').localize(datetime(2026, 10, 23, 12))


@pytest.fixture(autouse=True)
def db(monkeypatch):
    engine = create_engine('sqlite:///:memory:', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(wm, '_SessionFactory', sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(wm, '_engine', engine)
    monkeypatch.setattr(sales_metrics, '_ADMIN_API_KEY', 'outcomes-test')
    monkeypatch.setattr(sales_metrics, '_now_cst', lambda: NOW)
    monkeypatch.setenv('APP_RELEASE_ID', 'release-new')
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.setattr(sales_agent, 'fetch_live_stock', lambda *a, **k: None)
    monkeypatch.setattr(sales_leads, '_alert_lead_failure', lambda **k: None)
    # Any unexpected network attempt must fail the test without sending mail.
    monkeypatch.setattr(transport.smtplib, 'SMTP', lambda *a, **k: pytest.fail('Unmocked SMTP attempt'))
    monkeypatch.setattr(transport.smtplib, 'SMTP_SSL', lambda *a, **k: pytest.fail('Unmocked SMTPS attempt'))
    yield
    engine.dispose()


def metrics(query=''):
    app = FastAPI(); app.include_router(sales_metrics.router)
    with TestClient(app) as client:
        response = client.get('/admin/sales/metrics?period=all' + query, headers={'X-Admin-Key': 'outcomes-test'})
    assert response.status_code == 200, response.text
    return response.json()


def message(sid, intent, stage, *, age=0, handoff=None):
    return sm.SalesMessage(session_id=sid, role='assistant', content='Deidentified fixture',
        intent=intent, handoff=handoff, created_at=NOW - timedelta(hours=age),
        tools_used=json.dumps(['stage:' + stage] + (['catalog.resolve_product'] if stage == 'shop' else [])))


def test_purpose_denominators_and_normal_one_turn_answer_not_abandonment():
    with wm.warranty_db_session() as db:
        db.add_all([sm.SalesSession(session_id=sid, domain='osakiusa.com', created_at=NOW) for sid in ('direct', 'rec', 'active', 'idle', 'handoff')])
        db.add_all([
            message('direct', 'specs', 'shop', age=50),
            message('rec', 'recommend', 'recommend'),
            message('active', 'recommend', 'ask_height'),
            message('idle', 'recommend', 'ask_height', age=25),
            message('handoff', 'human', 'handoff', handoff='human'),
        ])
    data = metrics(); outcomes = data['outcomes']
    assert outcomes['recommendation']['requested_sessions'] == 3
    assert outcomes['recommendation']['completed_sessions'] == 1
    assert outcomes['recommendation']['completion_rate_pct'] == 33.3
    assert outcomes['conversation']['answer_provided_sessions'] == 2
    assert outcomes['conversation']['active_waiting_sessions'] == 1
    assert outcomes['conversation']['idle_missing_followup_sessions'] == 1
    assert outcomes['human']['requested_sessions'] == 1
    assert outcomes['human']['staff_confirmed_sessions'] is None
    height = next(row for row in data['question_funnel'] if row['stage'] == 'ask_height')
    assert height['active_waiting'] == height['idle_missing_followup'] == 1
    assert height['dropped'] is height['drop_off_pct'] is None


def test_question_branch_order_and_helpful_task_exit_are_observations():
    with wm.warranty_db_session() as db:
        db.add_all([sm.SalesSession(session_id=sid, domain='osakiusa.com', created_at=NOW) for sid in ('branch', 'answered')])
        db.add_all([
            message('branch', 'recommend', 'ask_doorway', age=4),
            message('branch', 'recommend', 'ask_goal', age=3),
            message('answered', 'recommend', 'ask_height', age=4),
            message('answered', 'specs', 'shop', age=3),
        ])
    data = metrics()
    rows = {row['stage']: row for row in data['question_funnel']}
    assert rows['ask_doorway']['continued'] == 1
    assert rows['ask_height']['answer_provided'] == 1
    assert rows['ask_height']['idle_missing_followup'] == 0


def test_repeated_unclear_and_feedback_coverage_have_separate_denominators():
    with wm.warranty_db_session() as db:
        db.add_all([sm.SalesSession(session_id=sid, domain='osakiusa.com', created_at=NOW) for sid in ('rated', 'unknown')])
        db.add_all([message('rated', 'specs', 'shop'), message('unknown', 'unclear', 'menu', age=1), message('unknown', 'unclear', 'menu')])
        db.add(sm.SalesFeedback(session_id='rated', rating='helpful', intent='specs', domain='osakiusa.com', created_at=NOW))
    quality = metrics()['outcomes']['quality']
    assert quality['helpful_rate_pct'] == 100
    assert quality['rating_coverage_pct'] == 50
    assert quality['rated'] == quality['rated_sessions'] == quality['unrated_sessions'] == 1
    assert quality['repeated_unclear_sessions'] == 1


def test_empty_samples_are_unknown_not_zero_percent():
    outcomes = metrics()['outcomes']
    assert outcomes['recommendation']['completion_rate_pct'] is None
    assert outcomes['quality']['helpful_rate_pct'] is None
    assert outcomes['quality']['rating_coverage_pct'] is None
    assert outcomes['follow_up']['staff_confirmed'] is None


def test_lead_submission_states_do_not_imply_staff_receipt_or_unique_people():
    with wm.warranty_db_session() as db:
        db.add(sm.SalesSession(session_id='lead', domain='osakiusa.com', created_at=NOW))
        for status in ('pending', 'sent', 'failed'):
            db.add(sm.SalesLead(session_id='lead', domain='osakiusa.com', forwarded=status,
                               forwarded_error='smtp_authentication_failed' if status == 'failed' else None, created_at=NOW))
    followup = metrics()['outcomes']['follow_up']
    assert followup['lead_records'] == 3
    assert followup['lead_sessions'] == 1
    assert followup['pending'] == followup['sent'] == followup['failed'] == 1
    assert followup['submission_failure_rate_pct'] == 50
    assert followup['staff_confirmed'] is None
    assert followup['failure_codes'] == [{'code': 'smtp_authentication_failed', 'count': 1}]


def test_actual_router_records_new_release_and_handoff_request_not_staff_confirmation():
    app = FastAPI(); app.include_router(sales_router.router)
    with TestClient(app) as client:
        response = client.post('/api/v1/sales/chat', json={'session_id': 'new-human', 'message': 'Talk to a human', 'domain': 'osakiusa.com'})
    assert response.status_code == 200, response.text
    assert response.json()['handoff_state'] == 'requested'
    assert 'one thing' not in response.json()['reply']
    with wm.warranty_db_session() as db:
        tag = db.query(journey.ConversationRelease).one()
        assert tag.initial_release == 'release-new'
        assert not tag.mixed
        events = db.query(journey.CustomerJourneyEvent).all()
        assert {(row.stage, row.event) for row in events} >= {('response', 'completed'), ('handoff', 'requested')}
        assert not any(row.stage == 'handoff' and row.event == 'completed' for row in events)
        assert all(not hasattr(row, 'content') for row in events)


def test_cohorts_exclude_old_and_mixed_records_without_relabeling(monkeypatch):
    with wm.warranty_db_session() as db:
        db.add(sm.SalesSession(session_id='old', domain='osakiusa.com', created_at=NOW))
    sm.get_or_create_session('new', domain='osakiusa.com')
    with wm.warranty_db_session() as db:
        db.query(sm.SalesSession).filter_by(session_id='new').one().created_at = NOW
        row = db.query(sm.SalesSession).filter_by(session_id='old').one()
        journey.register_conversation(db, surface='sales', conversation_id=str(row.id), session_id='old', domain='osakiusa.com', is_new=False)
    monkeypatch.setenv('APP_RELEASE_ID', 'release-next')
    from sales_journey import record_reply
    record_reply('new', 'osakiusa.com', sales_agent.respond('hello'))
    data = metrics('&cohort=mixed')
    assert data['totals']['started'] == 2
    assert metrics('&cohort=current')['totals']['started'] == 0
    assert metrics('&cohort=legacy')['totals']['started'] == 0


class FakeSMTP:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.tls = args, kwargs, False
        self.accepted = False
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def ehlo(self): pass
    def starttls(self, **kwargs): self.tls = True
    def login(self, *args): pass
    def send_message(self, message):
        self.accepted = True
        assert 'buyer@example.test' not in message['Subject']
        return {}


@pytest.mark.parametrize('port,implicit', [('587', False), ('465', True)])
def test_transport_is_bounded_and_uses_expected_tls(monkeypatch, port, implicit):
    monkeypatch.setenv('EMAIL_SENDER', 'sender@example.test')
    monkeypatch.setenv('EMAIL_PASSWORD', 'fixture-not-secret')
    monkeypatch.setenv('SMTP_PORT', port)
    fake = FakeSMTP()
    calls = []
    def connect(*args, **kwargs):
        calls.append(kwargs)
        return fake
    monkeypatch.setattr(transport.smtplib, 'SMTP_SSL' if implicit else 'SMTP', connect)
    assert transport.send_sales_notification('buyer@example.test', 'Authorized fixture summary', '', 'osakiusa.com')
    assert calls[0]['timeout'] == 8
    assert fake.tls is not implicit
    assert fake.accepted


@pytest.mark.parametrize('failure,code', [
    (smtplib.SMTPAuthenticationError(535, b'private user@example.test'), 'smtp_authentication_failed'),
    (TimeoutError('private user@example.test'), 'smtp_connection_failed'),
    (smtplib.SMTPRecipientsRefused({'private@example.test': (550, 'private')}), 'smtp_recipient_rejected'),
])
def test_delivery_records_sanitized_failure_code_without_send_or_retry(monkeypatch, failure, code):
    monkeypatch.setenv('EMAIL_SENDER', 'sender@example.test')
    monkeypatch.setenv('EMAIL_PASSWORD', 'fixture-not-secret')
    monkeypatch.setenv('SMTP_PORT', '587')
    calls = []
    def fail(*args, **kwargs):
        calls.append(1); raise failure
    monkeypatch.setattr(transport.smtplib, 'SMTP', fail)
    sm.get_or_create_session('lead', domain='osakiusa.com')
    lead_id = sales_leads.capture_lead(session_id='lead', domain='osakiusa.com', email='buyer@example.test')
    sales_leads.deliver_lead(lead_id=lead_id, domain='osakiusa.com', email='buyer@example.test')
    with wm.warranty_db_session() as db:
        lead = db.get(sm.SalesLead, lead_id)
        assert lead.forwarded == 'failed'
        assert lead.forwarded_error == code
        assert 'private' not in lead.forwarded_error
    assert calls == [1]


def test_unconfigured_transport_is_distinguished_without_any_connection(monkeypatch):
    monkeypatch.delenv('EMAIL_SENDER', raising=False)
    monkeypatch.delenv('EMAIL_PASSWORD', raising=False)
    assert transport.transport_health()['failure_code'] == 'smtp_not_configured'
    with pytest.raises(transport.SalesEmailError, match='smtp_not_configured'):
        transport.send_sales_notification('', 'Fixture', '', 'osakiusa.com')


@pytest.mark.parametrize('domain,key', [('titanchair.com', 'titanchair'), ('osakichair.com', 'titanchair'),
    ('osaki-titan.com', 'titanchair'), ('osakititan.com', 'titanchair'),
    ('osakiusa.com', 'osakiusa'), ('osakimassagechair.com', 'osakimassagechair')])
def test_supported_store_aliases_retain_the_existing_sales_inbox(monkeypatch, domain, key):
    from config import SALES_EMAIL_BY_DOMAIN
    submitted = []
    monkeypatch.setattr(transport, '_send', lambda recipient, *args: submitted.append(recipient) or True)
    assert transport.send_sales_notification('', 'Fixture request', '', domain)
    assert submitted == [SALES_EMAIL_BY_DOMAIN[key]]


def test_optional_contact_keeps_nonshopping_question_without_a_recommendation():
    reply = sales_agent.respond('I want to create an invoice for a massage chair')
    assert reply.handoff and reply.intent == 'human'
    prefs = reply.prefs_patch
    email_prompt = sales_agent.respond('', payload='lead:email', prefs=prefs)
    prefs = {**prefs, **email_prompt.prefs_patch}
    saved = sales_agent.respond('buyer@example.test', prefs=prefs)
    assert 'create an invoice' in saved.lead_capture['interest_summary']
    assert not saved.products
    assert 'your pick' not in saved.reply


def test_lead_capture_cannot_reassign_an_existing_store_conversation():
    sm.get_or_create_session('store-owned', domain='osakiusa.com')
    app = FastAPI(); app.include_router(sales_router.router)
    with TestClient(app) as client:
        response = client.post('/api/v1/sales/lead', json={'session_id': 'store-owned', 'domain': 'titanchair.com', 'email': 'buyer@example.test'})
    assert response.status_code == 409
    with wm.warranty_db_session() as db:
        assert db.query(sm.SalesSession).filter_by(session_id='store-owned').one().domain == 'osakiusa.com'
        assert db.query(sm.SalesLead).count() == 0
