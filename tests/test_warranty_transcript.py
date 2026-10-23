"""Final API replies must remain distinct from internal workflow references."""
from datetime import datetime
from pathlib import Path
import sys

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import warranty_models as wm
import warranty_transcript as transcript
import admin_history


@pytest.fixture
def chat(monkeypatch):
    engine = create_engine('sqlite:///:memory:', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(engine)
    monkeypatch.setattr(wm, '_SessionFactory', sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setenv('ADMIN_API_KEY', 'transcript-test-key')
    with wm.warranty_db_session() as db:
        db.add(wm.WarrantyTicket(ticket_id='case', session_id='session', domain='osakiusa.com', current_node_id='root'))
        db.add(wm.WarrantyChatConsent(session_id='session', domain='osakiusa.com', accepted_at=datetime(2026, 10, 23)))
        db.add(wm.WarrantyTurn(ticket_id='case', node_id='root', node_prompt='Internal workflow reference', customer_answer='Earlier answer'))
    app = FastAPI()
    effects = []
    router = __import__('fastapi').APIRouter(route_class=transcript.WarrantyConversationRoute)

    def payload(text):
        return {'session_id': 'session', 'ticket': {'ticket_id': 'case', 'current_node': {'prompt': 'Raw prompt'}}, 'assistant_message': text}

    @router.post('/api/v1/warranty/{ticket_id}/answer')
    async def answer(ticket_id: str, body: dict, background_tasks: BackgroundTasks):
        if body.get('fail'):
            raise HTTPException(422, 'Invalid answer')
        background_tasks.add_task(effects.append, 'background preserved')
        result = payload('Intermediate reply')
        result['assistant_message'] = body.get('reply', 'Final customer-facing answer')
        return result

    @router.post('/api/v1/warranty/session/{session_id}/delivery-tracking')
    async def tracking(session_id: str, body: dict):
        return payload('Latest carrier update: In transit')

    @router.post('/api/v1/warranty/session/{session_id}/handoff')
    async def handoff(session_id: str, body: dict):
        return payload('You can contact our support specialist.')

    @router.post('/api/v1/warranty/{ticket_id}/customer-note')
    @router.post('/api/v1/warranty/{ticket_id}/troubleshooting-outcome')
    async def terminal_followup(ticket_id: str, body: dict):
        return payload('Your support request has been saved for review.')

    @router.get('/api/v1/warranty/session/{session_id}')
    async def state(session_id: str):
        return payload('Read-only state')

    app.include_router(router)
    app.include_router(admin_history.router)
    with TestClient(app) as client:
        yield client, effects
    engine.dispose()


def stored():
    with wm.warranty_db_session() as db:
        return db.query(wm.WarrantyConversationMessage).order_by(wm.WarrantyConversationMessage.id).all()


def test_final_outward_reply_is_full_and_repetitions_are_not_deduplicated(chat):
    client, effects = chat
    text = 'Full customer answer: ' + 'x' * 6000
    for _ in range(2):
        response = client.post('/api/v1/warranty/case/answer', json={'answer': 'Please answer directly', 'reply': text})
        assert response.status_code == 200
        assert response.json()['assistant_message'] == text
    rows = stored()
    assert [(r.role, r.origin) for r in rows] == [('user', 'customer_input'), ('assistant', 'api_response')] * 2
    assert rows[1].content == rows[3].content == text
    assert rows[0].exchange_id == rows[1].exchange_id
    assert rows[0].exchange_id != rows[2].exchange_id
    assert effects == ['background preserved'] * 2


def test_reads_errors_and_unconsented_requests_do_not_create_history(chat):
    client, _ = chat
    assert client.get('/api/v1/warranty/session/session').status_code == 200
    assert client.post('/api/v1/warranty/case/answer', json={'fail': True}).status_code == 422
    with wm.warranty_db_session() as db:
        db.query(wm.WarrantyChatConsent).delete()
    assert client.post('/api/v1/warranty/case/answer', json={'answer': 'No consent'}).status_code == 200
    assert stored() == []


def test_delivery_identity_is_not_copied_and_wrong_session_cannot_record(chat):
    client, _ = chat
    body = {'email': 'private@example.test', 'order_number': 'PRIVATE-123'}
    assert client.post('/api/v1/warranty/session/session/delivery-tracking', json=body).status_code == 200
    rows = stored()
    assert rows[0].origin == 'user_action'
    assert rows[0].content == 'Requested latest delivery tracking.'
    assert all('private@example.test' not in row.content and 'PRIVATE-123' not in row.content for row in rows)
    assert client.post('/api/v1/warranty/session/wrong-session/delivery-tracking', json=body).status_code == 200
    assert len(stored()) == 2


def test_handoff_action_records_actual_reply(chat):
    client, _ = chat
    assert client.post('/api/v1/warranty/session/session/handoff', json={}).status_code == 200
    rows = stored()
    assert rows[0].origin == 'user_action'
    assert rows[1].content == 'You can contact our support specialist.'


@pytest.mark.parametrize('route,field', [('customer-note', 'note'), ('troubleshooting-outcome', 'outcome')])
def test_handoff_from_terminal_followup_preserves_customer_words_and_final_reply(chat, route, field):
    client, _ = chat
    assert client.post(f'/api/v1/warranty/case/{route}', json={field: 'Please let me speak to someone'}).status_code == 200
    rows = stored()
    assert [(row.role, row.content) for row in rows] == [
        ('user', 'Please let me speak to someone'),
        ('assistant', 'Your support request has been saved for review.'),
    ]


def test_recording_failure_does_not_break_reply_or_log_customer_text(chat, monkeypatch, caplog):
    client, _ = chat
    def broken(*args, **kwargs):
        raise RuntimeError('Sensitive private@example.test')
    monkeypatch.setattr(transcript, 'record_warranty_exchange', broken)
    result = client.post('/api/v1/warranty/case/answer', json={'answer': 'private@example.test'})
    assert result.status_code == 200
    assert 'RuntimeError' in caplog.text
    assert 'private@example.test' not in caplog.text


def test_history_prefers_recorded_replies_with_workflow_reference_still_accessible(chat):
    client, _ = chat
    headers = {'X-Admin-Key': 'transcript-test-key'}
    legacy = client.get('/admin/history/warranty/1', headers=headers).json()
    assert legacy['transcript_mode'] == 'legacy_reconstruction'
    assert [m['origin'] for m in legacy['messages']] == ['workflow_prompt', 'workflow_answer']
    client.post('/api/v1/warranty/case/answer', json={'answer': 'Current question', 'reply': 'Actual API reply'})
    recorded = client.get('/admin/history/warranty/1', headers=headers).json()
    assert recorded['transcript_mode'] == 'api_recorded'
    assert recorded['recorded_since']
    assert recorded['alternate_view_available'] is True
    assert [m['content'] for m in recorded['messages']] == ['Current question', 'Actual API reply']
    assert recorded['total'] == recorded['conversation']['message_count'] == 2
    listing = client.get('/admin/history?source=warranty', headers=headers).json()
    assert listing['items'][0]['message_count'] == 2
    reference = client.get('/admin/history/warranty/1?view=workflow', headers=headers).json()
    assert reference['transcript_mode'] == 'workflow_reference'
    assert [m['content'] for m in reference['messages']] == ['Internal workflow reference', 'Earlier answer']
    assert client.get('/admin/history/warranty/1?view=invalid', headers=headers).status_code == 422
    assert client.get('/admin/history/warranty/1?view=workflow').status_code == 401
