"""
tests/test_warranty_resume.py
=============================
Unit + HTTP tests for the "Save & continue later" warranty resume flow.
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402
import warranty_resume as wr  # noqa: E402


TEST_SECRET = "a" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ADMIN_SESSION_SECRET", TEST_SECRET)
    yield


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    import warranty_workflow as wf  # noqa: WPS433

    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    mem_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=mem_engine,
        expire_on_commit=False,
    )
    wm.Base.metadata.create_all(bind=mem_engine)

    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", mem_session_factory)
    monkeypatch.setattr(wf, "_SessionFactory", mem_session_factory)
    yield


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        wr,
        "_send_resume_email_async",
        lambda **kwargs: None,  # do not hit SMTP in tests
    )
    from warranty_router import router as warranty_router  # noqa: WPS433

    app = FastAPI()
    app.include_router(warranty_router)
    app.include_router(wr.router)
    return TestClient(app)


# --- Token round-trip ------------------------------------------------------


def test_token_roundtrip_ok():
    tok = wr.create_resume_token("t-abc", "s-1", ttl_secs=60)
    payload = wr.verify_resume_token(tok)
    assert payload is not None
    assert payload["tid"] == "t-abc"
    assert payload["sid"] == "s-1"


def test_token_rejects_tampered_signature():
    tok = wr.create_resume_token("t-abc", "s-1")
    body, _sig = tok.split(".")
    tampered = f"{body}.AAAA"
    assert wr.verify_resume_token(tampered) is None


def test_token_rejects_expired():
    tok = wr.create_resume_token("t-abc", "s-1", ttl_secs=-10)
    assert wr.verify_resume_token(tok) is None


def test_token_rejects_wrong_secret(monkeypatch):
    tok = wr.create_resume_token("t-abc", "s-1", ttl_secs=60)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "b" * 40)
    assert wr.verify_resume_token(tok) is None


def test_token_requires_min_secret_length(monkeypatch):
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "short")
    with pytest.raises(RuntimeError):
        wr.create_resume_token("t", "s")


def test_resume_signing_ready(monkeypatch):
    monkeypatch.setenv("ADMIN_SESSION_SECRET", TEST_SECRET)
    assert wr.resume_signing_ready() is True
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "short")
    assert wr.resume_signing_ready() is False
    monkeypatch.delenv("ADMIN_SESSION_SECRET", raising=False)
    assert wr.resume_signing_ready() is False


# --- HTTP end-to-end -------------------------------------------------------


def _register_and_start(client, session_id: str) -> str:
    client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": "OS-4000T", "domain": "osaki.com"},
    )
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    return resp.json()["ticket"]["ticket_id"]


def test_resume_link_requires_active_ticket(client):
    resp = client.post(
        "/api/v1/warranty/session/no-ticket/resume-link",
        json={"customer_email": "buyer@example.com"},
    )
    assert resp.status_code == 404


def test_resume_link_rejects_bad_email(client):
    session_id = "sess-resume-bad-email"
    _register_and_start(client, session_id)
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/resume-link",
        json={"customer_email": "not-an-email"},
    )
    assert resp.status_code == 422


def test_resume_link_and_verify_roundtrip(client):
    session_id = "sess-resume-happy"
    ticket_id = _register_and_start(client, session_id)

    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/resume-link",
        json={"customer_email": "buyer@example.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sent"] is False
    assert body["queued"] is True
    assert body["customer_email"] == "b***r@example.com"
    assert 'resume=' in body['resume_url']
    with wm.warranty_db_session() as db:
        saved = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one().get_collected()
        assert saved['resume_email'] == 'buyer@example.com'
        assert saved['resume_delivery_status'] == 'pending'

    # Directly generate a token — the SMTP side-effect is mocked, so we
    # cannot pluck the URL out of the email. We verify the round-trip via
    # the public token helpers instead.
    token = wr.create_resume_token(ticket_id, session_id)
    verify = client.get(f"/api/v1/warranty/resume/{token}")
    assert verify.status_code == 200
    data = verify.json()
    assert data["ticket_id"] == ticket_id
    assert data["session_id"] == session_id
    assert data["status"] == "in_progress"
    assert data["expires_at"] > int(time.time())


def test_resume_endpoint_rejects_bad_token(client):
    resp = client.get("/api/v1/warranty/resume/definitely-not-a-token")
    assert resp.status_code == 400


def test_resume_endpoint_404_when_ticket_missing(client):
    token = wr.create_resume_token("does-not-exist", "s-x")
    resp = client.get(f"/api/v1/warranty/resume/{token}")
    assert resp.status_code == 404


@pytest.mark.parametrize('sent,status', [(True, 'sent'), (False, 'failed')])
def test_resume_worker_records_transport_result_for_current_attempt(client, monkeypatch, sent, status):
    sid = 'resume-worker'
    tid = _register_and_start(client, sid)
    response = client.post(f'/api/v1/warranty/session/{sid}/resume-link', json={'customer_email': 'buyer@example.com'})
    assert response.status_code == 200
    with wm.warranty_db_session() as db:
        data = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()
    monkeypatch.setattr(wr, '_send_resume_email', lambda **kwargs: sent)
    wr._deliver_resume_email(attempt_id=data['resume_delivery_attempt_id'], customer_email='buyer@example.com',
        resume_url=response.json()['resume_url'], ticket_id=tid, model_name='OS-4000T', domain='osakiusa.com')
    with wm.warranty_db_session() as db:
        result = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()
        assert result['resume_delivery_status'] == status
    # A previous background attempt cannot overwrite the newer request's state.
    wr._deliver_resume_email(attempt_id='old-attempt', customer_email='buyer@example.com',
        resume_url=response.json()['resume_url'], ticket_id=tid, model_name='OS-4000T', domain='osakiusa.com')
    with wm.warranty_db_session() as db:
        assert db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()['resume_delivery_status'] == status


def test_resume_wakeup_failure_preserves_durable_queue_and_copy_link(client, monkeypatch):
    sid = 'resume-no-worker'
    tid = _register_and_start(client, sid)
    def unavailable(**kwargs):
        raise RuntimeError('Do not expose private@example.test')
    monkeypatch.setattr(wr, '_send_resume_email_async', unavailable)
    response = client.post(f'/api/v1/warranty/session/{sid}/resume-link', json={'customer_email': 'buyer@example.com'})
    assert response.status_code == 200
    assert response.json()['queued'] is True
    assert response.json()['sent'] is False
    assert response.json()['resume_url']
    assert 'private@example.test' not in response.text
    with wm.warranty_db_session() as db:
        from warranty_notification_jobs import WarrantyNotificationJob
        assert db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()['resume_delivery_status'] == 'pending'
        assert db.query(WarrantyNotificationJob).filter_by(kind='resume', ticket_id=tid, status='pending').count() == 1


def test_invalid_resume_token_is_safe_when_signing_is_unavailable(monkeypatch):
    token = wr.create_resume_token('ticket', 'session')
    monkeypatch.delenv('ADMIN_SESSION_SECRET')
    assert wr.verify_resume_token(token) is None
    assert wr.verify_resume_token('x' * 5000 + '.signature') is None


def test_followup_capability_cannot_be_used_as_resume_token():
    from warranty_quality import issue_outcome_token
    assert wr.verify_resume_token(issue_outcome_token('ticket', 'session')) is None


def test_resume_rejects_validly_signed_ticket_session_mismatch(client):
    tid = _register_and_start(client, 'real-session')
    token = wr.create_resume_token(tid, 'another-session')
    assert client.get(f'/api/v1/warranty/resume/{token}').status_code == 400


def test_repeat_save_requests_do_not_queue_repeat_emails(client, monkeypatch):
    sid = "resume-repeat-limit"
    _register_and_start(client, sid)
    queued = []
    monkeypatch.setattr(wr, "_send_resume_email_async", lambda **kwargs: queued.append(kwargs))
    responses = [client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "buyer@example.test"}) for _ in range(35)]
    assert responses[0].status_code == 200
    assert len(queued) == 1
    assert all(response.status_code == 429 for response in responses[1:])
    assert all(int(response.headers["Retry-After"]) > 0 for response in responses[1:])
    changed_recipient = client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "different@example.test"})
    assert changed_recipient.status_code == 429
    assert len(queued) == 1


def test_recipient_cooldown_applies_across_sessions_and_clients(client, monkeypatch):
    for sid in ("recipient-session-a", "recipient-session-b"):
        _register_and_start(client, sid)
    first = client.post("/api/v1/warranty/session/recipient-session-a/resume-link", json={"customer_email": "same@example.test"})
    assert first.status_code == 200
    # A caller-supplied forwarding header cannot reset the server-observed IP.
    second = client.post("/api/v1/warranty/session/recipient-session-b/resume-link", json={"customer_email": "SAME@example.test"}, headers={"X-Forwarded-For": "203.0.113.9"})
    assert second.status_code == 429


def test_delivery_failure_can_retry_but_is_still_bounded(client, monkeypatch):
    sid = "resume-failure-retry-limit"
    tid = _register_and_start(client, sid)
    monkeypatch.setattr(wr, "_send_resume_email", lambda **kwargs: False)
    responses = []
    for _ in range(4):
        response = client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "buyer@example.test"})
        responses.append(response)
        if response.status_code == 200:
            with wm.warranty_db_session() as db:
                attempt_id = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_attempt_id"]
            assert wr._deliver_resume_email(attempt_id=attempt_id) is False
    assert [response.status_code for response in responses] == [200, 200, 200, 429]
    assert all(response.json()["queued"] is True for response in responses[:3])
    assert all(response.json()["resume_url"] for response in responses[:3])


def test_ip_limit_blocks_recipient_and_session_rotation(client):
    responses = []
    for number in range(7):
        sid = f"resume-ip-rotation-{number}"
        _register_and_start(client, sid)
        responses.append(client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": f"buyer{number}@example.test"}))
    assert [response.status_code for response in responses] == [200] * 6 + [429]


def test_durable_worker_uses_reserved_recipient_and_cannot_send_completed_attempt_twice(client, monkeypatch):
    sid = "resume-delivery-idempotent"
    tid = _register_and_start(client, sid)
    response = client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "reserved@example.test"})
    assert response.status_code == 200
    with wm.warranty_db_session() as db:
        attempt_id = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_attempt_id"]
    sent = []
    monkeypatch.setattr(wr, "_send_resume_email", lambda **kwargs: sent.append(kwargs) or True)
    wr.deliver_resume_attempt(attempt_id)
    wr.deliver_resume_attempt(attempt_id)
    assert len(sent) == 1
    assert sent[0]["customer_email"] == "reserved@example.test"
    assert wr.verify_resume_token(sent[0]["resume_url"].split("resume=", 1)[1])["sid"] == sid


def test_durable_worker_raises_on_transport_failure_for_queue_retry(client, monkeypatch):
    sid = "resume-delivery-fails"
    tid = _register_and_start(client, sid)
    client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "reserved@example.test"})
    with wm.warranty_db_session() as db:
        attempt_id = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_attempt_id"]
    monkeypatch.setattr(wr, "_send_resume_email", lambda **kwargs: False)
    with pytest.raises(RuntimeError, match="Resume email delivery failed"):
        wr.deliver_resume_attempt(attempt_id)


def test_resume_queue_survives_scheduling_without_a_running_worker(client, monkeypatch):
    from warranty_notification_jobs import WarrantyNotificationJob, enqueue_notification, process_notification

    sid = "resume-durable-queue"
    tid = _register_and_start(client, sid)
    client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "durable@example.test"})
    with wm.warranty_db_session() as db:
        attempt_id = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_attempt_id"]
    job_id = enqueue_notification(kind="resume", ticket_id=tid, source_key=attempt_id)
    with wm.warranty_db_session() as db:
        row = db.get(WarrantyNotificationJob, job_id)
        assert row.source_key == attempt_id
        assert row.evidence_id is None
        assert row.status == "pending"
        assert "durable@example.test" not in str(row.__dict__)
    delivered = []
    monkeypatch.setattr(wr, "_send_resume_email", lambda **kwargs: delivered.append(kwargs) or True)
    assert process_notification(job_id) is True
    assert process_notification(job_id) is False
    assert len(delivered) == 1
    assert delivered[0]["customer_email"] == "durable@example.test"


def test_atomic_rate_reservations_survive_concurrency_and_reconnection(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4
    from fastapi import HTTPException
    import warranty_resume_limits as limits

    database_url = f"sqlite:///{tmp_path / 'resume-admission.db'}"
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    wm.Base.metadata.create_all(engine)
    monkeypatch.setattr(wm, "_engine", engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    with engine.begin() as connection:
        connection.execute(wm.WarrantyTicket.__table__.insert().values(
            ticket_id="atomic-ticket", session_id="atomic-session", current_node_id="root", collected_data="{}"))
    def reserve(_):
        try:
            limits.reserve_resume_attempt(attempt_id=str(uuid4()), ticket_id="atomic-ticket",
                session_id="atomic-session", recipient="atomic@example.test", client_ip="192.0.2.1")
            return 200
        except HTTPException as exc:
            return exc.status_code
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(reserve, range(16)))
    assert statuses.count(200) == 1
    assert statuses.count(429) == 15
    engine.dispose()
    reconnected = create_engine(database_url)
    monkeypatch.setattr(wm, "_engine", reconnected)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=reconnected, expire_on_commit=False))
    assert reserve(None) == 429


def test_failed_queue_insert_rolls_back_resume_admission_and_saved_metadata(client):
    from sqlalchemy import event, text
    from sqlalchemy.exc import OperationalError
    from warranty_notification_jobs import WarrantyNotificationJob, initialize_notification_storage
    import warranty_resume_limits as limits

    sid = "resume-atomic-rollback"
    tid = _register_and_start(client, sid)
    initialize_notification_storage()
    with limits._transaction():
        pass
    with wm.warranty_db_session() as db:
        before = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()
    def reject_queue_insert(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().startswith("INSERT INTO warranty_notification_jobs"):
            raise OperationalError(statement, parameters, RuntimeError("synthetic queue failure"))
    event.listen(wm._engine, "before_cursor_execute", reject_queue_insert)
    try:
        response = client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "buyer@example.test"})
    finally:
        event.remove(wm._engine, "before_cursor_execute", reject_queue_insert)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    with wm.warranty_db_session() as db:
        assert db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected() == before
        assert db.query(WarrantyNotificationJob).filter_by(kind="resume", ticket_id=tid).count() == 0
        assert db.execute(text("SELECT COUNT(*) FROM warranty_resume_requests WHERE ticket_id=:tid"), {"tid": tid}).scalar() == 0
    assert client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "buyer@example.test"}).status_code == 200


def test_expired_resume_worker_cannot_regress_a_new_workers_success(client, monkeypatch):
    from types import SimpleNamespace
    from sqlalchemy import text
    import warranty_resume_limits as limits

    clock = [1000.0]
    monkeypatch.setattr(limits, "time", SimpleNamespace(time=lambda: clock[0]))
    sid = "resume-fenced-worker"
    tid = _register_and_start(client, sid)
    assert client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "buyer@example.test"}).status_code == 200
    with wm.warranty_db_session() as db:
        attempt_id = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_attempt_id"]
    old = limits.claim_resume_attempt(attempt_id)
    clock[0] += 601
    recovered = limits.claim_resume_attempt(attempt_id)
    assert old["lease_token"] != recovered["lease_token"]
    limits.finish_resume_attempt(attempt_id, sent=True, lease_token=recovered["lease_token"])
    limits.finish_resume_attempt(attempt_id, sent=False, lease_token=old["lease_token"])
    with wm.warranty_db_session() as db:
        assert db.execute(text("SELECT status FROM warranty_resume_requests WHERE attempt_id=:id"), {"id": attempt_id}).scalar() == "sent"
        assert db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_status"] == "sent"


def test_later_save_replaces_an_older_unsent_recipient_request(client, monkeypatch):
    from sqlalchemy import text

    sid = "resume-replaced-request"
    tid = _register_and_start(client, sid)
    assert client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "old@example.test"}).status_code == 200
    with wm.warranty_db_session() as db:
        old_attempt = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_attempt_id"]
    monkeypatch.setattr(wr, "_send_resume_email", lambda **kwargs: False)
    assert wr._deliver_resume_email(attempt_id=old_attempt) is False
    assert client.post(f"/api/v1/warranty/session/{sid}/resume-link", json={"customer_email": "new@example.test"}).status_code == 200
    with wm.warranty_db_session() as db:
        new_attempt = db.query(wm.WarrantyTicket).filter_by(ticket_id=tid).one().get_collected()["resume_delivery_attempt_id"]
    deliveries = []
    monkeypatch.setattr(wr, "_send_resume_email", lambda **kwargs: deliveries.append(kwargs) or True)
    wr.deliver_resume_attempt(old_attempt)
    wr.deliver_resume_attempt(new_attempt)
    assert len(deliveries) == 1
    assert deliveries[0]["customer_email"] == "new@example.test"
    with wm.warranty_db_session() as db:
        assert db.execute(text("SELECT status FROM warranty_resume_requests WHERE attempt_id=:id"), {"id": old_attempt}).scalar() == "superseded"
