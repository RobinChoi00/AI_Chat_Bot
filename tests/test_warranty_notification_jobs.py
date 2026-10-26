"""Restart and delivery regressions for reference-only warranty notifications."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import sys
import threading

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import warranty_models as wm
import warranty_notification_jobs as jobs
import warranty_email as email


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}", connect_args={"check_same_thread": False, "timeout": 5})
    wm.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(wm, "_SessionFactory", factory)
    clock = [datetime(2026, 10, 26, 12)]
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock[0])
    with wm.warranty_db_session() as db:
        ticket = wm.WarrantyTicket(ticket_id="synthetic-ticket", session_id="synthetic-session", current_node_id="terminal", issue_type="defect", model_name="Synthetic model", status="awaiting_admin_review")
        ticket.set_collected("case_summary", "Synthetic issue summary")
        db.add(ticket)
        evidence = wm.WarrantyEvidence(ticket_id="synthetic-ticket", evidence_type="damage_photos", file_path=str(tmp_path / "photo.jpg"), original_filename="photo.jpg", mime_type="image/jpeg", file_size_bytes=4, customer_email="synthetic@example.test")
        db.add(evidence)
        db.flush()
        evidence_id = int(evidence.id)
    yield engine, clock, evidence_id
    engine.dispose()


def _enqueue(evidence_id, kind="evidence"):
    return jobs.enqueue_notification(kind=kind, evidence_id=evidence_id, ticket_id="synthetic-ticket")


def test_schedule_persists_references_and_deduplicates_without_sending(database, monkeypatch):
    _engine, _clock, evidence_id = database
    send = []
    monkeypatch.setattr(email, "send_evidence_upload_notification", lambda **kwargs: send.append(kwargs))
    email.notify_evidence_upload_async(evidence_id=evidence_id, ticket_id="synthetic-ticket", customer_email="synthetic@example.test", evidence_type="damage_photos", original_filename="photo.jpg", file_path="ignored", mime_type="image/jpeg", file_size_bytes=4)
    assert _enqueue(evidence_id) == _enqueue(evidence_id)
    with wm.warranty_db_session() as db:
        assert db.query(jobs.WarrantyNotificationJob).count() == 1
        job = db.query(jobs.WarrantyNotificationJob).one()
        assert job.status == "pending"
        assert not any("email" in column.name or "payload" in column.name or "password" in column.name for column in job.__table__.columns)
        assert db.get(wm.WarrantyEvidence, evidence_id).emailed == 0
    assert send == []


@pytest.mark.parametrize("kind,sender_name", [("evidence", "send_evidence_upload_notification"), ("contact", "send_email_only_contact_notification")])
def test_failed_delivery_retries_after_due_time_and_records_only_confirmed_send(database, monkeypatch, kind, sender_name):
    _engine, clock, evidence_id = database
    outcomes = iter([False, True])
    calls = []
    def sender(**kwargs):
        calls.append(kwargs)
        return next(outcomes)
    monkeypatch.setattr(email, sender_name, sender)
    job_id = _enqueue(evidence_id, kind)
    assert jobs.process_pending_notifications() == 0
    with wm.warranty_db_session() as db:
        assert db.get(jobs.WarrantyNotificationJob, job_id).status == "failed"
        assert db.get(wm.WarrantyEvidence, evidence_id).emailed == 0
    assert jobs.process_pending_notifications() == 0
    assert len(calls) == 1
    clock[0] += timedelta(seconds=11)
    assert jobs.process_pending_notifications() == 1
    with wm.warranty_db_session() as db:
        row = db.get(jobs.WarrantyNotificationJob, job_id)
        assert (row.status, row.attempts) == ("completed", 2)
        assert db.get(wm.WarrantyEvidence, evidence_id).emailed == 1
    assert calls[-1]["customer_email"] == "synthetic@example.test"
    assert calls[-1]["case_summary"] == "Synthetic issue summary"
    assert jobs.process_notification(job_id) is False
    assert len(calls) == 2


def test_unclaimed_job_survives_new_session_factory(database, monkeypatch):
    engine, _clock, evidence_id = database
    job_id = _enqueue(evidence_id)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(email, "send_evidence_upload_notification", lambda **kwargs: True)
    assert jobs.process_pending_notifications() == 1
    with wm.warranty_db_session() as db:
        assert db.get(jobs.WarrantyNotificationJob, job_id).status == "completed"


def test_source_update_and_job_share_commit_or_rollback(database):
    _engine, _clock, evidence_id = database
    jobs.initialize_notification_storage()
    with pytest.raises(RuntimeError):
        with wm.warranty_db_session() as db:
            ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id="synthetic-ticket").one()
            ticket.status = "resolved"
            jobs.enqueue_notification(kind="evidence", evidence_id=evidence_id, ticket_id="synthetic-ticket", db=db)
            raise RuntimeError("interrupted before commit")
    with wm.warranty_db_session() as db:
        assert db.query(jobs.WarrantyNotificationJob).count() == 0
        assert db.query(wm.WarrantyTicket).one().status == "awaiting_admin_review"
    with wm.warranty_db_session() as db:
        jobs.enqueue_notification(kind="evidence", evidence_id=evidence_id, ticket_id="synthetic-ticket", db=db)
    assert jobs.notification_job_stats()["pending"] == 1


def test_periodic_worker_recovers_expired_lease_and_ignores_old_acknowledgment(database, monkeypatch):
    _engine, clock, evidence_id = database
    job_id = _enqueue(evidence_id)
    first = jobs._claim_notification(job_id)
    assert first is not None
    monkeypatch.setattr(email, "send_evidence_upload_notification", lambda **kwargs: True)
    assert jobs.process_pending_notifications() == 0
    clock[0] += timedelta(seconds=jobs.NOTIFICATION_LEASE_SECONDS + 1)
    assert jobs.process_pending_notifications() == 1
    assert jobs._finish_notification(first, sent=False, error="late_failure") is False
    with wm.warranty_db_session() as db:
        assert db.get(jobs.WarrantyNotificationJob, job_id).status == "completed"
        assert db.get(jobs.WarrantyNotificationJob, job_id).attempts == 2


def test_last_interrupted_attempt_becomes_dead_letter_and_operator_can_retry(database, monkeypatch):
    _engine, clock, evidence_id = database
    job_id = _enqueue(evidence_id)
    claim = jobs._claim_notification(job_id)
    with wm.warranty_db_session() as db:
        db.get(jobs.WarrantyNotificationJob, job_id).attempts = jobs.MAX_NOTIFICATION_ATTEMPTS
    clock[0] += timedelta(seconds=jobs.NOTIFICATION_LEASE_SECONDS + 1)
    monkeypatch.setattr(email, "send_evidence_upload_notification", lambda **kwargs: True)
    assert jobs.process_pending_notifications() == 0
    assert jobs.notification_job_stats()["dead_letter"] == 1
    assert jobs.retry_notification(job_id) is True
    assert jobs._finish_notification(claim, sent=True) is False
    assert jobs.process_pending_notifications() == 1
    assert jobs.retry_notification(job_id) is False


def test_active_job_cannot_be_claimed_or_retried_twice(database, monkeypatch):
    _engine, _clock, evidence_id = database
    job_id = _enqueue(evidence_id)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def sender(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(5)
        return True
    monkeypatch.setattr(email, "send_evidence_upload_notification", sender)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(jobs.process_notification, job_id)
        assert entered.wait(5)
        assert pool.submit(jobs.process_notification, job_id).result(timeout=5) is False
        assert jobs.retry_notification(job_id) is False
        release.set()
        assert first.result(timeout=5) is True
    assert len(calls) == 1


def test_removed_source_does_not_send_and_is_visible_as_dead_letter(database, monkeypatch):
    _engine, _clock, evidence_id = database
    job_id = _enqueue(evidence_id)
    monkeypatch.setattr(email, "send_evidence_upload_notification", lambda **kwargs: pytest.fail("deleted evidence cannot be sent"))
    with wm.warranty_db_session() as db:
        db.delete(db.get(wm.WarrantyEvidence, evidence_id))
    assert jobs.process_notification(job_id) is False
    with wm.warranty_db_session() as db:
        row = db.get(jobs.WarrantyNotificationJob, job_id)
        assert (row.status, row.last_error) == ("dead_letter", "source_unavailable")


def test_resume_queue_stores_attempt_reference_and_uses_resolver(database, monkeypatch):
    import warranty_resume
    calls = []
    monkeypatch.setattr(warranty_resume, "deliver_resume_attempt", lambda attempt: calls.append(attempt), raising=False)
    job_id = jobs.enqueue_notification(kind="resume", ticket_id="synthetic-ticket", source_key="synthetic-attempt")
    assert jobs.process_notification(job_id) is True
    assert calls == ["synthetic-attempt"]
    with wm.warranty_db_session() as db:
        row = db.get(jobs.WarrantyNotificationJob, job_id)
        assert row.evidence_id is None
        assert row.source_key == "synthetic-attempt"


def test_handoff_notification_retries_unconfirmed_submission(database, monkeypatch):
    import sys
    from types import SimpleNamespace
    _engine, clock, _evidence_id = database
    calls = []
    outcomes = iter([False, True])
    def sender(ticket_id):
        calls.append(ticket_id)
        return next(outcomes)
    monkeypatch.setitem(sys.modules, "warranty_handoff_notifications", SimpleNamespace(send_handoff_notification=sender))
    job_id = jobs.enqueue_notification(kind="handoff", ticket_id="synthetic-ticket", source_key="synthetic-request")
    assert jobs.process_notification(job_id) is False
    clock[0] += timedelta(seconds=11)
    assert jobs.process_pending_notifications() == 1
    assert calls == ["synthetic-ticket", "synthetic-ticket"]


def test_provider_exception_details_are_not_copied_to_job_or_log(database, monkeypatch, caplog):
    _engine, _clock, evidence_id = database
    def sender(**kwargs):
        raise OSError("synthetic@example.test password=not-a-real-secret")
    monkeypatch.setattr(email, "send_evidence_upload_notification", sender)
    job_id = _enqueue(evidence_id)
    assert jobs.process_notification(job_id) is False
    with wm.warranty_db_session() as db:
        assert db.get(jobs.WarrantyNotificationJob, job_id).last_error == "OSError"
    assert "synthetic@example.test" not in caplog.text
    assert "not-a-real-secret" not in caplog.text
