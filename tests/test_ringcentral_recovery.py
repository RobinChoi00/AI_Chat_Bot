"""Crash recovery uses the same periodic worker path as production."""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import ringcentral_store as store
import warranty_models as wm


@pytest.fixture
def database(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    clock = [datetime(2026, 10, 26, 12)]
    monkeypatch.setattr(store, "_utcnow", lambda: clock[0])
    yield clock
    engine.dispose()


def test_periodic_worker_reclaims_callback_interrupted_before_restart(database):
    event_id, _ = store.enqueue_event("on-call-enter", {"sessionId": "synthetic-session"})
    original = store._claim_event(event_id)
    assert original is not None
    assert store.pending_event_ids() == []
    database[0] += timedelta(minutes=11)
    assert store.pending_event_ids() == [event_id]
    handled = []
    assert store.process_pending_events({"on-call-enter": lambda payload: handled.append(payload)}) == 1
    assert len(handled) == 1
    with wm.warranty_db_session() as db:
        row = db.get(wm.RingCentralWebhookEvent, event_id)
        assert (row.status, row.attempts) == ("completed", 2)
    assert store.process_pending_events({"on-call-enter": lambda payload: pytest.fail("callback already completed")}) == 0


def test_exhausted_interrupted_callback_becomes_visible_dead_letter(database):
    event_id, _ = store.enqueue_event("on-call-enter", {"sessionId": "synthetic-exhausted"})
    assert store._claim_event(event_id)
    with wm.warranty_db_session() as db:
        db.query(wm.RingCentralWebhookEvent).filter_by(id=event_id).update({
            wm.RingCentralWebhookEvent.attempts: store.MAX_EVENT_ATTEMPTS,
            wm.RingCentralWebhookEvent.updated_at: database[0],
        })
    database[0] += timedelta(minutes=11)
    assert store.process_pending_events({"on-call-enter": lambda payload: pytest.fail("attempt limit exceeded")}) == 0
    assert store.event_stats()["dead_letter"] == 1
    assert store.pending_event_ids() == []


def test_fresh_lease_cannot_be_claimed_by_duplicate_delivery(database):
    event_id, _ = store.enqueue_event("on-call-exit", {"sessionId": "synthetic-live"})
    assert store._claim_event(event_id)
    assert store._claim_event(event_id) is None
    assert store.process_event(event_id, {"on-call-exit": lambda payload: pytest.fail("worker still owns lease")}) is False


def test_late_worker_acknowledgment_cannot_overwrite_newer_claim(database):
    event_id, _ = store.enqueue_event("on-call-exit", {"sessionId": "synthetic-fenced"})
    first = store._claim_event(event_id)
    database[0] += timedelta(minutes=11)
    second = store._claim_event(event_id)
    assert first is not None and second is not None
    assert store._mark_completed(event_id, first[2]) is False
    store._mark_failed(event_id, first[2], RuntimeError("late"))
    with wm.warranty_db_session() as db:
        row = db.get(wm.RingCentralWebhookEvent, event_id)
        assert (row.status, row.attempts) == ("processing", 2)
    assert store._mark_completed(event_id, second[2]) is True
    assert store.event_stats()["processing"] == 0


def test_callback_failure_keeps_sensitive_exception_details_out_of_storage_and_logs(database, caplog):
    event_id, _ = store.enqueue_event("on-call-exit", {"sessionId": "synthetic-sensitive"})
    def fail(payload):
        raise RuntimeError("synthetic@example.test credential=not-a-secret")
    assert store.process_event(event_id, {"on-call-exit": fail}) is False
    with wm.warranty_db_session() as db:
        assert db.get(wm.RingCentralWebhookEvent, event_id).last_error == "RuntimeError"
    assert "synthetic@example.test" not in caplog.text
    assert "not-a-secret" not in caplog.text
