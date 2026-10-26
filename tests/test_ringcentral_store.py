"""Durable RingCentral webhook inbox tests."""

import sys
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import ringcentral_store  # noqa: E402
import warranty_models  # noqa: E402
from ringcentral_store import (  # noqa: E402
    MAX_EVENT_ATTEMPTS,
    enqueue_event,
    pending_event_ids,
    process_event,
    process_pending_events,
)
from warranty_models import RingCentralWebhookEvent, warranty_db_session  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_event_database(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'ringcentral.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    warranty_models.Base.metadata.create_all(engine)
    monkeypatch.setattr(
        warranty_models,
        "_SessionFactory",
        sessionmaker(bind=engine, expire_on_commit=False),
    )
    yield
    engine.dispose()


def _processing_event(*, age_minutes: int, attempts: int = 1) -> int:
    payload = {"sessionId": f"s-{uuid.uuid4().hex}", "partyId": "p-1"}
    event_id, _ = enqueue_event("on-call-enter", payload)
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        row.status = "processing"
        row.attempts = attempts
        row.updated_at = datetime.utcnow() - timedelta(minutes=age_minutes)
    return event_id


def test_exact_duplicate_is_enqueued_once():
    payload = {"sessionId": f"s-{uuid.uuid4().hex}", "partyId": "p-1"}
    first_id, first_created = enqueue_event("on-call-enter", payload)
    second_id, second_created = enqueue_event("on-call-enter", payload)
    assert first_created is True
    assert second_created is False
    assert second_id == first_id
    assert process_event(first_id, {"on-call-enter": lambda _payload: None}) is True


def test_failed_event_is_persisted_and_can_retry():
    payload = {"sessionId": f"s-{uuid.uuid4().hex}", "partyId": "p-1"}
    event_id, _ = enqueue_event("on-call-enter", payload)
    calls = {"count": 0}

    def flaky(_payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary")

    assert process_event(event_id, {"on-call-enter": flaky}) is False
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        assert row is not None
        assert row.status == "failed"
        row.next_attempt_at = None

    assert process_event(event_id, {"on-call-enter": flaky}) is True
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        assert row is not None
        assert row.status == "completed"
        assert row.attempts == 2


def test_worker_recovers_expired_processing_claim_without_webhook_redelivery():
    event_id = _processing_event(age_minutes=11)
    calls = []

    assert pending_event_ids() == [event_id]
    assert process_pending_events({"on-call-enter": calls.append}) == 1

    assert len(calls) == 1
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        assert row.status == "completed"
        assert row.attempts == 2
        assert row.last_error is None
    assert pending_event_ids() == []


def test_worker_and_direct_delivery_do_not_reclaim_active_processing_lease():
    event_id = _processing_event(age_minutes=5)
    calls = []
    handlers = {"on-call-enter": calls.append}

    assert pending_event_ids() == []
    assert process_pending_events(handlers) == 0
    assert process_event(event_id, handlers) is False

    assert calls == []
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        assert row.status == "processing"
        assert row.attempts == 1
        assert row.last_error is None


def test_worker_dead_letters_expired_processing_claim_at_attempt_limit():
    event_id = _processing_event(age_minutes=11, attempts=MAX_EVENT_ATTEMPTS)
    calls = []

    assert pending_event_ids() == [event_id]
    assert process_pending_events({"on-call-enter": calls.append}) == 0

    assert calls == []
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        assert row.status == "dead_letter"
        assert row.attempts == MAX_EVENT_ATTEMPTS
        assert row.next_attempt_at is None
        assert row.last_error == "stale_processing_claim"
    assert pending_event_ids() == []


def test_concurrent_workers_recover_expired_claim_only_once(monkeypatch):
    event_id = _processing_event(age_minutes=11)
    selected = threading.Barrier(2)
    handler_started = threading.Event()
    release_handler = threading.Event()
    calls = []
    original_pending = pending_event_ids

    def select_same_candidate(limit=25):
        candidates = original_pending(limit)
        assert candidates == [event_id]
        selected.wait(timeout=5)
        return candidates

    def handle(payload):
        calls.append(payload)
        handler_started.set()
        assert release_handler.wait(timeout=5)

    monkeypatch.setattr(ringcentral_store, "pending_event_ids", select_same_candidate)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(process_pending_events, {"on-call-enter": handle})
            for _ in range(2)
        ]
        try:
            assert handler_started.wait(timeout=5)
            completed, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            assert len(completed) == 1
            assert next(iter(completed)).result() == 0
        finally:
            release_handler.set()
        assert sorted(future.result(timeout=5) for future in futures) == [0, 1]

    assert len(calls) == 1
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        assert row.status == "completed"
        assert row.attempts == 2
