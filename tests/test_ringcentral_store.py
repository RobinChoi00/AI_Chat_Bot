"""Durable RingCentral webhook inbox tests."""

import sys
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_store import enqueue_event, process_event  # noqa: E402
from warranty_models import RingCentralWebhookEvent, warranty_db_session  # noqa: E402


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
