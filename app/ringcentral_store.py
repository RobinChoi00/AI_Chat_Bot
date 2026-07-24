"""Durable RingCentral webhook inbox and retry processing.

Webhook routes persist an event before acknowledging it.  Handlers then claim
rows from SQLite, so duplicate deliveries do not repeat side effects and a
process restart does not lose callbacks that were already accepted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional

from sqlalchemy import or_, text

from warranty_models import (
    RingCentralCallState,
    RingCentralWebhookEvent,
    warranty_db_session,
)

logger = logging.getLogger(__name__)

MAX_EVENT_ATTEMPTS = 8
_RETRY_DELAYS_SECONDS = (1, 2, 5, 10, 30, 60, 120, 300)


def _utcnow() -> datetime:
    return datetime.utcnow()


def event_key(route: str, payload: Mapping[str, Any]) -> str:
    """Return a stable idempotency key for an exact callback delivery."""
    canonical = json.dumps(
        {"route": route, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def enqueue_event(route: str, payload: dict[str, Any]) -> tuple[int, bool]:
    """Persist a callback and return ``(row_id, was_created)``."""
    key = event_key(route, payload)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    session_id = str(payload.get("sessionId") or "")[:255]
    with warranty_db_session() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        existing = (
            db.query(RingCentralWebhookEvent)
            .filter(RingCentralWebhookEvent.event_key == key)
            .first()
        )
        if existing is not None:
            return int(existing.id), False
        row = RingCentralWebhookEvent(
            event_key=key,
            route=route,
            session_id=session_id or None,
            payload_json=body,
            status="pending",
            attempts=0,
        )
        db.add(row)
        db.flush()
        return int(row.id), True


def _claim_event(event_id: int) -> Optional[tuple[str, dict[str, Any]]]:
    now = _utcnow()
    stale_before = now - timedelta(minutes=10)
    with warranty_db_session() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        # A worker can die after setting ``processing``. Reclaim that row after
        # a conservative timeout instead of leaving it stuck forever.
        db.query(RingCentralWebhookEvent).filter(
            RingCentralWebhookEvent.id == event_id,
            RingCentralWebhookEvent.status == "processing",
            RingCentralWebhookEvent.updated_at < stale_before,
        ).update(
            {
                RingCentralWebhookEvent.status: "failed",
                RingCentralWebhookEvent.next_attempt_at: now,
                RingCentralWebhookEvent.last_error: "stale_processing_claim",
            },
            synchronize_session=False,
        )
        row = (
            db.query(RingCentralWebhookEvent)
            .filter(RingCentralWebhookEvent.id == event_id)
            .first()
        )
        if row is None or row.status in {"completed", "processing", "dead_letter"}:
            return None
        if row.attempts >= MAX_EVENT_ATTEMPTS:
            row.status = "dead_letter"
            return None
        if row.next_attempt_at is not None and row.next_attempt_at > now:
            return None
        row.status = "processing"
        row.attempts = int(row.attempts or 0) + 1
        row.next_attempt_at = None
        route = str(row.route)
        try:
            payload = json.loads(str(row.payload_json))
        except (TypeError, ValueError) as exc:
            row.status = "dead_letter"
            row.last_error = f"invalid_stored_json:{type(exc).__name__}"[:500]
            return None
        if not isinstance(payload, dict):
            row.status = "dead_letter"
            row.last_error = "stored_payload_not_object"
            return None
        return route, payload


def _mark_completed(event_id: int) -> None:
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        if row is not None:
            row.status = "completed"
            row.last_error = None
            row.completed_at = _utcnow()


def _mark_failed(event_id: int, exc: BaseException) -> None:
    with warranty_db_session() as db:
        row = db.get(RingCentralWebhookEvent, event_id)
        if row is None:
            return
        attempts = int(row.attempts or 1)
        row.last_error = f"{type(exc).__name__}: {exc}"[:500]
        if attempts >= MAX_EVENT_ATTEMPTS:
            row.status = "dead_letter"
            row.next_attempt_at = None
        else:
            row.status = "failed"
            delay = _RETRY_DELAYS_SECONDS[min(attempts - 1, len(_RETRY_DELAYS_SECONDS) - 1)]
            row.next_attempt_at = _utcnow() + timedelta(seconds=delay)


def process_event(
    event_id: int,
    handlers: Mapping[str, Callable[[dict[str, Any]], None]],
) -> bool:
    """Claim and process one event. Returns True only on completion."""
    claimed = _claim_event(event_id)
    if claimed is None:
        return False
    route, payload = claimed
    handler = handlers.get(route)
    if handler is None:
        exc = RuntimeError(f"No RingCentral handler registered for {route}")
        _mark_failed(event_id, exc)
        return False
    try:
        handler(payload)
    except Exception as exc:
        logger.exception(
            "RingCentral event processing failed event_id=%s route=%s",
            event_id,
            route,
        )
        _mark_failed(event_id, exc)
        return False
    _mark_completed(event_id)
    return True


def pending_event_ids(limit: int = 25) -> list[int]:
    now = _utcnow()
    with warranty_db_session() as db:
        rows = (
            db.query(RingCentralWebhookEvent.id)
            .filter(
                RingCentralWebhookEvent.status.in_(("pending", "failed")),
                RingCentralWebhookEvent.attempts < MAX_EVENT_ATTEMPTS,
                or_(
                    RingCentralWebhookEvent.next_attempt_at.is_(None),
                    RingCentralWebhookEvent.next_attempt_at <= now,
                ),
            )
            .order_by(RingCentralWebhookEvent.id.asc())
            .limit(max(1, min(limit, 100)))
            .all()
        )
        return [int(row[0]) for row in rows]


def release_session_retries(session_id: str) -> None:
    """Make dependency failures for a newly-created call state retry now."""
    if not session_id:
        return
    with warranty_db_session() as db:
        db.query(RingCentralWebhookEvent).filter(
            RingCentralWebhookEvent.session_id == session_id,
            RingCentralWebhookEvent.status == "failed",
            RingCentralWebhookEvent.attempts < MAX_EVENT_ATTEMPTS,
        ).update(
            {RingCentralWebhookEvent.next_attempt_at: _utcnow()},
            synchronize_session=False,
        )


def process_pending_events(
    handlers: Mapping[str, Callable[[dict[str, Any]], None]],
    *,
    limit: int = 25,
) -> int:
    completed = 0
    for row_id in pending_event_ids(limit=limit):
        if process_event(row_id, handlers):
            completed += 1
    return completed


def event_stats() -> dict[str, int]:
    stats = {"pending": 0, "processing": 0, "failed": 0, "dead_letter": 0}
    with warranty_db_session() as db:
        rows = (
            db.query(RingCentralWebhookEvent.status, RingCentralWebhookEvent.id)
            .filter(RingCentralWebhookEvent.status != "completed")
            .all()
        )
    for status, _row_id in rows:
        key = str(status)
        stats[key] = stats.get(key, 0) + 1
    return stats


def last_webhook_received_at() -> Optional[str]:
    """ISO timestamp of the newest inbound RC webhook (any status), or None."""
    with warranty_db_session() as db:
        row = (
            db.query(RingCentralWebhookEvent)
            .order_by(RingCentralWebhookEvent.created_at.desc())
            .first()
        )
        if row is None or row.created_at is None:
            return None
        created = row.created_at
        if getattr(created, "tzinfo", None) is None:
            return f"{created.isoformat()}Z"
        return created.isoformat()


def call_state_stats() -> dict[str, int]:
    stale_before = _utcnow() - timedelta(hours=24)
    with warranty_db_session() as db:
        active = db.query(RingCentralCallState).count()
        stale = (
            db.query(RingCentralCallState)
            .filter(RingCentralCallState.updated_at < stale_before)
            .count()
        )
    return {"active": int(active), "stale_over_24h": int(stale)}


def cleanup_completed_events(retention_days: int = 30) -> int:
    cutoff = _utcnow() - timedelta(days=max(7, min(retention_days, 365)))
    with warranty_db_session() as db:
        return int(
            db.query(RingCentralWebhookEvent)
            .filter(
                RingCentralWebhookEvent.status == "completed",
                RingCentralWebhookEvent.completed_at < cutoff,
            )
            .delete(synchronize_session=False)
        )
