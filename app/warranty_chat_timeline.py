"""
Append-only chat timeline on warranty tickets (collected_data).

Stores non-flowchart events (side questions, enrichment tips, clarifies)
so Admin can review a richer conversation without a new DB table.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

_TIMELINE_KEY = "chat_timeline"
_MAX_EVENTS = 80


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_chat_timeline(ticket) -> list[dict[str, Any]]:
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    raw = collected.get(_TIMELINE_KEY) if isinstance(collected, dict) else None
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict)]


def append_chat_event(
    ticket,
    *,
    role: str,
    kind: str,
    text: str,
    node_id: str = "",
    persist: bool = True,
) -> list[dict[str, Any]]:
    """
    Append one timeline event onto ``ticket.collected_data['chat_timeline']``.

    Caps at ``_MAX_EVENTS``. Dedupes consecutive identical events.
    """
    body = (text or "").strip()
    if not body:
        return get_chat_timeline(ticket)

    ticket_id = str(getattr(ticket, "ticket_id", "") or "")
    event = {
        "role": role,
        "kind": kind,
        "text": body[:2000],
        "node_id": (node_id or "").strip(),
        "created_at": _now_iso(),
    }

    if not persist or not ticket_id:
        timeline = get_chat_timeline(ticket)
        if timeline and _same_event(timeline[-1], event):
            return timeline
        timeline = (timeline + [event])[-_MAX_EVENTS:]
        if hasattr(ticket, "set_collected"):
            ticket.set_collected(_TIMELINE_KEY, timeline)  # type: ignore[arg-type]
        return timeline

    try:
        from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            row = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if row is None:
                return get_chat_timeline(ticket)
            collected = row.get_collected()
            timeline = list(collected.get(_TIMELINE_KEY) or [])
            if not isinstance(timeline, list):
                timeline = []
            if timeline and isinstance(timeline[-1], dict) and _same_event(timeline[-1], event):
                return timeline
            timeline.append(event)
            timeline = timeline[-_MAX_EVENTS:]
            collected[_TIMELINE_KEY] = timeline
            row.collected_data = __import__("json").dumps(collected)
            db.commit()
            # Keep the caller's detached ticket in sync when possible.
            if hasattr(ticket, "set_collected"):
                ticket.set_collected(_TIMELINE_KEY, timeline)  # type: ignore[arg-type]
            return timeline
    except Exception:
        return get_chat_timeline(ticket)


def _same_event(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("role") == right.get("role")
        and left.get("kind") == right.get("kind")
        and left.get("node_id") == right.get("node_id")
        and left.get("text") == right.get("text")
    )
