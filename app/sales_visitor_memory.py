"""
Remember a returning Tidio shopper across conversations.

Tidio often opens a new ``session_id`` per chat while ``contact_id`` stays
stable. Prefs lived on the session, so the same visitor was asked height
again. This module copies the last chair / fit answers onto the new session
so greeting and recommend can pick up where they left off.

Nothing here invents a product, price, or date — it only copies values the
shopper already produced in a prior turn.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from warranty_models import _now_cst, warranty_db_session

# Keys that are useful on a later visit. Skip handoff/email-wait flags so a
# new chat does not inherit a half-finished transfer.
RESUME_COPY_KEYS = (
    "recommend_prefs",
    "pending_primary",
    "pending_product_url",
    "pending_pick_summary",
    "pending_tier_picks",
)

_MAX_AGE_DAYS = 30


def has_resume_memory(prefs: Optional[dict]) -> bool:
    """True when we can greet with a last chair or a finished height+goal."""
    data = prefs or {}
    if str(data.get("pending_primary") or "").strip():
        return True
    rec = data.get("recommend_prefs") or {}
    if not isinstance(rec, dict):
        return False
    return bool(str(rec.get("height") or "").strip() and str(rec.get("goal") or "").strip())


def _session_is_empty_of_memory(prefs: Optional[dict]) -> bool:
    data = prefs or {}
    if str(data.get("pending_primary") or "").strip():
        return False
    rec = data.get("recommend_prefs") or {}
    if isinstance(rec, dict) and any(str(v or "").strip() for v in rec.values()):
        return False
    return True


def find_prior_collected(
    visitor_id: str,
    *,
    exclude_session_id: str,
    max_age_days: int = _MAX_AGE_DAYS,
) -> Optional[dict[str, Any]]:
    """Latest other session for this Tidio contact that still has resume memory."""
    vid = (visitor_id or "").strip()
    if not vid:
        return None
    cutoff = _now_cst() - timedelta(days=max(1, int(max_age_days)))
    from sales_models import SalesSession  # noqa: WPS433

    with warranty_db_session() as db:
        rows = (
            db.query(SalesSession)
            .filter(
                SalesSession.tidio_visitor_id == vid,
                SalesSession.session_id != exclude_session_id,
                SalesSession.updated_at >= cutoff,
            )
            .order_by(SalesSession.updated_at.desc())
            .limit(12)
            .all()
        )
        for row in rows:
            collected = row.get_collected()
            if has_resume_memory(collected):
                return collected
    return None


def snapshot_from_prefs(prefs: Optional[dict]) -> dict[str, Any]:
    """Subset of prefs safe to copy onto a later session."""
    data = prefs or {}
    out: dict[str, Any] = {}
    for key in RESUME_COPY_KEYS:
        if key not in data:
            continue
        value = data[key]
        if value in (None, "", [], {}):
            continue
        out[key] = value
    return out


def hydrate_visitor_memory(
    session_id: str,
    visitor_id: Optional[str],
    *,
    max_age_days: int = _MAX_AGE_DAYS,
) -> dict:
    """
    Copy last-chair memory onto ``session_id`` when this chat is empty.

    Returns the session's collected_data after the merge. No-op when the
    visitor id is missing or this session already has fit/product prefs.
    """
    from sales_models import get_session_collected, merge_session_collected  # noqa: WPS433

    current = get_session_collected(session_id)
    if current.get("visitor_resume_declined"):
        return current
    if not _session_is_empty_of_memory(current):
        return current
    prior = find_prior_collected(
        visitor_id or "",
        exclude_session_id=session_id,
        max_age_days=max_age_days,
    )
    if not prior:
        return current
    patch = snapshot_from_prefs(prior)
    if not patch:
        return current
    patch["visitor_memory_hydrated"] = True
    return merge_session_collected(session_id, patch)
