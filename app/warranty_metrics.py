"""
warranty_metrics.py
===================
Admin-only funnel + completion-rate aggregation over the warranty ticket log.

Design contract
---------------
- ZERO LLM calls.
- Read-only: queries the same SQLite database used by ``warranty_workflow.py``.
- Reuses ``WarrantyEngine.get_ticket / get_tickets`` semantics, but goes
  straight to the ORM so we can efficiently ``GROUP BY`` without loading every
  ticket into memory.
- Returns plain-JSON dicts — the frontend (`/admin/warranty/dashboard`) renders
  them with a small server component.

Endpoint
--------
    GET /admin/warranty/metrics?days=30
        ⇒ Overall funnel + status/issue_type/domain/terminal breakdowns +
          daily started-count trend + contact submission rate.

Definitions
-----------
- **Started**            – any ticket created in the window.
- **Reached terminal**   – ``status != "in_progress"``. The workflow always
                           bumps status *away* from ``in_progress`` when a
                           terminal node is hit (see ``warranty_workflow.py``
                           terminal handling).
- **Contact captured**   – ticket.collected_data has a ``customer_email``.
- **Admin decided**      – ticket.admin_decision is set.
- **Resolved**           – ticket.status == "resolved" (admin finalised).
- **Abandoned**          – status == "in_progress" AND updated_at older than
                           ``ABANDON_THRESHOLD_HOURS`` (default 6 h).

All timestamps are stored in America/Chicago tz (see ``_now_cst``); we filter
by ``created_at`` cast to the local tz for consistent "day" bucketing.
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytz
from fastapi import APIRouter, Header, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin-warranty-metrics"])

_CST = pytz.timezone("America/Chicago")
_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
_ABANDON_THRESHOLD_HOURS = int(os.getenv("WARRANTY_ABANDON_HOURS", "6"))
_MAX_DAYS = 180


def _require_admin(x_admin_key: Optional[str]) -> None:
    if not _ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Admin API is not configured. Set ADMIN_API_KEY.",
        )
    if x_admin_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key.")


def _now_cst() -> datetime:
    return datetime.now(_CST)


def _to_cst(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        # Legacy rows in SQLite may be naive; assume they were written as CST.
        return _CST.localize(value)
    return value.astimezone(_CST)


def _iso_day(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _lazy_orm():
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    from warranty_models import (  # type: ignore  # noqa: WPS433
        WarrantyTicket,
        WarrantyTurn,
        warranty_db_session,
    )

    return WarrantyTicket, WarrantyTurn, warranty_db_session


def _percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100.0, 1)


@router.get("/admin/warranty/metrics")
async def warranty_metrics(
    days: int = Query(30, ge=1, le=_MAX_DAYS),
    domain: Optional[str] = None,
    x_admin_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """
    Return an aggregated snapshot of the warranty funnel for the last ``days``
    days (inclusive of today).

    Response shape
    --------------
    {
      "range": {"days": 30, "start": "...", "end": "..."},
      "totals": {
        "started": 128,
        "reached_terminal": 94,
        "completion_rate_pct": 73.4,
        "contact_captured": 41,
        "contact_rate_pct": 32.0,
        "admin_decided": 22,
        "resolved": 18,
        "abandoned": 12
      },
      "by_status":     [{"status": "resolved", "count": 18}, ...],
      "by_issue_type": [{"issue_type": "defect", "count": 87, ...}, ...],
      "by_domain":     [{"domain": "osakichair.com", "count": 60, ...}, ...],
      "top_terminals": [{"node_id": "...", "count": 12}, ...],
      "daily_started": [{"day": "2026-06-01", "started": 3, "completed": 2}, ...],
      "median_turns_to_terminal": 4
    }
    """
    _require_admin(x_admin_key)

    WarrantyTicket, WarrantyTurn, warranty_db_session = _lazy_orm()

    end = _now_cst()
    start = (end - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    abandon_cutoff = end - timedelta(hours=_ABANDON_THRESHOLD_HOURS)

    with warranty_db_session() as db:
        q = db.query(WarrantyTicket).filter(WarrantyTicket.created_at >= start)
        if domain:
            q = q.filter(WarrantyTicket.domain.contains(domain))
        tickets = q.all()

        # Preload turn counts per ticket in a single query.
        turn_counts: dict[str, int] = {}
        terminal_turns: list[int] = []
        if tickets:
            ids = [str(t.ticket_id) for t in tickets]
            rows = (
                db.query(WarrantyTurn.ticket_id)
                .filter(WarrantyTurn.ticket_id.in_(ids))
                .all()
            )
            for (tid,) in rows:
                turn_counts[tid] = turn_counts.get(tid, 0) + 1

    started = len(tickets)
    reached_terminal = 0
    contact_captured = 0
    admin_decided = 0
    resolved = 0
    abandoned = 0

    status_counts: Counter[str] = Counter()
    issue_totals: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "completed": 0}
    )
    domain_totals: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "completed": 0}
    )
    terminal_counts: Counter[str] = Counter()
    daily: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"started": 0, "completed": 0}
    )

    for t in tickets:
        status = (t.status or "in_progress").lower()
        issue_type = (t.issue_type or "unknown").lower()
        dom = (t.domain or "unknown").lower()
        created_local = _to_cst(t.created_at) or end
        updated_local = _to_cst(t.updated_at) or created_local
        day = _iso_day(created_local)
        collected = t.get_collected() if hasattr(t, "get_collected") else {}
        has_email = bool(str(collected.get("customer_email") or "").strip())
        is_completed = status != "in_progress"

        status_counts[status] += 1
        issue_totals[issue_type]["count"] += 1
        domain_totals[dom]["count"] += 1
        daily[day]["started"] += 1

        if is_completed:
            reached_terminal += 1
            issue_totals[issue_type]["completed"] += 1
            domain_totals[dom]["completed"] += 1
            daily[day]["completed"] += 1
            node_id = str(t.current_node_id or "").strip()
            if node_id:
                terminal_counts[node_id] += 1
            step_count = turn_counts.get(str(t.ticket_id), 0)
            if step_count > 0:
                terminal_turns.append(step_count)
        elif updated_local < abandon_cutoff:
            abandoned += 1

        if has_email:
            contact_captured += 1
        if t.admin_decision:
            admin_decided += 1
        if status == "resolved":
            resolved += 1

    # Ensure every day in the window shows up (even zero-days) for a smooth chart.
    trend: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        d = _iso_day(cursor)
        trend.append(
            {
                "day": d,
                "started": daily[d]["started"],
                "completed": daily[d]["completed"],
            }
        )
        cursor += timedelta(days=1)

    median_turns = 0
    if terminal_turns:
        terminal_turns.sort()
        mid = len(terminal_turns) // 2
        if len(terminal_turns) % 2 == 1:
            median_turns = terminal_turns[mid]
        else:
            median_turns = int(round((terminal_turns[mid - 1] + terminal_turns[mid]) / 2))

    return {
        "range": {
            "days": days,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "abandon_threshold_hours": _ABANDON_THRESHOLD_HOURS,
        },
        "totals": {
            "started": started,
            "reached_terminal": reached_terminal,
            "completion_rate_pct": _percent(reached_terminal, started),
            "contact_captured": contact_captured,
            "contact_rate_pct": _percent(contact_captured, started),
            "admin_decided": admin_decided,
            "resolved": resolved,
            "resolved_rate_pct": _percent(resolved, reached_terminal),
            "abandoned": abandoned,
            "abandoned_rate_pct": _percent(abandoned, started),
            "median_turns_to_terminal": median_turns,
        },
        "by_status": [
            {"status": s, "count": c}
            for s, c in sorted(status_counts.items(), key=lambda kv: kv[1], reverse=True)
        ],
        "by_issue_type": [
            {
                "issue_type": issue,
                "count": data["count"],
                "completed": data["completed"],
                "completion_rate_pct": _percent(data["completed"], data["count"]),
            }
            for issue, data in sorted(
                issue_totals.items(), key=lambda kv: kv[1]["count"], reverse=True
            )
        ],
        "by_domain": [
            {
                "domain": dom,
                "count": data["count"],
                "completed": data["completed"],
                "completion_rate_pct": _percent(data["completed"], data["count"]),
            }
            for dom, data in sorted(
                domain_totals.items(), key=lambda kv: kv[1]["count"], reverse=True
            )
        ],
        "top_terminals": [
            {"node_id": nid, "count": c}
            for nid, c in terminal_counts.most_common(10)
        ],
        "daily_started": trend,
    }
