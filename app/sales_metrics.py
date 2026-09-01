"""Admin-only Sales AI funnel and reliability metrics."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytz
from fastapi import APIRouter, Header, Query

from admin_auth import require_admin_key

router = APIRouter(tags=["admin-sales-metrics"])

_CST = pytz.timezone("America/Chicago")
_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
_MAX_DAYS = 180


def _require_admin(x_admin_key: Optional[str]) -> None:
    require_admin_key(x_admin_key, _ADMIN_API_KEY)


def _now_cst() -> datetime:
    return datetime.now(_CST)


def _to_cst(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return _CST.localize(value)
    return value.astimezone(_CST)


def _percent(part: int, whole: int) -> float:
    return round((part / whole) * 100.0, 1) if whole else 0.0


def _lazy_orm():
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    from sales_models import SalesLead, SalesMessage, SalesSession  # noqa: WPS433
    from warranty_models import warranty_db_session  # noqa: WPS433

    return SalesSession, SalesMessage, SalesLead, warranty_db_session


def _tools(raw: Optional[str]) -> set[str]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return set()
    return {str(item) for item in value} if isinstance(value, list) else set()


@router.get("/admin/sales/metrics")
async def sales_metrics(
    days: int = Query(30, ge=1, le=_MAX_DAYS),
    domain: Optional[str] = None,
    x_admin_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Aggregate recommendation, handoff, lead, and delivery-health metrics."""
    _require_admin(x_admin_key)
    SalesSession, SalesMessage, SalesLead, warranty_db_session = _lazy_orm()

    end = _now_cst()
    start = (end - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    with warranty_db_session() as db:
        session_q = db.query(SalesSession).filter(SalesSession.created_at >= start)
        lead_q = db.query(SalesLead).filter(SalesLead.created_at >= start)
        if domain:
            session_q = session_q.filter(SalesSession.domain.contains(domain))
            lead_q = lead_q.filter(SalesLead.domain.contains(domain))
        sessions = session_q.all()
        leads = lead_q.all()

        session_ids = [str(row.session_id) for row in sessions]
        messages = (
            db.query(SalesMessage)
            .filter(SalesMessage.session_id.in_(session_ids))
            .all()
            if session_ids
            else []
        )

    user_turns: Counter[str] = Counter()
    assistant_turns: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    recommended_sessions: set[str] = set()
    nofit_sessions: set[str] = set()
    handoff_sessions: set[str] = set()

    for msg in messages:
        sid = str(msg.session_id)
        role = str(msg.role or "").lower()
        if role == "user":
            user_turns[sid] += 1
        elif role == "assistant":
            assistant_turns[sid] += 1
            tools = _tools(msg.tools_used)
            if tools & {"cases.lookup", "catalog.recommend"}:
                recommended_sessions.add(sid)
            if "cases.nofit" in tools:
                nofit_sessions.add(sid)
        intent = str(msg.intent or "").strip().lower()
        if role == "assistant" and intent:
            intent_counts[intent] += 1
        if msg.handoff:
            handoff_sessions.add(sid)

    status_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    daily: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"started": 0, "recommended": 0, "leads": 0}
    )
    for session in sessions:
        sid = str(session.session_id)
        status = str(session.status or "active").lower()
        status_counts[status] += 1
        domain_counts[str(session.domain or "unknown").lower()] += 1
        channel_counts[str(session.channel or "unknown").lower()] += 1
        if status == "handoff":
            handoff_sessions.add(sid)
        created = _to_cst(session.created_at) or end
        day = created.strftime("%Y-%m-%d")
        daily[day]["started"] += 1
        if sid in recommended_sessions:
            daily[day]["recommended"] += 1

    lead_status_counts: Counter[str] = Counter()
    for lead in leads:
        lead_status_counts[str(lead.forwarded or "pending").lower()] += 1
        created = _to_cst(lead.created_at) or end
        daily[created.strftime("%Y-%m-%d")]["leads"] += 1

    started = len(sessions)
    engaged = sum(1 for s in sessions if user_turns[str(s.session_id)] >= 2)
    recommended = len(recommended_sessions)
    handoffs = len(handoff_sessions)
    nofit = len(nofit_sessions)
    lead_count = len(leads)
    failed_leads = lead_status_counts["failed"]
    from sales_spec_index import sales_artifact_health  # noqa: WPS433

    daily_rows = []
    cursor = start
    while cursor.date() <= end.date():
        day = cursor.strftime("%Y-%m-%d")
        values = daily[day]
        daily_rows.append({"day": day, **values})
        cursor += timedelta(days=1)

    return {
        "range": {
            "days": days,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "totals": {
            "started": started,
            "engaged": engaged,
            "engagement_rate_pct": _percent(engaged, started),
            "recommended": recommended,
            "recommend_rate_pct": _percent(recommended, started),
            "nofit": nofit,
            "nofit_rate_pct": _percent(nofit, recommended + nofit),
            "handoffs": handoffs,
            "handoff_rate_pct": _percent(handoffs, started),
            "leads": lead_count,
            "lead_rate_pct": _percent(lead_count, started),
            "lead_forward_failed": failed_leads,
            "lead_forward_failure_rate_pct": _percent(failed_leads, lead_count),
            "user_turns": sum(user_turns.values()),
            "assistant_turns": sum(assistant_turns.values()),
        },
        "by_status": [
            {"status": key, "count": count}
            for key, count in status_counts.most_common()
        ],
        "by_intent": [
            {"intent": key, "count": count}
            for key, count in intent_counts.most_common()
        ],
        "by_domain": [
            {"domain": key, "count": count}
            for key, count in domain_counts.most_common()
        ],
        "by_channel": [
            {"channel": key, "count": count}
            for key, count in channel_counts.most_common()
        ],
        "lead_delivery": [
            {"status": key, "count": count}
            for key, count in lead_status_counts.most_common()
        ],
        "daily": daily_rows,
        "artifacts": sales_artifact_health(),
    }
