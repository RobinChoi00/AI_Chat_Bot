"""Read-only queue for open support requests with no confirmed submission."""
from __future__ import annotations

from fastapi import APIRouter, Header, Query
from sqlalchemy import and_, case, func, literal, or_

from admin_auth import require_admin_key
from warranty_case_ref import case_reference_for_ticket
from warranty_models import WarrantyTicket, warranty_db_session
from warranty_notification_jobs import WarrantyNotificationJob

router = APIRouter(tags=["admin-notifications"])

_OPEN_REVIEW = ("awaiting_admin_review", "admin_reviewing", "need_more_information")


@router.get("/admin/warranty/handoff-attention")
def handoff_attention(
    limit: int = Query(8, ge=1, le=25), x_admin_key: str | None = Header(default=None),
) -> dict:
    """List only requests that need a human check; never resend from this view."""
    require_admin_key(x_admin_key)
    ticket, job = WarrantyTicket, WarrantyNotificationJob
    safe_data = case((func.json_valid(ticket.collected_data) == 1, ticket.collected_data), else_=literal("{}"))

    def field(key):
        return func.json_extract(safe_data, "$." + key)

    safety = field("human_support_reason") == "safety_hazard"
    with warranty_db_session() as db:
        matches = db.query(ticket, job.status).outerjoin(job, and_(
            job.kind == "handoff", job.ticket_id == ticket.ticket_id,
            job.source_key == field("human_support_notification_source"),
        )).filter(
            ticket.status.in_(_OPEN_REVIEW),
            field("human_support_requested") == 1,
            func.strftime("%s", field("human_support_staff_accepted_at")).is_(None),
            or_(job.id.is_(None), job.status.in_(("failed", "dead_letter"))),
        )
        total = matches.count()
        urgent = matches.filter(safety).count()
        rows = matches.order_by(case((safety, 0), else_=1), ticket.created_at, ticket.id).limit(limit).all()
        items = []
        for row, status in rows:
            data = row.get_collected()
            items.append({
                "ticket_id": row.ticket_id,
                "case_reference": case_reference_for_ticket(row),
                "safety_review": data.get("human_support_reason") == "safety_hazard",
                "department": "sales" if data.get("human_support_department") == "sales" else "warranty",
                "notification_state": status or "not_recorded",
            })
    return {"total": total, "safety_reviews": urgent, "items": items}
