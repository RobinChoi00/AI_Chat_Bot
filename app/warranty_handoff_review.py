"""Open support requests with no confirmed submission — review and staff ack."""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, case, func, literal, or_

from admin_auth import require_admin_key
from warranty_case_ref import case_reference_for_ticket
from warranty_models import WarrantyTicket, warranty_db_session, _now_cst
from warranty_notification_jobs import WarrantyNotificationJob

router = APIRouter(tags=["admin-notifications"])

_OPEN_REVIEW = ("awaiting_admin_review", "admin_reviewing", "need_more_information")


class HandoffAckRequest(BaseModel):
    """Record that staff reconciled a support request without changing claim status."""

    note: str = ""
    accepted_by: str = Field(default="admin", max_length=120)


@router.get("/admin/warranty/handoff-attention")
def handoff_attention(
    limit: int = Query(12, ge=1, le=25), x_admin_key: str | None = Header(default=None),
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
        # Safety first, then newest — ops should triage recent failures before old backlog.
        rows = matches.order_by(
            case((safety, 0), else_=1),
            ticket.created_at.desc(),
            ticket.id.desc(),
        ).limit(limit).all()
        items = []
        for row, status in rows:
            data = row.get_collected()
            items.append({
                "ticket_id": row.ticket_id,
                "case_reference": case_reference_for_ticket(row),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "safety_review": data.get("human_support_reason") == "safety_hazard",
                "department": "sales" if data.get("human_support_department") == "sales" else "warranty",
                "notification_state": status or "not_recorded",
            })
    return {"total": total, "safety_reviews": urgent, "items": items}


@router.post("/admin/warranty/{ticket_id}/handoff-ack")
def acknowledge_handoff(
    ticket_id: str,
    body: HandoffAckRequest,
    x_admin_key: str | None = Header(default=None),
) -> dict:
    """
    Clear the attention queue after Freshdesk/staff reconciliation.

    Does not approve/reject the warranty claim and does not resend notifications.
    """
    require_admin_key(x_admin_key)
    with warranty_db_session() as db:
        row = db.query(WarrantyTicket).filter(WarrantyTicket.ticket_id == ticket_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")
        data = row.get_collected()
        if not data.get("human_support_requested"):
            raise HTTPException(
                status_code=409,
                detail="This case has no saved human-support request to acknowledge.",
            )
        if data.get("human_support_staff_accepted_at"):
            return {
                "ticket_id": ticket_id,
                "already_acknowledged": True,
                "staff_acceptance_recorded": True,
                "accepted_at": data.get("human_support_staff_accepted_at"),
            }
        accepted_at = _now_cst().isoformat()
        row.set_collected("human_support_staff_accepted_at", accepted_at)
        row.set_collected("human_support_staff_accepted_by", (body.accepted_by or "admin")[:120])
        row.set_collected("human_support_staff_ack_note", (body.note or "").strip()[:1000])
        try:
            from customer_journey import record_server_event, register_conversation

            register_conversation(
                db,
                surface="warranty",
                conversation_id=str(ticket_id),
                session_id=str(row.session_id or ""),
                domain=str(row.domain or ""),
                is_new=False,
            )
            record_server_event(
                db,
                surface="warranty",
                conversation_id=str(ticket_id),
                session_id=str(row.session_id or ""),
                domain=str(row.domain or ""),
                stage="handoff",
                event="completed",
                event_id=f"staff-ack:{ticket_id}:{accepted_at}",
            )
        except Exception:
            pass
    return {
        "ticket_id": ticket_id,
        "already_acknowledged": False,
        "staff_acceptance_recorded": True,
        "accepted_at": accepted_at,
    }
