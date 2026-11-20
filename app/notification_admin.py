"""Authenticated delivery queue status and explicit retry operations."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query

from admin_auth import require_admin_key
from sales_email_transport import transport_health
from warranty_handoff_delivery import freshdesk_handoff_health
from warranty_models import warranty_db_session
from warranty_notification_jobs import WarrantyHandoffDispatch, WarrantyNotificationJob, notification_job_stats, notification_worker_health, retry_notification

router = APIRouter(tags=["admin-notifications"])


@router.get("/admin/warranty/notification-jobs")
def get_notification_jobs(
    status: Literal["pending", "processing", "failed", "dead_letter", "completed"] | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    kind: Literal["handoff", "evidence", "contact", "resume"] | None = None,
    offset: int = Query(default=0, ge=0),
    x_admin_key: str | None = Header(default=None),
):
    require_admin_key(x_admin_key)
    counts = notification_job_stats(kind=kind)
    with warranty_db_session() as db:
        query = db.query(WarrantyNotificationJob, WarrantyHandoffDispatch).outerjoin(
            WarrantyHandoffDispatch, WarrantyHandoffDispatch.job_id == WarrantyNotificationJob.id,
        )
        if status:
            query = query.filter(WarrantyNotificationJob.status == status)
        if kind:
            query = query.filter(WarrantyNotificationJob.kind == kind)
        rows = query.order_by(WarrantyNotificationJob.id.desc()).offset(offset).limit(limit + 1).all()
        items = [{
            "id": int(row.id), "kind": str(row.kind), "ticket_id": str(row.ticket_id),
            "status": str(row.status), "attempts": int(row.attempts),
            "failure_code": row.last_error,
            "next_attempt_at": row.next_attempt_at.isoformat() if row.next_attempt_at else None,
            "delivery_channel": "freshdesk" if dispatch is not None else "smtp",
            "submitted_via": ("freshdesk" if dispatch.state == "submitted" and dispatch.provider_ticket_id else None)
            if dispatch is not None else "smtp" if row.kind == "handoff" and row.status == "completed" else None,
            "dispatch_state": dispatch.state if dispatch is not None else None,
            "helpdesk_ticket_id": dispatch.provider_ticket_id if dispatch is not None else None,
        } for row, dispatch in rows[:limit]]
    return {"counts": counts, "worker": notification_worker_health(), "smtp": transport_health(), "handoff": freshdesk_handoff_health(),
            "items": items, "has_more": len(rows) > limit}


@router.post("/admin/warranty/notification-jobs/{job_id}/retry")
def retry_notification_job(job_id: int, x_admin_key: str | None = Header(default=None)):
    require_admin_key(x_admin_key)
    if not retry_notification(job_id):
        raise HTTPException(status_code=409, detail="Only failed jobs without an unresolved external submission can be retried. Reconcile any Freshdesk attempt first.")
    return {"job_id": job_id, "queued": True, "sent": False}
