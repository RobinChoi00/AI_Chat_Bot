"""Authenticated delivery queue status and explicit retry operations."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query

from admin_auth import require_admin_key
from warranty_models import warranty_db_session
from warranty_notification_jobs import WarrantyNotificationJob, notification_job_stats, notification_worker_health, retry_notification

router = APIRouter(tags=["admin-notifications"])


@router.get("/admin/warranty/notification-jobs")
def get_notification_jobs(
    status: Literal["pending", "processing", "failed", "dead_letter", "completed"] | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    x_admin_key: str | None = Header(default=None),
):
    require_admin_key(x_admin_key)
    counts = notification_job_stats()
    with warranty_db_session() as db:
        query = db.query(WarrantyNotificationJob)
        if status:
            query = query.filter_by(status=status)
        rows = query.order_by(WarrantyNotificationJob.id.desc()).limit(limit + 1).all()
        items = [{
            "id": int(row.id), "kind": str(row.kind), "ticket_id": str(row.ticket_id),
            "status": str(row.status), "attempts": int(row.attempts),
            "failure_code": row.last_error,
            "next_attempt_at": row.next_attempt_at.isoformat() if row.next_attempt_at else None,
        } for row in rows[:limit]]
    return {"counts": counts, "worker": notification_worker_health(), "items": items, "has_more": len(rows) > limit}


@router.post("/admin/warranty/notification-jobs/{job_id}/retry")
def retry_notification_job(job_id: int, x_admin_key: str | None = Header(default=None)):
    require_admin_key(x_admin_key)
    if not retry_notification(job_id):
        raise HTTPException(status_code=409, detail="Only failed or exhausted notifications can be retried.")
    return {"job_id": job_id, "queued": True, "sent": False}
