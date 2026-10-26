"""Durable delivery of warranty evidence and contact notifications.

Jobs contain database references, not copied customer messages, email addresses,
attachments, or credentials. A bounded lease recovers interrupted workers. SMTP
delivery is at least once: an interruption after SMTP accepts a message and
before the database commit can cause a retry. Only a confirmed send marks the
evidence as emailed.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint, and_, func, or_, text

from warranty_models import Base, WarrantyEvidence, WarrantyTicket, WarrantyTurn, warranty_db_session

logger = logging.getLogger(__name__)

MAX_NOTIFICATION_ATTEMPTS = 8
NOTIFICATION_LEASE_SECONDS = 600
_RETRY_DELAYS = (10, 30, 60, 300, 900, 3600, 21600, 86400)
_KINDS = frozenset({"evidence", "contact", "resume", "handoff"})
_schema_lock = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class WarrantyNotificationJob(Base):
    __tablename__ = "warranty_notification_jobs"
    __table_args__ = (UniqueConstraint("kind", "source_key", name="uq_warranty_notification_source"),)

    id = Column(Integer, primary_key=True)
    kind = Column(String(20), nullable=False)
    source_key = Column(String(100), nullable=False)
    evidence_id = Column(Integer, nullable=True, index=True)
    ticket_id = Column(String, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    attempts = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime, nullable=True, index=True)
    lease_token = Column(String(32), nullable=True)
    lease_expires_at = Column(DateTime, nullable=True, index=True)
    last_error = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)
    completed_at = Column(DateTime, nullable=True)


def _ensure_schema(db) -> None:
    # This table is owned by this module; existing deployments need no destructive
    # migration. Resolve the active bind so isolated test databases work too.
    with _schema_lock:
        WarrantyNotificationJob.__table__.create(bind=db.get_bind(), checkfirst=True)


def initialize_notification_storage() -> None:
    """Create additive queue storage before opening a shared writer transaction."""
    with warranty_db_session() as db:
        _ensure_schema(db)


def enqueue_notification(*, kind: str, ticket_id: str, evidence_id: int | None = None, source_key: str | None = None, db=None) -> int:
    """Persist a job, optionally in the caller's transaction with its source.

    Shared transactions must initialize storage before their first write and
    flush newly created source rows before passing their generated IDs. The
    caller owns commit/rollback; workers cannot see a job before that commit.
    """
    if kind not in _KINDS:
        raise ValueError("Unsupported warranty notification kind")
    key = source_key or (str(evidence_id) if evidence_id is not None else "")
    if not key or len(key) > 100 or (kind in {"evidence", "contact"} and evidence_id is None):
        raise ValueError("A notification source reference is required")

    def insert(session) -> int:
        evidence = session.get(WarrantyEvidence, evidence_id) if evidence_id is not None else None
        if kind in {"evidence", "contact"} and (evidence is None or evidence.ticket_id != ticket_id):
            raise ValueError("Notification evidence does not belong to the ticket")
        existing = session.query(WarrantyNotificationJob).filter_by(kind=kind, source_key=key).first()
        if existing is not None:
            if existing.ticket_id != ticket_id:
                raise ValueError("Notification source does not belong to the ticket")
            return int(existing.id)
        row = WarrantyNotificationJob(
            kind=kind,
            source_key=key,
            evidence_id=evidence_id,
            ticket_id=ticket_id,
            status="completed" if evidence is not None and evidence.emailed else "pending",
            completed_at=_utcnow() if evidence is not None and evidence.emailed else None,
        )
        session.add(row)
        session.flush()
        return int(row.id)

    if db is not None:
        job_id = insert(db)
    else:
        with warranty_db_session() as owned_db:
            _ensure_schema(owned_db)
            owned_db.execute(text("BEGIN IMMEDIATE"))
            job_id = insert(owned_db)
    _worker_wakeup.set()
    return job_id


@dataclass(frozen=True)
class _Claim:
    job_id: int
    kind: str
    evidence_id: int | None
    ticket_id: str
    source_key: str
    token: str


def _claim_notification(job_id: int) -> _Claim | None:
    now = _utcnow()
    with warranty_db_session() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        row = db.get(WarrantyNotificationJob, job_id)
        if row is None or row.status not in {"pending", "failed", "processing"}:
            return None
        if row.status == "processing" and row.lease_expires_at and row.lease_expires_at > now:
            return None
        if row.next_attempt_at and row.next_attempt_at > now and row.status != "processing":
            return None
        if int(row.attempts or 0) >= MAX_NOTIFICATION_ATTEMPTS:
            row.status = "dead_letter"
            row.last_error = "attempts_exhausted"
            row.lease_token = None
            row.lease_expires_at = None
            return None
        row.status = "processing"
        row.attempts = int(row.attempts or 0) + 1
        row.next_attempt_at = None
        row.lease_token = uuid.uuid4().hex
        row.lease_expires_at = now + timedelta(seconds=NOTIFICATION_LEASE_SECONDS)
        return _Claim(int(row.id), str(row.kind), int(row.evidence_id) if row.evidence_id is not None else None, str(row.ticket_id), str(row.source_key), str(row.lease_token))


class _PermanentDeliveryError(Exception):
    pass


def _deliver_notification(claim: _Claim) -> bool:
    if claim.kind == "handoff":
        from warranty_handoff_notifications import send_handoff_notification

        return bool(send_handoff_notification(claim.ticket_id))
    if claim.kind == "resume":
        from warranty_resume import deliver_resume_attempt

        deliver_resume_attempt(claim.source_key)
        return True
    # Load the authoritative records for each attempt instead of persisting
    # another copy of contact details, transcripts, or attachment bytes.
    with warranty_db_session() as db:
        evidence = db.get(WarrantyEvidence, claim.evidence_id)
        ticket = db.query(WarrantyTicket).filter_by(ticket_id=claim.ticket_id).first()
        if evidence is None or ticket is None or evidence.ticket_id != claim.ticket_id:
            raise _PermanentDeliveryError("source_unavailable")
        if evidence.emailed:
            return True
        turns = db.query(WarrantyTurn).filter_by(ticket_id=claim.ticket_id).order_by(WarrantyTurn.id).all()
        collected = ticket.get_collected()
        common = dict(
            ticket_id=claim.ticket_id,
            customer_email=str(evidence.customer_email or collected.get("customer_contact_email") or ""),
            issue_type=str(ticket.issue_type or ""),
            model_name=str(ticket.model_name or ""),
            turns=turns,
            terminal_node_id=str(ticket.current_node_id or ""),
            case_summary=str(collected.get("case_summary") or ""),
            case_summary_source=str(collected.get("case_summary_source") or ""),
        )
        if not common["customer_email"]:
            raise _PermanentDeliveryError("contact_unavailable")
        if claim.kind == "evidence":
            from warranty_email import send_evidence_upload_notification

            kwargs = dict(
                common,
                evidence_type=str(evidence.evidence_type),
                original_filename=str(evidence.original_filename or "attachment"),
                file_path=str(evidence.file_path or ""),
                mime_type=str(evidence.mime_type or "application/octet-stream"),
                file_size_bytes=int(evidence.file_size_bytes or 0),
            )
            sender = send_evidence_upload_notification
        elif claim.kind == "contact":
            from warranty_email import send_email_only_contact_notification

            kwargs = dict(common, session_id=str(ticket.session_id), domain=str(ticket.domain or ""), ticket_status=str(ticket.status))
            sender = send_email_only_contact_notification
        else:
            raise _PermanentDeliveryError("unsupported_kind")
    # Do not hold a database transaction open across SMTP/network work.
    return bool(sender(**kwargs))


def _finish_notification(claim: _Claim, *, sent: bool, error: str = "", permanent: bool = False) -> bool:
    with warranty_db_session() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        row = db.get(WarrantyNotificationJob, claim.job_id)
        if row is None or row.status != "processing" or row.lease_token != claim.token:
            return False
        row.lease_token = None
        row.lease_expires_at = None
        if sent:
            evidence = db.get(WarrantyEvidence, claim.evidence_id) if claim.evidence_id is not None else None
            if evidence is not None and evidence.ticket_id == claim.ticket_id:
                evidence.emailed = 1
            row.status = "completed"
            row.completed_at = _utcnow()
            row.last_error = None
            return True
        row.last_error = error[:100] or "send_not_confirmed"
        if permanent or int(row.attempts) >= MAX_NOTIFICATION_ATTEMPTS:
            row.status = "dead_letter"
            row.next_attempt_at = None
        else:
            row.status = "failed"
            row.next_attempt_at = _utcnow() + timedelta(seconds=_RETRY_DELAYS[min(int(row.attempts) - 1, len(_RETRY_DELAYS) - 1)])
        return False


def process_notification(job_id: int) -> bool:
    claim = _claim_notification(job_id)
    if claim is None:
        return False
    try:
        sent = _deliver_notification(claim)
    except Exception as exc:
        # Exception strings can contain recipients or provider credentials.
        code = str(exc) if isinstance(exc, _PermanentDeliveryError) else type(exc).__name__
        logger.warning("Warranty notification failed job_id=%s kind=%s error_type=%s", claim.job_id, claim.kind, type(exc).__name__)
        return _finish_notification(claim, sent=False, error=code, permanent=isinstance(exc, _PermanentDeliveryError))
    return _finish_notification(claim, sent=sent)


def process_pending_notifications(*, limit: int = 25) -> int:
    now = _utcnow()
    with warranty_db_session() as db:
        _ensure_schema(db)
        rows = db.query(WarrantyNotificationJob.id).filter(or_(
            and_(WarrantyNotificationJob.status.in_(("pending", "failed")), or_(WarrantyNotificationJob.next_attempt_at.is_(None), WarrantyNotificationJob.next_attempt_at <= now)),
            and_(WarrantyNotificationJob.status == "processing", or_(WarrantyNotificationJob.lease_expires_at.is_(None), WarrantyNotificationJob.lease_expires_at <= now)),
        )).order_by(WarrantyNotificationJob.id).limit(max(1, min(limit, 100))).all()
        job_ids = [int(row[0]) for row in rows]
    return sum(process_notification(job_id) for job_id in job_ids)


def notification_job_stats() -> dict[str, int]:
    counts = {"pending": 0, "processing": 0, "failed": 0, "dead_letter": 0, "completed": 0}
    with warranty_db_session() as db:
        _ensure_schema(db)
        for status, count in db.query(WarrantyNotificationJob.status, func.count(WarrantyNotificationJob.id)).group_by(WarrantyNotificationJob.status):
            counts[str(status)] = int(count)
    return counts


def notification_worker_health() -> dict[str, bool]:
    """Expose process state independently from persisted delivery counts."""
    enabled = os.getenv("WARRANTY_NOTIFICATION_WORKER_ENABLED", "true").strip().lower() not in {"0", "false", "off", "no"}
    return {"enabled": enabled, "alive": bool(_worker_thread is not None and _worker_thread.is_alive())}


def retry_notification(job_id: int) -> bool:
    """Explicit operator retry; never steal an active lease or resend completion."""
    with warranty_db_session() as db:
        _ensure_schema(db)
        db.execute(text("BEGIN IMMEDIATE"))
        row = db.get(WarrantyNotificationJob, job_id)
        if row is None or row.status not in {"failed", "dead_letter"}:
            return False
        row.status = "pending"
        row.attempts = 0
        row.next_attempt_at = None
        row.lease_token = None
        row.lease_expires_at = None
        row.last_error = None
    _worker_wakeup.set()
    return True


_worker_stop = threading.Event()
_worker_wakeup = threading.Event()
_worker_lock = threading.Lock()
_worker_thread: threading.Thread | None = None


def _worker_loop() -> None:
    while not _worker_stop.is_set():
        try:
            process_pending_notifications()
        except Exception as exc:
            logger.warning("Warranty notification worker failed error_type=%s", type(exc).__name__)
        _worker_wakeup.wait(2.0)
        _worker_wakeup.clear()


def start_notification_worker() -> None:
    """Start the single-host worker and recover jobs accepted before a restart."""
    global _worker_thread
    if os.getenv("WARRANTY_NOTIFICATION_WORKER_ENABLED", "true").strip().lower() in {"0", "false", "off", "no"}:
        return
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        with warranty_db_session() as db:
            _ensure_schema(db)
        _worker_stop.clear()
        _worker_thread = threading.Thread(target=_worker_loop, name="warranty-notification-worker", daemon=True)
        _worker_thread.start()


def stop_notification_worker() -> None:
    _worker_stop.set()
    _worker_wakeup.set()
    worker = _worker_thread
    if worker is not None:
        worker.join(timeout=5.0)
