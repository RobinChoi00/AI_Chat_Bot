"""
Persist customer live-chat privacy consent before messages are stored.

Consent is recorded at the session level (before a ticket exists) and copied
onto the ticket's collected_data when the workflow starts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, cast

from sqlalchemy.orm import Session

from warranty_models import WarrantyChatConsent, WarrantyTicket, _now_cst, warranty_db_session

EMAIL_GATE_PROVIDED = "provided"
EMAIL_GATE_SKIPPED = "skipped"


def record_chat_consent(
    session_id: str,
    *,
    domain: str,
    policy_store: str,
) -> datetime:
    """Upsert consent for a browser session. Returns accepted_at (CST)."""
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")

    now = _now_cst()
    with warranty_db_session() as db:
        row = (
            db.query(WarrantyChatConsent)
            .filter(WarrantyChatConsent.session_id == sid)
            .first()
        )
        if row is None:
            row = WarrantyChatConsent(
                session_id=sid,
                domain=(domain or "unknown").strip().lower() or "unknown",
                policy_store=(policy_store or "").strip().lower() or None,
                accepted_at=now,
            )
            db.add(row)
        else:
            row.domain = (domain or "unknown").strip().lower() or "unknown"
            row.policy_store = (policy_store or "").strip().lower() or None
            row.accepted_at = now
        return cast(datetime, row.accepted_at)


def record_session_contact_email(
    session_id: str,
    *,
    customer_email: str = "",
    skipped: bool = False,
) -> dict[str, Optional[str]]:
    """
    Store the post-consent email gate result on the session consent row.

    Soft-required UX: either a validated email (provided) or an explicit skip.
    """
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")

    from warranty_email import extract_email  # noqa: WPS433

    email = extract_email(customer_email or "") or ""
    if skipped:
        status = EMAIL_GATE_SKIPPED
        email = ""
    else:
        if not email:
            raise ValueError("A valid email address is required, or set skipped=true.")
        status = EMAIL_GATE_PROVIDED

    with warranty_db_session() as db:
        row = (
            db.query(WarrantyChatConsent)
            .filter(WarrantyChatConsent.session_id == sid)
            .first()
        )
        if row is None:
            # Consent should already exist; create a minimal row so email is not lost.
            row = WarrantyChatConsent(
                session_id=sid,
                domain="unknown",
                accepted_at=_now_cst(),
            )
            db.add(row)
        row.contact_email = email or None
        row.email_gate_status = status

    # Keep any already-open ticket in sync (reload / race with start_session).
    _sync_email_gate_to_active_ticket(sid, status=status, email=email)

    return {
        "session_id": sid,
        "customer_email": email or None,
        "email_gate_status": status,
    }


def _sync_email_gate_to_active_ticket(
    session_id: str,
    *,
    status: str,
    email: str,
) -> None:
    """Copy gate status onto the active in-progress ticket for this session."""
    with warranty_db_session() as db:
        ticket = (
            db.query(WarrantyTicket)
            .filter(
                WarrantyTicket.session_id == session_id,
                WarrantyTicket.status == "in_progress",
            )
            .order_by(WarrantyTicket.created_at.desc())
            .first()
        )
        if ticket is None:
            return
        ticket.set_collected("intake_email_gate_status", status)
        if status == EMAIL_GATE_PROVIDED and email:
            ticket.set_collected("customer_contact_email", email)
            ticket.set_collected("intake_email_skipped", "")
        elif status == EMAIL_GATE_SKIPPED:
            ticket.set_collected("intake_email_skipped", "1")


def get_chat_consent(session_id: str) -> Optional[WarrantyChatConsent]:
    sid = (session_id or "").strip()
    if not sid:
        return None
    with warranty_db_session() as db:
        return (
            db.query(WarrantyChatConsent)
            .filter(WarrantyChatConsent.session_id == sid)
            .first()
        )


def attach_consent_to_ticket(db: Session, session_id: str, ticket: WarrantyTicket) -> None:
    """Copy session consent metadata onto the ticket collected_data bag."""
    row = (
        db.query(WarrantyChatConsent)
        .filter(WarrantyChatConsent.session_id == session_id)
        .first()
    )
    if row is None or row.accepted_at is None:
        return

    accepted_at = cast(datetime, row.accepted_at).isoformat()
    ticket.set_collected("chat_consent_accepted_at", accepted_at)
    if row.policy_store:
        ticket.set_collected("chat_consent_policy_store", str(row.policy_store))
    if row.domain:
        ticket.set_collected("chat_consent_domain", str(row.domain))

    status = str(getattr(row, "email_gate_status", None) or "").strip().lower()
    email = str(getattr(row, "contact_email", None) or "").strip().lower()
    # Consent row may have an email before status was backfilled.
    if not status and email:
        status = EMAIL_GATE_PROVIDED
    if status:
        ticket.set_collected("intake_email_gate_status", status)
    if status == EMAIL_GATE_PROVIDED and email:
        existing = str(ticket.get_collected().get("customer_contact_email") or "").strip()
        if not existing:
            ticket.set_collected("customer_contact_email", email)
    elif status == EMAIL_GATE_SKIPPED:
        ticket.set_collected("intake_email_skipped", "1")
