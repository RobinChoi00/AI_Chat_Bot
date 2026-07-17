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
