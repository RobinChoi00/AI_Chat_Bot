"""
Public warranty case-status lookup.

Customers prove they own a case with the shareable WR- reference plus the
email they left during intake. The payload is customer-safe: no ticket UUID,
admin notes, or Freshdesk URLs.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any, Optional

from warranty_case_ref import (
    case_reference_for_ticket,
    parse_case_reference,
)
from warranty_email import extract_email, resolve_customer_email

logger = logging.getLogger(__name__)

LOOKUP_NOT_FOUND = "We couldn't find a case with those details."

_STATUS_COPY = {
    "in_progress": (
        "In progress",
        "Your chat is still open. Continue in the warranty assistant when you can.",
    ),
    "awaiting_admin_review": (
        "Under review",
        "Our warranty team has your case and typically replies within 1 business day.",
    ),
    "admin_reviewing": (
        "Being reviewed",
        "A specialist is reviewing your case. We will email you when we have an update.",
    ),
    "awaiting_evidence": (
        "Photos needed",
        "Please add photos or a video in the warranty chat, or reply to your confirmation email.",
    ),
    "need_more_information": (
        "More information needed",
        "Our team asked a follow-up question. Reply from the warranty chat or email.",
    ),
    "send_info": (
        "Info sent",
        "Self-help steps were shared. Start a new case if you still need help.",
    ),
    "sales_handoff": (
        "Sent to sales",
        "This request was routed to our sales team.",
    ),
    "resolved": (
        "Resolved",
        "This case is closed. Contact warranty support if you need to reopen it.",
    ),
    "self_resolved": (
        "Resolved",
        "This case was closed after the troubleshooting steps.",
    ),
}

_DECISION_LABELS = {
    "approved": "Approved",
    "rejected": "Not approved",
    "closed": "Closed",
}


def emails_match(left: str, right: str) -> bool:
    """Constant-time compare of normalized email addresses."""
    a = (extract_email(left) or (left or "").strip().lower())
    b = (extract_email(right) or (right or "").strip().lower())
    if not a or not b or "@" not in a or "@" not in b:
        return False
    return hmac.compare_digest(a, b)


def public_status_label(status: str, admin_decision: str = "") -> str:
    """Customer-facing status label."""
    decision = (admin_decision or "").strip().lower()
    if (status or "").strip().lower() == "resolved" and decision in _DECISION_LABELS:
        return _DECISION_LABELS[decision]
    copy = _STATUS_COPY.get((status or "").strip().lower())
    if copy:
        return copy[0]
    return "Update available"


def public_next_step(status: str, admin_decision: str = "") -> str:
    """One-line 'what happens next' for the customer."""
    key = (status or "").strip().lower()
    decision = (admin_decision or "").strip().lower()
    if key == "resolved" and decision == "need_more_information":
        return _STATUS_COPY["need_more_information"][1]
    copy = _STATUS_COPY.get(key)
    if copy:
        return copy[1]
    return "Our warranty team will follow up by email."


def _ticket_emails(ticket, turns=None, evidences=None) -> list[str]:
    emails: list[str] = []
    found = resolve_customer_email(ticket, turns=turns, evidences=evidences)
    if found:
        emails.append(found)
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    for key in (
        "customer_contact_email",
        "order_or_email",
        "resume_email",
    ):
        extra = extract_email(str(collected.get(key) or ""))
        if extra:
            emails.append(extra)
    session_id = str(getattr(ticket, "session_id", "") or "")
    if session_id:
        try:
            from warranty_consent import get_chat_consent  # noqa: WPS433

            row = get_chat_consent(session_id)
            contact = getattr(row, "contact_email", None) if row is not None else None
            extra = extract_email(str(contact or ""))
            if extra:
                emails.append(extra)
        except Exception:
            pass
    return emails


def ticket_matches_email(ticket, email: str, *, turns=None, evidences=None) -> bool:
    """True when *email* matches a contact address stored on the case."""
    wanted = extract_email(email) or ""
    if not wanted:
        return False
    return any(emails_match(wanted, stored) for stored in _ticket_emails(ticket, turns, evidences))


def find_ticket_by_case_reference(case_reference: str):
    """Return the WarrantyTicket whose shareable reference matches, or None."""
    parsed = parse_case_reference(case_reference)
    if parsed is None:
        return None
    _date_part, suffix = parsed
    from sqlalchemy import func  # noqa: WPS433

    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    compact = func.replace(func.upper(WarrantyTicket.ticket_id), "-", "")
    with warranty_db_session() as db:
        rows = (
            db.query(WarrantyTicket)
            .filter(compact.like(f"{suffix}%"))
            .order_by(WarrantyTicket.updated_at.desc())
            .all()
        )
        if not rows:
            rows = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.collected_data.contains(suffix))
                .order_by(WarrantyTicket.updated_at.desc())
                .all()
            )
        for ticket in rows:
            stored = str((ticket.get_collected() or {}).get("case_reference") or "").strip()
            computed = case_reference_for_ticket(ticket)
            if stored.upper() == f"WR-{_date_part}-{suffix}" or computed == f"WR-{_date_part}-{suffix}":
                db.expunge(ticket)
                return ticket
    return None


def build_public_status_payload(ticket, *, turns=None) -> dict[str, Any]:
    """Customer-safe status card. Never includes ticket UUID or admin notes."""
    status = str(getattr(ticket, "status", "") or "")
    decision = str(getattr(ticket, "admin_decision", "") or "")
    customer_message = ""
    if status in {"resolved", "need_more_information"}:
        customer_message = str(getattr(ticket, "customer_message", "") or "").strip()
    from warranty_case_ref import case_reference_for_ticket  # noqa: WPS433

    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    case_ref = str(collected.get("case_reference") or "").strip() or case_reference_for_ticket(
        ticket
    )
    return {
        "found": True,
        "case_reference": case_ref,
        "status": status,
        "status_label": public_status_label(status, decision),
        "next_step": public_next_step(status, decision),
        "model_name": str(getattr(ticket, "model_name", "") or "") or None,
        "issue_type": str(getattr(ticket, "issue_type", "") or "") or None,
        "customer_message": customer_message or None,
    }


def lookup_public_case(
    *,
    case_reference: str,
    email: str,
    engine=None,
) -> Optional[dict[str, Any]]:
    """
    Return a public status payload when the case reference and email match.

    Returns None for unknown / unauthorized lookups (same miss).
    """
    parsed = parse_case_reference(case_reference)
    wanted = extract_email(email) or ""
    if parsed is None or not wanted:
        return None

    ticket = find_ticket_by_case_reference(case_reference)
    if ticket is None:
        return None

    turns = None
    evidences = None
    if engine is not None:
        ticket_id = str(ticket.ticket_id)
        try:
            turns = engine.get_turns(ticket_id)
            evidences = engine.get_evidences(ticket_id)
        except Exception:
            turns = None
            evidences = None

    if not ticket_matches_email(ticket, wanted, turns=turns, evidences=evidences):
        logger.info("warranty status lookup email mismatch case=%s", parsed)
        return None

    return build_public_status_payload(ticket, turns=turns)
