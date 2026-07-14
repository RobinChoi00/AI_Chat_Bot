"""
ringcentral_followup.py
=======================
Post-call SMS follow-up for after-hours warranty IVR callers.

Environment
-----------
RC_SMS_FROM_NUMBER          E.164 sender (Roman Warranty line)
RC_SMS_FOLLOWUP_ENABLED     true/false (default true when FROM is set)
RC_SMS_FOLLOWUP_MESSAGE     Optional override for the SMS body
RC_IVR_TEAM_EMAIL_ENABLED   true/false (default true when EMAIL_SENDER is set)

Idempotency: per-channel delivery timestamps prevent duplicate SMS/email.
Failed channels remain retryable and a crashed delivery claim expires.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from ringcentral_client import RC_SMS_FROM_NUMBER, send_sms

logger = logging.getLogger(__name__)
_followup_lock = threading.Lock()

DEFAULT_FOLLOWUP_SMS = (
    "Thank you for calling Osaki and Titan warranty support. "
    "If you need additional help, please email service@osakititan.com "
    "with details about your current issue and we will contact you within 24 hours."
)


def build_followup_sms_message(
    *,
    case_ref: str = "",
    resume_url: str = "",
) -> str:
    custom = os.getenv("RC_SMS_FOLLOWUP_MESSAGE", "").strip()
    if custom:
        return custom

    parts = [
        "Thank you for calling Osaki and Titan warranty support.",
    ]
    if case_ref:
        parts.append(f"Reference: {case_ref}.")
    if resume_url:
        parts.append(f"Continue online: {resume_url}")
    parts.append(
        "Email service@osakititan.com with details and we will contact you within 24 hours."
    )
    return " ".join(parts)


def _sms_from_number() -> str:
    return os.getenv("RC_SMS_FROM_NUMBER", RC_SMS_FROM_NUMBER).strip()


def _followup_enabled() -> bool:
    flag = os.getenv("RC_SMS_FOLLOWUP_ENABLED", "true").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return bool(_sms_from_number())


def _followup_context(ticket_id: str, session_id: str) -> tuple[str, str]:
    """Return (case_reference, resume_url) for follow-up messages."""
    if not ticket_id:
        return "", ""

    from warranty_case_ref import case_reference_for_ticket  # noqa: WPS433
    from warranty_resume import build_warranty_resume_url  # noqa: WPS433
    from warranty_workflow import WarrantyEngine  # noqa: WPS433

    ticket = WarrantyEngine.get_ticket(ticket_id)
    if ticket is None:
        return "", ""

    case_ref = case_reference_for_ticket(ticket)
    resume_url = ""
    if session_id:
        resume_url = build_warranty_resume_url(
            ticket_id,
            session_id,
            str(ticket.domain or ""),
        ) or ""
    return case_ref, resume_url


def send_call_followup_sms(
    caller_phone: str,
    *,
    ticket_id: str = "",
    session_id: str = "",
) -> bool:
    """Send the standard warranty follow-up SMS. Returns True when sent."""
    if not _followup_enabled():
        logger.debug("RC follow-up SMS skipped (disabled or no FROM number)")
        return False

    phone = (caller_phone or "").strip()
    if not phone.startswith("+"):
        logger.warning(
            "RC follow-up SMS skipped — invalid caller phone ticket=%s",
            ticket_id,
        )
        return False

    case_ref, resume_url = _followup_context(ticket_id, session_id)
    message = build_followup_sms_message(case_ref=case_ref, resume_url=resume_url)

    try:
        send_sms(
            to=phone,
            text=message,
            from_number=_sms_from_number(),
        )
    except Exception:
        logger.exception(
            "RC follow-up SMS failed ticket=%s",
            ticket_id,
        )
        return False

    logger.info("RC follow-up SMS sent ticket=%s", ticket_id)
    return True


def _team_email_enabled() -> bool:
    flag = os.getenv("RC_IVR_TEAM_EMAIL_ENABLED", "true").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    from config import EMAIL_SENDER  # noqa: WPS433

    return bool((EMAIL_SENDER or "").strip())


def send_call_followup_team_email(
    *,
    caller_phone: str,
    ticket_id: str,
    session_id: str,
    sms_sent: bool = False,
) -> bool:
    """Email the warranty team with the phone IVR case summary. Returns True when sent."""
    if not _team_email_enabled():
        logger.debug("RC follow-up team email skipped (disabled or no EMAIL_SENDER)")
        return False

    if not ticket_id:
        logger.warning("RC follow-up team email skipped — missing ticket_id session=%s", session_id)
        return False

    from warranty_case_ref import case_reference_for_ticket  # noqa: WPS433
    from warranty_email import send_phone_ivr_team_email  # noqa: WPS433
    from warranty_workflow import WarrantyEngine  # noqa: WPS433

    ticket = WarrantyEngine.get_ticket(ticket_id)
    if ticket is None:
        logger.warning(
            "RC follow-up team email skipped — ticket not found ticket=%s session=%s",
            ticket_id,
            session_id,
        )
        return False

    turns = WarrantyEngine.get_turns(ticket_id)
    case_ref = case_reference_for_ticket(ticket)

    try:
        return send_phone_ivr_team_email(
            caller_phone=caller_phone,
            session_id=session_id,
            ticket_id=ticket_id,
            case_reference=case_ref,
            ticket_status=str(ticket.status or ""),
            issue_type=str(ticket.issue_type or ""),
            model_name=str(ticket.model_name or ""),
            current_node_id=str(ticket.current_node_id or ""),
            turns=turns,
            sms_sent=sms_sent,
        )
    except Exception:
        logger.exception(
            "RC follow-up team email failed ticket=%s session=%s",
            ticket_id,
            session_id,
        )
        return False


def try_claim_phone_followup(ticket_id: str) -> bool:
    """
    Atomically claim follow-up delivery for *ticket_id*.

    The claim expires after ten minutes so a crashed worker can be retried.
    Final delivery timestamps are written only after each channel succeeds.
    """
    if not ticket_id:
        return False

    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    with _followup_lock:
        with warranty_db_session() as db:
            # SQLite serializes competing workers here, preventing two exit
            # callbacks from both sending the same SMS.
            db.execute(text("BEGIN IMMEDIATE"))
            row = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if row is None:
                logger.warning("RC follow-up claim skipped — ticket not found ticket=%s", ticket_id)
                return False
            collected = row.get_collected()
            if collected.get("followup_sent_at"):
                logger.info("RC follow-up already completed ticket=%s", ticket_id)
                return False

            now = datetime.utcnow().replace(microsecond=0)
            claimed_at_raw = str(collected.get("followup_claimed_at") or "")
            if claimed_at_raw:
                try:
                    claimed_at = datetime.fromisoformat(claimed_at_raw)
                    if now - claimed_at < timedelta(minutes=10):
                        return False
                except ValueError:
                    pass

            collected["followup_claimed_at"] = now.isoformat()
            collected["followup_attempts"] = int(collected.get("followup_attempts") or 0) + 1
            row.collected_data = json.dumps(collected)
    return True


def _delivery_state(ticket_id: str) -> dict[str, bool]:
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    with warranty_db_session() as db:
        row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        collected = row.get_collected() if row is not None else {}
    return {
        "sms_sent": bool(collected.get("followup_sms_sent_at")),
        "email_sent": bool(collected.get("followup_email_sent_at")),
    }


def _finish_phone_followup(
    ticket_id: str,
    *,
    sms_sent: bool,
    email_sent: bool,
    sms_required: bool,
    email_required: bool,
) -> None:
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    with _followup_lock:
        with warranty_db_session() as db:
            db.execute(text("BEGIN IMMEDIATE"))
            row = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if row is None:
                return
            collected = row.get_collected()
            now = datetime.utcnow().replace(microsecond=0).isoformat()
            if sms_sent:
                collected["followup_sms_sent_at"] = now
            if email_sent:
                collected["followup_email_sent_at"] = now
            collected.pop("followup_claimed_at", None)

            sms_done = (not sms_required) or bool(collected.get("followup_sms_sent_at"))
            email_done = (not email_required) or bool(collected.get("followup_email_sent_at"))
            if sms_done and email_done:
                collected["followup_sent_at"] = now
                collected.pop("followup_last_error_at", None)
            else:
                collected["followup_last_error_at"] = now
            row.collected_data = json.dumps(collected)


def send_phone_call_followups(
    *,
    caller_phone: str,
    ticket_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Send post-call SMS + team email once per ticket (idempotent)."""
    if not try_claim_phone_followup(ticket_id):
        return {"skipped": True, "sms_sent": False, "email_sent": False}

    state = _delivery_state(ticket_id)
    sms_required = _followup_enabled()
    email_required = _team_email_enabled()
    sms_sent = state["sms_sent"]
    email_sent = state["email_sent"]

    if sms_required and not sms_sent:
        sms_sent = send_call_followup_sms(
            caller_phone,
            ticket_id=ticket_id,
            session_id=session_id,
        )
    if email_required and not email_sent:
        email_sent = send_call_followup_team_email(
            caller_phone=caller_phone,
            ticket_id=ticket_id,
            session_id=session_id,
            sms_sent=sms_sent,
        )
    _finish_phone_followup(
        ticket_id,
        sms_sent=sms_sent,
        email_sent=email_sent,
        sms_required=sms_required,
        email_required=email_required,
    )
    return {"skipped": False, "sms_sent": sms_sent, "email_sent": email_sent}
