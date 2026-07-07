"""
ringcentral_followup.py
=======================
Post-call SMS follow-up for after-hours warranty IVR callers.

Environment
-----------
RC_SMS_FROM_NUMBER          E.164 sender (Roman Warranty line)
RC_SMS_FOLLOWUP_ENABLED     true/false (default true when FROM is set)
RC_SMS_FOLLOWUP_MESSAGE     Optional override for the SMS body
"""

from __future__ import annotations

import logging
import os

from ringcentral_client import RC_SMS_FROM_NUMBER, send_sms

logger = logging.getLogger(__name__)

DEFAULT_FOLLOWUP_SMS = (
    "Thank you for calling Osaki and Titan warranty support. "
    "If you need additional help, please email service@osakititan.com "
    "with details about your current issue and we will contact you within 24 hours."
)


def build_followup_sms_message() -> str:
    custom = os.getenv("RC_SMS_FOLLOWUP_MESSAGE", "").strip()
    return custom or DEFAULT_FOLLOWUP_SMS


def _sms_from_number() -> str:
    return os.getenv("RC_SMS_FROM_NUMBER", RC_SMS_FROM_NUMBER).strip()


def _followup_enabled() -> bool:
    flag = os.getenv("RC_SMS_FOLLOWUP_ENABLED", "true").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return bool(_sms_from_number())


def send_call_followup_sms(caller_phone: str, *, ticket_id: str = "") -> bool:
    """Send the standard warranty follow-up SMS. Returns True when sent."""
    if not _followup_enabled():
        logger.debug("RC follow-up SMS skipped (disabled or no FROM number)")
        return False

    phone = (caller_phone or "").strip()
    if not phone.startswith("+"):
        logger.warning(
            "RC follow-up SMS skipped — invalid caller phone ticket=%s phone=%r",
            ticket_id,
            phone,
        )
        return False

    try:
        send_sms(
            to=phone,
            text=build_followup_sms_message(),
            from_number=_sms_from_number(),
        )
    except Exception:
        logger.exception(
            "RC follow-up SMS failed ticket=%s session_caller=%s",
            ticket_id,
            phone,
        )
        return False

    logger.info("RC follow-up SMS sent ticket=%s to=%s", ticket_id, phone)
    return True
