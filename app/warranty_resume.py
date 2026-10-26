"""
warranty_resume.py
==================
"Save & continue later" links for the guided warranty chat.

Design
------
- The customer taps *Save my progress*, enters an email address, and the
  server responds by emailing a signed resume URL.
- The URL contains HMAC-signed case/session identifiers and expiry, without a
  contact email. It is a bearer link; its payload is signed, not encrypted.
  Signing uses ``ADMIN_SESSION_SECRET`` — the same secret already
  used by the admin dashboard cookie.
- When the customer opens the link, the frontend calls
  ``GET /api/v1/warranty/resume/{token}`` which validates the signature and
  returns the ``ticket_id`` + original ``session_id``. The frontend then
  restores that session locally and the existing session endpoint hydrates
  chat state.

Uses the existing SQLAlchemy database and durable notification queue. Signed
tokens use the standard-library ``hmac`` + ``base64`` + ``json`` modules.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from pii_redact import mask_email

logger = logging.getLogger(__name__)

_TOKEN_MAX_AGE_SECS = 30 * 24 * 3600  # 30 days
_MIN_SECRET_LEN = 32
_SECRET_ENV = "ADMIN_SESSION_SECRET"


# ---------------------------------------------------------------------------
# Token helpers (HMAC-SHA256 over JSON payload)
# ---------------------------------------------------------------------------

def _get_secret() -> bytes:
    secret = os.getenv(_SECRET_ENV, "").strip()
    if not secret:
        raise RuntimeError(
            f"{_SECRET_ENV} must be set to enable warranty resume links."
        )
    if len(secret) < _MIN_SECRET_LEN:
        raise RuntimeError(
            f"{_SECRET_ENV} must be at least {_MIN_SECRET_LEN} characters."
        )
    return secret.encode("utf-8")


def resume_signing_ready() -> bool:
    """True when HMAC resume URLs can be minted (ADMIN_SESSION_SECRET is valid)."""
    try:
        _get_secret()
        return True
    except RuntimeError:
        return False


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def create_resume_token(
    ticket_id: str,
    session_id: str,
    ttl_secs: int = _TOKEN_MAX_AGE_SECS,
) -> str:
    """Return an HMAC-signed opaque token used in the resume URL."""
    payload = {
        "tid": ticket_id,
        "sid": session_id,
        "purpose": "warranty_resume",
        "exp": int(time.time()) + ttl_secs,
    }
    body = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64u_encode(
        hmac.new(_get_secret(), body.encode("utf-8"), hashlib.sha256).digest()
    )
    return f"{body}.{sig}"


def verify_resume_token(token: str) -> Optional[dict]:
    """Return the decoded payload or ``None`` if invalid / expired."""
    if not token or len(token) > 4096 or not token.isascii() or "." not in token:
        return None
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    try:
        expected = _b64u_encode(
            hmac.new(_get_secret(), body.encode("utf-8"), hashlib.sha256).digest()
        )
    except RuntimeError:
        return None
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_b64u_decode(body))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('purpose') not in (None, 'warranty_resume'):
        return None
    try:
        if int(payload.get("exp", 0)) <= int(time.time()):
            return None
    except (ValueError, TypeError, OverflowError):
        return None
    if not isinstance(payload.get("tid"), str) or not isinstance(payload.get("sid"), str):
        return None
    return payload


# ---------------------------------------------------------------------------
# Email transport and durable worker adapter
# ---------------------------------------------------------------------------

def _send_resume_email(
    customer_email: str,
    resume_url: str,
    ticket_id: str,
    model_name: str,
    domain: str,
) -> bool:
    """Send the resume link to ``customer_email`` over SMTP."""
    from config import EMAIL_PASSWORD, EMAIL_SENDER, SMTP_PORT, SMTP_SERVER  # noqa: WPS433

    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.warning(
            "warranty_resume: EMAIL_SENDER/EMAIL_PASSWORD not set — skipping send."
        )
        return False

    subject = "Your Osaki & Titan warranty progress — continue later"
    model_line = f"Chair model on file: {model_name}\n" if model_name else ""

    body = (
        "Hi,\n\n"
        "We saved your warranty case so you can pick up where you left off. "
        "Use the link below to continue on any device:\n\n"
        f"    {resume_url}\n\n"
        f"{model_line}"
        "This link is valid for 30 days. If you did not request it, you can "
        "safely ignore this message.\n\n"
        "— Osaki & Titan Warranty Team\n"
    )

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = customer_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        transport = smtplib.SMTP_SSL if int(SMTP_PORT) == 465 else smtplib.SMTP
        options = {"timeout": 15}
        if int(SMTP_PORT) == 465:
            options["context"] = ssl.create_default_context()
        with transport(SMTP_SERVER, SMTP_PORT, **options) as server:
            server.ehlo()
            if int(SMTP_PORT) != 465:
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            rejected = server.send_message(msg)
            if rejected:
                return False
        logger.info(
            "warranty_resume sent ticket=%s domain=%s to=%s",
            ticket_id,
            domain,
            mask_email(customer_email),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("warranty_resume send failed ticket=%s kind=%s", ticket_id, type(exc).__name__)
        return False


def _deliver_resume_email(*, attempt_id: str, **_legacy_kwargs) -> bool:
    """Resolve the reserved recipient and deliver at most once per active lease."""
    from warranty_resume_limits import claim_resume_attempt, finish_resume_attempt, supersede_resume_attempt

    attempt = claim_resume_attempt(attempt_id)
    if attempt is None:
        return True
    lease_token = attempt["lease_token"]
    ticket = _lazy_engine().get_ticket(attempt["ticket_id"])
    if ticket is None:
        finish_resume_attempt(attempt_id, sent=False, lease_token=lease_token)
        return False
    if ticket.get_collected().get("resume_delivery_attempt_id") != attempt_id:
        supersede_resume_attempt(attempt_id, lease_token=lease_token)
        return True
    resume_url = build_warranty_resume_url(str(ticket.ticket_id), str(ticket.session_id), str(ticket.domain or ""))
    if not resume_url:
        finish_resume_attempt(attempt_id, sent=False, lease_token=lease_token)
        return False
    try:
        sent = _send_resume_email(
            customer_email=attempt["recipient_email"], resume_url=resume_url,
            ticket_id=str(ticket.ticket_id), model_name=str(ticket.model_name or ""),
            domain=str(ticket.domain or ""),
        )
    except Exception as exc:
        logger.warning("Resume email attempt failed (%s)", type(exc).__name__)
        sent = False
    finish_resume_attempt(attempt_id, sent=bool(sent), lease_token=lease_token)
    return bool(sent)


def deliver_resume_attempt(attempt_id: str) -> None:
    """Durable notification-worker entry point; failures must remain retryable."""
    if not _deliver_resume_email(attempt_id=attempt_id):
        raise RuntimeError("Resume email delivery failed")


def _send_resume_email_async(**kwargs) -> None:
    from warranty_notification_jobs import enqueue_notification

    enqueue_notification(kind="resume", ticket_id=kwargs["ticket_id"], source_key=kwargs["attempt_id"])


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

router = APIRouter(tags=["warranty-resume"])


class ResumeLinkRequest(BaseModel):
    customer_email: str = Field(min_length=3, max_length=254)


def _lazy_engine():
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    from warranty_workflow import WarrantyEngine  # type: ignore

    return WarrantyEngine


def _resolve_resume_base_url(domain: str) -> str:
    """Pick the storefront base URL for the resume link."""
    explicit = os.getenv("WARRANTY_RESUME_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return explicit

    d = (domain or "").lower()
    if d == "phone":
        return os.getenv("WARRANTY_PHONE_RESUME_BASE_URL", "https://titanchair.com").rstrip("/")

    from store_config import get_storefront_base_url  # noqa: WPS433
    from warranty_defaults import normalize_warranty_domain  # noqa: WPS433

    return get_storefront_base_url(normalize_warranty_domain(domain))


def build_warranty_resume_url(
    ticket_id: str,
    session_id: str,
    domain: str = "",
) -> Optional[str]:
    """Return a signed resume URL for SMS/email, or None when signing is unavailable."""
    try:
        token = create_resume_token(ticket_id, session_id)
    except RuntimeError as exc:
        logger.warning("warranty resume URL unavailable ticket=%s: %s", ticket_id, exc)
        return None
    base = _resolve_resume_base_url(domain)
    return f"{base}/warranty?resume={token}"


@router.post("/api/v1/warranty/session/{session_id}/resume-link")
async def send_warranty_resume_link(request: Request, session_id: str, body: ResumeLinkRequest):
    """Email the customer a signed URL to resume the active warranty case."""
    from warranty_email import extract_email  # noqa: WPS433

    email = body.customer_email.strip().lower()
    if not email or extract_email(email) != email:
        raise HTTPException(status_code=422, detail="A valid email address is required.")

    engine = _lazy_engine()
    ticket = engine.get_active_session_ticket(session_id)
    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="No active warranty case for this session.",
        )
    ticket_id = str(ticket.ticket_id)
    from warranty_router import _require_web_chat_privacy_for_ticket

    _require_web_chat_privacy_for_ticket(ticket)

    resume_url = build_warranty_resume_url(ticket_id, session_id, str(ticket.domain or ""))
    if not resume_url:
        raise HTTPException(
            status_code=503,
            detail="Save & continue is temporarily unavailable.",
        )

    from warranty_resume_limits import reserve_resume_attempt

    attempt_id = str(uuid4())
    try:
        reserve_resume_attempt(
            attempt_id=attempt_id, ticket_id=ticket_id, session_id=session_id,
            recipient=email, client_ip=request.client.host if request.client else "unknown",
        )
    except SQLAlchemyError as exc:
        logger.warning("Resume request persistence failed kind=%s", type(exc).__name__)
        raise HTTPException(503, "Save & continue is temporarily unavailable. Please try again.",
                            headers={"Retry-After": "5"}) from None
    queued = True
    try:
        _send_resume_email_async(
            attempt_id=attempt_id,
            ticket_id=ticket_id,
        )
    except Exception as exc:
        # Admission already committed the queue row atomically. Its periodic
        # worker can recover even if the optional wake-up fails here.
        logger.warning("Resume worker wake-up failed (%s)", type(exc).__name__)

    return {
        "sent": False,
        "queued": queued,
        "resume_url": resume_url,
        "customer_email": mask_email(email),
        "email_saved": True,
        "expires_in_days": _TOKEN_MAX_AGE_SECS // 86400,
    }


@router.get("/api/v1/warranty/resume/{token}")
async def resume_from_token(token: str):
    """Validate a resume token and return the ticket + session mapping."""
    payload = verify_resume_token(token)
    if payload is None:
        raise HTTPException(status_code=400, detail="Invalid or expired resume link.")

    engine = _lazy_engine()
    ticket = engine.get_ticket(payload["tid"])
    if ticket is None:
        raise HTTPException(status_code=404, detail="Warranty case not found.")
    if str(ticket.session_id) != str(payload['sid']):
        raise HTTPException(status_code=400, detail="Invalid resume link.")

    return {
        "ticket_id": payload["tid"],
        "session_id": payload["sid"],
        "status": str(ticket.status),
        "domain": str(ticket.domain or ""),
        "expires_at": int(payload.get("exp", 0)),
    }
