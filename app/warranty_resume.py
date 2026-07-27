"""
warranty_resume.py
==================
"Save & continue later" links for the guided warranty chat.

Design
------
- The customer taps *Save my progress*, enters an email address, and the
  server responds by emailing a signed resume URL.
- The URL contains an opaque HMAC-signed token (no plaintext PII / ticket ID
  exposed). Signing uses ``ADMIN_SESSION_SECRET`` — the same secret already
  used by the admin dashboard cookie.
- When the customer opens the link, the frontend calls
  ``GET /api/v1/warranty/resume/{token}`` which validates the signature and
  returns the ``ticket_id`` + original ``session_id``. The frontend then
  restores that session locally and the existing session endpoint hydrates
  chat state.

No new dependencies — pure ``hmac`` + ``base64`` + ``json``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
        "exp": int(time.time()) + ttl_secs,
    }
    body = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64u_encode(
        hmac.new(_get_secret(), body.encode("utf-8"), hashlib.sha256).digest()
    )
    return f"{body}.{sig}"


def verify_resume_token(token: str) -> Optional[dict]:
    """Return the decoded payload or ``None`` if invalid / expired."""
    if not token or "." not in token:
        return None
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    expected = _b64u_encode(
        hmac.new(_get_secret(), body.encode("utf-8"), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_b64u_decode(body))
    except Exception:  # noqa: BLE001
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    if not isinstance(payload.get("tid"), str) or not isinstance(payload.get("sid"), str):
        return None
    return payload


# ---------------------------------------------------------------------------
# Email helper (fire-and-forget)
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
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(
            "warranty_resume sent ticket=%s domain=%s to=%s",
            ticket_id,
            domain,
            mask_email(customer_email),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("warranty_resume send failed ticket=%s err=%s", ticket_id, exc)
        return False


def _send_resume_email_async(**kwargs) -> None:
    threading.Thread(target=_send_resume_email, kwargs=kwargs, daemon=True).start()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

router = APIRouter(tags=["warranty-resume"])


class ResumeLinkRequest(BaseModel):
    customer_email: str


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
async def send_warranty_resume_link(session_id: str, body: ResumeLinkRequest):
    """Email the customer a signed URL to resume the active warranty case."""
    from warranty_email import extract_email  # noqa: WPS433

    email = extract_email(body.customer_email or "")
    if not email:
        raise HTTPException(status_code=422, detail="A valid email address is required.")

    engine = _lazy_engine()
    ticket = engine.get_active_session_ticket(session_id)
    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="No active warranty case for this session.",
        )
    ticket_id = str(ticket.ticket_id)

    resume_url = build_warranty_resume_url(ticket_id, session_id, str(ticket.domain or ""))
    if not resume_url:
        raise HTTPException(
            status_code=503,
            detail="Save & continue is temporarily unavailable.",
        )

    ticket.set_collected("resume_email", email)  # detached ORM copy update (best-effort)
    _send_resume_email_async(
        customer_email=email,
        resume_url=resume_url,
        ticket_id=ticket_id,
        model_name=str(ticket.model_name or ""),
        domain=str(ticket.domain or ""),
    )

    return {
        "sent": True,
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

    return {
        "ticket_id": payload["tid"],
        "session_id": payload["sid"],
        "status": str(ticket.status),
        "domain": str(ticket.domain or ""),
        "expires_at": int(payload.get("exp", 0)),
    }
