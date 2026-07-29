"""
sales_tidio.py
==============
Tidio ↔ Sales AI glue: webhook signature verification + OpenAPI client.

Secrets (server ``.env`` only — never commit):
  TIDIO_PUBLIC_KEY              Project data → Public key
  TIDIO_PRIVATE_KEY             Project data → Private key (fallback webhook secret)
  TIDIO_WEBHOOK_SECRET          Webhooks stack secret (preferred for signatures)
  TIDIO_OPENAPI_CLIENT_ID       Developer → OpenAPI → ci_…
  TIDIO_OPENAPI_CLIENT_SECRET   Developer → OpenAPI → cs_…
  TIDIO_OPERATOR_ID             Operator UUID used when replying to tickets
  TIDIO_DOMAIN                  Default storefront domain (osakiusa.com)

How replies get back into Tidio
-------------------------------
Tidio's public OpenAPI does **not** expose a first-class "post as bot in live
chat" endpoint. Practical paths we support:

1. **Tidio Flow / Bot HTTP Request** → ``POST /api/v1/sales/tidio/turn``
   Returns ``{reply, handoff, …}``. The Flow's next step sends that text
   as the bot message. This is the recommended MVP for live chat.

2. **Ticket channel** → webhook ``ticket.replied`` (contact) → we reply with
   ``POST /tickets/{id}/reply`` as the configured operator.

3. OpenAPI changes do **not** re-trigger webhooks (Tidio docs), so we won't
   loop on our own replies.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_TIDIO_API_BASE = os.getenv("TIDIO_API_BASE", "https://api.tidio.co").rstrip("/")
_ACCEPT = "application/json; version=1"
_MAX_SIGNATURE_SKEW_SECONDS = 5 * 60


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def tidio_enabled() -> bool:
    return os.getenv("TIDIO_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def tidio_domain() -> str:
    return (os.getenv("TIDIO_DOMAIN") or "osakiusa.com").strip()


def _public_key() -> str:
    return (os.getenv("TIDIO_PUBLIC_KEY") or "").strip()


def _webhook_secret() -> str:
    """Prefer the Webhooks-stack secret; fall back to Project private key."""
    return (
        (os.getenv("TIDIO_WEBHOOK_SECRET") or "").strip()
        or (os.getenv("TIDIO_PRIVATE_KEY") or "").strip()
    )


def _openapi_headers() -> Optional[dict[str, str]]:
    client_id = (os.getenv("TIDIO_OPENAPI_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("TIDIO_OPENAPI_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        return None
    return {
        "X-Tidio-Openapi-Client-Id": client_id,
        "X-Tidio-Openapi-Client-Secret": client_secret,
        "Accept": _ACCEPT,
        "Content-Type": "application/json",
    }


def openapi_configured() -> bool:
    return _openapi_headers() is not None


def operator_id() -> str:
    return (os.getenv("TIDIO_OPERATOR_ID") or "").strip()


# ---------------------------------------------------------------------------
# Webhook signature (Tidio docs: body + '_' + timestamp, HMAC-SHA256)
# ---------------------------------------------------------------------------


def _extract_timestamp(header: str) -> int:
    for item in (header or "").split(","):
        key, _, value = item.partition("=")
        if key.strip() == "t":
            if not value.strip().isdigit():
                raise ValueError("invalid tidio signature timestamp")
            return int(value.strip())
    raise ValueError("missing tidio signature timestamp")


def _extract_signatures(header: str) -> list[str]:
    out: list[str] = []
    for item in (header or "").split(","):
        key, _, value = item.partition("=")
        if key.strip() == "s" and value.strip():
            out.append(value.strip())
    if not out:
        raise ValueError("missing tidio signatures")
    return out


def verify_tidio_signature(
    *,
    body: bytes | str,
    signature_header: str,
    secret: Optional[str] = None,
    now: Optional[float] = None,
) -> bool:
    """Return True when ``x-tidio-signature`` matches the webhook secret."""
    secret = (secret if secret is not None else _webhook_secret()).strip()
    if not secret:
        return False
    if isinstance(body, bytes):
        body_text = body.decode("utf-8")
    else:
        body_text = body

    try:
        timestamp = _extract_timestamp(signature_header)
        signatures = _extract_signatures(signature_header)
    except ValueError:
        return False

    skew = abs((now if now is not None else time.time()) - timestamp)
    if skew > _MAX_SIGNATURE_SKEW_SECONDS:
        logger.warning("tidio webhook signature timestamp skew=%.0fs", skew)
        return False

    payload = f"{body_text}_{timestamp}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, digestmod="sha256").hexdigest()
    return any(hmac.compare_digest(expected, sig) for sig in signatures)


def require_tidio_signature(body: bytes, signature_header: str) -> None:
    """Raise ValueError when signature verification fails (caller → HTTP 401)."""
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    secret = _webhook_secret()
    if not secret:
        if app_env == "production":
            raise ValueError("TIDIO_WEBHOOK_SECRET / TIDIO_PRIVATE_KEY not configured")
        logger.warning("tidio webhook signature skipped — no secret configured")
        return
    if not verify_tidio_signature(body=body, signature_header=signature_header, secret=secret):
        raise ValueError("invalid tidio webhook signature")


def project_key_matches(payload_public_key: str) -> bool:
    expected = _public_key()
    if not expected:
        return True
    return hmac.compare_digest(expected, (payload_public_key or "").strip())


# ---------------------------------------------------------------------------
# OpenAPI client
# ---------------------------------------------------------------------------


def _request(
    method: str,
    path: str,
    *,
    json_body: Optional[dict] = None,
    timeout: float = 15.0,
) -> tuple[int, Any]:
    headers = _openapi_headers()
    if headers is None:
        return 503, {"error": "openapi_not_configured"}
    url = f"{_TIDIO_API_BASE}{path}"
    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            json=json_body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logger.exception("tidio openapi request failed: %s %s", method, path)
        return 502, {"error": str(exc)}
    try:
        data = resp.json() if resp.content else {}
    except ValueError:
        data = {"raw": resp.text[:500]}
    return resp.status_code, data


def fetch_contact_messages(contact_id: str) -> tuple[int, Any]:
    return _request("GET", f"/contacts/{contact_id}/messages")


def reply_to_ticket(
    ticket_id: int | str,
    *,
    content: str,
    operator_uuid: Optional[str] = None,
) -> tuple[int, Any]:
    """Post a public operator reply on a Tidio ticket."""
    op = (operator_uuid or operator_id()).strip()
    body: dict[str, Any] = {
        "author_type": "operator",
        "content": content[:5000],
        "message_type": "public",
    }
    if op:
        body["operator_id"] = op
    return _request("POST", f"/tickets/{ticket_id}/reply", json_body=body)


def send_contact_side_message(contact_id: str, message: str) -> tuple[int, Any]:
    """
    POST /contacts/{id}/messages — documented as *on behalf of the contact*.

    Kept for completeness; not used for bot replies (wrong author).
    """
    return _request(
        "POST",
        f"/contacts/{contact_id}/messages",
        json_body={"message": message[:5000]},
    )


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def extract_visitor_text(content: dict) -> Optional[str]:
    """Best-effort extract of customer-authored text from a webhook content blob."""
    if not isinstance(content, dict):
        return None

    # conversation.operator_replied shape (operator) — skip unless forced.
    message = content.get("message")
    if isinstance(message, dict):
        # ticket.replied unified shape
        author = str(message.get("author_type") or "").lower()
        text = (
            message.get("message_content")
            or message.get("message")
            or message.get("content")
            or ""
        )
        text = str(text).strip()
        if text and author in ("", "contact"):
            return text
        if text and author == "operator":
            return None

    # ticket.* legacy: messages[] array — take last contact public message
    messages = content.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            author = str(item.get("author_type") or "").lower()
            if author and author != "contact":
                continue
            text = str(
                item.get("message_content")
                or item.get("message")
                or item.get("content")
                or ""
            ).strip()
            if text:
                return text

    for key in ("message", "text", "body", "content"):
        val = content.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_contact_id(content: dict) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    for key in ("contact_id", "id"):
        val = content.get(key)
        if isinstance(val, str) and val.strip():
            # contact.created uses "id"; conversation.* uses contact_id
            if key == "id" and content.get("email") is None and content.get("contact_id"):
                continue
            return val.strip()
    message = content.get("message")
    if isinstance(message, dict):
        author = str(message.get("author_type") or "").lower()
        author_id = message.get("author_id")
        if author == "contact" and isinstance(author_id, str) and author_id.strip():
            return author_id.strip()
    return None


def extract_ticket_id(content: dict) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    if content.get("ticket_id") is not None:
        return str(content["ticket_id"]).strip()
    # ticket.* payloads often use numeric "id" alongside messages[]
    if "messages" in content or (
        isinstance(content.get("message"), dict)
        and "message_content" in (content.get("message") or {})
    ):
        raw = content.get("id")
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None
