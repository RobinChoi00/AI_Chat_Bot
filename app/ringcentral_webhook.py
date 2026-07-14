"""
ringcentral_webhook.py
======================
Validate inbound RingCentral IVR webhook requests.

RingCentral setup
-----------------
1. **Validation-Token** — echoed on subscription / URL verification (required).
2. **Verification-Token** — optional shared secret configured in the RC app;
   set ``RC_WEBHOOK_VERIFICATION_TOKEN`` in ``.env`` to enforce on every POST.

Development may run without a verification token. Production fails closed
until the shared token is configured in RingCentral and the application.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BODY_BYTES = 256 * 1024


def _max_body_bytes() -> int:
    try:
        value = int(os.getenv("RC_WEBHOOK_MAX_BODY_BYTES", str(_DEFAULT_MAX_BODY_BYTES)))
    except ValueError:
        value = _DEFAULT_MAX_BODY_BYTES
    return max(1024, min(value, 1024 * 1024))


def _header(request: Request, name: str) -> str:
    return (request.headers.get(name) or request.headers.get(name.lower()) or "").strip()


def validation_token_response(request: Request) -> Optional[Response]:
    """
    Echo ``Validation-Token`` for RingCentral URL verification.

    Returns a 200 response when the header is present, else None.
    """
    token = _header(request, "Validation-Token")
    if not token:
        return None
    logger.info("RC webhook validation token echoed")
    return Response(
        status_code=200,
        headers={
            "Validation-Token": token,
            "Content-Type": "application/json",
        },
    )


def verify_webhook_request(request: Request) -> None:
    """Raise HTTP 401 when ``RC_WEBHOOK_VERIFICATION_TOKEN`` does not match."""
    expected = os.getenv("RC_WEBHOOK_VERIFICATION_TOKEN", "").strip()
    if not expected:
        if os.getenv("APP_ENV", "development").strip().lower() == "production":
            logger.error("RC webhook rejected — verification token is not configured")
            raise HTTPException(
                status_code=503,
                detail="RingCentral webhook authentication is not configured.",
            )
        return

    received = _header(request, "Verification-Token")
    if not received or not hmac.compare_digest(received, expected):
        logger.warning("RC webhook rejected — invalid Verification-Token")
        raise HTTPException(status_code=401, detail="Invalid verification token.")


async def parse_rc_webhook_json(request: Request) -> dict:
    """
    Validate and parse a RingCentral webhook POST body.

    Raises HTTPException on authentication, size, or JSON-shape failure.
    """
    verify_webhook_request(request)
    content_length = request.headers.get("content-length", "").strip()
    if content_length:
        try:
            if int(content_length) > _max_body_bytes():
                raise HTTPException(status_code=413, detail="Webhook body is too large.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

    body = await request.body()
    if len(body) > _max_body_bytes():
        raise HTTPException(status_code=413, detail="Webhook body is too large.")
    if not body:
        raise HTTPException(status_code=400, detail="Webhook JSON body is required.")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Invalid webhook JSON.")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Webhook JSON must be an object.")
    return payload


def validate_event_payload(route: str, payload: dict) -> None:
    """Reject malformed callbacks before they enter the durable inbox."""
    session_id = str(payload.get("sessionId") or "").strip()
    if not session_id or len(session_id) > 255:
        raise HTTPException(status_code=422, detail="sessionId is required.")
    if route == "on-call-enter":
        in_party = payload.get("inParty") or {}
        party_id = str(in_party.get("id") or payload.get("partyId") or "").strip()
        if not party_id or len(party_id) > 255:
            raise HTTPException(status_code=422, detail="partyId is required.")
    elif route == "on-command-update":
        command = str(payload.get("command") or "").strip()
        status = str(payload.get("status") or "").strip()
        if not command or not status:
            raise HTTPException(status_code=422, detail="command and status are required.")
