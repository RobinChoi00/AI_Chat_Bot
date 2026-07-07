"""
ringcentral_webhook.py
======================
Validate inbound RingCentral IVR webhook requests.

RingCentral setup
-----------------
1. **Validation-Token** — echoed on subscription / URL verification (required).
2. **Verification-Token** — optional shared secret configured in the RC app;
   set ``RC_WEBHOOK_VERIFICATION_TOKEN`` in ``.env`` to enforce on every POST.

When ``RC_WEBHOOK_VERIFICATION_TOKEN`` is unset, verification is skipped so
existing deployments keep working until the token is configured in RC + .env.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)


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
        return

    received = _header(request, "Verification-Token")
    if not received or not hmac.compare_digest(received, expected):
        logger.warning("RC webhook rejected — invalid Verification-Token")
        raise HTTPException(status_code=401, detail="Invalid verification token.")


async def parse_rc_webhook_json(request: Request) -> dict:
    """
    Validate and parse a RingCentral webhook POST body.

    Raises HTTPException on auth failure; returns {} for empty bodies.
    """
    verify_webhook_request(request)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        return payload
    return {}
