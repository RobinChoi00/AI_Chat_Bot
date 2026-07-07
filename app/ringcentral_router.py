"""
ringcentral_router.py
=====================
RingCentral Automated Voice App webhook endpoints.

Register these callback URLs on the ApplicationExtension (IVR App):
  POST {PUBLIC_BASE_URL}/rc/on-call-enter
  POST {PUBLIC_BASE_URL}/rc/on-command-update
  POST {PUBLIC_BASE_URL}/rc/on-call-exit

Also serves cached TTS audio for the play API:
  GET /rc/audio/{cache_key}.wav

Webhook security (see ringcentral_webhook.py):
  RC_WEBHOOK_VERIFICATION_TOKEN — optional; must match Verification-Token header
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ringcentral"])

try:
    from app.ringcentral_ivr import (
        handle_call_enter,
        handle_call_exit,
        handle_command_update,
    )
    from app.ringcentral_voice import RC_AUDIO_CACHE_DIR
    from app.ringcentral_webhook import parse_rc_webhook_json, validation_token_response
except ImportError:
    from ringcentral_ivr import (  # type: ignore
        handle_call_enter,
        handle_call_exit,
        handle_command_update,
    )
    from ringcentral_voice import RC_AUDIO_CACHE_DIR  # type: ignore
    from ringcentral_webhook import parse_rc_webhook_json, validation_token_response  # type: ignore


def _run_safe(fn: Any, payload: dict[str, Any]) -> None:
    try:
        fn(payload)
    except Exception:
        logger.exception("RingCentral IVR handler failed")


async def _rc_webhook_payload(request: Request) -> dict[str, Any] | Response:
    """Validation-token echo or parsed JSON payload."""
    validation = validation_token_response(request)
    if validation is not None:
        return validation
    return await parse_rc_webhook_json(request)


@router.post("/rc/on-call-enter")
async def rc_on_call_enter(request: Request, background_tasks: BackgroundTasks):
    """
    RingCentral calls this when a customer reaches the IVR App extension.
    Must respond 204 immediately; play/collect run in a background task.
    """
    payload = await _rc_webhook_payload(request)
    if isinstance(payload, Response):
        return payload
    background_tasks.add_task(_run_safe, handle_call_enter, payload)
    return Response(status_code=204)


@router.post("/rc/on-command-update")
async def rc_on_command_update(request: Request, background_tasks: BackgroundTasks):
    payload = await _rc_webhook_payload(request)
    if isinstance(payload, Response):
        return payload
    background_tasks.add_task(_run_safe, handle_command_update, payload)
    return Response(status_code=204)


@router.post("/rc/on-call-exit")
async def rc_on_call_exit(request: Request, background_tasks: BackgroundTasks):
    payload = await _rc_webhook_payload(request)
    if isinstance(payload, Response):
        return payload
    background_tasks.add_task(_run_safe, handle_call_exit, payload)
    return Response(status_code=204)


@router.get("/rc/audio/{cache_key}.wav")
def rc_audio_file(cache_key: str):
    """Public HTTPS audio files for RingCentral play commands."""
    if not cache_key.isalnum() or len(cache_key) > 64:
        raise HTTPException(status_code=400, detail="Invalid cache key.")
    path = RC_AUDIO_CACHE_DIR / f"{cache_key}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(path, media_type="audio/wav")


@router.get("/rc/health")
def rc_health():
    """Quick check that the RingCentral IVR routes are mounted."""
    return {"status": "ok", "service": "ringcentral-ivr"}
