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
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ringcentral"])
_worker_lock = threading.Lock()
_worker_stop = threading.Event()
_worker_thread: threading.Thread | None = None
_session_locks_guard = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}

try:
    from app.ringcentral_ivr import (
        handle_call_enter,
        handle_call_exit,
        handle_command_update,
    )
    from app.ringcentral_voice import RC_AUDIO_CACHE_DIR
    from app.ringcentral_store import (
        enqueue_event,
        call_state_stats,
        cleanup_completed_events,
        event_stats,
        last_webhook_received_at,
        process_event,
        process_pending_events,
        release_session_retries,
    )
    from app.ringcentral_webhook import (
        parse_rc_webhook_json,
        validate_event_payload,
        validation_token_response,
    )
except ImportError:
    from ringcentral_ivr import (  # type: ignore
        handle_call_enter,
        handle_call_exit,
        handle_command_update,
    )
    from ringcentral_voice import RC_AUDIO_CACHE_DIR  # type: ignore
    from ringcentral_store import (  # type: ignore
        enqueue_event,
        call_state_stats,
        cleanup_completed_events,
        event_stats,
        last_webhook_received_at,
        process_event,
        process_pending_events,
        release_session_retries,
    )
    from ringcentral_webhook import (  # type: ignore
        parse_rc_webhook_json,
        validate_event_payload,
        validation_token_response,
    )


def _session_lock(session_id: str) -> threading.Lock:
    key = session_id or "unknown"
    with _session_locks_guard:
        lock = _session_locks.get(key)
        if lock is None:
            # Bound lock bookkeeping without removing locks currently held by
            # in-flight callbacks.
            if len(_session_locks) >= 10_000:
                for old_key, old_lock in list(_session_locks.items())[:1_000]:
                    if not old_lock.locked():
                        _session_locks.pop(old_key, None)
            lock = threading.Lock()
            _session_locks[key] = lock
        return lock


def _serialized(handler: Any, payload: dict[str, Any]) -> None:
    with _session_lock(str(payload.get("sessionId") or "")):
        handler(payload)


def _handlers() -> dict[str, Any]:
    # Build dynamically so tests and emergency instrumentation can patch a
    # handler without replacing the durable-inbox plumbing.
    return {
        "on-call-enter": lambda payload: _serialized(handle_call_enter, payload),
        "on-command-update": lambda payload: _serialized(handle_command_update, payload),
        "on-call-exit": lambda payload: _serialized(handle_call_exit, payload),
    }


def _worker_loop() -> None:
    last_cleanup = 0.0
    while not _worker_stop.is_set():
        try:
            process_pending_events(_handlers(), limit=50)
            if time.monotonic() - last_cleanup >= 3600:
                try:
                    retention = int(os.getenv("RC_EVENT_RETENTION_DAYS", "30"))
                except ValueError:
                    retention = 30
                cleanup_completed_events(retention_days=retention)
                last_cleanup = time.monotonic()
        except Exception:
            logger.exception("RingCentral durable event worker iteration failed")
        _worker_stop.wait(2.0)


def start_event_worker() -> None:
    """Recover persisted callbacks, including rows left by a process restart."""
    global _worker_thread
    if os.getenv("RC_EVENT_WORKER_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_stop.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            name="ringcentral-event-worker",
            daemon=True,
        )
        _worker_thread.start()


def stop_event_worker() -> None:
    _worker_stop.set()
    worker = _worker_thread
    if worker is not None:
        worker.join(timeout=3.0)


def _process_accepted_event(event_id: int, route_name: str, session_id: str) -> None:
    handlers = _handlers()
    completed = process_event(event_id, handlers)
    if completed and route_name == "on-call-enter":
        release_session_retries(session_id)
    # Also recover older due rows. This makes a new valid callback self-heal
    # events accepted immediately before a process restart or transient outage.
    process_pending_events(handlers, limit=24)


async def _rc_webhook_payload(request: Request) -> dict[str, Any] | Response:
    """Validation-token echo or parsed JSON payload."""
    validation = validation_token_response(request)
    if validation is not None:
        return validation
    return await parse_rc_webhook_json(request)


async def _accept_event(
    route_name: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    payload = await _rc_webhook_payload(request)
    if isinstance(payload, Response):
        return payload
    validate_event_payload(route_name, payload)
    event_id, _created = enqueue_event(route_name, payload)
    background_tasks.add_task(
        _process_accepted_event,
        event_id,
        route_name,
        str(payload.get("sessionId") or ""),
    )
    return Response(status_code=204)


@router.post("/rc/on-call-enter")
async def rc_on_call_enter(request: Request, background_tasks: BackgroundTasks):
    """
    RingCentral calls this when a customer reaches the IVR App extension.
    Must respond 204 immediately; play/collect run in a background task.
    """
    return await _accept_event("on-call-enter", request, background_tasks)


@router.post("/rc/on-command-update")
async def rc_on_command_update(request: Request, background_tasks: BackgroundTasks):
    return await _accept_event("on-command-update", request, background_tasks)


@router.post("/rc/on-call-exit")
async def rc_on_call_exit(request: Request, background_tasks: BackgroundTasks):
    return await _accept_event("on-call-exit", request, background_tasks)


@router.get("/rc/audio/{cache_key}.wav")
def rc_audio_file(cache_key: str):
    """Public HTTPS audio files for RingCentral play commands."""
    if not cache_key.isalnum() or len(cache_key) > 64:
        raise HTTPException(status_code=400, detail="Invalid cache key.")
    path = RC_AUDIO_CACHE_DIR / f"{cache_key}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/rc/health")
def rc_health():
    """Configuration and durable-inbox health for monitoring."""
    worker_enabled = os.getenv("RC_EVENT_WORKER_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    worker_alive = bool(_worker_thread is not None and _worker_thread.is_alive())
    required = {
        "RC_CLIENT_ID": bool(os.getenv("RC_CLIENT_ID", "").strip()),
        "RC_CLIENT_SECRET": bool(os.getenv("RC_CLIENT_SECRET", "").strip()),
        "RC_AUTH": bool(
            os.getenv("RC_USER_JWT", "").strip()
            or os.getenv("RC_USER_JWT_FILE", "").strip()
            or os.getenv("RC_JWT_PRIVATE_KEY", "").strip()
        ),
        "PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL", "").strip().startswith("https://"),
        "RC_TRANSFER_TARGET": bool(
            os.getenv("RC_WARRANTY_TRANSFER_EXTENSION", "").strip()
            or os.getenv("RC_WARRANTY_TRANSFER_TO", "").strip()
        ),
        "RC_SMS_FROM_NUMBER": bool(os.getenv("RC_SMS_FROM_NUMBER", "").strip()),
        "EVENT_WORKER": worker_enabled and worker_alive,
    }
    if os.getenv("APP_ENV", "development").strip().lower() == "production":
        required["RC_WEBHOOK_VERIFICATION_TOKEN"] = bool(
            os.getenv("RC_WEBHOOK_VERIFICATION_TOKEN", "").strip()
        )
    stats = event_stats()
    calls = call_state_stats()
    last_webhook = last_webhook_received_at()
    healthy = (
        all(required.values())
        and stats.get("dead_letter", 0) == 0
        and calls.get("stale_over_24h", 0) == 0
    )
    payload = {
        "status": "ok" if healthy else "degraded",
        "service": "ringcentral-ivr",
        "checks": required,
        "worker": {"enabled": worker_enabled, "alive": worker_alive},
        "events": stats,
        "calls": calls,
        "last_webhook_received_at": last_webhook,
        "live_call_blocker": (
            "ApplicationExtension must be enabled by RingCentral and routed "
            "from the main menu Warranty option before live after-hours calls work."
        ),
        "live_e2e_checklist": [
            "Confirm RC ApplicationExtension activation email completed",
            "Roman: route after-hours Warranty key to Osaki Warranty IVR app",
            "Place a closed-hours test call to 888-848-2630 → Warranty",
            "Confirm SMS follow-up + team email after hangup",
            "Confirm last_webhook_received_at updates on /rc/health",
        ],
    }
    if not healthy and os.getenv("APP_ENV", "development").strip().lower() == "production":
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=payload)
    return payload
