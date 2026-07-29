"""
sales_tidio_router.py
=====================
HTTP surface that connects Tidio (OsakiUSA) to the Sales AI.

Endpoints
---------
POST /api/v1/sales/tidio/turn
    Called from a **Tidio Flow / Bot HTTP Request** step.
    Body: { contact_id?, message, session_id?, payload? }
    Returns: { reply, intent, handoff, quick_replies, … }
    The Flow then posts ``reply`` as the bot's chat message.

POST /api/v1/sales/tidio/webhook
    Tidio Webhooks stack target. Verifies ``x-tidio-signature``.
    Handles:
      - ticket.replied (contact) → Sales AI → OpenAPI ticket reply
      - conversation.operator_replied → mark session handoff (human took over)
      - other topics → ack only
    Responds 200 quickly; heavy work runs in BackgroundTasks.

GET  /api/v1/sales/tidio/health
    Config checklist (no secrets leaked).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel, Field

from sales_agent import respond
from sales_models import (
    get_or_create_session,
    record_message,
    update_session_last_intent,
)
from sales_tidio import (
    extract_contact_id,
    extract_ticket_id,
    extract_visitor_text,
    openapi_configured,
    operator_id,
    project_key_matches,
    reply_to_ticket,
    require_tidio_signature,
    tidio_domain,
    tidio_enabled,
)

logger = logging.getLogger(__name__)

try:
    from cost_guard import limiter
except ImportError:  # pragma: no cover

    class _NoopLimiter:
        def limit(self, *_args, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

    limiter = _NoopLimiter()


_SALES_CHAT_RATE = os.getenv("SALES_CHAT_RATE_LIMIT", "60/minute")
_MAX_MESSAGE_LEN = 2000

router = APIRouter(tags=["sales-tidio"])

# Topics we actively process (everything else is acknowledged).
_TICKET_REPLY_TOPICS = frozenset(
    {
        "ticket.replied",
        "ticket.contact_replied",
    }
)
_HUMAN_TAKEOVER_TOPICS = frozenset(
    {
        "conversation.operator_replied",
        "conversation.solved_by_operator",
    }
)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


class TidioTurnRequest(BaseModel):
    message: str = Field(default="", max_length=_MAX_MESSAGE_LEN)
    payload: Optional[str] = Field(default=None, max_length=200)
    contact_id: Optional[str] = Field(default=None, max_length=200)
    session_id: Optional[str] = Field(default=None, max_length=200)
    domain: Optional[str] = Field(default=None, max_length=200)


class TidioTurnResponse(BaseModel):
    reply: str
    intent: str
    handoff: bool
    handoff_reason: Optional[str] = None
    quick_replies: list[dict[str, str]] = Field(default_factory=list)
    session_id: str
    contact_id: Optional[str] = None
    # Flat text for Tidio Flow "use response body path" mappers that can't nest.
    reply_plain: str = ""


# ---------------------------------------------------------------------------
# Shared turn runner
# ---------------------------------------------------------------------------


def _run_sales_turn(
    *,
    session_id: str,
    message: str,
    payload: Optional[str],
    contact_id: Optional[str],
    domain: str,
) -> dict[str, Any]:
    get_or_create_session(
        session_id,
        domain=domain,
        channel="tidio",
        tidio_visitor_id=contact_id,
    )
    result = respond(message, payload=payload)

    if message:
        record_message(session_id, role="user", content=message, intent=result.intent)
    if payload:
        record_message(
            session_id,
            role="user",
            content=f"[button:{payload}]",
            intent=result.intent,
            tools_used=["quick_reply"],
        )
    record_message(
        session_id,
        role="assistant",
        content=result.reply,
        intent=result.intent,
        handoff=result.handoff_reason if result.handoff else None,
        tools_used=result.tools_used + ["tidio"],
    )
    update_session_last_intent(
        session_id,
        intent=result.intent,
        last_message=result.reply,
        status="handoff" if result.handoff else None,
    )
    return {
        "reply": result.reply,
        "reply_plain": result.reply,
        "intent": result.intent,
        "handoff": result.handoff,
        "handoff_reason": result.handoff_reason,
        "quick_replies": [
            {"label": q.label, "payload": q.payload} for q in result.quick_replies
        ],
        "session_id": session_id,
        "contact_id": contact_id,
    }


def _process_ticket_reply_event(payload: dict) -> None:
    content = payload.get("content") or {}
    if not isinstance(content, dict):
        return

    # Unified ticket.replied: only react to contact-authored public messages.
    message_obj = content.get("message")
    if isinstance(message_obj, dict):
        author = str(message_obj.get("author_type") or "").lower()
        if author and author != "contact":
            logger.info("tidio ticket.replied ignored author=%s", author)
            return

    text = extract_visitor_text(content)
    if not text:
        return

    contact_id = extract_contact_id(content) or "unknown"
    ticket_id = extract_ticket_id(content)
    session_id = f"tidio:{contact_id}"

    turn = _run_sales_turn(
        session_id=session_id,
        message=text[:_MAX_MESSAGE_LEN],
        payload=None,
        contact_id=contact_id if contact_id != "unknown" else None,
        domain=tidio_domain(),
    )

    if not ticket_id:
        logger.warning("tidio ticket event without ticket_id — reply not pushed")
        return
    if not openapi_configured():
        logger.warning("tidio openapi not configured — cannot push ticket reply")
        return

    status, data = reply_to_ticket(ticket_id, content=turn["reply"])
    logger.info(
        "tidio ticket reply pushed ticket=%s status=%s intent=%s",
        ticket_id,
        status,
        turn["intent"],
    )
    if status >= 400:
        logger.error("tidio ticket reply failed: %s", data)


def _process_human_takeover(payload: dict) -> None:
    content = payload.get("content") or {}
    contact_id = extract_contact_id(content) if isinstance(content, dict) else None
    if not contact_id:
        return
    session_id = f"tidio:{contact_id}"
    get_or_create_session(session_id, domain=tidio_domain(), channel="tidio", tidio_visitor_id=contact_id)
    update_session_last_intent(
        session_id,
        intent="human_takeover",
        last_message="[tidio operator joined]",
        status="handoff",
    )
    logger.info("tidio human takeover session=%s", session_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/api/v1/sales/tidio/health")
async def tidio_health():
    """Non-secret config checklist for ops."""
    return {
        "enabled": tidio_enabled(),
        "domain": tidio_domain(),
        "public_key_set": bool(os.getenv("TIDIO_PUBLIC_KEY", "").strip()),
        "webhook_secret_set": bool(
            os.getenv("TIDIO_WEBHOOK_SECRET", "").strip()
            or os.getenv("TIDIO_PRIVATE_KEY", "").strip()
        ),
        "openapi_configured": openapi_configured(),
        "operator_id_set": bool(operator_id()),
        "webhook_url_hint": "/api/v1/sales/tidio/webhook",
        "turn_url_hint": "/api/v1/sales/tidio/turn",
        "recommended_live_chat_path": "tidio_flow_http_request",
    }


@router.post("/api/v1/sales/tidio/turn", response_model=TidioTurnResponse)
@limiter.limit(_SALES_CHAT_RATE)
async def tidio_turn(request: Request, body: TidioTurnRequest):
    """
    Tidio Flow / Bot HTTP Request target.

    Map the Flow's visitor message into ``message`` (and optional ``contact_id``).
    Use ``reply`` (or ``reply_plain``) as the bot's outgoing text.
    """
    if not tidio_enabled():
        raise HTTPException(status_code=503, detail="Tidio sales adapter disabled.")

    message = (body.message or "").strip()[:_MAX_MESSAGE_LEN]
    payload = (body.payload or "").strip()[:200] or None
    if not message and not payload:
        raise HTTPException(status_code=422, detail="Provide message or payload.")

    contact_id = (body.contact_id or "").strip() or None
    session_id = (body.session_id or "").strip() or (
        f"tidio:{contact_id}" if contact_id else f"tidio:anon:{os.urandom(8).hex()}"
    )
    domain = (body.domain or tidio_domain()).strip()

    result = _run_sales_turn(
        session_id=session_id,
        message=message,
        payload=payload,
        contact_id=contact_id,
        domain=domain,
    )
    return TidioTurnResponse(**result)


@router.post("/api/v1/sales/tidio/webhook")
@limiter.limit(_SALES_CHAT_RATE)
async def tidio_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_tidio_signature: Optional[str] = Header(default=None),
):
    """Inbound Tidio webhook — must return 2xx within a few seconds."""
    if not tidio_enabled():
        raise HTTPException(status_code=503, detail="Tidio sales adapter disabled.")

    raw = await request.body()
    try:
        require_tidio_signature(raw, x_tidio_signature or "")
    except ValueError as exc:
        logger.warning("tidio webhook rejected: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid Tidio signature.") from exc

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Webhook body must be an object.")

    public_key = str(payload.get("project_public_key") or "")
    if not project_key_matches(public_key):
        raise HTTPException(status_code=401, detail="Unexpected project_public_key.")

    topic = str(payload.get("topic") or "")
    webhook_id = str(payload.get("webhook_id") or "")

    logger.info("tidio webhook topic=%s id=%s", topic, webhook_id)

    if topic in _TICKET_REPLY_TOPICS:
        background_tasks.add_task(_process_ticket_reply_event, payload)
    elif topic in _HUMAN_TAKEOVER_TOPICS:
        background_tasks.add_task(_process_human_takeover, payload)
    # Ack everything else so Tidio doesn't retry forever.

    return {"ok": True, "topic": topic, "webhook_id": webhook_id}
