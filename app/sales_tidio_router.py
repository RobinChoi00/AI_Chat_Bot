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

import hmac
import logging
import os
import re
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel, Field

from sales_agent import respond
from sales_intent import (
    INTENT_DISCOUNT,
    INTENT_HUMAN,
    WARRANTY_ROUTE_INTENTS,
)
from sales_models import (
    SalesLead,
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
from sales_tidio_buttons import (
    append_numbered_menu,
    flatten_buttons_for_flow,
    normalize_stored_buttons,
    prioritize_quick_replies,
    resolve_button_choice,
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
    # Flat text for Tidio Flow mappers (markdown stars stripped + numbered menu).
    reply_plain: str = ""
    # Flow branch helper:
    #   reply              → send reply_plain, stay in bot
    #   transfer_operator  → send reply_plain, then Transfer to operator
    #   warranty_redirect  → send reply_plain only (point to Warranty chat)
    next_action: str = "reply"
    # Boolean shortcut for Tidio Flows that can only branch on a single
    # variable at a time. True for Warranty-path intents (defect / parts /
    # shipping / tracking). Cancel/refund is False → transfer to an agent.
    # Tidio should show ``reply_plain`` and END the flow when True.
    is_warranty_route: bool = False
    # Static Decision branching (free Tidio plan — no dynamic buttons).
    # menu | ask_height | ask_weight | ask_space | ask_goal | recommend | …
    flow_stage: str = "menu"
    # Flat button fields for Flow session variables / static Decision nodes.
    button_count: int = 0
    button_1_label: str = ""
    button_1_payload: str = ""
    button_1_url: str = ""
    button_2_label: str = ""
    button_2_payload: str = ""
    button_2_url: str = ""
    button_3_label: str = ""
    button_3_payload: str = ""
    button_3_url: str = ""
    button_4_label: str = ""
    button_4_payload: str = ""
    button_4_url: str = ""
    button_5_label: str = ""
    button_5_payload: str = ""
    button_5_url: str = ""
    # True when the visitor message matched a prior button label/number.
    resolved_from_button: bool = False


def _strip_md(text: str) -> str:
    """Tidio chat shows plain text better without **bold** / markdown links."""
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", text or "")
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1: \2", out)
    out = re.sub(r"_([^_]+)_", r"\1", out)
    return out.strip()


def _next_action(intent: str, handoff: bool) -> str:
    if intent in WARRANTY_ROUTE_INTENTS:
        return "warranty_redirect"
    if intent in (INTENT_DISCOUNT, INTENT_HUMAN) or (
        handoff and intent not in WARRANTY_ROUTE_INTENTS
    ):
        return "transfer_operator"
    return "reply"


def _require_turn_secret(request: Request) -> None:
    """
    Optional shared secret for the Flow API Call node.

    Set TIDIO_TURN_SECRET on the server and the same value in the Flow
    header ``X-Tidio-Turn-Secret``. When unset, the endpoint stays open
    (rate-limited) so local/dev Flows keep working.
    """
    expected = (os.getenv("TIDIO_TURN_SECRET") or "").strip()
    if not expected:
        return
    received = (
        request.headers.get("X-Tidio-Turn-Secret")
        or request.headers.get("x-tidio-turn-secret")
        or ""
    ).strip()
    if not received or not hmac.compare_digest(expected, received):
        raise HTTPException(status_code=401, detail="Invalid turn secret.")


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
    from sales_models import get_session_collected, merge_session_collected

    prefs = get_session_collected(session_id)
    last_buttons = normalize_stored_buttons(prefs.get("last_quick_replies"))

    resolved_from_button = False
    effective_payload = (payload or "").strip() or None
    effective_message = (message or "").strip()
    if not effective_payload and effective_message:
        matched = resolve_button_choice(effective_message, last_buttons)
        if matched:
            effective_payload = matched
            resolved_from_button = True
            # Button label / number is not free-text preference content.
            effective_message = ""

    result = respond(
        effective_message,
        payload=effective_payload,
        domain=domain,
        prefs=prefs,
    )
    if result.prefs_patch:
        merge_session_collected(session_id, result.prefs_patch)

    if result.lead_capture and result.lead_capture.get("email"):
        from sales_models import SalesSession
        from warranty_models import warranty_db_session

        email = str(result.lead_capture.get("email") or "").strip()
        interest = str(result.lead_capture.get("interest_summary") or "").strip() or None
        reason = str(result.lead_capture.get("reason") or "save_pick").strip() or "save_pick"
        with warranty_db_session() as db:
            db.add(
                SalesLead(
                    session_id=session_id,
                    email=email or None,
                    phone=None,
                    domain=domain,
                    interest_summary=interest,
                    reason=reason,
                    forwarded="pending",
                )
            )
            session = (
                db.query(SalesSession)
                .filter(SalesSession.session_id == session_id)
                .one_or_none()
            )
            if session is not None:
                if email and not session.contact_email:
                    session.contact_email = email
                session.status = "handoff"

    buttons = prioritize_quick_replies(result.quick_replies)
    flat = flatten_buttons_for_flow(buttons)
    merge_session_collected(session_id, {"last_quick_replies": buttons})

    plain = append_numbered_menu(_strip_md(result.reply), buttons)
    action = _next_action(result.intent, result.handoff)
    is_warranty_route = result.intent in WARRANTY_ROUTE_INTENTS

    if message:
        record_message(
            session_id,
            role="user",
            content=message,
            intent=result.intent,
            tools_used=["quick_reply"] if resolved_from_button else None,
        )
    elif effective_payload:
        record_message(
            session_id,
            role="user",
            content=f"[button:{effective_payload}]",
            intent=result.intent,
            tools_used=["quick_reply"],
        )
    record_message(
        session_id,
        role="assistant",
        content=result.reply,
        intent=result.intent,
        handoff=result.handoff_reason if result.handoff else None,
        tools_used=result.tools_used + ["tidio", f"action:{action}", "tidio.buttons"],
    )
    update_session_last_intent(
        session_id,
        intent=result.intent,
        last_message=result.reply,
        status="handoff" if result.handoff else None,
    )
    return {
        "reply": result.reply,
        "reply_plain": plain,
        "intent": result.intent,
        "handoff": result.handoff,
        "handoff_reason": result.handoff_reason,
        "quick_replies": buttons,
        "session_id": session_id,
        "contact_id": contact_id,
        "next_action": action,
        "is_warranty_route": is_warranty_route,
        "resolved_from_button": resolved_from_button,
        "flow_stage": result.flow_stage or "menu",
        **flat,
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

    status, data = reply_to_ticket(ticket_id, content=turn["reply_plain"] or turn["reply"])
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
        "turn_secret_set": bool(os.getenv("TIDIO_TURN_SECRET", "").strip()),
        "openapi_configured": openapi_configured(),
        "operator_id_set": bool(operator_id()),
        "webhook_url_hint": "/api/v1/sales/tidio/webhook",
        "turn_url_hint": "/api/v1/sales/tidio/turn",
        "recommended_live_chat_path": "tidio_flow_http_request",
        "goal": "ai_first_24_7_before_human_agent",
        "buttons": {
            "max_quick_replies": int(os.getenv("TIDIO_MAX_QUICK_REPLIES", "5")),
            "reply_plain_includes_numbered_menu": True,
            "label_or_number_resolves_to_payload": True,
            "flow_note": (
                "Tidio Decision nodes are static — map button_N_label in Flow, "
                "or rely on numbered menu in reply_plain. See docs/tidio_flow_sales_buttons.md"
            ),
        },
    }


@router.post("/api/v1/sales/tidio/turn", response_model=TidioTurnResponse)
@limiter.limit(_SALES_CHAT_RATE)
async def tidio_turn(request: Request, body: TidioTurnRequest):
    """
    Tidio Flow / Bot HTTP Request target.

    Map the Flow's visitor message into ``message`` (and optional ``contact_id``).
    Use ``reply_plain`` as the bot's outgoing text, then branch on ``next_action``.
    """
    if not tidio_enabled():
        raise HTTPException(status_code=503, detail="Tidio sales adapter disabled.")
    _require_turn_secret(request)

    message = (body.message or "").strip()[:_MAX_MESSAGE_LEN]
    payload = (body.payload or "").strip()[:200] or None
    if not message and not payload:
        raise HTTPException(status_code=422, detail="Provide message or payload.")

    # Tidio Flow bug: Body typed as {{visitor_question}} instead of a variable chip.
    if re.fullmatch(r"\{\{[^}]+\}\}", message or ""):
        raise HTTPException(
            status_code=422,
            detail=(
                "message looks like an unsubstituted Tidio template "
                f"({message!r}). In the API Call Body, insert message via the {{}} "
                "chip picker (Ask → visitor_question), do not type braces by hand."
            ),
        )

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
    logger.info(
        "tidio_turn session=%s contact=%s msg=%r payload=%r intent=%s resolved=%s stage=%s",
        session_id,
        contact_id,
        (message or "")[:120],
        payload,
        result.get("intent"),
        result.get("resolved_from_button"),
        result.get("flow_stage"),
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
