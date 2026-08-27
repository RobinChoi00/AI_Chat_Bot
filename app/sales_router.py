"""
sales_router.py
===============
FastAPI router for the Sales AI (Tidio-backed) chat.

Endpoints
---------
Public (rate-limited):
    POST   /api/v1/sales/chat           – main turn endpoint used by Tidio
    POST   /api/v1/sales/lead           – capture email/phone for handoff
    GET    /api/v1/sales/session/{id}   – resume: return latest messages + status

Admin (X-Admin-Key required):
    GET    /admin/sales/sessions        – list recent conversations
    GET    /admin/sales/sessions/{id}   – full transcript + leads
    GET    /admin/sales/leads           – lead queue

Design principles
-----------------
- Zero LLM calls in this router — all replies are produced by
  ``sales_agent.respond`` which is deterministic and testable.
- Guardrail intents (warranty/cancel/parts/discount/ETA) are flagged as
  ``handoff=True`` in the response so the storefront can surface a
  "Talk to a human" CTA and/or reveal the warranty chat launcher.
- PII masking is applied to every email echoed back to the client and to
  every value logged.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel, Field

try:
    from app.admin_auth import require_admin_key  # type: ignore
except ImportError:  # pragma: no cover
    from admin_auth import require_admin_key

from pii_redact import mask_email, mask_phone
from sales_agent import SalesReply, respond
from sales_intent import HANDOFF_INTENTS
from sales_models import (
    SalesLead,
    SalesMessage,
    SalesSession,
    get_or_create_session,
    record_message,
    update_session_last_intent,
)
from warranty_models import warranty_db_session

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
_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

_MAX_MESSAGE_LEN = 2000
_EMAIL_RE = re.compile(r"^[\w\.\-+]+@[\w\.\-]+\.\w{2,}$")


router = APIRouter(tags=["sales"])


# ---------------------------------------------------------------------------
# Request / Response DTOs
# ---------------------------------------------------------------------------


class SalesChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=200)
    message: str = Field(default="", max_length=_MAX_MESSAGE_LEN)
    payload: Optional[str] = Field(default=None, max_length=200)
    tidio_visitor_id: Optional[str] = Field(default=None, max_length=200)
    domain: Optional[str] = Field(default=None, max_length=200)
    channel: Optional[str] = Field(default=None, max_length=50)


class QuickReplyDTO(BaseModel):
    label: str
    payload: str


class SalesChatResponse(BaseModel):
    reply: str
    intent: str
    handoff: bool
    handoff_reason: Optional[str] = None
    quick_replies: list[QuickReplyDTO] = Field(default_factory=list)
    products: list[dict] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    session_id: str


class SalesLeadRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=200)
    email: Optional[str] = Field(default=None, max_length=200)
    phone: Optional[str] = Field(default=None, max_length=50)
    interest_summary: Optional[str] = Field(default=None, max_length=2000)
    reason: Optional[str] = Field(default=None, max_length=50)
    domain: Optional[str] = Field(default=None, max_length=200)


class SalesLeadResponse(BaseModel):
    ok: bool
    lead_id: int
    email: Optional[str] = None
    phone: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_admin(x_admin_key: Optional[str]) -> None:
    require_admin_key(x_admin_key, _ADMIN_API_KEY)


def _sanitize_message(text: str) -> str:
    return (text or "").strip()[:_MAX_MESSAGE_LEN]


def _reply_to_response(reply: SalesReply, session_id: str) -> SalesChatResponse:
    return SalesChatResponse(
        reply=reply.reply,
        intent=reply.intent,
        handoff=reply.handoff or reply.intent in HANDOFF_INTENTS,
        handoff_reason=reply.handoff_reason,
        quick_replies=[
            QuickReplyDTO(label=q.label, payload=q.payload) for q in reply.quick_replies
        ],
        products=reply.products,
        tools_used=reply.tools_used,
        session_id=session_id,
    )


def _fire_lead_email(
    *,
    email: str,
    interest_summary: str,
    domain: str,
    lead_id: int,
) -> None:
    """Send the lead notification email in the background — best-effort."""
    try:
        from main import send_sales_lead_email  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from app.main import send_sales_lead_email  # type: ignore
        except ImportError:
            logger.warning("sales lead capture: email transport unavailable")
            _mark_lead_status(lead_id, status="failed", error="transport_unavailable")
            return

    try:
        ok = bool(
            send_sales_lead_email(
                email,
                interest_summary or "Sales AI (Tidio) — customer requested follow-up",
                "",
                domain or "",
            )
        )
    except Exception as exc:  # pragma: no cover — email side-effects
        logger.exception("sales lead email failed: %s", exc)
        _mark_lead_status(lead_id, status="failed", error=str(exc)[:500])
        return

    _mark_lead_status(lead_id, status="sent" if ok else "failed", error=None if ok else "smtp_returned_false")


def _mark_lead_status(lead_id: int, *, status: str, error: Optional[str]) -> None:
    with warranty_db_session() as db:
        row = db.query(SalesLead).filter(SalesLead.id == lead_id).one_or_none()
        if row is None:
            return
        row.forwarded = status
        row.forwarded_error = error


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------


@router.post("/api/v1/sales/chat", response_model=SalesChatResponse)
@limiter.limit(_SALES_CHAT_RATE)
async def sales_chat(
    request: Request,
    body: SalesChatRequest,
    background_tasks: BackgroundTasks,
):  # noqa: WPS231 — router glue
    """Main sales chat turn — deterministic, no LLM."""
    message = _sanitize_message(body.message)
    payload = (body.payload or "").strip()[:200] or None

    if not message and not payload:
        raise HTTPException(
            status_code=422,
            detail="Provide `message` or `payload` (button choice).",
        )

    domain = body.domain or "osakiusa.com"
    session_row = get_or_create_session(
        session_id=body.session_id,
        domain=domain,
        channel=body.channel or "tidio",
        tidio_visitor_id=body.tidio_visitor_id,
    )

    from sales_models import get_session_collected, merge_session_collected

    prefs = get_session_collected(body.session_id)
    from sales_tidio_buttons import (
        normalize_stored_buttons,
        prioritize_quick_replies,
        resolve_button_choice,
    )

    effective_payload = payload
    effective_message = message
    if not effective_payload and effective_message:
        matched = resolve_button_choice(
            effective_message,
            normalize_stored_buttons(prefs.get("last_quick_replies")),
        )
        if matched:
            effective_payload = matched
            effective_message = ""

    reply = respond(
        effective_message,
        payload=effective_payload,
        domain=domain,
        prefs=prefs,
    )
    if reply.prefs_patch:
        merge_session_collected(body.session_id, reply.prefs_patch)
    merge_session_collected(
        body.session_id,
        {"last_quick_replies": prioritize_quick_replies(reply.quick_replies)},
    )

    if reply.lead_capture and reply.lead_capture.get("email"):
        email = str(reply.lead_capture.get("email") or "").strip()
        interest = str(reply.lead_capture.get("interest_summary") or "").strip() or None
        reason = str(reply.lead_capture.get("reason") or "save_pick").strip() or "save_pick"
        with warranty_db_session() as db:
            row = SalesLead(
                session_id=body.session_id,
                email=email or None,
                phone=None,
                domain=domain,
                interest_summary=interest,
                reason=reason,
                forwarded="pending",
            )
            db.add(row)
            db.flush()
            lead_id = row.id
            session = (
                db.query(SalesSession)
                .filter(SalesSession.session_id == body.session_id)
                .one_or_none()
            )
            if session is not None:
                if email and not session.contact_email:
                    session.contact_email = email
                session.status = "handoff"
        if email:
            background_tasks.add_task(
                _fire_lead_email,
                email=email,
                interest_summary=interest or "",
                domain=domain,
                lead_id=lead_id,
            )

    if message:
        record_message(body.session_id, role="user", content=message, intent=reply.intent)
    if payload:
        record_message(
            body.session_id,
            role="user",
            content=f"[button:{payload}]",
            intent=reply.intent,
            tools_used=["quick_reply"],
        )
    record_message(
        body.session_id,
        role="assistant",
        content=reply.reply,
        intent=reply.intent,
        handoff=reply.handoff_reason if reply.handoff else None,
        tools_used=reply.tools_used,
    )

    session_status = "handoff" if reply.handoff else session_row.status
    update_session_last_intent(
        body.session_id,
        intent=reply.intent,
        last_message=reply.reply,
        status=session_status if session_status != "closed" else None,
    )

    logger.info(
        "sales_chat session=%s intent=%s handoff=%s tools=%s",
        body.session_id,
        reply.intent,
        reply.handoff,
        reply.tools_used,
    )
    return _reply_to_response(reply, body.session_id)


@router.post("/api/v1/sales/lead", response_model=SalesLeadResponse)
@limiter.limit(_SALES_CHAT_RATE)
async def sales_lead(
    request: Request,
    body: SalesLeadRequest,
    background_tasks: BackgroundTasks,
):
    """Capture a sales lead for human follow-up."""
    email = (body.email or "").strip()
    phone = (body.phone or "").strip()
    if not email and not phone:
        raise HTTPException(
            status_code=422, detail="Provide at least one of `email` or `phone`."
        )
    if email and not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Invalid email format.")

    domain = (body.domain or "unknown").strip()
    reason = (body.reason or "").strip().lower() or None

    # Make sure a session row exists so admin dashboards can pivot on it.
    get_or_create_session(body.session_id, domain=domain, channel="tidio")

    from sales_models import get_session_collected

    collected = get_session_collected(body.session_id)
    interest = (body.interest_summary or "").strip() or (
        str(collected.get("pending_pick_summary") or "").strip() or None
    )

    with warranty_db_session() as db:
        row = SalesLead(
            session_id=body.session_id,
            email=email or None,
            phone=phone or None,
            domain=domain,
            interest_summary=interest,
            reason=reason,
            forwarded="pending",
        )
        db.add(row)
        db.flush()
        lead_id = row.id

        session = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == body.session_id)
            .one_or_none()
        )
        if session is not None:
            if email and not session.contact_email:
                session.contact_email = email
            if phone and not session.contact_phone:
                session.contact_phone = phone
            session.status = "handoff"

    if email:
        background_tasks.add_task(
            _fire_lead_email,
            email=email,
            interest_summary=interest or "",
            domain=domain,
            lead_id=lead_id,
        )

    logger.info(
        "sales_lead captured session=%s email=%s phone=%s reason=%s",
        body.session_id,
        mask_email(email) if email else "—",
        mask_phone(phone) if phone else "—",
        reason,
    )

    return SalesLeadResponse(
        ok=True,
        lead_id=lead_id,
        email=mask_email(email) if email else None,
        phone=mask_phone(phone) if phone else None,
    )


@router.get("/api/v1/sales/session/{session_id}")
async def sales_session(session_id: str, limit: int = 20):
    """Resume a conversation — return latest messages and current status."""
    limit = max(1, min(limit, 100))
    with warranty_db_session() as db:
        session = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == session_id)
            .one_or_none()
        )
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        messages = (
            db.query(SalesMessage)
            .filter(SalesMessage.session_id == session_id)
            .order_by(SalesMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        messages_dicts = [m.to_dict() for m in reversed(messages)]

    public_session = session.to_dict()
    if public_session.get("contact_email"):
        public_session["contact_email"] = mask_email(public_session["contact_email"])
    if public_session.get("contact_phone"):
        public_session["contact_phone"] = mask_phone(public_session["contact_phone"])

    return {
        "session": public_session,
        "messages": messages_dicts,
    }


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@router.get("/admin/sales/sessions")
async def admin_list_sessions(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    x_admin_key: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_key)
    limit = max(1, min(limit, 500))
    with warranty_db_session() as db:
        query = db.query(SalesSession)
        if status:
            query = query.filter(SalesSession.status == status.lower())
        rows = (
            query.order_by(SalesSession.updated_at.desc())
            .offset(max(0, offset))
            .limit(limit)
            .all()
        )
    return {
        "total": len(rows),
        "offset": offset,
        "rows": [row.to_dict() for row in rows],
    }


@router.get("/admin/sales/sessions/{session_id}")
async def admin_session_detail(
    session_id: str,
    x_admin_key: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_key)
    with warranty_db_session() as db:
        session = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == session_id)
            .one_or_none()
        )
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        messages = (
            db.query(SalesMessage)
            .filter(SalesMessage.session_id == session_id)
            .order_by(SalesMessage.created_at.asc())
            .all()
        )
        leads = (
            db.query(SalesLead)
            .filter(SalesLead.session_id == session_id)
            .order_by(SalesLead.created_at.asc())
            .all()
        )
    return {
        "session": session.to_dict(),
        "messages": [m.to_dict() for m in messages],
        "leads": [lead.to_dict() for lead in leads],
    }


@router.get("/admin/sales/leads")
async def admin_list_leads(
    forwarded: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    x_admin_key: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_key)
    limit = max(1, min(limit, 500))
    with warranty_db_session() as db:
        query = db.query(SalesLead)
        if forwarded:
            query = query.filter(SalesLead.forwarded == forwarded.lower())
        rows = (
            query.order_by(SalesLead.created_at.desc())
            .offset(max(0, offset))
            .limit(limit)
            .all()
        )
    return {
        "total": len(rows),
        "offset": offset,
        "rows": [row.to_dict() for row in rows],
    }
