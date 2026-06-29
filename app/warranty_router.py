"""
warranty_router.py
==================
FastAPI router for warranty-specific HTTP endpoints.

Phase D-lite  — Evidence upload
    POST   /api/v1/warranty/{ticket_id}/evidence
    GET    /api/v1/warranty/{ticket_id}/evidence

Phase E-lite  — Admin management
    GET    /admin/warranty/tickets
    GET    /admin/warranty/tickets/{ticket_id}
    POST   /admin/warranty/{ticket_id}/decision
    POST   /admin/warranty/{ticket_id}/note
    POST   /admin/warranty/sync-freshdesk

Authentication
--------------
Admin endpoints check the `X-Admin-Key` request header against the
`ADMIN_API_KEY` environment variable.

TODO: Replace the static API-key check with a proper JWT / OAuth2 flow
      once the authentication service is ready. The current design is a
      temporary secret-in-header approach suitable only for internal use.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import requests
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router — all endpoints registered here are included by main.py
# ---------------------------------------------------------------------------
router = APIRouter()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "uploaded_evidence"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".mp4", ".mov"}
_ALLOWED_MIME_PREFIXES = {
    "image/jpeg", "image/png", "application/pdf",
    "video/mp4", "video/quicktime",
}
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB

# Admin API key — set ADMIN_API_KEY in your .env (or server environment).
# TODO: Replace with proper JWT / session-based auth.
_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lazy_engine():
    """
    Import WarrantyEngine lazily to avoid circular imports at module load.

    Always imports from the same module instance as warranty_workflow.py itself
    (no 'app.' prefix) so that monkeypatching warranty_models in tests works
    correctly — avoids the "split module" problem where app.warranty_models and
    warranty_models would be two different module objects.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from warranty_workflow import WarrantyEngine  # type: ignore
    return WarrantyEngine


def _require_admin(x_admin_key: Optional[str]) -> None:
    """
    Verify the X-Admin-Key header.

    Raises HTTP 401 if the key is missing or wrong.
    If ADMIN_API_KEY is not set in the environment, the endpoint is still
    protected — a missing key always fails, preventing accidental exposure.
    """
    if not _ADMIN_API_KEY:
        # Key not configured → refuse all access with a clear message.
        raise HTTPException(
            status_code=503,
            detail=(
                "Admin API is not configured. "
                "Set ADMIN_API_KEY in the server environment first."
            ),
        )
    if x_admin_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key.")


def _safe_filename(original: str) -> str:
    """Sanitise a user-supplied filename to prevent path traversal."""
    name = Path(original).name  # strip any directory components
    # Keep only safe characters
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "upload"


# ---------------------------------------------------------------------------
# Phase D-lite — Evidence endpoints
# ---------------------------------------------------------------------------

@router.post("/api/v1/warranty/{ticket_id}/evidence", tags=["warranty"])
async def upload_evidence(
    ticket_id: str,
    evidence_type: str = Form(...),
    customer_email: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload an evidence file for a warranty ticket.

    Accepts: jpg, jpeg, png, pdf, mp4, mov (max 20 MB).
    Saves to:  uploaded_evidence/warranty/{ticket_id}/{uuid}_{filename}
    Stores metadata in WarrantyEvidence table.
    Requires customer_email and notifies the warranty evidence distribution list.
    """
    from warranty_email import extract_email, notify_evidence_upload_async

    engine = _lazy_engine()

    # --- Ticket existence check ---
    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

    normalized_email = extract_email(customer_email.strip())
    if not normalized_email:
        raise HTTPException(
            status_code=422,
            detail="A valid customer email address is required to upload evidence.",
        )

    # --- File type validation ---
    original_filename = file.filename or "upload"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"File type {suffix!r} is not allowed. "
                f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
            ),
        )

    # Validate MIME type reported by the browser (best-effort)
    content_type = file.content_type or ""
    if content_type and not any(
        content_type.startswith(p) for p in _ALLOWED_MIME_PREFIXES
    ):
        logger.warning(
            f"Evidence upload: suspicious MIME {content_type!r} for ticket {ticket_id}"
        )
        # We warn but don't block — the extension check is the hard gate.

    # --- Read and size-check ---
    data = await file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data):,} bytes). Max is {_MAX_FILE_BYTES:,} bytes (20 MB).",
        )

    # --- Save to disk (path-traversal safe) ---
    safe_name = _safe_filename(original_filename)
    dest_dir = _UPLOAD_ROOT / "warranty" / ticket_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the destination and verify it is still inside _UPLOAD_ROOT
    dest_path = (dest_dir / f"{uuid.uuid4().hex}_{safe_name}").resolve()
    if not str(dest_path).startswith(str(_UPLOAD_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Path traversal detected — request rejected.")

    dest_path.write_bytes(data)
    logger.info(
        f"📎 Evidence saved — ticket={ticket_id} type={evidence_type} "
        f"file={safe_name} size={len(data):,}B path={dest_path}"
    )

    # --- Persist metadata ---
    mime = content_type or (mimetypes.guess_type(safe_name)[0] or "application/octet-stream")
    ev = engine.record_evidence(
        ticket_id=ticket_id,
        evidence_type=evidence_type,
        file_path=str(dest_path),
        original_filename=original_filename,
        mime_type=mime,
        file_size_bytes=len(data),
        customer_email=normalized_email,
    )

    turns = engine.get_turns(ticket_id)
    current_node = engine.get_current_node(ticket_id)
    terminal_node_id = str(current_node.get("node_id") or "") if current_node else ""

    notify_evidence_upload_async(
        evidence_id=cast(int, ev.id),
        ticket_id=ticket_id,
        customer_email=normalized_email,
        evidence_type=evidence_type,
        original_filename=original_filename,
        file_path=str(dest_path),
        mime_type=mime,
        file_size_bytes=len(data),
        issue_type=str(ticket.issue_type or ""),
        model_name=str(ticket.model_name or ""),
        turns=turns,
        terminal_node_id=terminal_node_id,
    )

    return {
        "evidence_id":       ev.id,
        "ticket_id":         ticket_id,
        "ticket_status":     str(ticket.status),
        "evidence_type":     evidence_type,
        "original_filename": original_filename,
        "customer_email":    normalized_email,
        "saved_path":        str(dest_path),
        "mime_type":         mime,
        "file_size_bytes":   len(data),
    }


class WarrantyContactRequest(BaseModel):
    """Final-step customer contact — email required; photos/videos optional (N/A)."""
    customer_email: str
    evidence_na: bool = True


@router.post("/api/v1/warranty/{ticket_id}/contact", tags=["warranty"])
async def submit_warranty_contact(ticket_id: str, body: WarrantyContactRequest):
    """
    Final step: customer leaves their email without uploading photo/video (N/A).

    Records a not_available evidence row, emails the warranty inbox transcript,
    and notifies the evidence distribution list (no attachment).
    """
    from warranty_email import (  # noqa: WPS433
        extract_email,
        notify_email_only_contact_async,
        send_warranty_transcript_email,
    )
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    engine = _lazy_engine()
    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

    node = engine.get_current_node(ticket_id)
    if not node or node.get("type") != "terminal":
        raise HTTPException(
            status_code=422,
            detail="Contact can only be submitted after the warranty workflow is complete.",
        )

    normalized_email = extract_email(body.customer_email.strip())
    if not normalized_email:
        raise HTTPException(
            status_code=422,
            detail="A valid customer email address is required.",
        )

    turns = engine.get_turns(ticket_id)
    terminal_node_id = str(node.get("node_id") or "")
    from warranty_summary import summarize_warranty_case  # noqa: WPS433

    summary_payload = summarize_warranty_case(
        issue_type=str(ticket.issue_type or ""),
        model_name=str(ticket.model_name or ""),
        turns=turns,
        terminal_node_id=terminal_node_id,
    )
    case_summary = summary_payload["summary"]

    ev = engine.record_evidence(
        ticket_id=ticket_id,
        evidence_type="not_available",
        file_path="",
        original_filename="N/A",
        mime_type="",
        file_size_bytes=0,
        customer_email=normalized_email,
    )

    with warranty_db_session() as db:
        ticket_row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket_row:
            ticket_row.set_collected("customer_contact_email", normalized_email)
            ticket_row.set_collected("evidence_na", "1")
            ticket_row.set_collected("case_summary", case_summary)
            ticket_row.set_collected("case_summary_source", summary_payload.get("source", ""))

    send_warranty_transcript_email(
        customer_email=normalized_email,
        session_id=str(ticket.session_id),
        ticket_id=ticket_id,
        domain=str(ticket.domain or "unknown"),
        ticket_status=str(ticket.status or ""),
        issue_type=str(ticket.issue_type or ""),
        model_name=str(ticket.model_name or ""),
        turns=turns,
        case_summary=case_summary,
        case_summary_source=summary_payload.get("source", ""),
        terminal_node_id=terminal_node_id,
    )

    notify_email_only_contact_async(
        evidence_id=cast(int, ev.id),
        ticket_id=ticket_id,
        customer_email=normalized_email,
        session_id=str(ticket.session_id),
        domain=str(ticket.domain or "unknown"),
        ticket_status=str(ticket.status or ""),
        issue_type=str(ticket.issue_type or ""),
        model_name=str(ticket.model_name or ""),
        turns=turns,
        terminal_node_id=terminal_node_id,
        case_summary=case_summary,
        case_summary_source=summary_payload.get("source", ""),
    )

    return {
        "ticket_id": ticket_id,
        "customer_email": normalized_email,
        "evidence_type": "not_available",
        "evidence_na": True,
        "email_notified": True,
        "case_summary": case_summary,
        "case_summary_source": summary_payload.get("source", ""),
    }


class WarrantyAnswerRequest(BaseModel):
    """Customer answer for the current workflow node (answer_key, label, or text)."""
    answer: str


class WarrantyQuickStartRequest(BaseModel):
    """Skip the root menu and jump straight to a top-level warranty issue type."""
    issue_type: str  # installation | delivery | defect
    domain: str = "osaki.com"


class WarrantyRegisterModelRequest(BaseModel):
    """Register chair model before issue-type selection."""
    model: str
    domain: str = "osaki.com"


class WarrantyNaturalStartRequest(BaseModel):
    """Start warranty intake from free-text (LLM maps to issue type)."""
    message: str
    domain: str = "osaki.com"


class WarrantySmartStartRequest(BaseModel):
    """
    Start warranty intake from free-text and fast-forward as many flowchart
    steps as the LLM can confidently extract.
    """
    message: str
    domain: str = "osaki.com"


class WarrantyEmailNotifyRequest(BaseModel):
    """Notify the warranty team when a customer leaves their email in chat."""
    message: str = ""
    chat_messages: Optional[List[Dict[str, str]]] = None


class WarrantyRestartRequest(BaseModel):
    """Abandon any in-progress ticket so the customer can start over."""
    domain: str = "osaki.com"


_QUICK_START_ISSUE_KEYS = frozenset({"installation", "delivery", "defect"})


def _require_registered_model(ticket) -> None:
    if not str(getattr(ticket, "model_name", "") or "").strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "Please tell us your chair model first (for example OS-4000T or Solo Flex), "
                "then choose the type of issue."
            ),
        )


def _maybe_skip_install_model(engine, ticket_id: str) -> None:
    """If model was registered upfront, skip the install_model question."""
    ticket = engine.get_ticket(ticket_id)
    node = engine.get_current_node(ticket_id)
    if ticket is None or node is None:
        return
    if node.get("node_id") != "install_model":
        return
    model = str(getattr(ticket, "model_name", "") or "").strip()
    if model:
        engine.submit_answer(ticket_id, model)


def _quick_start_ticket(
    engine,
    session_id: str,
    issue_type: str,
    domain: str,
) -> Dict[str, Any]:
    """Shared quick-start logic for button and natural-language entry."""
    issue_type = issue_type.strip().lower()
    if issue_type not in _QUICK_START_ISSUE_KEYS:
        raise HTTPException(
            status_code=422,
            detail=f"issue_type must be one of: {sorted(_QUICK_START_ISSUE_KEYS)}",
        )

    ticket = engine.get_active_session_ticket(session_id)
    ticket_id: str

    if ticket is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Please tell us your chair model first (for example OS-4000T or Solo Flex), "
                "then choose the type of issue."
            ),
        )

    _require_registered_model(ticket)
    ticket_id = str(ticket.ticket_id)
    node = engine.get_current_node(ticket_id)
    node_id = node.get("node_id") if node else None
    if node_id == "root":
        engine.submit_answer(ticket_id, "warranty")
        engine.submit_answer(ticket_id, issue_type)
    elif node_id == "issue_type":
        engine.submit_answer(ticket_id, issue_type)

    if issue_type == "installation":
        _maybe_skip_install_model(engine, ticket_id)

    ticket = engine.get_ticket(ticket_id)
    node = engine.get_current_node(ticket_id)
    return _serialize_ticket_state(session_id, ticket, node, engine=engine)


def _register_model_ticket(
    engine,
    session_id: str,
    model: str,
    domain: str,
) -> Dict[str, Any]:
    from product_catalog import resolve_model_name  # noqa: WPS433

    raw = model.strip()
    if not raw:
        raise HTTPException(status_code=422, detail="model must not be empty")

    lower = raw.lower()
    issue_markers = (
        "not working",
        "not inflating",
        "won't",
        "wont",
        "doesn't",
        "doesnt",
        "broken",
        "damaged",
        "no air",
        "not turn",
        "not power",
        "false trigger",
        "too hot",
        "too loud",
        "making noise",
        "delivery",
        "tracking",
        "installation",
    )
    if any(marker in lower for marker in issue_markers) or len(raw.split()) >= 4:
        raise HTTPException(
            status_code=422,
            detail=(
                "That sounds like a problem description, not a chair model. "
                "Try describing model and issue together (e.g. 'OS-4000T footrest air not inflating'), "
                "or enter just your model (e.g. OS-4000T)."
            ),
        )

    resolved = resolve_model_name(raw) or raw
    ticket = engine.get_active_session_ticket(session_id)
    ticket_id: str

    if ticket is None:
        ticket_id, _root = engine.start_session(session_id, domain)
        engine.submit_answer(ticket_id, "warranty")
    else:
        ticket_id = str(ticket.ticket_id)
        node = engine.get_current_node(ticket_id)
        node_id = node.get("node_id") if node else None
        if node_id == "root":
            engine.submit_answer(ticket_id, "warranty")

    engine.set_model_name(ticket_id, resolved)
    ticket = engine.get_ticket(ticket_id)
    node = engine.get_current_node(ticket_id)
    payload = _serialize_ticket_state(session_id, ticket, node, engine=engine)
    payload["model_registered"] = True
    payload["resolved_model"] = resolved
    return payload


def _submit_answer_with_nlp(engine, ticket_id: str, answer: str) -> tuple[dict, bool]:
    """
    Submit a workflow answer; on option mismatch, map natural language via NLP.

    Returns (submit_result, nlp_interpreted).
    """
    try:
        return engine.submit_answer(ticket_id, answer), False
    except ValueError as exc:
        msg = str(exc)
        if "did not match any option" not in msg:
            raise

        node = engine.get_current_node(ticket_id)
        if not node:
            raise

        from warranty_nlp import interpret_warranty_answer  # noqa: WPS433

        mapped = interpret_warranty_answer(node, answer)
        if not mapped:
            raise ValueError(
                "I couldn't match your answer to the current question. "
                "Please tap one of the options above, or rephrase more clearly."
            ) from exc

        if node.get("type") == "question_text":
            return (
                engine.submit_answer(
                    ticket_id,
                    mapped,
                    customer_display=answer,
                ),
                True,
            )

        if mapped == answer:
            raise

        return (
            engine.submit_answer(
                ticket_id,
                mapped,
                customer_display=answer,
            ),
            True,
        )


def _finalize_answer_response(
    engine,
    ticket_id: str,
    answer: str,
    result: dict,
    *,
    nlp_interpreted: bool = False,
) -> Dict[str, Any]:
    """Build the browser payload after a successful submit_answer."""
    tracking_summary: Optional[Dict[str, Any]] = None
    previous_node = result.get("previous_node_id")
    if previous_node in ("delivery_get_tracking_number", "delivery_get_name"):
        from delivery_lookup import (  # noqa: WPS433
            format_warranty_tracking_message,
            lookup_by_order_or_email,
            lookup_by_tracking_number,
            persist_snapshot,
        )

        ticket_for_domain = engine.get_ticket(ticket_id)
        domain = str(ticket_for_domain.domain if ticket_for_domain else "osaki.com")

        lookup_text = answer
        if previous_node == "delivery_get_tracking_number":
            snapshot = lookup_by_tracking_number(lookup_text, domain)
        else:
            snapshot = lookup_by_order_or_email(lookup_text, domain)

        persist_snapshot(ticket_id, snapshot)
        tracking_summary = {
            "available": snapshot.available,
            "message": format_warranty_tracking_message(snapshot),
            "snapshot": snapshot.to_dict(),
        }

    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

    from warranty_email import maybe_send_warranty_transcript  # noqa: WPS433
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    email_notified = False
    with warranty_db_session() as db:
        ticket_row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket_row:
            turns = engine.get_turns(ticket_id)
            _detected, sent_now = maybe_send_warranty_transcript(
                ticket=ticket_row,
                answer_text=answer,
                turns=turns,
            )
            email_notified = sent_now

    node = engine.get_current_node(ticket_id)
    payload = _serialize_ticket_state(str(ticket.session_id), ticket, node, engine=engine)
    if tracking_summary is not None:
        payload["tracking_summary"] = tracking_summary
    if email_notified:
        payload["email_notified"] = True
    if nlp_interpreted:
        payload["nlp_interpreted"] = True
    return payload


def _serialize_ticket_state(
    session_id: str,
    ticket,
    node,
    *,
    engine=None,
) -> Dict[str, Any]:
    """Build the browser-safe session payload shared by GET/POST warranty endpoints."""
    if ticket is None:
        return {"session_id": session_id, "ticket": None}

    ticket_id = str(ticket.ticket_id)
    options: List[Dict[str, Any]] = []
    node_prompt: Optional[str] = None
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    is_terminal = False
    evidence_required: List[str] = []
    evidence_email: Optional[str] = None

    if node:
        node_id = str(node.get("node_id", ""))
        node_type = str(node.get("type", ""))
        node_prompt = str(node.get("prompt", ""))
        is_terminal = node_type == "terminal"
        if is_terminal:
            evidence_required = list(node.get("evidence_required", []))
            evidence_email = node.get("evidence_email") or "service@osakititan.com"
        for opt in node.get("options", []):
            options.append({
                "answer_key": opt.get("answer_key", ""),
                "label": opt.get("label", ""),
            })

    terminal_enrichment: Optional[Dict[str, Any]] = None
    if node and node.get("type") == "terminal":
        from warranty_terminal_enrichment import build_terminal_enrichment  # noqa: WPS433

        if engine is None:
            engine = _lazy_engine()
        terminal_enrichment = build_terminal_enrichment(engine, ticket, node)

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "ticket": {
            "ticket_id":    ticket_id,
            "status":       str(ticket.status),
            "issue_type":   str(ticket.issue_type or ""),
            "model_name":   str(ticket.model_name or ""),
            "model_confirmed": bool(str(ticket.model_name or "").strip()),
            "ready_for_issue_type": (
                node_id == "issue_type" and bool(str(ticket.model_name or "").strip())
            ),
            "current_node": {
                "node_id":            node_id,
                "node_type":          node_type,
                "prompt":             node_prompt,
                "options":            options,
                "is_terminal":        is_terminal,
                "evidence_required":  evidence_required,
                "evidence_email":     evidence_email,
            } if node else None,
        },
    }
    if terminal_enrichment:
        payload["terminal_enrichment"] = terminal_enrichment
        payload["assistant_message"] = terminal_enrichment.get("message")
    return payload


def _get_open_session_ticket(engine, session_id: str):
    """Return the latest non-resolved ticket for this chat session, if any."""
    ticket = engine.get_active_session_ticket(session_id)
    if ticket is not None:
        return ticket

    from warranty_models import warranty_db_session, WarrantyTicket  # type: ignore
    with warranty_db_session() as db:
        return (
            db.query(WarrantyTicket)
            .filter(
                WarrantyTicket.session_id == session_id,
                WarrantyTicket.status != "resolved",
            )
            .order_by(WarrantyTicket.created_at.desc())
            .first()
        )


@router.get("/api/v1/warranty/session/{session_id}", tags=["warranty"])
async def get_warranty_session_state(session_id: str):
    """
    Return the current warranty session state for a given session_id.

    Called by the Next.js frontend after each chat turn to determine:
    - whether an active warranty ticket exists
    - the current node prompt and answer options (for rendering clickable buttons)
    - the current ticket status (for showing status badges)

    Returns 200 with ticket=null if no open ticket exists.
    This endpoint is SAFE to call from the browser — it exposes no admin data.
    """
    engine = _lazy_engine()
    ticket = _get_open_session_ticket(engine, session_id)
    if ticket is None:
        return {"session_id": session_id, "ticket": None}

    node = engine.get_current_node(str(ticket.ticket_id))
    return _serialize_ticket_state(session_id, ticket, node, engine=engine)


@router.post("/api/v1/warranty/session/{session_id}/restart", tags=["warranty"])
async def restart_warranty_session(session_id: str, body: WarrantyRestartRequest):
    """
    Close any open ticket on this chat session so the customer can start over.

    Behavior:
      - Marks every non-resolved ticket on this session as `resolved` with
        `admin_decision='abandoned'` so they drop out of the admin queue but
        remain queryable for audit.
      - Returns the same shape as `GET /session/{id}` (with ticket=null) so the
        frontend can refresh its state from one response.
    """
    engine = _lazy_engine()
    closed = engine.abandon_session_tickets(session_id)
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "ticket": None,
        "restarted": True,
        "closed_ticket_count": closed,
        "domain": body.domain,
    }
    return payload


@router.post("/api/v1/warranty/session/{session_id}/register-model", tags=["warranty"])
async def register_warranty_model(session_id: str, body: WarrantyRegisterModelRequest):
    """
    Step 1 of warranty intake: confirm chair model, then show issue-type options.
    """
    engine = _lazy_engine()
    return _register_model_ticket(engine, session_id, body.model, body.domain)


@router.post("/api/v1/warranty/session/{session_id}/quick-start", tags=["warranty"])
async def quick_start_warranty(session_id: str, body: WarrantyQuickStartRequest):
    """
    Start (or resume) a warranty ticket and jump to Installation / Delivery / Defect
    without any LLM call. Used by the frontend landing buttons on /warranty.
    """
    engine = _lazy_engine()
    return _quick_start_ticket(engine, session_id, body.issue_type, body.domain)


@router.post("/api/v1/warranty/session/{session_id}/natural-start", tags=["warranty"])
async def natural_start_warranty(session_id: str, body: WarrantyNaturalStartRequest):
    """
    Start warranty intake from free-text — LLM maps message to issue type, then
    runs the same deterministic flowchart as quick-start.
    """
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    from warranty_nlp import interpret_issue_type  # noqa: WPS433

    issue_type = interpret_issue_type(message)
    if not issue_type:
        raise HTTPException(
            status_code=422,
            detail=(
                "I couldn't tell whether this is an installation, delivery, or "
                "defect issue. Please pick one of the options above or describe "
                "your issue more specifically."
            ),
        )

    engine = _lazy_engine()
    ticket = engine.get_active_session_ticket(session_id)
    if ticket is not None:
        _require_registered_model(ticket)
    payload = _quick_start_ticket(engine, session_id, issue_type, body.domain)
    payload["nlp_interpreted"] = True
    payload["interpreted_issue_type"] = issue_type
    return payload


@router.post("/api/v1/warranty/session/{session_id}/smart-start", tags=["warranty"])
async def smart_start_warranty(session_id: str, body: WarrantySmartStartRequest):
    """
    Multi-step free-text intake.

    LLM reads the customer's one-line description and produces an ordered
    sequence of valid flowchart answer_keys. We auto-submit those keys so the
    customer can skip 2~6 multiple-choice questions when their description is
    clear (e.g. "OS-4000T footrest air not inflating" → defect → air →
    footrest → terminal).

    Behavior:
      - On any failure / low confidence: behaves like quick-start defect (safe).
      - Only auto-submits answer_keys that match the live flowchart options.
      - Returns the same ticket-state payload as other warranty endpoints, plus
        `smart_start` metadata explaining what was inferred.
    """
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    from warranty_intake import (  # noqa: WPS433
        apply_prefill_to_engine,
        extract_workflow_prefill,
    )
    from warranty_workflow import _NODES  # type: ignore  # noqa: WPS433

    engine = _lazy_engine()
    existing = engine.get_active_session_ticket(session_id)
    if existing is not None:
        _require_registered_model(existing)
        ticket_id = str(existing.ticket_id)
    else:
        ticket_id, _root = engine.start_session(session_id, body.domain)

    extraction = extract_workflow_prefill(free_text=message, nodes=_NODES)
    answer_keys: list[str] = list(extraction.get("answer_keys") or [])

    apply_result: dict[str, Any] = {
        "applied": [],
        "skipped": [],
        "stopped_reason": "empty",
        "final_node": engine.get_current_node(ticket_id),
    }
    if answer_keys:
        apply_result = apply_prefill_to_engine(
            engine=engine,
            ticket_id=ticket_id,
            nodes=_NODES,
            answer_keys=answer_keys,
        )

    model_hint = str(extraction.get("model_name") or "").strip()
    if model_hint:
        from product_catalog import resolve_model_name  # noqa: WPS433

        resolved_model = resolve_model_name(model_hint) or model_hint
        engine.set_model_name(ticket_id, resolved_model)

    if not apply_result["applied"]:
        # Nothing usable — fall back to a safe defect quick-start so the
        # frontend still progresses past the root menu.
        node = engine.get_current_node(ticket_id)
        node_id = node.get("node_id") if node else None
        try:
            if node_id == "root":
                engine.submit_answer(ticket_id, "warranty")
                engine.submit_answer(ticket_id, "defect")
            elif node_id == "issue_type":
                engine.submit_answer(ticket_id, "defect")
        except ValueError:
            pass

    ticket = engine.get_ticket(ticket_id)
    node = engine.get_current_node(ticket_id)
    payload = _serialize_ticket_state(session_id, ticket, node, engine=engine)
    payload["smart_start"] = {
        "source": extraction.get("source", "empty"),
        "summary": extraction.get("summary", ""),
        "applied_keys": apply_result["applied"],
        "skipped_keys": apply_result["skipped"],
        "stopped_reason": apply_result["stopped_reason"],
        "model_name_hint": extraction.get("model_name", ""),
    }
    return payload


@router.post("/api/v1/warranty/{ticket_id}/answer", tags=["warranty"])
async def submit_warranty_answer(ticket_id: str, body: WarrantyAnswerRequest):
    """
    Advance the warranty workflow by one step.

    Accepts an answer_key, option label, free-text (question_text nodes), or
    natural language (mapped to the closest option via NLP when needed).
    """
    engine = _lazy_engine()
    answer = body.answer.strip()
    if not answer:
        raise HTTPException(status_code=422, detail="answer must not be empty")

    try:
        result, nlp_interpreted = _submit_answer_with_nlp(engine, ticket_id, answer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _finalize_answer_response(
        engine,
        ticket_id,
        answer,
        result,
        nlp_interpreted=nlp_interpreted,
    )


@router.post("/api/v1/warranty/session/{session_id}/notify-email", tags=["warranty"])
async def notify_warranty_email(session_id: str, body: WarrantyEmailNotifyRequest):
    """
    Send the warranty chat transcript to the warranty inbox when the customer
    leaves their email address in free-text chat.
    """
    from warranty_email import extract_email, maybe_send_warranty_transcript, send_warranty_transcript_email  # noqa: WPS433
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    engine = _lazy_engine()

    email = extract_email(body.message)
    if not email and body.chat_messages:
        for msg in reversed(body.chat_messages):
            if msg.get("role") == "user":
                email = extract_email(msg.get("content", ""))
                if email:
                    break

    if not email:
        return {"sent": False, "reason": "no_email_found"}

    ticket = _get_open_session_ticket(engine, session_id)
    if ticket is not None:
        with warranty_db_session() as db:
            ticket_row = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket.ticket_id)
                .first()
            )
            if ticket_row is None:
                return {"sent": False, "reason": "ticket_not_found"}

            turns = engine.get_turns(str(ticket.ticket_id))
            _detected, sent_now = maybe_send_warranty_transcript(
                ticket=ticket_row,
                answer_text=email,
                turns=turns,
                chat_messages=body.chat_messages,
            )
            sent = sent_now
        return {"sent": sent, "customer_email": email}

    sent = send_warranty_transcript_email(
        customer_email=email,
        session_id=session_id,
        chat_messages=body.chat_messages,
    )
    return {"sent": sent, "customer_email": email}


@router.get("/api/v1/warranty/{ticket_id}/evidence", tags=["warranty"])
async def list_evidence(ticket_id: str):
    """List all evidence files attached to a warranty ticket."""
    engine = _lazy_engine()

    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

    evidences = engine.get_evidences(ticket_id)
    return {
        "ticket_id":    ticket_id,
        "ticket_status": str(ticket.status),
        "evidence":     [e.to_dict() for e in evidences],
    }


# ---------------------------------------------------------------------------
# Phase E-lite — Admin endpoints
# ---------------------------------------------------------------------------

class AdminDecisionRequest(BaseModel):
    decision: str
    """
    Allowed values:
      admin_reviewing      – admin has picked up the ticket
      need_more_information – admin needs more info from the customer
      approved             – admin approves the warranty action
      rejected             – admin rejects the warranty claim
      closed               – case closed without further action
    """
    note: str = ""
    customer_message: str = ""
    decided_by: str = "admin"


class AdminNoteRequest(BaseModel):
    note: str
    added_by: str = "admin"


_ADMIN_ALLOWED_STATUSES = {
    "admin_reviewing", "need_more_information",
    "approved", "rejected", "closed",
}


def _serialize_admin_ticket(ticket, turns=None, evidences=None) -> dict:
    from warranty_email import resolve_customer_email  # noqa: WPS433

    payload = ticket.to_dict()
    payload["customer_email"] = resolve_customer_email(ticket, turns=turns, evidences=evidences)
    return payload


@router.get("/admin/warranty/tickets", tags=["admin-warranty"])
async def list_warranty_tickets(
    status: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    x_admin_key: Optional[str] = Header(default=None),
):
    """
    List warranty tickets.  Admin-only.

    Optional query params: status, domain, limit, offset.
    """
    _require_admin(x_admin_key)
    engine = _lazy_engine()
    tickets = engine.get_tickets(
        status=status,
        domain=domain,
        limit=min(limit, 200),
        offset=offset,
    )
    return {
        "total":   len(tickets),
        "offset":  offset,
        "tickets": [_serialize_admin_ticket(t) for t in tickets],
    }


@router.get("/admin/warranty/tickets/{ticket_id}", tags=["admin-warranty"])
async def get_warranty_ticket_detail(
    ticket_id: str,
    x_admin_key: Optional[str] = Header(default=None),
):
    """Return full ticket detail including all turns and evidence.  Admin-only."""
    _require_admin(x_admin_key)
    engine = _lazy_engine()

    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

    turns = engine.get_turns(ticket_id)
    evidences = engine.get_evidences(ticket_id)

    return {
        "ticket":   _serialize_admin_ticket(ticket, turns=turns, evidences=evidences),
        "turns":    [t.to_dict() for t in turns],
        "evidence": [e.to_dict() for e in evidences],
    }


@router.post("/admin/warranty/{ticket_id}/decision", tags=["admin-warranty"])
async def admin_warranty_decision(
    ticket_id: str,
    body: AdminDecisionRequest,
    x_admin_key: Optional[str] = Header(default=None),
):
    """
    Record an admin decision on a warranty ticket.  Admin-only.

    This is the ONLY endpoint that may set status=approved or status=rejected.
    Customer-facing chat must never call this path.
    """
    _require_admin(x_admin_key)

    if body.decision not in _ADMIN_ALLOWED_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid decision {body.decision!r}. "
                f"Must be one of: {sorted(_ADMIN_ALLOWED_STATUSES)}"
            ),
        )

    engine = _lazy_engine()
    try:
        ticket = engine.admin_decision(
            ticket_id=ticket_id,
            decision=body.decision,
            note=body.note,
            decided_by=body.decided_by,
            customer_message=body.customer_message,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    turns = engine.get_turns(ticket_id)
    evidences = engine.get_evidences(ticket_id)

    from warranty_email import maybe_send_admin_decision_customer_email  # noqa: WPS433

    customer_email_sent, customer_email_skip_reason = maybe_send_admin_decision_customer_email(
        ticket=ticket,
        decision=body.decision,
        customer_message=body.customer_message or "",
        turns=turns,
        evidences=evidences,
    )

    logger.info(
        f"⚖️  Admin decision — ticket={ticket_id} decision={body.decision} "
        f"decided_by={body.decided_by} customer_email_sent={customer_email_sent}"
    )
    return {
        "ticket": _serialize_admin_ticket(ticket, turns=turns, evidences=evidences),
        "customer_email_sent": customer_email_sent,
        "customer_email_skip_reason": customer_email_skip_reason,
    }


@router.post("/admin/warranty/{ticket_id}/note", tags=["admin-warranty"])
async def admin_warranty_note(
    ticket_id: str,
    body: AdminNoteRequest,
    x_admin_key: Optional[str] = Header(default=None),
):
    """
    Append a note to a warranty ticket without changing its status.  Admin-only.
    """
    _require_admin(x_admin_key)

    engine = _lazy_engine()
    try:
        ticket = engine.add_admin_note(
            ticket_id=ticket_id,
            note=body.note,
            added_by=body.added_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"ticket": ticket.to_dict()}


@router.post("/admin/warranty/sync-freshdesk", tags=["admin-warranty"])
async def admin_sync_freshdesk(
    x_admin_key: Optional[str] = Header(default=None),
    max_pages: int = 5,
):
    """
    Pull resolved Freshdesk tickets into data/freshdesk_tickets.json and
    reload warranty self-help knowledge cache. Admin-only.
    """
    _require_admin(x_admin_key)

    try:
        from freshdesk_sync import sync_freshdesk_knowledge  # noqa: WPS433
        from warranty_knowledge import clear_knowledge_cache, load_knowledge_entries  # noqa: WPS433
    except ImportError:
        from app.freshdesk_sync import sync_freshdesk_knowledge  # type: ignore  # noqa: WPS433
        from app.warranty_knowledge import (  # type: ignore  # noqa: WPS433
            clear_knowledge_cache,
            load_knowledge_entries,
        )

    pages = max(1, min(int(max_pages), 20))
    try:
        result = sync_freshdesk_knowledge(max_pages=pages)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Freshdesk API error: {exc}") from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    clear_knowledge_cache()
    entries = load_knowledge_entries()
    freshdesk_entries = sum(1 for entry in entries if entry.source == "freshdesk")

    logger.info(
        "Freshdesk sync — ok=%s tickets=%s knowledge_freshdesk=%s",
        result.get("ok"),
        result.get("ticket_count"),
        freshdesk_entries,
    )

    return {
        **result,
        "knowledge_freshdesk_entries": freshdesk_entries,
        "knowledge_total_entries": len(entries),
    }


@router.get(
    "/admin/warranty/{ticket_id}/evidence/{evidence_id}/download",
    tags=["admin-warranty"],
)
async def admin_download_evidence(
    ticket_id: str,
    evidence_id: int,
    x_admin_key: Optional[str] = Header(default=None),
):
    """
    Stream/download an uploaded evidence file.  Admin-only.

    Security guarantees
    -------------------
    * Requires a valid X-Admin-Key header.
    * Verifies the evidence row belongs to *ticket_id* — prevents cross-ticket
      access by guessing evidence IDs.
    * Resolves the stored path and confirms it remains inside _UPLOAD_ROOT —
      prevents path-traversal attacks introduced by a corrupted DB row.
    * Never returns the raw file-system path to the caller.
    * Returns HTTP 404 (not 403) for missing resources to avoid information
      leakage about the filesystem layout.
    """
    _require_admin(x_admin_key)

    engine = _lazy_engine()

    # --- Evidence existence and ownership check ---
    evidence = engine.get_evidence_by_id(evidence_id)
    if evidence is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evidence {evidence_id!r} not found.",
        )

    if str(evidence.ticket_id) != ticket_id:
        # Return 404, not 403, to avoid revealing that the evidence_id is valid
        raise HTTPException(
            status_code=404,
            detail="Evidence does not belong to the specified ticket.",
        )

    # --- Path resolution and traversal guard ---
    raw_path = str(evidence.file_path or "")
    if not raw_path:
        raise HTTPException(status_code=404, detail="Evidence file path is not recorded.")

    try:
        resolved = Path(raw_path).resolve()
    except (OSError, ValueError) as exc:
        logger.warning(f"Evidence path resolution failed for id={evidence_id}: {exc}")
        raise HTTPException(status_code=404, detail="Evidence file not accessible.")

    upload_root_resolved = _UPLOAD_ROOT.resolve()
    if not str(resolved).startswith(str(upload_root_resolved)):
        logger.error(
            f"Path traversal rejected — evidence_id={evidence_id} "
            f"path={resolved!r} upload_root={upload_root_resolved!r}"
        )
        raise HTTPException(status_code=404, detail="Evidence file not accessible.")

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evidence file no longer exists on disk.",
        )

    # --- Determine MIME type ---
    original_filename = str(evidence.original_filename or "evidence_file")
    stored_mime = str(evidence.mime_type or "")
    mime_type = (
        stored_mime
        or mimetypes.guess_type(original_filename)[0]
        or "application/octet-stream"
    )

    logger.info(
        f"📥 Evidence download — ticket={ticket_id} evidence_id={evidence_id} "
        f"file={original_filename} mime={mime_type}"
    )

    return FileResponse(
        path=str(resolved),
        media_type=mime_type,
        filename=original_filename,
    )
