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
`ADMIN_API_KEY` environment variable. The public Next.js admin application
adds this server-side only after validating its signed, HTTP-only session.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pytz
import requests
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel

from warranty_defaults import DEFAULT_WARRANTY_DOMAIN, normalize_warranty_domain

try:
    from app.admin_auth import require_admin_key
except ImportError:  # pragma: no cover - direct module execution in tests
    from admin_auth import require_admin_key  # type: ignore


def _now_cst_iso() -> str:
    return datetime.now(pytz.timezone("America/Chicago")).isoformat()

logger = logging.getLogger(__name__)


def _chat_privacy_enforced() -> bool:
    return os.getenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _require_web_chat_privacy(session_id: str) -> None:
    """Block web chat mutators until consent + email gate are recorded server-side."""
    if not _chat_privacy_enforced():
        return
    from warranty_consent import (  # noqa: WPS433
        EMAIL_GATE_PROVIDED,
        get_chat_consent,
    )

    row = get_chat_consent(session_id)
    if row is None:
        raise HTTPException(
            status_code=403,
            detail="Please agree to the privacy notice before continuing.",
        )
    status = str(getattr(row, "email_gate_status", "") or "").strip().lower()
    if status != EMAIL_GATE_PROVIDED:
        raise HTTPException(
            status_code=403,
            detail="Please provide your email so we can follow up on your case securely.",
        )


def _require_web_chat_privacy_for_ticket(ticket) -> None:
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    if str(collected.get("channel") or "").strip().lower() == "phone":
        return
    _require_web_chat_privacy(str(getattr(ticket, "session_id", "") or ""))


def _masked_public_email(value: str | None) -> Optional[str]:
    from pii_redact import mask_email  # noqa: WPS433

    text = (value or "").strip()
    if not text:
        return None
    return mask_email(text)

# ---------------------------------------------------------------------------
# Router — all endpoints registered here are included by main.py
# ---------------------------------------------------------------------------
router = APIRouter()

try:
    from cost_guard import limiter  # noqa: WPS433
except ImportError:  # pragma: no cover — tests without full deps
    class _NoopLimiter:  # noqa: WPS441
        def limit(self, *_args, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

    limiter = _NoopLimiter()

_WARRANTY_LLM_RATE = os.getenv("WARRANTY_LLM_RATE_LIMIT", "20/minute")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "uploaded_evidence"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".pdf", ".mp4", ".mov", ".avi", ".webm"
}
_ALLOWED_MIME_PREFIXES = {
    "image/jpeg", "image/png", "image/webp", "application/pdf",
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/avi", "video/webm",
}
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_UPLOAD_RATE = os.getenv("WARRANTY_UPLOAD_RATE_LIMIT", "10/hour")

# Backend-to-backend admin credential. Browser clients never receive this key.
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
    require_admin_key(x_admin_key, _ADMIN_API_KEY)


def _safe_filename(original: str) -> str:
    """Sanitise a user-supplied filename to prevent path traversal."""
    name = Path(original).name  # strip any directory components
    # Keep only safe characters
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "upload"


def _matches_file_signature(suffix: str, header: bytes) -> bool:
    """Validate actual file bytes instead of trusting extension/MIME headers."""
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".webp":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    if suffix == ".pdf":
        return header.startswith(b"%PDF-")
    if suffix in {".mp4", ".mov"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if suffix == ".avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    if suffix == ".webm":
        return header.startswith(b"\x1aE\xdf\xa3")
    return False


# ---------------------------------------------------------------------------
# Phase D-lite — Evidence endpoints
# ---------------------------------------------------------------------------

@router.post("/api/v1/warranty/{ticket_id}/evidence", tags=["warranty"])
@limiter.limit(_UPLOAD_RATE)
async def upload_evidence(
    request: Request,
    ticket_id: str,
    evidence_type: str = Form(...),
    customer_email: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload an evidence file for a warranty ticket.

    Accepts: jpg, jpeg, png, webp, pdf, mp4, mov, avi, webm (max 20 MB).
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
    _require_web_chat_privacy_for_ticket(ticket)

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

    # Reject obviously incompatible browser-reported types. Actual bytes are
    # validated below as the authoritative gate.
    content_type = file.content_type or ""
    if content_type and not any(
        content_type.startswith(p) for p in _ALLOWED_MIME_PREFIXES
    ):
        raise HTTPException(
            status_code=422,
            detail=f"File MIME type {content_type!r} is not allowed.",
        )

    # --- Stream to disk with bounded memory and path-traversal protection ---
    safe_name = _safe_filename(original_filename)
    dest_dir = _UPLOAD_ROOT / "warranty" / ticket_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the destination and verify it is still inside _UPLOAD_ROOT
    dest_path = (dest_dir / f"{uuid.uuid4().hex}_{safe_name}").resolve()
    if not dest_path.is_relative_to(_UPLOAD_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal detected — request rejected.")

    file_size = 0
    header = b""
    try:
        with dest_path.open("xb") as output:
            os.chmod(dest_path, 0o600)
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > _MAX_FILE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max is {_MAX_FILE_BYTES:,} bytes (20 MB).",
                    )
                if len(header) < 32:
                    header += chunk[: 32 - len(header)]
                output.write(chunk)

        if file_size == 0 or not _matches_file_signature(suffix, header):
            raise HTTPException(
                status_code=422,
                detail="File contents do not match the declared file type.",
            )
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    logger.info(
        "evidence_saved ticket=%s type=%s size_bytes=%s",
        ticket_id,
        evidence_type,
        file_size,
    )

    # --- Persist metadata ---
    mime = content_type or (mimetypes.guess_type(safe_name)[0] or "application/octet-stream")
    ev = engine.record_evidence(
        ticket_id=ticket_id,
        evidence_type=evidence_type,
        file_path=str(dest_path),
        original_filename=original_filename,
        mime_type=mime,
        file_size_bytes=file_size,
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
        file_size_bytes=file_size,
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
        "customer_email":    _masked_public_email(normalized_email),
        "email_saved":       True,
        "mime_type":         mime,
        "file_size_bytes":   file_size,
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
        "customer_email": _masked_public_email(normalized_email),
        "email_saved": True,
        "evidence_type": "not_available",
        "evidence_na": True,
        "email_notified": True,
        "case_summary": case_summary,
        "case_summary_source": summary_payload.get("source", ""),
    }


class WarrantyCustomerNoteRequest(BaseModel):
    """Free-text follow-up note the customer types after reaching a terminal node."""
    note: str


_MAX_CUSTOMER_NOTE_LEN = 1000
_MAX_CUSTOMER_NOTES = 20


@router.post("/api/v1/warranty/{ticket_id}/customer-note", tags=["warranty"])
async def append_customer_note(ticket_id: str, body: WarrantyCustomerNoteRequest):
    """
    Append a customer-typed follow-up note to a ticket after the workflow ends.

    Notes land in ``collected_data["customer_notes"]`` as a list of
    ``{text, created_at}`` entries so the admin can see extra context the
    customer added while the contact form was hidden.
    """
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    note = (body.note or "").strip()
    if not note:
        raise HTTPException(status_code=422, detail="Note text must not be empty.")
    if len(note) > _MAX_CUSTOMER_NOTE_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"Note is too long (max {_MAX_CUSTOMER_NOTE_LEN} characters).",
        )

    engine = _lazy_engine()
    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

    now_iso = _now_cst_iso()

    with warranty_db_session() as db:
        ticket_row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket_row is None:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

        collected = ticket_row.get_collected()
        existing = collected.get("customer_notes")
        notes: List[Dict[str, str]] = list(existing) if isinstance(existing, list) else []
        notes.append({"text": note, "created_at": now_iso})
        if len(notes) > _MAX_CUSTOMER_NOTES:
            notes = notes[-_MAX_CUSTOMER_NOTES:]
        collected["customer_notes"] = notes
        import json as _json  # noqa: WPS433
        ticket_row.collected_data = _json.dumps(collected)
        stored = notes
        ticket_status = str(ticket_row.status)

    turns = engine.get_turns(ticket_id)
    ticket = engine.get_ticket(ticket_id)

    team_notified = False
    if ticket_status == "need_more_information" and ticket is not None:
        from warranty_email import send_customer_followup_notification  # noqa: WPS433
        from warranty_freshdesk_case import maybe_add_freshdesk_customer_reply  # noqa: WPS433

        team_notified = send_customer_followup_notification(
            ticket=ticket,
            note_text=note,
            turns=turns,
        )
        maybe_add_freshdesk_customer_reply(ticket_id, note, engine=engine)

    logger.info(
        "warranty customer note appended ticket=%s len=%d total=%d",
        ticket_id,
        len(note),
        len(stored),
    )

    return {
        "ticket_id": ticket_id,
        "customer_notes": stored,
        "team_notified": team_notified,
    }


class WarrantyTroubleshootingOutcomeRequest(BaseModel):
    """Customer progress through the resolution-first terminal experience."""

    outcome: str


_TROUBLESHOOTING_OUTCOMES = frozenset(
    {"steps_completed", "resolved", "unresolved", "unable_to_attempt"}
)


@router.post(
    "/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
    tags=["warranty"],
)
async def record_troubleshooting_outcome(
    ticket_id: str,
    body: WarrantyTroubleshootingOutcomeRequest,
):
    """Persist self-service progress before exposing team-review options.

    A confirmed resolution closes the ticket as ``self_resolved`` so it cannot
    remain in an admin replacement/shipping queue. Other outcomes preserve the
    workflow status and allow the customer to continue to team review.
    """
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    outcome = (body.outcome or "").strip().lower()
    if outcome not in _TROUBLESHOOTING_OUTCOMES:
        raise HTTPException(
            status_code=422,
            detail="Unsupported troubleshooting outcome.",
        )

    engine = _lazy_engine()
    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")
    node = engine.get_current_node(ticket_id)
    if not node or str(node.get("type") or "") != "terminal":
        raise HTTPException(
            status_code=409,
            detail="Troubleshooting outcome is only available after diagnosis.",
        )

    previous_outcome = str(
        ticket.get_collected().get("troubleshooting_outcome") or ""
    ).strip().lower()
    ticket_status = str(ticket.status or "")
    decision = str(ticket.admin_decision or "")
    if ticket_status == "resolved":
        if decision == "self_resolved" and outcome == "resolved":
            return {
                "ticket_id": ticket_id,
                "outcome": "resolved",
                "status": "resolved",
                "self_service_resolved": True,
            }
        raise HTTPException(status_code=409, detail="This ticket is already resolved.")
    if outcome in {"resolved", "unresolved"} and previous_outcome not in {
        "steps_completed",
        outcome,
    }:
        raise HTTPException(
            status_code=409,
            detail="Complete the recommended steps before recording the result.",
        )
    if previous_outcome in {"unresolved", "unable_to_attempt"} and outcome == "steps_completed":
        raise HTTPException(
            status_code=409,
            detail="This ticket has already continued to team review.",
        )
    if previous_outcome == outcome:
        return {
            "ticket_id": ticket_id,
            "outcome": outcome,
            "status": ticket_status,
            "self_service_resolved": outcome == "resolved",
        }

    now_iso = _now_cst_iso()
    terminal_node_id = str(node.get("node_id") or "")
    with warranty_db_session() as db:
        ticket_row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket_row is None:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

        collected = ticket_row.get_collected()
        history_value = collected.get("troubleshooting_history")
        history: List[Dict[str, str]] = (
            list(history_value) if isinstance(history_value, list) else []
        )
        history.append(
            {
                "outcome": outcome,
                "created_at": now_iso,
                "terminal_node_id": terminal_node_id,
            }
        )
        collected["troubleshooting_outcome"] = outcome
        collected["troubleshooting_updated_at"] = now_iso
        collected["troubleshooting_history"] = history[-20:]

        import json as _json  # noqa: WPS433

        ticket_row.collected_data = _json.dumps(collected)
        if outcome == "resolved":
            ticket_row.status = "resolved"
            ticket_row.admin_decision = "self_resolved"
            resolution_note = (
                f"[system] Customer confirmed the issue was resolved after "
                f"self-service steps at {now_iso}."
            )
            existing_note = str(ticket_row.admin_note or "").strip()
            ticket_row.admin_note = (
                f"{existing_note}\n{resolution_note}".strip()
            )
        status = str(ticket_row.status)

    logger.info(
        "warranty troubleshooting outcome ticket=%s outcome=%s terminal=%s",
        ticket_id,
        outcome,
        terminal_node_id,
    )
    return {
        "ticket_id": ticket_id,
        "outcome": outcome,
        "status": status,
        "self_service_resolved": outcome == "resolved",
    }


@router.post(
    "/api/v1/warranty/{ticket_id}/troubleshooting-back",
    tags=["warranty"],
)
async def go_back_from_warranty_contact(ticket_id: str):
    """Return from the final contact form to the preceding resolution step.

    This is deliberately separate from workflow ``/back``: the workflow is
    already terminal here, so only the reversible troubleshooting outcome is
    changed. Submitted/resolved cases remain immutable.
    """
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    engine = _lazy_engine()
    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")
    node = engine.get_current_node(ticket_id)
    if not node or str(node.get("type") or "") != "terminal":
        raise HTTPException(
            status_code=409,
            detail="The contact step is only available after diagnosis.",
        )
    if str(ticket.status or "") == "resolved":
        raise HTTPException(status_code=409, detail="This ticket is already resolved.")

    now_iso = _now_cst_iso()
    with warranty_db_session() as db:
        ticket_row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket_row is None:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

        collected = ticket_row.get_collected()
        if str(collected.get("customer_contact_email") or "").strip():
            raise HTTPException(
                status_code=409,
                detail="Contact information has already been submitted.",
            )

        previous_outcome = str(
            collected.get("troubleshooting_outcome") or ""
        ).strip().lower()
        if previous_outcome == "unresolved":
            restored_outcome: Optional[str] = "steps_completed"
            restored_stage = "outcome"
        elif previous_outcome == "unable_to_attempt":
            restored_outcome = None
            restored_stage = "review"
        else:
            raise HTTPException(
                status_code=409,
                detail="There is no contact-form step to go back from.",
            )

        history_value = collected.get("troubleshooting_history")
        history: List[Dict[str, str]] = (
            list(history_value) if isinstance(history_value, list) else []
        )
        history.append(
            {
                "outcome": restored_outcome or "review",
                "action": "back",
                "created_at": now_iso,
                "terminal_node_id": str(node.get("node_id") or ""),
            }
        )
        if restored_outcome:
            collected["troubleshooting_outcome"] = restored_outcome
        else:
            collected.pop("troubleshooting_outcome", None)
        collected["troubleshooting_updated_at"] = now_iso
        collected["troubleshooting_history"] = history[-20:]

        import json as _json  # noqa: WPS433

        ticket_row.collected_data = _json.dumps(collected)

    logger.info(
        "warranty contact back ticket=%s previous=%s restored=%s",
        ticket_id,
        previous_outcome,
        restored_outcome or "review",
    )
    return {
        "ticket_id": ticket_id,
        "stage": restored_stage,
        "outcome": restored_outcome,
    }


class WarrantyAnswerRequest(BaseModel):
    """Customer answer for the current workflow node (answer_key, label, or text)."""
    answer: str


class WarrantyQuickStartRequest(BaseModel):
    """Skip the root menu and jump straight to a top-level warranty issue type."""
    issue_type: str  # installation | delivery | defect
    domain: str = DEFAULT_WARRANTY_DOMAIN


class WarrantyRegisterModelRequest(BaseModel):
    """Register chair model before issue-type selection."""
    model: str
    domain: str = DEFAULT_WARRANTY_DOMAIN


class WarrantyConfirmModelRequest(BaseModel):
    """Confirm or correct inferred chair model after smart-start."""
    confirmed: bool = True
    model: Optional[str] = None
    domain: str = DEFAULT_WARRANTY_DOMAIN


class WarrantyNaturalStartRequest(BaseModel):
    """Start warranty intake from free-text (LLM maps to issue type)."""
    message: str
    domain: str = DEFAULT_WARRANTY_DOMAIN


class WarrantySmartStartRequest(BaseModel):
    """
    Start warranty intake from free-text and fast-forward as many flowchart
    steps as the LLM can confidently extract.
    """
    message: str
    domain: str = DEFAULT_WARRANTY_DOMAIN


class WarrantyEmailNotifyRequest(BaseModel):
    """Notify the warranty team when a customer leaves their email in chat."""
    message: str = ""
    chat_messages: Optional[List[Dict[str, str]]] = None


class WarrantyRestartRequest(BaseModel):
    """Abandon any in-progress ticket so the customer can start over."""
    domain: str = DEFAULT_WARRANTY_DOMAIN


class WarrantyConsentRequest(BaseModel):
    """Record live-chat privacy / recording consent for a browser session."""
    domain: str = DEFAULT_WARRANTY_DOMAIN
    policy_store: str = ""


class WarrantySessionContactEmailRequest(BaseModel):
    """Post-consent email gate — soft-required contact email before chat."""
    customer_email: str = ""
    skipped: bool = False


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
    from warranty_intake_context import mark_model_confirmed  # noqa: WPS433

    if ticket is not None:
        mark_model_confirmed(ticket)
        _persist_ticket_row(ticket_id, ticket)
    node = engine.get_current_node(ticket_id)
    payload = _serialize_ticket_state(session_id, ticket, node, engine=engine)
    payload["model_registered"] = True
    payload["resolved_model"] = resolved
    return payload


_DEFECT_MODEL_REQUIRED_MSG = (
    "Please tell us your chair model first (for example OS-4000T or 3D LTX), "
    "then continue with warranty / defect."
)


def _answer_selects_defect(node: Optional[dict], answer: str) -> bool:
    if not node or str(node.get("node_id") or "") != "issue_type":
        return False
    key = str(answer or "").strip().lower()
    if key in {"defect", "warranty / defect", "warranty/defect"}:
        return True
    for opt in node.get("options") or []:
        opt_key = str(opt.get("answer_key") or "").strip().lower()
        label = str(opt.get("label") or "").strip().lower()
        if opt_key != "defect":
            continue
        if key in {opt_key, label}:
            return True
        if key and (key in label or label in key):
            return True
    return False


def _guard_defect_requires_model(engine, ticket_id: str, answer: str) -> Optional[str]:
    node = engine.get_current_node(ticket_id)
    if not _answer_selects_defect(node, answer):
        return None
    ticket = engine.get_ticket(ticket_id)
    if ticket is not None and str(getattr(ticket, "model_name", "") or "").strip():
        return None
    return _DEFECT_MODEL_REQUIRED_MSG


def _validate_text_before_submit(engine, ticket_id: str, answer: str) -> None:
    node = engine.get_current_node(ticket_id)
    if not node or node.get("type") != "question_text":
        return

    node_id = str(node.get("node_id") or "")
    if node_id not in _DELIVERY_TEXT_NODES:
        return

    ticket = engine.get_ticket(ticket_id)
    model_name = str(getattr(ticket, "model_name", None) or "")

    from delivery_intake import validate_delivery_text_answer  # noqa: WPS433

    validate_delivery_text_answer(node_id, answer, model_name=model_name)


_DELIVERY_TEXT_NODES = frozenset(
    {
        "delivery_get_name",
        "delivery_get_tracking_number",
        "delivery_status_get_order_email",
        "delivery_status_get_tracking",
    }
)


def _persist_ticket_row(ticket_id: str, ticket) -> None:
    """Flush in-memory ticket field changes (e.g. collected_data) to SQLite."""
    if ticket is None:
        return
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    with warranty_db_session() as db:
        row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if row is None:
            return
        row.collected_data = ticket.collected_data
        if getattr(ticket, "model_name", None):
            row.model_name = ticket.model_name


def _maybe_side_question_message(engine, ticket_id: str, answer: str) -> Optional[str]:
    from warranty_error_code_gate import (  # noqa: WPS433
        format_midflow_error_code_help,
        is_gate_node,
    )
    from warranty_intake_context import try_side_question_for_ticket  # noqa: WPS433
    from warranty_scope import (  # noqa: WPS433
        build_warranty_scope_refusal,
        evaluate_warranty_scope,
        is_sales_workflow_answer,
    )

    if is_sales_workflow_answer(answer):
        return build_warranty_scope_refusal("sales")

    node = engine.get_current_node(ticket_id)
    node_id = str(node.get("node_id") or "") if node else ""
    ticket = engine.get_ticket(ticket_id)
    issue_type = str(getattr(ticket, "issue_type", "") or "") if ticket else ""

    if node and not is_gate_node(node_id):
        from warranty_side_questions import _looks_like_valid_workflow_answer  # noqa: WPS433

        if ticket is not None and not _looks_like_valid_workflow_answer(node, answer):
            scope = evaluate_warranty_scope(
                answer,
                node_id=node_id,
                issue_type=issue_type or None,
            )
            if scope.is_blocked:
                return build_warranty_scope_refusal(scope.reason)

            prompt = str(node.get("prompt") or "").strip()
            mid = format_midflow_error_code_help(
                ticket,
                answer,
                reprompt=prompt or "Please choose one of the options above to continue.",
            )
            if mid:
                _persist_ticket_row(ticket_id, ticket)
                return mid

    if ticket is not None:
        scope = evaluate_warranty_scope(
            answer,
            node_id=node_id,
            issue_type=issue_type or None,
        )
        if scope.is_blocked:
            return build_warranty_scope_refusal(scope.reason)

    return try_side_question_for_ticket(engine, ticket_id, answer)


def _build_side_question_response(
    engine,
    ticket_id: str,
    message: str,
    *,
    customer_text: str = "",
) -> Dict[str, Any]:
    ticket = engine.get_ticket(ticket_id)
    node = engine.get_current_node(ticket_id)
    node_id = str((node or {}).get("node_id") or getattr(ticket, "current_node_id", "") or "")
    try:
        from warranty_chat_timeline import append_chat_event  # noqa: WPS433

        if customer_text.strip():
            append_chat_event(
                ticket,
                role="user",
                kind="side_question",
                text=customer_text,
                node_id=node_id,
            )
        if message.strip():
            append_chat_event(
                ticket,
                role="assistant",
                kind="side_question",
                text=message,
                node_id=node_id,
            )
            # Refresh after persist.
            ticket = engine.get_ticket(ticket_id) or ticket
    except Exception:
        pass
    session_id = str(getattr(ticket, "session_id", "") or "")
    payload = _serialize_ticket_state(session_id, ticket, node, engine=engine)
    payload["side_question"] = True
    payload["assistant_message"] = message
    return payload


def _submit_answer_with_nlp(
    engine,
    ticket_id: str,
    answer: str,
) -> tuple[Optional[dict], bool, Optional[str]]:
    """
    Submit a workflow answer; on option mismatch, map natural language via NLP.

    Returns (submit_result, nlp_interpreted, clarifying_message).
    When clarifying_message is set, the workflow does not advance.
    """
    _validate_text_before_submit(engine, ticket_id, answer)

    node = engine.get_current_node(ticket_id)
    node_id = str(node.get("node_id") or "") if node else ""

    from warranty_error_code_gate import is_gate_node, map_gate_free_text  # noqa: WPS433
    from warranty_nlp import build_clarifying_workflow_message, interpret_warranty_answer  # noqa: WPS433

    if node and is_gate_node(node_id):
        mapped = map_gate_free_text(node_id, answer)
        if mapped:
            return (
                engine.submit_answer(
                    ticket_id,
                    mapped,
                    customer_display=answer,
                ),
                True,
                None,
            )
        try:
            return engine.submit_answer(ticket_id, answer), False, None
        except ValueError as exc:
            if "did not match any option" not in str(exc):
                raise
            return (
                None,
                False,
                (
                    "Please tap one of the buttons above, type **yes** or **no**, "
                    "or enter the error code exactly as shown (for example: C6, E5)."
                ),
            )

    try:
        return engine.submit_answer(ticket_id, answer), False, None
    except ValueError as exc:
        msg = str(exc)
        if "did not match any option" not in msg:
            raise

        node = engine.get_current_node(ticket_id)
        if not node:
            raise

        mapped = interpret_warranty_answer(node, answer)
        if not mapped or mapped == answer:
            return None, False, build_clarifying_workflow_message(node, answer)

        if node.get("type") == "question_text":
            return (
                engine.submit_answer(
                    ticket_id,
                    mapped,
                    customer_display=answer,
                ),
                True,
                None,
            )

        return (
            engine.submit_answer(
                ticket_id,
                mapped,
                customer_display=answer,
            ),
            True,
            None,
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
    if previous_node in (
        "delivery_get_tracking_number",
        "delivery_get_name",
        "delivery_status_get_tracking",
        "delivery_status_get_order_email",
    ):
        from delivery_lookup import (  # noqa: WPS433
            append_eligibility_to_tracking_message,
            build_self_service_lookup_links,
            format_warranty_tracking_message,
            lookup_by_order_or_email,
            lookup_by_tracking_number,
            persist_snapshot,
            public_tracking_summary_payload,
        )

        ticket_for_domain = engine.get_ticket(ticket_id)
        domain = normalize_warranty_domain(
            str(ticket_for_domain.domain if ticket_for_domain else "")
        )

        lookup_text = answer
        if previous_node in (
            "delivery_get_tracking_number",
            "delivery_status_get_tracking",
        ):
            lookup_kind = "tracking"
            snapshot = lookup_by_tracking_number(lookup_text, domain)
        else:
            lookup_kind = "order"
            snapshot = lookup_by_order_or_email(lookup_text, domain)

        persist_snapshot(ticket_id, snapshot)
        self_service_links = build_self_service_lookup_links(
            domain=domain,
            lookup_kind=lookup_kind,
            raw_input=lookup_text,
        )
        tracking_message = format_warranty_tracking_message(
            snapshot,
            domain=domain,
            lookup_kind=lookup_kind,
            raw_input=lookup_text,
            continue_with_questions=previous_node
            in (
                "delivery_get_tracking_number",
                "delivery_get_name",
            ),
        )
        tracking_message = append_eligibility_to_tracking_message(
            tracking_message, ticket_id
        )
        tracking_summary = public_tracking_summary_payload(
            available=snapshot.available,
            message=tracking_message,
            self_service_links=[
                {"label": label, "url": url} for label, url in self_service_links
            ],
        )

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

    if result.get("is_terminal") or str(ticket.status) in (
        "awaiting_admin_review",
        "awaiting_evidence",
    ):
        from warranty_freshdesk_case import schedule_freshdesk_case_creation  # noqa: WPS433
        from warranty_email import maybe_send_customer_receipt_email  # noqa: WPS433

        freshdesk_result = schedule_freshdesk_case_creation(ticket_id)
        if freshdesk_result.get("freshdesk_ticket_id"):
            # Public clients only need the customer case reference — not agent portal URLs.
            payload["freshdesk"] = {
                "case_reference": freshdesk_result.get("case_reference"),
                "linked": True,
            }
        elif freshdesk_result.get("scheduled"):
            payload["freshdesk_scheduled"] = True
        elif freshdesk_result.get("error") and not freshdesk_result.get("skipped"):
            payload["freshdesk_error"] = {
                "linked": False,
                "case_reference": freshdesk_result.get("case_reference"),
            }

        # Customer receipt (idempotent) — works even when Freshdesk create is skipped.
        with warranty_db_session() as db:
            ticket_row = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if ticket_row:
                turns = engine.get_turns(ticket_id)
                receipt_sent, receipt_skip = maybe_send_customer_receipt_email(
                    ticket=ticket_row,
                    turns=turns,
                    case_reference=str(
                        (freshdesk_result or {}).get("case_reference") or ""
                    ),
                    freshdesk_url=str((freshdesk_result or {}).get("freshdesk_url") or ""),
                )
                if receipt_sent:
                    payload["receipt_email_sent"] = True
                elif receipt_skip:
                    payload["receipt_email_skip_reason"] = receipt_skip

    return payload


def _case_reference_for(ticket) -> str:
    from warranty_case_ref import case_reference_for_ticket  # noqa: WPS433

    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    stored = str(collected.get("case_reference") or "").strip()
    if stored:
        return stored
    return case_reference_for_ticket(ticket)


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
        if node_id == "root":
            from warranty_scope import filter_warranty_menu_options  # noqa: WPS433

            menu_options = filter_warranty_menu_options(node)
        else:
            menu_options = list(node.get("options") or [])
        for opt in menu_options:
            options.append({
                "answer_key": opt.get("answer_key", ""),
                "label": opt.get("label", ""),
            })

    if engine is None:
        engine = _lazy_engine()

    from warranty_assistant_message import build_assistant_message_bundle  # noqa: WPS433
    from warranty_intake_context import (  # noqa: WPS433
        is_model_confirmed,
        needs_model_confirmation,
    )

    enrichment = build_assistant_message_bundle(
        engine=engine,
        ticket=ticket,
        node=node,
    )
    terminal_enrichment = enrichment.get("terminal_enrichment")
    assistant_message = enrichment.get("assistant_message")

    # Persist enriched assistant tips once per node (best-effort).
    try:
        base_prompt = (node_prompt or "").strip()
        enriched = (assistant_message or "").strip()
        if enriched and enriched != base_prompt and node_id:
            from warranty_chat_timeline import append_chat_event  # noqa: WPS433

            append_chat_event(
                ticket,
                role="assistant",
                kind="enrichment",
                text=enriched,
                node_id=str(node_id),
            )
    except Exception:
        pass

    collected = ticket.get_collected()
    troubleshooting_outcome = str(
        collected.get("troubleshooting_outcome") or ""
    ).strip() or None

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "ticket": {
            "ticket_id":    ticket_id,
            "case_reference": _case_reference_for(ticket),
            "status":       str(ticket.status),
            "issue_type":   str(ticket.issue_type or ""),
            "model_name":   str(ticket.model_name or ""),
            "model_confirmed": is_model_confirmed(ticket),
            "needs_model_confirmation": needs_model_confirmation(ticket),
            "ready_for_issue_type": (
                node_id == "issue_type" and bool(str(ticket.model_name or "").strip())
            ),
            "needs_customer_reply": str(ticket.status) == "need_more_information",
            "can_go_back": engine.can_go_back(ticket_id),
            "customer_message": (
                str(ticket.customer_message or "").strip() or None
            ),
            "admin_decision": str(ticket.admin_decision or "").strip() or None,
            "troubleshooting_outcome": troubleshooting_outcome,
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
        # Public responses need the customer message and step count, not the
        # internal source names, ticket titles, or match metadata.
        public_terminal_enrichment = dict(terminal_enrichment)
        diagnosis = public_terminal_enrichment.get("diagnosis")
        if isinstance(diagnosis, dict):
            public_terminal_enrichment["diagnosis"] = {
                key: diagnosis[key]
                for key in ("summary", "steps")
                if key in diagnosis
            }
        payload["terminal_enrichment"] = public_terminal_enrichment
    if assistant_message:
        payload["assistant_message"] = assistant_message
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


@router.post("/api/v1/warranty/session/{session_id}/consent", tags=["warranty"])
async def record_warranty_chat_consent(session_id: str, body: WarrantyConsentRequest):
    """
    Persist customer acceptance of the live-chat privacy / recording notice.

    Called when the customer taps **I Agree** in the widget, before any message
    is stored. The timestamp is copied onto the ticket when the workflow starts.
    """
    from warranty_consent import record_chat_consent  # noqa: WPS433

    try:
        accepted_at = record_chat_consent(
            session_id,
            domain=body.domain,
            policy_store=body.policy_store,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "session_id": session_id,
        "consent_recorded": True,
        "accepted_at": accepted_at.isoformat(),
        "policy_store": (body.policy_store or "").strip().lower() or None,
    }


@router.post("/api/v1/warranty/session/{session_id}/contact-email", tags=["warranty"])
async def record_warranty_session_contact_email(
    session_id: str,
    body: WarrantySessionContactEmailRequest,
):
    """
    Soft-required email after I Agree: store on the session so it copies onto
    the ticket when the warranty workflow starts. Skip is disabled when
    ``WARRANTY_REQUIRE_CHAT_PRIVACY`` is on (default).
    """
    from warranty_consent import (  # noqa: WPS433
        get_chat_consent,
        record_session_contact_email,
    )

    if get_chat_consent(session_id) is None:
        raise HTTPException(
            status_code=403,
            detail="Please agree to the privacy notice before providing your email.",
        )
    if bool(body.skipped) and _chat_privacy_enforced():
        raise HTTPException(
            status_code=422,
            detail="Email is required for warranty chat follow-up.",
        )

    try:
        result = record_session_contact_email(
            session_id,
            customer_email=body.customer_email,
            skipped=bool(body.skipped),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "session_id": session_id,
        "recorded": True,
        "email_saved": bool(result.get("customer_email")),
        "customer_email": _masked_public_email(
            str(result.get("customer_email") or "")
        ),
        "email_gate_status": result.get("email_gate_status"),
        "skipped": result.get("email_gate_status") == "skipped",
    }


@router.post("/api/v1/warranty/session/{session_id}/register-model", tags=["warranty"])
async def register_warranty_model(session_id: str, body: WarrantyRegisterModelRequest):
    """
    Step 1 of warranty intake: confirm chair model, then show issue-type options.
    """
    _require_web_chat_privacy(session_id)
    engine = _lazy_engine()
    return _register_model_ticket(engine, session_id, body.model, body.domain)


@router.post("/api/v1/warranty/session/{session_id}/confirm-model", tags=["warranty"])
async def confirm_warranty_model(session_id: str, body: WarrantyConfirmModelRequest):
    """
    Confirm or correct the chair model after smart-start inferred it from free text.
    """
    _require_web_chat_privacy(session_id)
    from product_catalog import resolve_model_name  # noqa: WPS433
    from warranty_intake_context import mark_model_confirmed  # noqa: WPS433

    engine = _lazy_engine()
    ticket = engine.get_active_session_ticket(session_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="No active warranty session found.")

    ticket_id = str(ticket.ticket_id)
    corrected = str(body.model or "").strip()
    if corrected:
        resolved = resolve_model_name(corrected) or corrected
        engine.set_model_name(ticket_id, resolved)
    elif not body.confirmed:
        raise HTTPException(
            status_code=422,
            detail="Set confirmed=true or provide a corrected model name.",
        )

    ticket = engine.get_ticket(ticket_id)
    if ticket is not None:
        mark_model_confirmed(ticket)
        _persist_ticket_row(ticket_id, ticket)

    ticket = engine.get_ticket(ticket_id)
    node = engine.get_current_node(ticket_id)
    return _serialize_ticket_state(session_id, ticket, node, engine=engine)


@router.post("/api/v1/warranty/session/{session_id}/quick-start", tags=["warranty"])
async def quick_start_warranty(session_id: str, body: WarrantyQuickStartRequest):
    """
    Start (or resume) a warranty ticket and jump to Installation / Delivery / Defect
    without any LLM call. Used by the frontend landing buttons on /warranty.
    """
    _require_web_chat_privacy(session_id)
    engine = _lazy_engine()
    return _quick_start_ticket(engine, session_id, body.issue_type, body.domain)


@router.post("/api/v1/warranty/session/{session_id}/natural-start", tags=["warranty"])
@limiter.limit(_WARRANTY_LLM_RATE)
async def natural_start_warranty(
    request: Request,
    session_id: str,
    body: WarrantyNaturalStartRequest,
):
    """
    Start warranty intake from free-text — LLM maps message to issue type, then
    runs the same deterministic flowchart as quick-start.
    """
    _require_web_chat_privacy(session_id)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    from warranty_scope import build_warranty_scope_refusal, evaluate_warranty_scope  # noqa: WPS433

    scope = evaluate_warranty_scope(message)
    if scope.is_blocked:
        engine = _lazy_engine()
        existing = engine.get_active_session_ticket(session_id)
        if existing is None:
            ticket_id, _root = engine.start_session(session_id, body.domain)
            try:
                engine.submit_answer(ticket_id, "warranty")
            except ValueError:
                pass
        else:
            ticket_id = str(existing.ticket_id)
        return _build_side_question_response(
            engine,
            ticket_id,
            build_warranty_scope_refusal(scope.reason),
        )

    from warranty_nlp import (  # noqa: WPS433
        build_clarifying_issue_type_message,
        interpret_issue_type,
    )
    from warranty_intake_context import (  # noqa: WPS433
        mark_model_confirmed,
        needs_model_confirmation,
    )

    engine = _lazy_engine()
    ticket = engine.get_active_session_ticket(session_id)
    if ticket is not None:
        _require_registered_model(ticket)

    if ticket is not None and needs_model_confirmation(ticket):
        from product_catalog import looks_like_model_only, resolve_model_name  # noqa: WPS433

        issue_type = interpret_issue_type(message)
        model_candidate = looks_like_model_only(message)
        if not model_candidate and not issue_type:
            model_candidate = resolve_model_name(message)

        if model_candidate or (not issue_type and len(message.split()) <= 6):
            ticket_id = str(ticket.ticket_id)
            resolved = model_candidate or resolve_model_name(message) or message
            engine.set_model_name(ticket_id, resolved)
            ticket = engine.get_ticket(ticket_id)
            if ticket is not None:
                mark_model_confirmed(ticket)
                _persist_ticket_row(ticket_id, ticket)
            node = engine.get_current_node(ticket_id)
            payload = _serialize_ticket_state(session_id, ticket, node, engine=engine)
            payload["model_corrected"] = True
            payload["resolved_model"] = resolved
            return payload

    issue_type = interpret_issue_type(message)
    if not issue_type:
        if ticket is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Please tell us your chair model first (for example OS-4000T), "
                    "then describe the type of issue."
                ),
            )
        model_name = str(getattr(ticket, "model_name", "") or "")
        clarify = build_clarifying_issue_type_message(message, model_name=model_name)
        return _build_side_question_response(
            engine,
            str(ticket.ticket_id),
            clarify,
        )
    payload = _quick_start_ticket(engine, session_id, issue_type, body.domain)
    from warranty_intake_context import persist_intake_summary  # noqa: WPS433

    ticket = engine.get_ticket(str(payload.get("ticket", {}).get("ticket_id") or ""))
    if ticket is None:
        active = engine.get_active_session_ticket(session_id)
        ticket = active
    persist_intake_summary(ticket, raw_message=message)
    if ticket is not None:
        _persist_ticket_row(str(ticket.ticket_id), ticket)
    payload["nlp_interpreted"] = True
    payload["interpreted_issue_type"] = issue_type
    return payload


@router.post("/api/v1/warranty/session/{session_id}/smart-start", tags=["warranty"])
@limiter.limit(_WARRANTY_LLM_RATE)
async def smart_start_warranty(
    request: Request,
    session_id: str,
    body: WarrantySmartStartRequest,
):
    """
    Multi-step free-text intake.

    LLM reads the customer's one-line description and produces an ordered
    sequence of valid flowchart answer_keys. We auto-submit those keys so the
    customer can skip 2~6 multiple-choice questions when their description is
    clear (e.g. "OS-4000T footrest air not inflating" → defect → air →
    footrest → terminal).

    Behavior:
      - On failure / low confidence: advances only to the issue-type menu (never
        silently defaults to defect).
      - Only auto-submits answer_keys that match the live flowchart options.
      - Returns the same ticket-state payload as other warranty endpoints, plus
        `smart_start` metadata explaining what was inferred.
    """
    _require_web_chat_privacy(session_id)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    from warranty_scope import build_warranty_scope_refusal, evaluate_warranty_scope  # noqa: WPS433

    scope = evaluate_warranty_scope(message)
    if scope.is_blocked:
        engine = _lazy_engine()
        ticket = engine.get_active_session_ticket(session_id)
        if ticket is None:
            ticket_id, _root = engine.start_session(session_id, body.domain)
            try:
                engine.submit_answer(ticket_id, "warranty")
            except ValueError:
                pass
        else:
            ticket_id = str(ticket.ticket_id)
        return _build_side_question_response(
            engine,
            ticket_id,
            build_warranty_scope_refusal(scope.reason),
        )

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

    applied_keys_list: list[str] = list(apply_result.get("applied") or [])
    if not applied_keys_list:
        # Nothing usable — open the warranty menu only; never guess defect.
        node = engine.get_current_node(ticket_id)
        node_id = node.get("node_id") if node else None
        try:
            if node_id == "root":
                engine.submit_answer(ticket_id, "warranty")
                applied_keys_list = ["warranty"]
                apply_result["applied"] = applied_keys_list
                apply_result["stopped_reason"] = "done"
        except ValueError:
            pass

    from warranty_intake_context import persist_intake_summary  # noqa: WPS433

    ticket = engine.get_ticket(ticket_id)
    persist_intake_summary(
        ticket,
        summary=str(extraction.get("summary") or "").strip(),
        raw_message=message,
    )
    if ticket is not None:
        _persist_ticket_row(ticket_id, ticket)

    ticket = engine.get_ticket(ticket_id)
    node = engine.get_current_node(ticket_id)
    payload = _serialize_ticket_state(session_id, ticket, node, engine=engine)

    from warranty_intake_context import (  # noqa: WPS433
        build_model_confirmation_message,
        needs_model_confirmation,
    )

    inferred_issue_type: Optional[str] = None
    if len(applied_keys_list) >= 2 and applied_keys_list[1] in (
        "installation",
        "delivery",
        "defect",
    ):
        inferred_issue_type = applied_keys_list[1]

    routing_confirmation: Optional[Dict[str, Any]] = None
    if inferred_issue_type and len(applied_keys_list) >= 2:
        summary = str(extraction.get("summary") or "").strip()
        routing_confirmation = {
            "inferred_issue_type": inferred_issue_type,
            "applied_count": len(applied_keys_list),
            "summary": summary,
            "message": (
                f"We're treating this as a {inferred_issue_type} issue"
                + (f": {summary}" if summary else "")
                + ". If that's wrong, tap **Start over** and choose again."
            ),
        }

    payload["smart_start"] = {
        "source": extraction.get("source", "empty"),
        "summary": extraction.get("summary", ""),
        "applied_keys": applied_keys_list,
        "skipped_keys": apply_result["skipped"],
        "stopped_reason": apply_result["stopped_reason"],
        "model_name_hint": extraction.get("model_name", ""),
        "routing_confirmation": routing_confirmation,
    }
    if ticket and needs_model_confirmation(ticket):
        model_display = str(ticket.model_name or model_hint or "").strip()
        payload["model_confirmation"] = {
            "model_name": model_display,
            "message": build_model_confirmation_message(model_display),
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
    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")
    _require_web_chat_privacy_for_ticket(ticket)

    answer = body.answer.strip()
    if not answer:
        raise HTTPException(status_code=422, detail="answer must not be empty")

    side_message = _maybe_side_question_message(engine, ticket_id, answer)
    if side_message:
        return _build_side_question_response(
            engine, ticket_id, side_message, customer_text=answer
        )

    defect_guard = _guard_defect_requires_model(engine, ticket_id, answer)
    if defect_guard:
        return _build_side_question_response(
            engine, ticket_id, defect_guard, customer_text=answer
        )

    try:
        result, nlp_interpreted, clarify = _submit_answer_with_nlp(
            engine,
            ticket_id,
            answer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if clarify:
        return _build_side_question_response(
            engine, ticket_id, clarify, customer_text=answer
        )

    if result is None:
        raise HTTPException(status_code=422, detail="Could not process answer.")

    return _finalize_answer_response(
        engine,
        ticket_id,
        answer,
        result,
        nlp_interpreted=nlp_interpreted,
    )


@router.post("/api/v1/warranty/{ticket_id}/back", tags=["warranty"])
async def go_back_warranty(ticket_id: str):
    """
    Undo the last workflow answer and restore the previous question.

    Only available while the ticket is ``in_progress`` and at least one turn
    exists. Returns the same session payload shape as ``/answer``.
    """
    engine = _lazy_engine()
    try:
        rewind = engine.go_back(ticket_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    node = engine.get_current_node(ticket_id)
    payload = _serialize_ticket_state(str(ticket.session_id), ticket, node, engine=engine)
    payload["went_back"] = True
    payload["rewind"] = rewind
    return payload


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
        "evidence":     [e.to_dict_public() for e in evidences],
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
    from warranty_error_code_gate import build_admin_fonz_payload  # noqa: WPS433

    payload = ticket.to_dict()
    payload["case_reference"] = _case_reference_for(ticket)
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    payload["freshdesk_ticket_id"] = collected.get("freshdesk_ticket_id")
    payload["freshdesk_url"] = collected.get("freshdesk_url")
    payload["freshdesk_create_error"] = collected.get("freshdesk_create_error") or None
    payload["freshdesk_create_error_detail"] = (
        collected.get("freshdesk_create_error_detail") or None
    )
    payload["freshdesk_create_failed_at"] = (
        collected.get("freshdesk_create_failed_at") or None
    )
    payload["freshdesk_create_attempt_count"] = collected.get(
        "freshdesk_create_attempt_count"
    )
    payload["channel"] = collected.get("channel")
    payload["caller_phone"] = collected.get("caller_phone")
    payload["customer_email"] = resolve_customer_email(ticket, turns=turns, evidences=evidences)
    payload["intake_email_gate_status"] = collected.get("intake_email_gate_status")
    payload["fonz_diagnostics"] = build_admin_fonz_payload(ticket)

    # Current node prompt so admins can see what the customer is looking at
    # even before that node is answered (no turn yet).
    current_prompt = None
    try:
        engine = _lazy_engine()
        ticket_id = str(getattr(ticket, "ticket_id", "") or "")
        node = engine.get_current_node(ticket_id) if ticket_id else None
        if isinstance(node, dict):
            current_prompt = str(node.get("prompt") or "").strip() or None
        if not current_prompt:
            node_id = str(getattr(ticket, "current_node_id", "") or "")
            nodes = engine.get_flowchart_nodes() if hasattr(engine, "get_flowchart_nodes") else {}
            raw = nodes.get(node_id) if isinstance(nodes, dict) and node_id else None
            if isinstance(raw, dict):
                current_prompt = str(raw.get("prompt") or "").strip() or None
    except Exception:
        current_prompt = None
    payload["current_node_prompt"] = current_prompt
    return payload


@router.get("/admin/warranty/tickets", tags=["admin-warranty"])
async def list_warranty_tickets(
    status: Optional[str] = None,
    domain: Optional[str] = None,
    channel: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    x_admin_key: Optional[str] = Header(default=None),
):
    """
    List warranty tickets.  Admin-only.

    Optional query params: status, domain, channel, limit, offset.
    """
    _require_admin(x_admin_key)
    engine = _lazy_engine()
    tickets = engine.get_tickets(
        status=status,
        domain=domain,
        channel=channel,
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

    from warranty_freshdesk_case import maybe_sync_admin_decision_to_freshdesk  # noqa: WPS433

    freshdesk_sync = maybe_sync_admin_decision_to_freshdesk(
        ticket_id,
        decision=body.decision,
        note=body.note or "",
        customer_message=body.customer_message or "",
        decided_by=body.decided_by or "",
        engine=engine,
    )
    if freshdesk_sync.get("synced"):
        ticket = engine.get_ticket(ticket_id) or ticket

    logger.info(
        f"⚖️  Admin decision — ticket={ticket_id} decision={body.decision} "
        f"decided_by={body.decided_by} customer_email_sent={customer_email_sent} "
        f"freshdesk_synced={freshdesk_sync.get('synced')}"
    )
    return {
        "ticket": _serialize_admin_ticket(ticket, turns=turns, evidences=evidences),
        "customer_email_sent": customer_email_sent,
        "customer_email_skip_reason": customer_email_skip_reason,
        "freshdesk_sync": freshdesk_sync,
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


@router.get("/admin/warranty/freshdesk-status", tags=["admin-warranty"])
async def admin_freshdesk_status(
    x_admin_key: Optional[str] = Header(default=None),
    probe: bool = True,
):
    """Freshdesk connection, sync history, and on-disk knowledge snapshot."""
    _require_admin(x_admin_key)

    try:
        from freshdesk_status import get_freshdesk_dashboard  # noqa: WPS433
    except ImportError:
        from app.freshdesk_status import get_freshdesk_dashboard  # type: ignore  # noqa: WPS433

    return get_freshdesk_dashboard(probe_connection=probe)


@router.get("/admin/warranty/freshdesk-field-catalog", tags=["admin-warranty"])
async def admin_freshdesk_field_catalog(
    x_admin_key: Optional[str] = Header(default=None),
    refresh: bool = False,
):
    """
    Official Freshdesk status + custom dropdown ID maps (for Ticket Queue labeling).

    Set ``refresh=1`` to pull live from ``/api/v2/admin/ticket_fields`` and update
    ``data/freshdesk_field_choices.json``.
    """
    _require_admin(x_admin_key)

    try:
        from freshdesk_field_catalog import get_field_catalog  # noqa: WPS433
    except ImportError:
        from app.freshdesk_field_catalog import get_field_catalog  # type: ignore  # noqa: WPS433

    try:
        catalog = get_field_catalog(refresh=refresh)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not catalog.get("lookup"):
        raise HTTPException(
            status_code=503,
            detail="Freshdesk field catalog is empty. Set FRESHDESK credentials and refresh=1.",
        )
    return catalog


@router.post("/admin/warranty/{ticket_id}/freshdesk-link", tags=["admin-warranty"])
async def admin_freshdesk_link(
    ticket_id: str,
    x_admin_key: Optional[str] = Header(default=None),
):
    """Create or return the Freshdesk ticket linked to a warranty case."""
    _require_admin(x_admin_key)

    engine = _lazy_engine()
    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found.")

    try:
        from warranty_freshdesk_case import ensure_freshdesk_link  # noqa: WPS433
    except ImportError:
        from app.warranty_freshdesk_case import ensure_freshdesk_link  # type: ignore  # noqa: WPS433

    result = ensure_freshdesk_link(ticket_id, engine=engine)
    ticket = engine.get_ticket(ticket_id) or ticket
    turns = engine.get_turns(ticket_id)
    evidences = engine.get_evidences(ticket_id)

    if result.get("error") and not result.get("freshdesk_ticket_id"):
        raise HTTPException(
            status_code=502,
            detail=result.get("detail") or result.get("error") or "Freshdesk link failed.",
        )

    ok = bool(
        result.get("created")
        or result.get("reason") == "already_linked"
        or result.get("freshdesk_ticket_id")
    )
    return {
        "ok": ok,
        "freshdesk": result,
        "ticket": _serialize_admin_ticket(ticket, turns=turns, evidences=evidences),
    }


@router.post("/admin/warranty/sync-freshdesk", tags=["admin-warranty"])
async def admin_sync_freshdesk(
    background_tasks: BackgroundTasks,
    x_admin_key: Optional[str] = Header(default=None),
    max_pages: int = 30,
    months_back: int = 12,
    llm_rescue: bool = True,
    rebuild_faiss: bool = True,
):
    """
    Pull resolved Freshdesk tickets into data/freshdesk_tickets.json, then
    optionally run the LLM rescue pass so tickets whose agent replies don't
    yield extractable steps still land in the knowledge base. Admin-only.

    Query params
    ------------
    max_pages : total Freshdesk Search pages to fetch (1..60, default 30).
        Each page returns up to 30 Resolved/Closed tickets only.
    months_back : calendar months to scan, newest first (1..36, default 12).
    llm_rescue : run ``freshdesk_ticket_summarizer`` after sync when
        OPENAI_API_KEY is set (default True). Set false to skip the LLM cost.
    rebuild_faiss : rebuild the ``freshdesk_qa`` FAISS index after a successful
        sync (default True). Set false to skip the embedding cost.
    """
    _require_admin(x_admin_key)

    try:
        from freshdesk_sync import _OUTPUT_PATH, sync_freshdesk_knowledge  # noqa: WPS433
        from warranty_knowledge import clear_knowledge_cache, load_knowledge_entries  # noqa: WPS433
        from freshdesk_ticket_summarizer import (  # noqa: WPS433
            is_enabled as summarizer_enabled,
            summarize_missing_tickets,
        )
    except ImportError:
        from app.freshdesk_sync import (  # type: ignore  # noqa: WPS433
            _OUTPUT_PATH,
            sync_freshdesk_knowledge,
        )
        from app.warranty_knowledge import (  # type: ignore  # noqa: WPS433
            clear_knowledge_cache,
            load_knowledge_entries,
        )
        from app.freshdesk_ticket_summarizer import (  # type: ignore  # noqa: WPS433
            is_enabled as summarizer_enabled,
            summarize_missing_tickets,
        )

    pages = max(1, min(int(max_pages), 60))
    months = max(1, min(int(months_back), 36))
    try:
        result = sync_freshdesk_knowledge(max_pages=pages, months_back=months)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Freshdesk API error: {exc}") from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    summarize_stats: Optional[Dict[str, Any]] = None
    if llm_rescue and result.get("ok") and summarizer_enabled():
        try:
            with open(_OUTPUT_PATH, encoding="utf-8") as handle:
                import json as _json

                raw_tickets = _json.load(handle)
        except (OSError, ValueError):
            raw_tickets = []
        if raw_tickets:
            try:
                summarize_stats = summarize_missing_tickets(raw_tickets)
                logger.info(
                    "Freshdesk LLM rescue — %s",
                    summarize_stats,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Freshdesk LLM rescue failed: %s", exc)
                summarize_stats = {"error": str(exc)}

    from freshdesk_knowledge_refresh import (  # noqa: WPS433
        build_knowledge_yield_stats,
        invalidate_warranty_knowledge_caches,
        log_ticket_sync_yield,
        schedule_faiss_rebuild,
    )

    invalidate_warranty_knowledge_caches()
    yield_stats = build_knowledge_yield_stats(
        synced_ticket_rows=int(result.get("ticket_count") or 0),
        resolved_scanned=int(result.get("resolved_scanned") or 0),
    )
    log_ticket_sync_yield(
        ok=bool(result.get("ok")),
        ticket_count=int(result.get("ticket_count") or 0),
        resolved_scanned=int(result.get("resolved_scanned") or 0),
        stats=yield_stats,
    )

    response: Dict[str, Any] = {
        **result,
        **yield_stats,
        "llm_rescue_enabled": bool(llm_rescue and summarizer_enabled()),
    }
    if summarize_stats is not None:
        response["llm_rescue_stats"] = summarize_stats

    response.update(
        schedule_faiss_rebuild(
            background_tasks,
            enabled=rebuild_faiss,
            sync_ok=bool(result.get("ok")),
        )
    )

    return response


@router.get("/admin/warranty/freshdesk-solutions/probe", tags=["admin-warranty"])
async def admin_probe_freshdesk_solutions(
    x_admin_key: Optional[str] = Header(default=None),
):
    """
    Read-only probe of the Freshdesk Solutions/KB. Cheap enough to run
    interactively so admins can decide whether to invest in KB ingest.
    """
    _require_admin(x_admin_key)
    try:
        from freshdesk_sync import probe_freshdesk_solutions  # noqa: WPS433
    except ImportError:
        from app.freshdesk_sync import probe_freshdesk_solutions  # type: ignore  # noqa: WPS433

    try:
        return probe_freshdesk_solutions()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Freshdesk API error: {exc}") from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/admin/warranty/sync-freshdesk-solutions",
    tags=["admin-warranty"],
)
async def admin_sync_freshdesk_solutions(
    background_tasks: BackgroundTasks,
    x_admin_key: Optional[str] = Header(default=None),
    max_articles: int = 500,
    rebuild_faiss: bool = True,
):
    """
    Ingest Freshdesk KB(Solutions) articles into
    ``data/freshdesk_solutions.json`` and refresh the warranty knowledge
    cache. Admin-only.
    """
    _require_admin(x_admin_key)

    try:
        from freshdesk_sync import sync_freshdesk_solutions  # noqa: WPS433
        from warranty_knowledge import clear_knowledge_cache, load_knowledge_entries  # noqa: WPS433
    except ImportError:
        from app.freshdesk_sync import sync_freshdesk_solutions  # type: ignore  # noqa: WPS433
        from app.warranty_knowledge import (  # type: ignore  # noqa: WPS433
            clear_knowledge_cache,
            load_knowledge_entries,
        )

    n = max(1, min(int(max_articles), 5000))
    try:
        result = sync_freshdesk_solutions(max_articles=n)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Freshdesk API error: {exc}") from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    from freshdesk_knowledge_refresh import (  # noqa: WPS433
        build_knowledge_yield_stats,
        invalidate_warranty_knowledge_caches,
        log_kb_sync_yield,
        schedule_faiss_rebuild,
    )

    invalidate_warranty_knowledge_caches()
    yield_stats = build_knowledge_yield_stats(
        synced_kb_articles=int(result.get("article_count") or 0),
    )
    log_kb_sync_yield(
        ok=bool(result.get("ok")),
        article_count=int(result.get("article_count") or 0),
        stats=yield_stats,
    )

    response: Dict[str, Any] = {
        **result,
        **yield_stats,
    }
    response.update(
        schedule_faiss_rebuild(
            background_tasks,
            enabled=rebuild_faiss,
            sync_ok=bool(result.get("ok")),
        )
    )
    return response


@router.get("/admin/warranty/faiss/status", tags=["admin-warranty"])
async def admin_faiss_status(
    x_admin_key: Optional[str] = Header(default=None),
):
    """Report the last freshdesk_qa FAISS rebuild status (admin-only)."""
    _require_admin(x_admin_key)
    try:
        from warranty_faiss_rebuilder import get_status  # noqa: WPS433
    except ImportError:
        from app.warranty_faiss_rebuilder import get_status  # type: ignore  # noqa: WPS433
    return get_status()


@router.post("/admin/warranty/rebuild-faiss", tags=["admin-warranty"])
async def admin_rebuild_faiss(
    background_tasks: BackgroundTasks,
    x_admin_key: Optional[str] = Header(default=None),
    wait: bool = False,
):
    """
    Rebuild the freshdesk_qa FAISS index used by the general-purpose
    ``/api/v1/chat`` endpoint. Admin-only.

    By default returns immediately (``202 Accepted``-style) and runs the
    rebuild in a BackgroundTask; poll ``/admin/warranty/faiss/status`` for
    progress. Set ``?wait=true`` to block until it finishes (mostly for the
    CLI + tests).
    """
    _require_admin(x_admin_key)
    try:
        from warranty_faiss_rebuilder import get_status, rebuild_freshdesk_qa_index  # noqa: WPS433
    except ImportError:
        from app.warranty_faiss_rebuilder import (  # type: ignore  # noqa: WPS433
            get_status,
            rebuild_freshdesk_qa_index,
        )

    current = get_status()
    if current.get("running"):
        return {"scheduled": False, **current, "message": "Already running."}

    if wait:
        return rebuild_freshdesk_qa_index()

    background_tasks.add_task(rebuild_freshdesk_qa_index)
    return {"scheduled": True, "message": "FAISS rebuild scheduled in background."}


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
