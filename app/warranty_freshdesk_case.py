"""
Create and update Freshdesk tickets for warranty chat cases.

Inbound sync (Freshdesk → JSON) lives in ``freshdesk_sync.py``.
This module handles the outbound path: warranty ticket → Freshdesk case.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

import requests

from warranty_case_ref import case_reference_for_ticket

logger = logging.getLogger(__name__)


def _create_enabled() -> bool:
    return os.getenv("WARRANTY_FRESHDESK_CREATE_CASE", "1") == "1"


def _freshdesk_enabled() -> bool:
    if not _create_enabled():
        return False
    return bool(os.getenv("FRESHDESK_DOMAIN", "").strip()) and bool(
        os.getenv("FRESHDESK_API_KEY", "").strip()
    )


def _client():
    from freshdesk_sync import normalize_freshdesk_domain  # noqa: WPS433

    domain = normalize_freshdesk_domain(os.getenv("FRESHDESK_DOMAIN", ""))
    api_key = os.getenv("FRESHDESK_API_KEY", "").strip()
    return {
        "domain": domain,
        "auth": (api_key, "X"),
        "headers": {"Content-Type": "application/json"},
        "base_url": f"https://{domain}/api/v2",
    }


def freshdesk_ticket_url(domain: str, ticket_id: int | str) -> str:
    return f"https://{domain}/a/tickets/{ticket_id}"


def _build_case_description(ticket, *, case_ref: str, turns=None) -> str:
    from warranty_email import resolve_customer_email  # noqa: WPS433

    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    channel = str(collected.get("channel") or "").strip().lower()
    if channel == "phone":
        opener = "Warranty case submitted via after-hours phone IVR."
    else:
        opener = "Warranty chat case submitted via the Osaki/Titan warranty bot."

    lines = [
        opener,
        "",
        f"Case reference: {case_ref}",
        f"Internal ticket ID: {ticket.ticket_id}",
        f"Session: {ticket.session_id}",
        f"Domain: {ticket.domain}",
        f"Status: {ticket.status}",
        f"Issue type: {ticket.issue_type or '—'}",
        f"Model: {ticket.model_name or '—'}",
        f"Terminal node: {ticket.current_node_id}",
    ]
    caller_phone = str(collected.get("caller_phone") or "").strip()
    if caller_phone:
        lines.append(f"Caller phone: {caller_phone}")

    email = resolve_customer_email(ticket, turns=turns)
    if email:
        lines.append(f"Customer email: {email}")

    error_code = str(collected.get("error_code") or "").strip()
    if error_code:
        lines.append(f"Customer error code: {error_code}")
        fonz_meaning = str(collected.get("fonz_meaning") or "").strip()
        fonz_parts = str(collected.get("fonz_parts_internal") or "").strip()
        if fonz_meaning:
            lines.append(f"Fonz meaning: {fonz_meaning[:400]}")
        if fonz_parts:
            lines.append(f"Fonz parts (internal): {fonz_parts[:300]}")

    tracking_raw = collected.get("tracking_snapshot")
    if tracking_raw:
        lines.append("")
        lines.append("Tracking snapshot was captured during delivery intake.")

    if turns:
        lines.append("")
        lines.append("Recent workflow answers:")
        for turn in list(turns)[-8:]:
            prompt = (getattr(turn, "node_prompt", None) or "")[:120]
            answer = getattr(turn, "customer_answer", None) or ""
            lines.append(f"- Q: {prompt}")
            lines.append(f"  A: {answer}")

    lines.append("")
    lines.append("Review this case in the warranty admin portal.")
    return "\n".join(lines)


def _apply_freshdesk_routing(payload: Dict[str, Any]) -> None:
    group_id = os.getenv("FRESHDESK_WARRANTY_GROUP_ID", "").strip()
    if group_id.isdigit():
        payload["group_id"] = int(group_id)
    product_id = os.getenv("FRESHDESK_WARRANTY_PRODUCT_ID", "").strip()
    if product_id.isdigit():
        payload["product_id"] = int(product_id)


def _post_private_note(fd_id: str, body: str) -> Dict[str, Any]:
    cfg = _client()
    try:
        response = requests.post(
            f"{cfg['base_url']}/tickets/{fd_id}/notes",
            auth=cfg["auth"],
            headers=cfg["headers"],
            json={"body": body, "private": True},
            timeout=15,
        )
    except requests.RequestException as exc:
        logger.warning("Freshdesk note failed for fd=%s: %s", fd_id, exc)
        return {"posted": False, "error": str(exc)}

    if response.status_code >= 400:
        return {
            "posted": False,
            "error": f"http_{response.status_code}",
            "detail": (response.text or "")[:300],
        }
    return {"posted": True, "freshdesk_ticket_id": fd_id}


def maybe_create_freshdesk_case(
    ticket_id: str,
    *,
    engine=None,
    allow_any_status: bool = False,
) -> Dict[str, Any]:
    """
    Create a Freshdesk ticket once when a warranty case reaches admin review.

    Idempotent — skips when ``collected_data.freshdesk_ticket_id`` is already set.
    """
    if not _freshdesk_enabled():
        return {"created": False, "skipped": True, "reason": "freshdesk_disabled"}

    if engine is None:
        from warranty_workflow import WarrantyEngine  # noqa: WPS433

        engine = WarrantyEngine

    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        return {"created": False, "error": "ticket_not_found"}

    collected = ticket.get_collected()
    existing = str(collected.get("freshdesk_ticket_id") or "").strip()
    if existing:
        domain = str(collected.get("freshdesk_domain") or "")
        url = str(collected.get("freshdesk_url") or "")
        if not url and domain and existing.isdigit():
            url = freshdesk_ticket_url(domain, int(existing))
        return {
            "created": False,
            "skipped": True,
            "reason": "already_linked",
            "freshdesk_ticket_id": existing,
            "freshdesk_url": url,
        }

    if not allow_any_status and str(ticket.status) not in (
        "awaiting_admin_review",
        "awaiting_evidence",
        "admin_reviewing",
        "need_more_information",
    ):
        return {"created": False, "skipped": True, "reason": "status_not_eligible"}

    from warranty_email import resolve_customer_email  # noqa: WPS433
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    turns = engine.get_turns(ticket_id)
    case_ref = case_reference_for_ticket(ticket)
    customer_email = resolve_customer_email(ticket, turns=turns)

    subject_parts = [case_ref]
    if ticket.model_name:
        subject_parts.append(str(ticket.model_name))
    if ticket.issue_type:
        subject_parts.append(str(ticket.issue_type))
    subject = " | ".join(subject_parts) + " — Warranty Bot"

    payload: Dict[str, Any] = {
        "subject": subject[:255],
        "description": _build_case_description(ticket, case_ref=case_ref, turns=turns),
        "priority": 2,
        "status": 2,
        "tags": ["warranty-bot", case_ref],
        "source": 2,
    }
    channel = str(collected.get("channel") or "").strip().lower()
    if channel == "phone":
        payload["tags"].append("phone-ivr")
    if customer_email:
        payload["email"] = customer_email
    else:
        caller_phone = str(collected.get("caller_phone") or "").strip()
        if caller_phone:
            payload["phone"] = caller_phone
            payload["name"] = "Warranty Phone Caller"
        else:
            fallback = os.getenv("WARRANTY_FRESHDESK_FALLBACK_EMAIL", "service@osakititan.com").strip()
            payload["email"] = fallback or "service@osakititan.com"

    _apply_freshdesk_routing(payload)

    cfg = _client()
    try:
        response = requests.post(
            f"{cfg['base_url']}/tickets",
            auth=cfg["auth"],
            headers=cfg["headers"],
            json=payload,
            timeout=20,
        )
    except requests.RequestException as exc:
        logger.warning("Freshdesk case create failed for %s: %s", ticket_id, exc)
        return {"created": False, "error": str(exc)}

    if response.status_code >= 400:
        logger.warning(
            "Freshdesk case create HTTP %s for %s: %s",
            response.status_code,
            ticket_id,
            (response.text or "")[:400],
        )
        return {
            "created": False,
            "error": f"http_{response.status_code}",
            "detail": (response.text or "")[:400],
        }

    data = response.json()
    fd_id = str(data.get("id") or "")
    fd_url = freshdesk_ticket_url(cfg["domain"], fd_id)

    with warranty_db_session() as db:
        row = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if row:
            c = row.get_collected()
            c["freshdesk_ticket_id"] = fd_id
            c["freshdesk_url"] = fd_url
            c["freshdesk_domain"] = cfg["domain"]
            c["case_reference"] = case_ref
            import json as _json  # noqa: WPS433

            row.collected_data = _json.dumps(c)

    logger.info(
        "Freshdesk case created ticket=%s fd_id=%s ref=%s",
        ticket_id,
        fd_id,
        case_ref,
    )
    return {
        "created": True,
        "freshdesk_ticket_id": fd_id,
        "freshdesk_url": fd_url,
        "case_reference": case_ref,
    }


_DECISION_LABELS: Dict[str, str] = {
    "admin_reviewing": "Reviewing",
    "need_more_information": "Need more information",
    "approved": "Approved",
    "rejected": "Rejected",
    "closed": "Closed",
}


def maybe_sync_admin_decision_to_freshdesk(
    ticket_id: str,
    *,
    decision: str,
    note: str = "",
    customer_message: str = "",
    decided_by: str = "",
    engine=None,
) -> Dict[str, Any]:
    """Ensure a Freshdesk link exists and append the admin decision as a private note."""
    if not _freshdesk_enabled():
        return {"synced": False, "skipped": True, "reason": "freshdesk_disabled"}

    link = maybe_create_freshdesk_case(
        ticket_id,
        engine=engine,
        allow_any_status=True,
    )
    fd_id = str(link.get("freshdesk_ticket_id") or "").strip()
    if not fd_id:
        if engine is None:
            from warranty_workflow import WarrantyEngine  # noqa: WPS433

            engine = WarrantyEngine
        ticket = engine.get_ticket(ticket_id)
        if ticket is not None:
            collected = ticket.get_collected()
            fd_id = str(collected.get("freshdesk_ticket_id") or "").strip()

    if not fd_id or not fd_id.isdigit():
        return {
            "synced": False,
            "error": link.get("error") or "no_freshdesk_link",
            "detail": link.get("detail"),
        }

    label = _DECISION_LABELS.get(decision, decision)
    lines = [
        "Admin decision recorded in the warranty portal:",
        "",
        f"Decision: {label} ({decision})",
    ]
    if decided_by:
        lines.append(f"Decided by: {decided_by}")
    if note.strip():
        lines.append("")
        lines.append("Internal note:")
        lines.append(note.strip())
    if customer_message.strip():
        lines.append("")
        lines.append("Customer message (also emailed when configured):")
        lines.append(customer_message.strip())

    posted = _post_private_note(fd_id, "\n".join(lines))
    if not posted.get("posted"):
        return {"synced": False, **posted}

    return {
        "synced": True,
        "freshdesk_ticket_id": fd_id,
        "freshdesk_url": freshdesk_ticket_url(_client()["domain"], fd_id),
        "link_created": bool(link.get("created")),
    }


def ensure_freshdesk_link(ticket_id: str, *, engine=None) -> Dict[str, Any]:
    """Manual retry: create or return the linked Freshdesk ticket."""
    return maybe_create_freshdesk_case(
        ticket_id,
        engine=engine,
        allow_any_status=True,
    )


def maybe_add_freshdesk_customer_reply(
    ticket_id: str,
    note_text: str,
    *,
    engine=None,
) -> Dict[str, Any]:
    """Append a customer follow-up note to the linked Freshdesk ticket."""
    if not _freshdesk_enabled() or not (note_text or "").strip():
        return {"posted": False, "skipped": True}

    if engine is None:
        from warranty_workflow import WarrantyEngine  # noqa: WPS433

        engine = WarrantyEngine

    ticket = engine.get_ticket(ticket_id)
    if ticket is None:
        return {"posted": False, "error": "ticket_not_found"}

    collected = ticket.get_collected()
    fd_id = str(collected.get("freshdesk_ticket_id") or "").strip()
    if not fd_id or not fd_id.isdigit():
        return {"posted": False, "skipped": True, "reason": "no_freshdesk_link"}

    body = (
        "Customer follow-up via warranty chat:\n\n"
        f"{note_text.strip()}\n\n"
        f"Case reference: {collected.get('case_reference') or case_reference_for_ticket(ticket)}"
    )
    return _post_private_note(fd_id, body)


def schedule_freshdesk_case_creation(
    ticket_id: str,
    *,
    allow_any_status: bool = False,
) -> None:
    """Fire-and-forget Freshdesk case creation (non-blocking HTTP handler)."""
    if not _freshdesk_enabled():
        return

    def _run() -> None:
        try:
            maybe_create_freshdesk_case(ticket_id, allow_any_status=allow_any_status)
        except Exception as exc:
            logger.warning("background Freshdesk create failed for %s: %s", ticket_id, exc)

    threading.Thread(target=_run, daemon=True).start()
