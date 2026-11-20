"""At-most-one Freshdesk create attempt for explicitly routed human handoffs.

Only an approved internal requester and support group receive this task. Names,
contact data, model/serial information, transcripts and attachments stay in the
authenticated warranty portal. This is not the customer-requester Freshdesk case.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from urllib.parse import quote, urlparse

import requests
from sqlalchemy import text

from warranty_case_ref import case_reference_for_ticket, parse_case_reference
from warranty_models import WarrantyTicket, warranty_db_session
from warranty_notification_jobs import WarrantyHandoffDispatch, WarrantyNotificationJob, _Claim, _PermanentDeliveryError


@dataclass(frozen=True)
class _Routing:
    api_url: str
    api_key: str
    requester_id: int
    group_id: int
    portal: str


def _positive_id(value: object) -> int | None:
    if type(value) is int:
        return value if 0 < value < 2**63 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,18}", value):
        return int(value) or None
    return None


def _routing() -> _Routing | None:
    # The legacy tenant-wide create switch is also the emergency stop for the
    # explicitly selected internal channel; do not silently reroute its jobs.
    if os.getenv("WARRANTY_FRESHDESK_CREATE_CASE", "1") != "1":
        return None
    # Refuse arbitrary hosts, redirect forwarding, and customer-provided URLs.
    host_value = os.getenv("FRESHDESK_DOMAIN", "").strip()
    parsed = urlparse(host_value if "://" in host_value else f"https://{host_value}")
    host = (parsed.hostname or "").lower()
    if host and "." not in host:
        host += ".freshdesk.com"
    if not (
        parsed.scheme == "https"
        and not parsed.username and not parsed.password
        and not parsed.params and not parsed.query and not parsed.fragment
        and parsed.path in {"", "/"}
        and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.freshdesk\.com", host)
        and ":" not in parsed.netloc
    ):
        return None
    key = os.getenv("FRESHDESK_API_KEY", "").strip()
    group = _positive_id(os.getenv("FRESHDESK_HANDOFF_GROUP_ID", "").strip())
    requester = _positive_id(os.getenv("FRESHDESK_HANDOFF_REQUESTER_ID", "").strip())
    portal = os.getenv("FRESHDESK_HANDOFF_ADMIN_ORIGIN", "").strip().rstrip("/")
    if not (key and group and requester and re.fullmatch(r"https://[A-Za-z0-9.-]+", portal)):
        return None
    return _Routing(f"https://{host}/api/v2/tickets", key, requester, group, portal)


def freshdesk_handoff_health() -> dict[str, str | bool]:
    route = "freshdesk" if os.getenv("WARRANTY_HANDOFF_CHANNEL", "smtp").strip().lower() == "freshdesk" else "smtp"
    try:
        configured = _routing() is not None
    except ValueError:
        configured = False
    return {"new_job_channel": route, "settings_present": configured, "acceptance_is_staff_confirmation": False}


def _payload(ticket, *, department: str, safety: bool, routing: _Routing) -> dict:
    reference = case_reference_for_ticket(ticket)
    if parse_case_reference(reference) is None or not re.fullmatch(r"[a-fA-F0-9-]{8,64}", str(ticket.ticket_id)):
        raise _PermanentDeliveryError("handoff_case_reference_invalid")
    recipient = "Sales" if department == "sales" and not safety else "Warranty"
    reason = "Safety review" if safety else "Human support requested"
    link = routing.portal + "/admin/warranty/" + quote(str(ticket.ticket_id), safe="")
    body = (
        f"<p>Internal chat handoff: {html.escape(reason)}.</p>"
        f"<p>Case: {html.escape(reference)}; intended department: {recipient}.</p>"
        f'<p><a href="{html.escape(link, quote=True)}">Open the authenticated warranty case</a>.</p>'
        "<p>This ticket was created for internal triage only. Review the warranty case for customer "
        "follow-up details. Do not reply to the requester on this internal ticket.</p>"
    )
    return {
        "subject": f"{reference} — {recipient} — {reason}",
        "description": body,
        "requester_id": routing.requester_id,
        "group_id": routing.group_id,
        "priority": 4 if safety else 2,
        "status": 2,
        "source": 2,
        "type": "Inquiry",
        "tags": ["warranty-handoff-internal", reference.lower(), f"handoff-{recipient.lower()}"],
    }


def _post_once(routing: _Routing, payload: dict) -> tuple[str, int | None, str]:
    """Do not retry a non-idempotent POST even when the response looks transient."""
    try:
        response = requests.post(
            routing.api_url,
            auth=(routing.api_key, "X"),
            json=payload,
            timeout=(3.0, 8.0),
            allow_redirects=False,
        )
        status = int(response.status_code)
        if status in {401, 403}:
            return "rejected", None, "freshdesk_authentication_failed"
        if status == 429:
            return "rejected", None, "freshdesk_rate_limited"
        if status in {400, 404, 405, 422}:
            return "rejected", None, "freshdesk_request_rejected"
        if 200 <= status < 300:
            body = response.json()
            ticket_id = _positive_id(body.get("id")) if isinstance(body, dict) else None
            if ticket_id:
                return "submitted", ticket_id, ""
    except (requests.RequestException, ValueError, TypeError):
        pass
    return "uncertain", None, "freshdesk_submission_uncertain"


def deliver_freshdesk_handoff(claim: _Claim) -> bool | None:
    """None selects the legacy SMTP route; True confirms helpdesk acceptance."""
    with warranty_db_session() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        dispatch = db.get(WarrantyHandoffDispatch, claim.job_id)
        if dispatch is None:
            return None
        if dispatch.state == "submitted" and dispatch.provider_ticket_id:
            return True
        if dispatch.state != "queued":
            raise _PermanentDeliveryError("freshdesk_submission_uncertain")
        current = db.get(WarrantyNotificationJob, claim.job_id)
        if current is None or current.status != "processing" or current.lease_token != claim.token:
            return False
        ticket = db.query(WarrantyTicket).filter_by(ticket_id=claim.ticket_id).first()
        if ticket is None or ticket.status != "awaiting_admin_review":
            raise _PermanentDeliveryError("handoff_source_no_longer_pending")
        data = ticket.get_collected()
        if not data.get("human_support_requested") or data.get("human_support_notification_source") != claim.source_key:
            raise _PermanentDeliveryError("handoff_source_no_longer_pending")
        try:
            routing = _routing()
        except ValueError:
            routing = None
        if routing is None:
            raise _PermanentDeliveryError("freshdesk_handoff_not_configured")
        payload = _payload(
            ticket,
            department=str(data.get("human_support_department") or ""),
            safety=data.get("human_support_reason") == "safety_hazard",
            routing=routing,
        )
        dispatch.state = "submitting"

    # The durable marker is committed before entering the non-idempotent call.
    # A timeout, bad response, or crash will never make this job create again.
    state, provider_id, code = _post_once(routing, payload)
    with warranty_db_session() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        dispatch = db.get(WarrantyHandoffDispatch, claim.job_id)
        if dispatch is None or dispatch.state != "submitting":
            raise _PermanentDeliveryError("freshdesk_submission_uncertain")
        dispatch.state = state
        dispatch.provider_ticket_id = provider_id
    if state == "submitted":
        return True
    raise _PermanentDeliveryError(code)
