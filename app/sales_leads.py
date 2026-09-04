"""
sales_leads.py
==============
Single owner of sales-lead persistence and delivery.

Every channel that captures a shopper's contact details routes through
``capture_lead`` + ``deliver_lead`` so a lead can never be stored without
also being forwarded to the sales inbox. The Tidio Flow path, the generic
chat endpoint, and the explicit lead endpoint all share this module — the
duplication that previously existed let the Tidio path (our only production
channel) save leads that nobody was ever notified about.

Delivery is best-effort: a failed send is recorded on the row and alerted to
ops, never raised into the customer's turn.
"""

from __future__ import annotations

import logging
from typing import Optional

from sales_models import SalesLead, SalesSession
from warranty_models import warranty_db_session

logger = logging.getLogger(__name__)

_FALLBACK_SUMMARY = "Sales AI — customer requested follow-up"


def capture_lead(
    *,
    session_id: str,
    domain: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    interest_summary: Optional[str] = None,
    reason: str = "save_pick",
) -> Optional[int]:
    """Persist a lead and flip the session to ``handoff``.

    Returns the new lead id, or ``None`` when neither an email nor a phone
    number was supplied (nothing actionable to forward).
    """
    email = (email or "").strip()
    phone = (phone or "").strip()
    if not email and not phone:
        return None

    summary = (interest_summary or "").strip() or None
    reason = (reason or "save_pick").strip() or "save_pick"

    with warranty_db_session() as db:
        row = SalesLead(
            session_id=session_id,
            email=email or None,
            phone=phone or None,
            domain=domain,
            interest_summary=summary,
            reason=reason,
            forwarded="pending",
        )
        db.add(row)
        db.flush()
        lead_id = int(row.id)

        session = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == session_id)
            .one_or_none()
        )
        if session is not None:
            if email and not session.contact_email:
                session.contact_email = email
            if phone and not session.contact_phone:
                session.contact_phone = phone
            session.status = "handoff"

    return lead_id


def deliver_lead(
    *,
    lead_id: int,
    domain: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    interest_summary: Optional[str] = None,
) -> None:
    """Notify the sales inbox and record the outcome on the lead row."""
    email = (email or "").strip()
    phone = (phone or "").strip()
    sender = _resolve_transport()
    if sender is None:
        _record_failure(lead_id=lead_id, domain=domain, error="transport_unavailable")
        return

    # Subject identifies the contact; the body carries whichever we have.
    contact_label = email or phone
    details = [(interest_summary or "").strip() or _FALLBACK_SUMMARY]
    if email:
        details.append(f"Email: {email}")
    if phone:
        details.append(f"Phone: {phone}")

    try:
        ok = bool(sender(contact_label, "\n".join(details), "", domain or ""))
    except Exception as exc:  # pragma: no cover — SMTP side-effects
        logger.exception("sales lead email failed for lead_id=%s", lead_id)
        _record_failure(lead_id=lead_id, domain=domain, error=str(exc))
        return

    if ok:
        _mark_lead_status(lead_id, status="sent", error=None)
        _send_shopper_receipt(
            email=email,
            interest_summary=interest_summary,
            domain=domain,
        )
        return
    _record_failure(lead_id=lead_id, domain=domain, error="smtp_returned_false")


def schedule_lead_delivery(
    background_tasks,
    *,
    lead_id: int,
    domain: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    interest_summary: Optional[str] = None,
) -> None:
    """Queue delivery when a request has BackgroundTasks, else send inline."""
    kwargs = {
        "lead_id": lead_id,
        "domain": domain,
        "email": email,
        "phone": phone,
        "interest_summary": interest_summary,
    }
    if background_tasks is None:
        deliver_lead(**kwargs)
        return
    background_tasks.add_task(deliver_lead, **kwargs)


def _resolve_transport():
    """Return ``send_sales_lead_email`` regardless of how the app was imported."""
    try:
        from main import send_sales_lead_email  # type: ignore
    except ImportError:  # pragma: no cover — packaged import layout
        try:
            from app.main import send_sales_lead_email  # type: ignore
        except ImportError:
            logger.warning("sales lead capture: email transport unavailable")
            return None
    return send_sales_lead_email


def _send_shopper_receipt(
    *,
    email: Optional[str],
    interest_summary: Optional[str],
    domain: str,
) -> None:
    """Best-effort confirmation to the shopper. Never fails the sales notify."""
    to_addr = (email or "").strip()
    if not to_addr:
        return
    try:
        from main import send_sales_shopper_receipt_email  # type: ignore
    except ImportError:
        try:
            from app.main import send_sales_shopper_receipt_email  # type: ignore
        except ImportError:
            logger.warning("sales shopper receipt: email transport unavailable")
            return
    try:
        send_sales_shopper_receipt_email(
            to_addr,
            (interest_summary or "").strip() or _FALLBACK_SUMMARY,
            domain or "",
        )
    except Exception:
        logger.exception("sales shopper receipt failed")


def _mark_lead_status(lead_id: int, *, status: str, error: Optional[str]) -> None:
    with warranty_db_session() as db:
        row = db.query(SalesLead).filter(SalesLead.id == lead_id).one_or_none()
        if row is None:
            return
        row.forwarded = status
        row.forwarded_error = error


def _record_failure(*, lead_id: int, domain: str, error: str) -> None:
    _mark_lead_status(lead_id, status="failed", error=error[:500])
    _alert_lead_failure(lead_id=lead_id, domain=domain, error=error)


def _alert_lead_failure(*, lead_id: int, domain: str, error: str) -> None:
    """Best-effort ops alert, deliberately free of customer PII."""
    try:
        from ops_notify import send_ops_alert  # noqa: WPS433

        send_ops_alert(
            "[Sales AI] Lead forwarding failed",
            (
                f"Lead ID: {lead_id}\n"
                f"Domain: {domain or 'unknown'}\n"
                f"Error: {(error or 'unknown')[:500]}\n\n"
                "Review the failed lead in /admin/sales."
            ),
        )
    except Exception:  # pragma: no cover — alerting must not break chat
        logger.exception("sales lead failure alert failed for lead_id=%s", lead_id)
