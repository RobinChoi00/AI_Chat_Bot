"""Submit requested human support to the correct inbox using saved case data."""

from __future__ import annotations

from warranty_models import WarrantyConversationMessage, WarrantyTicket, warranty_db_session


def send_handoff_notification(ticket_id: str) -> bool:
    """SMTP submission is not a staff response, appointment, or callback booking."""
    from warranty_case_ref import case_reference_for_ticket

    with warranty_db_session() as db:
        ticket = db.query(WarrantyTicket).filter_by(ticket_id=ticket_id).first()
        if ticket is None:
            return False
        data = ticket.get_collected()
        if not data.get("human_support_requested"):
            return False
        department = str(data.get("human_support_department") or "warranty")
        reason = str(data.get("human_support_reason") or "customer_request")
        contact = str(data.get("customer_contact_email") or "")
        domain = str(ticket.domain or "")
        lines = [
            "A customer requested human support in the chat.",
            f"Case: {case_reference_for_ticket(ticket)}",
            f"Department: {department}", f"Reason: {reason}",
            f"Customer email: {contact or 'Not provided'}",
            f"Model: {ticket.model_name or 'Not provided'}",
            f"Issue: {ticket.issue_type or 'Not classified'}", "",
            "Customer request:", str(data.get("human_support_request") or ""), "",
        ]
        original_safety_report = str(data.get("human_support_safety_report") or "")
        if original_safety_report and original_safety_report != str(data.get("human_support_request") or ""):
            lines.extend([
                "Original safety report:", original_safety_report, "",
            ])
        lines.append("Recent customer conversation:")
        # Only actual customer exchanges are included. Staff notes, source
        # documents and legacy workflow references are never copied.
        rows = db.query(WarrantyConversationMessage).filter(
            WarrantyConversationMessage.ticket_id == ticket_id,
            WarrantyConversationMessage.session_id == ticket.session_id,
            WarrantyConversationMessage.role.in_(("user", "assistant")),
            WarrantyConversationMessage.origin.in_(("customer_input", "user_action", "api_response")),
        ).order_by(WarrantyConversationMessage.id.desc()).limit(100).all()
        for row in reversed(rows):
            lines.append(f"{str(row.role).capitalize()}: {row.content}")
        lines.extend(["", "No callback or appointment has been booked automatically."])
        text = "\n".join(lines)
    if department == "sales" and reason != "safety_hazard":
        from sales_email_transport import send_sales_notification

        return bool(send_sales_notification(contact, text, "", domain))
    from config import WARRANTY_TEAM_EMAIL
    from sales_email_transport import _send

    subject = "[Support review] Safety concern" if reason == "safety_hazard" else "[Support review] Customer requested a specialist"
    return bool(_send(WARRANTY_TEAM_EMAIL, subject, text))


def handoff_notification_state(ticket_id: str, source_key: str) -> str:
    """Return delivery state without exposing a recipient or message body."""
    if not source_key:
        return "not_scheduled"
    from warranty_notification_jobs import WarrantyNotificationJob

    try:
        with warranty_db_session() as db:
            WarrantyNotificationJob.__table__.create(bind=db.get_bind(), checkfirst=True)
            row = db.query(WarrantyNotificationJob).filter_by(
                kind="handoff", ticket_id=ticket_id, source_key=source_key,
            ).first()
            return str(row.status) if row is not None else "not_scheduled"
    except Exception:
        # Delivery telemetry must not replace the customer's safety response.
        return "unknown"
