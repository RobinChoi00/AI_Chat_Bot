"""
warranty_email.py
=================
Send warranty chat transcripts to the warranty team inbox when a customer
leaves their email address during the /warranty guided flow.
"""

from __future__ import annotations

import logging
import re
import smtplib
import threading
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import (
    EMAIL_PASSWORD,
    EMAIL_SENDER,
    SMTP_PORT,
    SMTP_SERVER,
    WARRANTY_BUSINESS_HOURS,
    WARRANTY_EVIDENCE_NOTIFY_RECIPIENTS,
    WARRANTY_PHONE,
    WARRANTY_TEAM_EMAIL,
)

# Admin decisions that may trigger a customer email (when message + email exist).
_ADMIN_DECISION_NOTIFY = frozenset(
    {"approved", "rejected", "need_more_information", "closed"}
)

_ADMIN_DECISION_SUBJECT = {
    "approved": "Approved",
    "rejected": "Not approved",
    "need_more_information": "Additional information needed",
    "closed": "Case closed",
}

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.IGNORECASE)


def extract_email(text: str) -> Optional[str]:
    """Return the first email address found in *text*, or None."""
    match = _EMAIL_RE.search((text or "").strip())
    return match.group(0).lower() if match else None


def resolve_customer_email(
    ticket,
    turns: Optional[Sequence[Any]] = None,
    evidences: Optional[Sequence[Any]] = None,
) -> Optional[str]:
    """Best-effort customer email from collected data, turns, or evidence uploads."""
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    for key in ("customer_contact_email", "order_or_email"):
        found = extract_email(str(collected.get(key, "")))
        if found:
            return found

    if turns:
        for turn in reversed(list(turns)):
            answer = getattr(turn, "customer_answer", None)
            if answer is None and isinstance(turn, dict):
                answer = turn.get("customer_answer")
            found = extract_email(str(answer or ""))
            if found:
                return found

    if evidences:
        for ev in reversed(list(evidences)):
            raw = getattr(ev, "customer_email", None)
            if raw is None and isinstance(ev, dict):
                raw = ev.get("customer_email")
            found = extract_email(str(raw or ""))
            if found:
                return found

    return None


def _case_summary_email_lines(summary: str, source: str) -> list[str]:
    from warranty_summary import (  # noqa: WPS433
        format_case_summary_for_email,
        format_case_summary_section_header,
    )

    if not (summary or "").strip():
        return []
    return [
        "",
        format_case_summary_section_header(source),
        format_case_summary_for_email(summary, source),
        "",
    ]


def build_transcript_body(
    *,
    ticket_id: Optional[str],
    session_id: str,
    customer_email: str,
    domain: str,
    ticket_status: str = "",
    issue_type: str = "",
    model_name: str = "",
    turns: Sequence[Any],
    chat_messages: Optional[List[Dict[str, str]]] = None,
    case_summary: str = "",
    case_summary_source: str = "",
) -> str:
    """Format a plain-text transcript for the warranty team."""
    lines = [
        "New warranty chat — customer left their email address.",
        "",
        f"Customer Email : {customer_email}",
        f"Session ID     : {session_id}",
    ]
    if ticket_id:
        lines.append(f"Ticket ID      : {ticket_id}")
    if domain:
        lines.append(f"Site / Domain  : {domain}")
    if issue_type:
        lines.append(f"Issue Type     : {issue_type}")
    if model_name:
        lines.append(f"Model          : {model_name}")
    if ticket_status:
        lines.append(f"Ticket Status  : {ticket_status}")

    if case_summary:
        lines.extend(_case_summary_email_lines(case_summary, case_summary_source))

    if turns:
        lines.extend(["", "--- Workflow steps ---"])
        for turn in turns:
            node_id = getattr(turn, "node_id", "") or turn.get("node_id", "")
            prompt = getattr(turn, "node_prompt", "") or turn.get("node_prompt", "")
            answer = getattr(turn, "customer_answer", "") or turn.get("customer_answer", "")
            lines.append(f"[{node_id}]")
            if prompt:
                lines.append(f"Q: {prompt}")
            if answer:
                lines.append(f"A: {answer}")
            lines.append("")

    if chat_messages:
        lines.extend(["--- Chat messages ---"])
        for msg in chat_messages:
            role = (msg.get("role") or "unknown").capitalize()
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        lines.append("")

    lines.append("-- Sent automatically by Osaki/Titan Warranty Chat --")
    return "\n".join(lines)


def _resolve_case_summary(
    *,
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
    case_summary: str = "",
    case_summary_source: str = "",
) -> dict[str, str]:
    if case_summary.strip():
        from warranty_summary import suggested_subject_from_summary  # noqa: WPS433

        summary = case_summary.strip()
        source = (case_summary_source or "provided").strip() or "provided"
        return {
            "summary": summary,
            "suggested_subject": suggested_subject_from_summary(
                issue_type=issue_type,
                model_name=model_name,
                summary=summary,
            ),
            "source": source,
        }

    from warranty_summary import summarize_warranty_case  # noqa: WPS433

    return summarize_warranty_case(
        issue_type=issue_type,
        model_name=model_name,
        turns=turns or [],
        terminal_node_id=terminal_node_id,
    )


def send_warranty_transcript_email(
    *,
    customer_email: str,
    session_id: str,
    ticket_id: Optional[str] = None,
    domain: str = "unknown",
    ticket_status: str = "",
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    chat_messages: Optional[List[Dict[str, str]]] = None,
    case_summary: str = "",
    case_summary_source: str = "",
    terminal_node_id: str = "",
) -> bool:
    """Email the warranty team inbox with the customer's chat transcript."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.error(
            "Warranty transcript not sent — EMAIL_SENDER / EMAIL_PASSWORD not configured."
        )
        return False

    summary_payload = _resolve_case_summary(
        issue_type=issue_type,
        model_name=model_name,
        turns=turns,
        terminal_node_id=terminal_node_id,
        case_summary=case_summary,
        case_summary_source=case_summary_source,
    )
    summary_text = summary_payload["summary"]
    summary_source = summary_payload.get("source", "")
    subject_hint = summary_payload.get("suggested_subject") or ""

    subject_ref = ticket_id or session_id[:8]
    if subject_hint:
        subject = f"[Warranty Chat] {subject_hint}"
    else:
        subject = f"[Warranty Chat] Customer contact — {customer_email} ({subject_ref})"
    body = build_transcript_body(
        ticket_id=ticket_id,
        session_id=session_id,
        customer_email=customer_email,
        domain=domain,
        ticket_status=ticket_status,
        issue_type=issue_type,
        model_name=model_name,
        turns=turns or [],
        chat_messages=chat_messages,
        case_summary=summary_text,
        case_summary_source=summary_source,
    )

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = WARRANTY_TEAM_EMAIL
    msg["Reply-To"] = customer_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(
            "Warranty transcript emailed to %s for ticket=%s session=%s",
            WARRANTY_TEAM_EMAIL,
            ticket_id,
            session_id,
        )
        return True
    except smtplib.SMTPException as exc:
        logger.error("Warranty transcript email failed: %s", exc)
        return False


def maybe_send_warranty_transcript(
    *,
    ticket,
    answer_text: str,
    turns: Sequence[Any],
    chat_messages: Optional[List[Dict[str, str]]] = None,
) -> tuple[Optional[str], bool]:
    """
    If *answer_text* contains an email and we have not emailed yet for this
    ticket, send the transcript and record customer_contact_email.

    Returns (detected_email, sent_now).
    """
    email = extract_email(answer_text)
    if not email:
        return None, False

    ticket.set_collected("customer_contact_email", email)

    collected = ticket.get_collected()
    if collected.get("transcript_emailed") == "1":
        return email, False

    sent = send_warranty_transcript_email(
        customer_email=email,
        session_id=str(ticket.session_id),
        ticket_id=str(ticket.ticket_id),
        domain=str(ticket.domain or "unknown"),
        ticket_status=str(ticket.status or ""),
        issue_type=str(ticket.issue_type or ""),
        model_name=str(ticket.model_name or ""),
        turns=turns,
        chat_messages=chat_messages,
    )
    if sent:
        ticket.set_collected("transcript_emailed", "1")
        return email, True
    return email, False


def build_evidence_notification_body(
    *,
    ticket_id: str,
    customer_email: str,
    evidence_type: str,
    original_filename: str,
    file_size_bytes: int,
    issue_type: str = "",
    model_name: str = "",
    case_summary: str = "",
    case_summary_source: str = "",
) -> str:
    lines = [
        "New warranty evidence uploaded by a customer.",
        "",
        f"Ticket ID       : {ticket_id}",
        f"Customer Email  : {customer_email}",
        f"Evidence Type   : {evidence_type}",
        f"File Name       : {original_filename}",
        f"File Size       : {file_size_bytes:,} bytes",
    ]
    if issue_type:
        lines.append(f"Issue Type      : {issue_type}")
    if model_name:
        lines.append(f"Model           : {model_name}")
    if case_summary:
        lines.extend(_case_summary_email_lines(case_summary, case_summary_source))
    lines.extend(
        [
            "",
            "The uploaded file is attached to this email.",
            "",
            "-- Sent automatically by Osaki/Titan Warranty Chat --",
        ]
    )
    return "\n".join(lines)


def build_email_only_contact_body(
    *,
    ticket_id: str,
    customer_email: str,
    issue_type: str = "",
    model_name: str = "",
    ticket_status: str = "",
    session_id: str = "",
    domain: str = "",
    turns: Optional[Sequence[Any]] = None,
    case_summary: str = "",
    case_summary_source: str = "",
) -> str:
    """Plain-text body when the customer submits email without photo/video (N/A)."""
    lines = [
        "Warranty case — customer contact submitted (no photo/video).",
        "",
        "The customer selected N/A and could not or did not wish to upload photos or videos.",
        "",
        f"Ticket ID       : {ticket_id}",
        f"Customer Email  : {customer_email}",
    ]
    if session_id:
        lines.append(f"Session ID      : {session_id}")
    if domain:
        lines.append(f"Site / Domain   : {domain}")
    if issue_type:
        lines.append(f"Issue Type      : {issue_type}")
    if model_name:
        lines.append(f"Model           : {model_name}")
    if ticket_status:
        lines.append(f"Ticket Status   : {ticket_status}")

    if case_summary:
        lines.extend(_case_summary_email_lines(case_summary, case_summary_source))

    if turns:
        lines.extend(["", "--- Workflow steps ---"])
        for turn in turns:
            node_id = getattr(turn, "node_id", "") or turn.get("node_id", "")
            prompt = getattr(turn, "node_prompt", "") or turn.get("node_prompt", "")
            answer = getattr(turn, "customer_answer", "") or turn.get("customer_answer", "")
            lines.append(f"[{node_id}]")
            if prompt:
                lines.append(f"Q: {prompt}")
            if answer:
                lines.append(f"A: {answer}")
            lines.append("")

    lines.extend(
        [
            "No file attachment — follow up with the customer by email if photos or videos are needed.",
            "",
            "-- Sent automatically by Osaki/Titan Warranty Chat --",
        ]
    )
    return "\n".join(lines)


def send_evidence_upload_notification(
    *,
    ticket_id: str,
    customer_email: str,
    evidence_type: str,
    original_filename: str,
    file_path: str,
    mime_type: str,
    file_size_bytes: int,
    issue_type: str = "",
    model_name: str = "",
    recipients: Optional[List[tuple[str, str]]] = None,
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
    case_summary: str = "",
    case_summary_source: str = "",
) -> bool:
    """Email the warranty team distribution list with the uploaded file attached."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.error(
            "Evidence notification not sent — EMAIL_SENDER / EMAIL_PASSWORD not configured."
        )
        return False

    notify_list = recipients if recipients is not None else WARRANTY_EVIDENCE_NOTIFY_RECIPIENTS
    if not notify_list:
        logger.warning("Evidence notification skipped — no recipients configured.")
        return False

    path = Path(file_path)
    if not path.is_file():
        logger.error("Evidence notification failed — file not found: %s", file_path)
        return False

    summary_payload = _resolve_case_summary(
        issue_type=issue_type,
        model_name=model_name,
        turns=turns,
        terminal_node_id=terminal_node_id,
        case_summary=case_summary,
        case_summary_source=case_summary_source,
    )
    summary_text = summary_payload["summary"]
    summary_source = summary_payload.get("source", "")

    to_addrs = [email for _name, email in notify_list]
    subject = f"[Warranty Evidence] {ticket_id} — {original_filename}"
    body = build_evidence_notification_body(
        ticket_id=ticket_id,
        customer_email=customer_email,
        evidence_type=evidence_type,
        original_filename=original_filename,
        file_size_bytes=file_size_bytes,
        issue_type=issue_type,
        model_name=model_name,
        case_summary=summary_text,
        case_summary_source=summary_source,
    )

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(to_addrs)
    msg["Reply-To"] = customer_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    maintype, _, subtype = (mime_type or "application/octet-stream").partition("/")
    if not subtype:
        maintype, subtype = "application", "octet-stream"
    with path.open("rb") as handle:
        attachment = MIMEBase(maintype, subtype)
        attachment.set_payload(handle.read())
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename=original_filename)
    msg.attach(attachment)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(
            "Evidence notification emailed for ticket=%s file=%s to %d recipients",
            ticket_id,
            original_filename,
            len(to_addrs),
        )
        return True
    except smtplib.SMTPException as exc:
        logger.error("Evidence notification email failed: %s", exc)
        return False


def notify_evidence_upload_async(
    *,
    evidence_id: int,
    ticket_id: str,
    customer_email: str,
    evidence_type: str,
    original_filename: str,
    file_path: str,
    mime_type: str,
    file_size_bytes: int,
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
    case_summary: str = "",
    case_summary_source: str = "",
) -> None:
    """Send evidence notification in a background thread and mark emailed=1 on success."""

    def _worker() -> None:
        sent = send_evidence_upload_notification(
            ticket_id=ticket_id,
            customer_email=customer_email,
            evidence_type=evidence_type,
            original_filename=original_filename,
            file_path=file_path,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            issue_type=issue_type,
            model_name=model_name,
            turns=turns,
            terminal_node_id=terminal_node_id,
            case_summary=case_summary,
            case_summary_source=case_summary_source,
        )
        if not sent:
            return
        from warranty_models import WarrantyEvidence, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            row = db.query(WarrantyEvidence).filter(WarrantyEvidence.id == evidence_id).first()
            if row:
                row.emailed = 1

    threading.Thread(target=_worker, daemon=True).start()


def send_email_only_contact_notification(
    *,
    ticket_id: str,
    customer_email: str,
    session_id: str = "",
    domain: str = "",
    ticket_status: str = "",
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
    case_summary: str = "",
    case_summary_source: str = "",
    recipients: Optional[List[tuple[str, str]]] = None,
) -> bool:
    """Notify warranty team inboxes when the customer submits email only (N/A evidence)."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.error(
            "Email-only contact not sent — EMAIL_SENDER / EMAIL_PASSWORD not configured."
        )
        return False

    notify_list = recipients if recipients is not None else WARRANTY_EVIDENCE_NOTIFY_RECIPIENTS
    if not notify_list:
        logger.warning("Email-only contact skipped — no recipients configured.")
        return False

    summary_payload = _resolve_case_summary(
        issue_type=issue_type,
        model_name=model_name,
        turns=turns,
        terminal_node_id=terminal_node_id,
        case_summary=case_summary,
        case_summary_source=case_summary_source,
    )
    summary_text = summary_payload["summary"]
    summary_source = summary_payload.get("source", "")
    subject_hint = summary_payload.get("suggested_subject") or ""

    to_addrs = [email for _name, email in notify_list]
    if subject_hint:
        subject = f"[Warranty Contact] {subject_hint}"
    else:
        subject = f"[Warranty Contact] {ticket_id} — email only (no media)"
    body = build_email_only_contact_body(
        ticket_id=ticket_id,
        customer_email=customer_email,
        issue_type=issue_type,
        model_name=model_name,
        ticket_status=ticket_status,
        session_id=session_id,
        domain=domain,
        turns=turns,
        case_summary=summary_text,
        case_summary_source=summary_source,
    )

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(to_addrs)
    msg["Reply-To"] = customer_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(
            "Email-only warranty contact sent for ticket=%s to %d recipients",
            ticket_id,
            len(to_addrs),
        )
        return True
    except smtplib.SMTPException as exc:
        logger.error("Email-only contact notification failed: %s", exc)
        return False


def notify_email_only_contact_async(
    *,
    evidence_id: int,
    ticket_id: str,
    customer_email: str,
    session_id: str = "",
    domain: str = "",
    ticket_status: str = "",
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
    case_summary: str = "",
    case_summary_source: str = "",
) -> None:
    """Send email-only contact notification in a background thread."""

    def _worker() -> None:
        sent = send_email_only_contact_notification(
            ticket_id=ticket_id,
            customer_email=customer_email,
            session_id=session_id,
            domain=domain,
            ticket_status=ticket_status,
            issue_type=issue_type,
            model_name=model_name,
            turns=turns,
            terminal_node_id=terminal_node_id,
            case_summary=case_summary,
            case_summary_source=case_summary_source,
        )
        if not sent:
            return
        from warranty_models import WarrantyEvidence, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            row = db.query(WarrantyEvidence).filter(WarrantyEvidence.id == evidence_id).first()
            if row:
                row.emailed = 1

    threading.Thread(target=_worker, daemon=True).start()


def build_admin_decision_customer_body(
    *,
    ticket_id: str,
    customer_message: str,
    model_name: str = "",
    issue_type: str = "",
) -> str:
    """Plain-text email body sent to the customer after an admin decision."""
    lines = [
        "Dear Customer,",
        "",
        customer_message.strip(),
        "",
        f"Case reference: {ticket_id}",
    ]
    if model_name:
        lines.append(f"Product: {model_name}")
    if issue_type:
        lines.append(f"Issue: {issue_type}")
    lines.extend(
        [
            "",
            "If you have questions, contact our warranty team:",
            f"  Phone : {WARRANTY_PHONE}",
            f"  Email : {WARRANTY_TEAM_EMAIL}",
            f"  Hours : {WARRANTY_BUSINESS_HOURS}",
            "",
            "-- Osaki / Titan Warranty Support --",
        ]
    )
    return "\n".join(lines)


def send_admin_decision_customer_email(
    *,
    to_email: str,
    ticket_id: str,
    decision: str,
    customer_message: str,
    model_name: str = "",
    issue_type: str = "",
) -> bool:
    """Email the customer with the admin-written message only (never internal notes)."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.error(
            "Admin decision customer email not sent — EMAIL_SENDER / EMAIL_PASSWORD not configured."
        )
        return False

    label = _ADMIN_DECISION_SUBJECT.get(decision, "Update")
    subject = f"[Osaki/Titan Warranty] Case {ticket_id} — {label}"
    body = build_admin_decision_customer_body(
        ticket_id=ticket_id,
        customer_message=customer_message,
        model_name=model_name,
        issue_type=issue_type,
    )

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg["Reply-To"] = WARRANTY_TEAM_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(
            "Admin decision customer email sent — ticket=%s decision=%s to=%s",
            ticket_id,
            decision,
            to_email,
        )
        return True
    except smtplib.SMTPException as exc:
        logger.error("Admin decision customer email failed: %s", exc)
        return False


def maybe_send_admin_decision_customer_email(
    *,
    ticket,
    decision: str,
    customer_message: str,
    turns: Optional[Sequence[Any]] = None,
    evidences: Optional[Sequence[Any]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Send the admin's customer_message to the customer when appropriate.

    Returns (sent, skip_reason).  skip_reason is set when no email was sent.
    Internal admin notes are never included.
    """
    normalized_decision = decision.strip().lower()
    if normalized_decision not in _ADMIN_DECISION_NOTIFY:
        return False, "decision_not_notifiable"

    message = (customer_message or "").strip()
    if not message:
        return False, "no_customer_message"

    to_email = resolve_customer_email(ticket, turns=turns, evidences=evidences)
    if not to_email:
        return False, "no_customer_email"

    sent = send_admin_decision_customer_email(
        to_email=to_email,
        ticket_id=str(ticket.ticket_id),
        decision=normalized_decision,
        customer_message=message,
        model_name=str(ticket.model_name or ""),
        issue_type=str(ticket.issue_type or ""),
    )
    if sent:
        return True, None
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        return False, "smtp_not_configured"
    return False, "send_failed"
