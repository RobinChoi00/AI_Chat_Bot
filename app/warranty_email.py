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
    WARRANTY_EVIDENCE_NOTIFY_RECIPIENTS,
    WARRANTY_TEAM_EMAIL,
)

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.IGNORECASE)


def extract_email(text: str) -> Optional[str]:
    """Return the first email address found in *text*, or None."""
    match = _EMAIL_RE.search((text or "").strip())
    return match.group(0).lower() if match else None


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
) -> bool:
    """Email the warranty team inbox with the customer's chat transcript."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.error(
            "Warranty transcript not sent — EMAIL_SENDER / EMAIL_PASSWORD not configured."
        )
        return False

    subject_ref = ticket_id or session_id[:8]
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
        ticket.set_collected("customer_contact_email", email)
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
    lines.extend(
        [
            "",
            "The uploaded file is attached to this email.",
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
        )
        if not sent:
            return
        from warranty_models import WarrantyEvidence, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            row = db.query(WarrantyEvidence).filter(WarrantyEvidence.id == evidence_id).first()
            if row:
                row.emailed = 1

    threading.Thread(target=_worker, daemon=True).start()
