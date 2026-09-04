"""
Human-readable warranty case references for customers and ops.

Format: ``WR-YYYYMMDD-XXXXXX`` (date from ticket creation, suffix from ticket UUID).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

_CASE_REF_RE = re.compile(r"^WR-(\d{8})-([A-F0-9]{4,12})$", re.IGNORECASE)
_BARE_REF_RE = re.compile(r"^(\d{8})-([A-F0-9]{4,12})$", re.IGNORECASE)


def format_case_reference(
    ticket_id: str,
    created_at: Optional[datetime] = None,
) -> str:
    """Build a stable, shareable case reference from ticket metadata."""
    tid = (ticket_id or "").replace("-", "").upper()
    suffix = tid[:6] if len(tid) >= 6 else tid or "000000"
    if created_at is not None:
        date_part = created_at.strftime("%Y%m%d")
    else:
        date_part = datetime.now().strftime("%Y%m%d")
    return f"WR-{date_part}-{suffix}"


def case_reference_for_ticket(ticket) -> str:
    """Return case reference for a WarrantyTicket ORM object or dict-like row."""
    ticket_id = str(getattr(ticket, "ticket_id", "") or "")
    created = getattr(ticket, "created_at", None)
    return format_case_reference(ticket_id, created_at=created)


def normalize_case_reference(value: str) -> str:
    """Uppercase and strip a customer-typed case reference."""
    raw = (value or "").strip().upper()
    raw = raw.replace(" ", "").replace("CASE#", "").replace("CASE:", "")
    raw = raw.replace("REF:", "").strip("-")
    if raw.startswith("WR"):
        rest = raw[2:].lstrip("-")
        match = _BARE_REF_RE.match(rest)
        if match:
            return f"WR-{match.group(1)}-{match.group(2).upper()}"
        return raw if raw.startswith("WR-") else f"WR-{rest}"
    match = _BARE_REF_RE.match(raw)
    if match:
        return f"WR-{match.group(1)}-{match.group(2).upper()}"
    return raw


def parse_case_reference(value: str) -> Optional[tuple[str, str]]:
    """Return ``(YYYYMMDD, suffix)`` when *value* is a valid case reference."""
    match = _CASE_REF_RE.match(normalize_case_reference(value))
    if not match:
        return None
    return match.group(1), match.group(2).upper()


def persist_case_reference(ticket, case_reference: str = "") -> str:
    """Store the shareable case reference on the ticket and return it."""
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    stored = str(collected.get("case_reference") or "").strip()
    case_ref = (
        (case_reference or "").strip()
        or stored
        or case_reference_for_ticket(ticket)
    )
    if case_ref and hasattr(ticket, "set_collected") and case_ref != stored:
        ticket.set_collected("case_reference", case_ref)
    return case_ref
