"""
Human-readable warranty case references for customers and ops.

Format: ``WR-YYYYMMDD-XXXXXX`` (date from ticket creation, suffix from ticket UUID).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


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
