"""
Warranty eligibility helpers from a known purchase date.

This is a soft operational signal — it does not hard-block the flowchart.
Default window is configurable via WARRANTY_ELIGIBILITY_YEARS (default 3).
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional


@dataclass
class EligibilityResult:
    status: str  # in_warranty | possibly_expired | unknown
    purchase_date: str = ""
    purchase_date_iso: str = ""
    eligibility_years: int = 3
    expires_on: str = ""
    days_remaining: Optional[int] = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def eligibility_years() -> int:
    raw = os.getenv("WARRANTY_ELIGIBILITY_YEARS", "3").strip()
    try:
        years = int(raw)
    except ValueError:
        return 3
    return max(1, min(years, 15))


def parse_purchase_date(raw: str) -> Optional[date]:
    text = (raw or "").strip()
    if not text:
        return None

    # ISO / Shopify style
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.date()
    except ValueError:
        pass

    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    # "March 15, 2025"
    m = re.match(
        r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$",
        text,
    )
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(2)))
            except ValueError:
                return None
    return None


def evaluate_purchase_eligibility(
    purchase_date_raw: str,
    *,
    as_of: Optional[date] = None,
    years: Optional[int] = None,
) -> EligibilityResult:
    window = years if years is not None else eligibility_years()
    parsed = parse_purchase_date(purchase_date_raw)
    if parsed is None:
        return EligibilityResult(
            status="unknown",
            purchase_date=(purchase_date_raw or "").strip(),
            eligibility_years=window,
            summary="Purchase date unknown — eligibility not evaluated.",
        )

    today = as_of or datetime.now(timezone.utc).date()
    try:
        expires = parsed.replace(year=parsed.year + window)
    except ValueError:
        # Feb 29 → Feb 28 in non-leap target years
        expires = parsed.replace(year=parsed.year + window, day=28)

    days_remaining = (expires - today).days
    purchase_iso = parsed.isoformat()
    expires_iso = expires.isoformat()
    pretty = parsed.strftime("%B %d, %Y")

    if days_remaining >= 0:
        status = "in_warranty"
        summary = (
            f"Purchase date {pretty}. Within the default {window}-year window "
            f"(expires {expires.strftime('%B %d, %Y')}; {days_remaining} days left). "
            "Confirm exact coverage with the warranty team for parts/labor."
        )
    else:
        status = "possibly_expired"
        summary = (
            f"Purchase date {pretty}. Outside the default {window}-year window "
            f"(expired {expires.strftime('%B %d, %Y')}). "
            "Escalate for coverage review — some parts may still be covered."
        )

    return EligibilityResult(
        status=status,
        purchase_date=pretty,
        purchase_date_iso=purchase_iso,
        eligibility_years=window,
        expires_on=expires_iso,
        days_remaining=days_remaining,
        summary=summary,
    )


def admin_eligibility_note(result: EligibilityResult) -> str:
    """Internal copy — always set so admin can see unknown as well as dated cases."""
    if result.status == "unknown":
        return (
            "Purchase date unknown — confirm coverage with the warranty team. "
            "This does not block the case."
        )
    return result.summary or ""


def customer_eligibility_note(result: EligibilityResult) -> str:
    if result.status == "in_warranty":
        return (
            f"\n\n_Purchase date on file: **{result.purchase_date}**. "
            f"This looks within our default {result.eligibility_years}-year review window._"
        )
    if result.status == "possibly_expired":
        return (
            f"\n\n_Purchase date on file: **{result.purchase_date}**. "
            "This may be outside the standard warranty window — "
            "our team can still review coverage for your case._"
        )
    return ""
