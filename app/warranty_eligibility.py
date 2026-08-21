"""
Warranty eligibility helpers from a known purchase date.

This is a soft operational signal — it does not hard-block the flowchart.

Real coverage is plan-based (not a single flat window):
  - Standard: 1 year labor + parts, then parts-only (typically through year 2)
  - Extended: 3 years labor + parts, then additional parts-only years
  - Adjusted: 2 years labor + parts, then 2 years parts-only (rarely used)
  - Brand extended (Mattress Firm / Johnson Fitness): 5 years labor + parts
  - Third-party / unauthorized dealer / private sale: no service, no parts sales

The numeric ``years`` heuristic (default ``WARRANTY_ELIGIBILITY_YEARS``, usually 3)
is only a rough review cue until plan + purchase channel are confirmed in NetSuite
or by the warranty team.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional


# Shown to admin / Freshdesk — keep in sync with ops policy.
WARRANTY_PLAN_REFERENCE = (
    "Confirm plan in NetSuite (or with the customer): "
    "Standard = 1 yr labor+parts then parts-only; "
    "Extended = 3 yr labor+parts then parts-only; "
    "Adjusted = 2 yr labor+parts + 2 yr parts (rare); "
    "Brand extended (Mattress Firm / Johnson Fitness) = 5 yr labor+parts. "
    "Unauthorized / third-party / private-sale purchases: no service and no parts sales."
)


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
            summary=(
                "Purchase date unknown — eligibility not evaluated. "
                + WARRANTY_PLAN_REFERENCE
            ),
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
            f"Purchase date {pretty}. Within the rough {window}-year review cue "
            f"(horizon {expires.strftime('%B %d, %Y')}; {days_remaining} days left). "
            "This is NOT a coverage decision — labor vs parts depends on the plan "
            "(Standard / Extended / Adjusted / Brand extended). "
            + WARRANTY_PLAN_REFERENCE
        )
    else:
        status = "possibly_expired"
        summary = (
            f"Purchase date {pretty}. Outside the rough {window}-year review cue "
            f"(horizon was {expires.strftime('%B %d, %Y')}). "
            "Still confirm plan — Extended / Brand extended may still cover labor+parts; "
            "Standard may already be parts-only or ended. "
            + WARRANTY_PLAN_REFERENCE
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
            "Purchase date unknown — confirm plan and purchase channel in NetSuite. "
            "Soft signal only; does not block the case. "
            + WARRANTY_PLAN_REFERENCE
        )
    return result.summary or ""


def customer_eligibility_note(result: EligibilityResult) -> str:
    """Customer-facing soft note — never claims a hard deny or a specific plan."""
    if result.status == "in_warranty":
        return (
            f"\n\n_Purchase date on file: **{result.purchase_date}**. "
            "Coverage still depends on your warranty plan "
            "(Standard, Extended, or brand program) — "
            "labor and parts years can differ. Our team will confirm._"
        )
    if result.status == "possibly_expired":
        return (
            f"\n\n_Purchase date on file: **{result.purchase_date}**. "
            "This may be outside a standard review window, but coverage depends on "
            "your plan (some plans run longer for labor and/or parts). "
            "Our team can still review your case._"
        )
    return ""
