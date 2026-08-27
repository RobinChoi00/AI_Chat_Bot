"""
sales_cta.py
============
Post-recommendation CTAs and sales-rep lead cards.

Keeps URLs, after-hours copy, and fit-guide summaries out of the big
orchestrator so recommend replies stay readable.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from store_config import get_store_key_prefix, get_storefront_base_url

_EMAIL_RE = re.compile(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)

# America/Chicago by default — Osaki HQ / most USA sales coverage.
_TZ_NAME = os.getenv("SALES_BUSINESS_TZ", "America/Chicago")
# Inclusive start hour, exclusive end hour (local). Default Mon–Fri 09:00–18:00.
_OPEN_HOUR = int(os.getenv("SALES_BUSINESS_OPEN_HOUR", "9"))
_CLOSE_HOUR = int(os.getenv("SALES_BUSINESS_CLOSE_HOUR", "18"))

_SHOWROOM_ADDRESS = "1001 W Crosby Rd, Carrollton, TX 75006"


def product_page_url(domain: str, handle: str) -> Optional[str]:
    handle = (handle or "").strip().strip("/")
    if not handle:
        return None
    base = get_storefront_base_url(domain).rstrip("/")
    return f"{base}/products/{handle}"


def financing_page_url(domain: str, *, product_url: Optional[str] = None) -> Optional[str]:
    """
    Affirm / pay-over-time lives at checkout on our storefronts.

    Prefer an explicit financing page via env; otherwise send shoppers to the
    product page so they can open Affirm at checkout (no invented APR/terms).
    """
    prefix = get_store_key_prefix(domain)
    for key in (f"{prefix}_FINANCING_URL", "SALES_FINANCING_URL"):
        override = (os.getenv(key) or "").strip()
        if override:
            return override
    return product_url


def showroom_address() -> str:
    try:
        from config import COMPANY_ADDRESS  # type: ignore

        return (COMPANY_ADDRESS or _SHOWROOM_ADDRESS).strip() or _SHOWROOM_ADDRESS
    except Exception:
        return _SHOWROOM_ADDRESS


def showroom_blurb() -> str:
    addr = showroom_address()
    return (
        f"You're welcome to visit our **showroom** in Carrollton, TX:\n"
        f"{addr}\n\n"
        "Please call ahead so we can confirm availability. "
        "On-site specialists can walk you through fit, financing at checkout, "
        "and current promotions (I won't invent discount amounts here)."
    )


def extract_email(text: str) -> Optional[str]:
    match = _EMAIL_RE.search(text or "")
    return match.group(1).strip() if match else None


def is_sales_after_hours(now: Optional[datetime] = None) -> bool:
    """True when a human sales follow-up is unlikely to be immediate."""
    try:
        tz = ZoneInfo(_TZ_NAME)
    except Exception:
        tz = ZoneInfo("America/Chicago")
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    else:
        current = current.astimezone(tz)
    if current.weekday() >= 5:  # Sat/Sun
        return True
    return not (_OPEN_HOUR <= current.hour < _CLOSE_HOUR)


def after_hours_blurb() -> str:
    return (
        "After hours — I can recommend from our fit guide now. "
        "A specialist follows up next business day on **pricing, delivery dates, "
        "and discounts** (I won't invent those)."
    )


def format_defaults_note(applied: list[str], prefs: dict[str, str]) -> Optional[str]:
    """Short note when we skipped intensity/foot questions."""
    if not applied:
        return None
    bits = []
    if "intensity" in applied and prefs.get("intensity"):
        bits.append(prefs["intensity"].lower())
    if "foot" in applied and prefs.get("foot"):
        bits.append(f"foot {prefs['foot'].lower()}")
    if not bits:
        return None
    return f"_Assumed {', '.join(bits)} — say if that should change._"


def format_fit_guide_summary(
    *,
    domain: str,
    prefs: dict[str, str],
    primary: str,
    alternatives: list[str],
    product_url: Optional[str] = None,
    stock_label: Optional[str] = None,
) -> str:
    """One card a sales rep can act on without re-reading the chat."""
    bits = [
        prefs.get("budget"),
        prefs.get("height"),
        prefs.get("weight"),
        prefs.get("goal"),
        prefs.get("intensity"),
        prefs.get("foot"),
        prefs.get("space"),
    ]
    fit = " / ".join(b for b in bits if b) or "fit details n/a"
    alts = ", ".join(a for a in alternatives if a) or "—"
    stock = f" ({stock_label})" if stock_label else ""
    lines = [
        f"Store: {domain}",
        f"Fit: {fit}",
        f"Primary: {primary}{stock}",
        f"Alternatives: {alts}",
    ]
    if product_url:
        lines.append(f"Product URL: {product_url}")
    lines.append("Note: AI must not quote unofficial discounts or hard ETAs.")
    return "\n".join(lines)


def is_strong_buy_path(*, product_url: Optional[str], stock_label: Optional[str]) -> bool:
    """In-stock (or low stock) + product URL → push shop / financing / showroom."""
    if not product_url:
        return False
    label = (stock_label or "").lower()
    if not label or "unchecked" in label:
        # Still offer shop link; treat as soft conversion.
        return True
    if "out of stock" in label:
        return False
    return "in stock" in label or "low stock" in label
