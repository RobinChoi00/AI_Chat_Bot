"""
Warranty phone-line business hours (America/Chicago).

Matches ``WARRANTY_BUSINESS_HOURS`` in config.py:
  Mon–Fri, 10:00 AM – 6:00 PM CST
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

import pytz

_WARRANTY_TZ = pytz.timezone("America/Chicago")
_WARRANTY_OPEN = time(10, 0)
_WARRANTY_CLOSE = time(18, 0)


def _normalize(now: datetime) -> datetime:
    if now.tzinfo is None:
        return _WARRANTY_TZ.localize(now)
    return now.astimezone(_WARRANTY_TZ)


def is_warranty_business_hours(now: Optional[datetime] = None) -> bool:
    """Return True during Mon–Fri 10:00–17:59:59 America/Chicago."""
    moment = _normalize(now or datetime.now(_WARRANTY_TZ))
    if moment.weekday() >= 5:
        return False
    current = moment.time()
    return _WARRANTY_OPEN <= current < _WARRANTY_CLOSE


def warranty_hours_text() -> str:
    try:
        from config import WARRANTY_BUSINESS_HOURS  # noqa: WPS433

        return str(WARRANTY_BUSINESS_HOURS or "Monday through Friday, 10 AM to 6 PM Central time")
    except Exception:
        return "Monday through Friday, 10 AM to 6 PM Central time"


def sales_hours_text() -> str:
    try:
        from config import SALES_BUSINESS_HOURS  # noqa: WPS433

        return str(SALES_BUSINESS_HOURS or "").strip()
    except Exception:
        return ""


def next_warranty_open_phrase(now: Optional[datetime] = None) -> str:
    """Human phrase for when the warranty phone line next opens."""
    moment = _normalize(now or datetime.now(_WARRANTY_TZ))
    open_today = moment.replace(hour=10, minute=0, second=0, microsecond=0)

    if moment.weekday() < 5 and moment.time() < _WARRANTY_OPEN:
        return "later today at 10 AM Central time"

    probe = moment
    for _ in range(8):
        probe = probe + timedelta(days=1)
        if probe.weekday() < 5:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            day = day_names[probe.weekday()]
            if probe.date() == (moment + timedelta(days=1)).date() and moment.weekday() < 4:
                return "tomorrow at 10 AM Central time"
            return f"{day} at 10 AM Central time"

    return "the next weekday at 10 AM Central time"
