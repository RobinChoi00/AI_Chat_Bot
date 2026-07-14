"""
Warranty phone-line business hours (America/Chicago).

Matches ``WARRANTY_BUSINESS_HOURS`` in config.py:
  Mon–Fri, 10:00 AM – 6:00 PM CST
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import os
from typing import Optional

import pytz

_WARRANTY_TZ = pytz.timezone("America/Chicago")
_WARRANTY_OPEN = time(10, 0)
_WARRANTY_CLOSE = time(18, 0)


def _parse_time(name: str, fallback: time) -> time:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        hour, minute = (int(part) for part in raw.split(":", 1))
        return time(hour, minute)
    except (TypeError, ValueError):
        return fallback


def _open_days() -> set[int]:
    raw = os.getenv("RC_WARRANTY_OPEN_WEEKDAYS", "0,1,2,3,4")
    days: set[int] = set()
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if 0 <= value <= 6:
            days.add(value)
    return days or {0, 1, 2, 3, 4}


def _closed_dates() -> set[date]:
    dates: set[date] = set()
    for token in os.getenv("RC_WARRANTY_CLOSED_DATES", "").split(","):
        try:
            if token.strip():
                dates.add(date.fromisoformat(token.strip()))
        except ValueError:
            continue
    return dates


def _format_time(value: time) -> str:
    hour = value.hour % 12 or 12
    suffix = "AM" if value.hour < 12 else "PM"
    return f"{hour}:{value.minute:02d} {suffix}" if value.minute else f"{hour} {suffix}"


def _normalize(now: datetime) -> datetime:
    if now.tzinfo is None:
        return _WARRANTY_TZ.localize(now)
    return now.astimezone(_WARRANTY_TZ)


def is_warranty_business_hours(now: Optional[datetime] = None) -> bool:
    """Return True during Mon–Fri 10:00–17:59:59 America/Chicago."""
    moment = _normalize(now or datetime.now(_WARRANTY_TZ))
    open_time = _parse_time("RC_WARRANTY_OPEN_TIME", _WARRANTY_OPEN)
    close_time = _parse_time("RC_WARRANTY_CLOSE_TIME", _WARRANTY_CLOSE)
    if moment.weekday() not in _open_days() or moment.date() in _closed_dates():
        return False
    current = moment.time()
    return open_time <= current < close_time


def warranty_hours_text() -> str:
    if os.getenv("RC_WARRANTY_OPEN_TIME") or os.getenv("RC_WARRANTY_CLOSE_TIME"):
        open_time = _parse_time("RC_WARRANTY_OPEN_TIME", _WARRANTY_OPEN)
        close_time = _parse_time("RC_WARRANTY_CLOSE_TIME", _WARRANTY_CLOSE)
        return f"weekdays, {_format_time(open_time)} to {_format_time(close_time)} Central time"
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
    open_time = _parse_time("RC_WARRANTY_OPEN_TIME", _WARRANTY_OPEN)
    open_label = _format_time(open_time)
    open_days = _open_days()
    closed_dates = _closed_dates()

    if (
        moment.weekday() in open_days
        and moment.date() not in closed_dates
        and moment.time() < open_time
    ):
        return f"later today at {open_label} Central time"

    probe = moment
    for _ in range(8):
        probe = probe + timedelta(days=1)
        if probe.weekday() in open_days and probe.date() not in closed_dates:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day = day_names[probe.weekday()]
            if probe.date() == (moment + timedelta(days=1)).date():
                return f"tomorrow at {open_label} Central time"
            return f"{day} at {open_label} Central time"

    return f"the next open day at {open_label} Central time"
