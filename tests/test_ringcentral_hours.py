"""Tests for warranty IVR business-hours gate."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytz

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_hours import is_warranty_business_hours  # noqa: E402

_CST = pytz.timezone("America/Chicago")


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return _CST.localize(datetime(year, month, day, hour, minute))


def test_business_hours_weekday_morning_closed():
    assert is_warranty_business_hours(_dt(2026, 7, 1, 9, 59)) is False


def test_business_hours_weekday_open():
    assert is_warranty_business_hours(_dt(2026, 7, 1, 10, 0)) is True
    assert is_warranty_business_hours(_dt(2026, 7, 1, 15, 30)) is True


def test_business_hours_weekday_at_close():
    assert is_warranty_business_hours(_dt(2026, 7, 1, 18, 0)) is False


def test_business_hours_weekend_closed():
    assert is_warranty_business_hours(_dt(2026, 7, 4, 12, 0)) is False  # Saturday


def test_next_warranty_open_phrase_friday_evening():
    from ringcentral_hours import next_warranty_open_phrase  # noqa: WPS433

    phrase = next_warranty_open_phrase(_dt(2026, 7, 3, 19, 0))  # Friday 7 PM
    assert "monday" in phrase.lower()


def test_next_warranty_open_phrase_monday_morning_before_open():
    from ringcentral_hours import next_warranty_open_phrase  # noqa: WPS433

    phrase = next_warranty_open_phrase(_dt(2026, 7, 6, 8, 0))  # Monday 8 AM
    assert "today" in phrase.lower()
