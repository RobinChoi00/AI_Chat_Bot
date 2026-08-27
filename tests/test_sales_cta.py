"""Tests for Sales AI CTAs and fit-guide lead cards."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_cta import (  # noqa: E402
    after_hours_blurb,
    extract_email,
    financing_page_url,
    format_defaults_note,
    format_fit_guide_summary,
    is_sales_after_hours,
    is_strong_buy_path,
    product_page_url,
    showroom_blurb,
)


def test_product_page_url_by_store():
    assert (
        product_page_url("osakiusa.com", "titan-ecabin-3d-massage-chair")
        == "https://osakiusa.com/products/titan-ecabin-3d-massage-chair"
    )
    assert (
        product_page_url("titanchair.com", "titan-ecabin-3d")
        == "https://titanchair.com/products/titan-ecabin-3d"
    )
    assert product_page_url("osakiusa.com", "") is None


def test_extract_email():
    assert extract_email("please use me@example.com thanks") == "me@example.com"
    assert extract_email("no email here") is None


def test_fit_guide_summary_card():
    card = format_fit_guide_summary(
        domain="osakiusa.com",
        prefs={
            "budget": "Under $3,000",
            "height": 'Petite (<5\'4")',
            "weight": "≤180 lb",
            "goal": "Neck & Shoulders",
        },
        primary="Titan eCabin 3D",
        alternatives=["Titan Ignite Sync 3D+2D"],
        product_url="https://osakiusa.com/products/titan-ecabin-3d-massage-chair",
        stock_label="in stock",
    )
    assert "Primary: Titan eCabin 3D (in stock)" in card
    assert "Store: osakiusa.com" in card
    assert "Product URL:" in card
    assert "discounts" in card.lower() or "ETA" in card or "AI must not" in card


def test_after_hours_weekend():
    saturday = datetime(2026, 8, 22, 12, 0, tzinfo=ZoneInfo("America/Chicago"))
    assert is_sales_after_hours(saturday) is True
    assert "After hours" in after_hours_blurb()


def test_business_hours_weekday_afternoon():
    weekday = datetime(2026, 8, 24, 14, 0, tzinfo=ZoneInfo("America/Chicago"))
    assert is_sales_after_hours(weekday) is False


def test_financing_falls_back_to_product_url():
    url = "https://osakiusa.com/products/titan-ecabin-3d-massage-chair"
    assert financing_page_url("osakiusa.com", product_url=url) == url


def test_strong_buy_path_in_stock_only():
    url = "https://osakiusa.com/products/x"
    assert is_strong_buy_path(product_url=url, stock_label="in stock") is True
    assert is_strong_buy_path(product_url=url, stock_label="low stock (2)") is True
    assert is_strong_buy_path(product_url=url, stock_label="out of stock") is False
    assert is_strong_buy_path(product_url=None, stock_label="in stock") is False


def test_defaults_note_and_showroom():
    note = format_defaults_note(
        ["intensity", "foot"],
        {"intensity": "Balanced", "foot": "Not Important"},
    )
    assert note is not None
    assert "balanced" in note.lower()
    assert "Carrollton" in showroom_blurb()
