"""Tests for product_catalog model name resolution."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import product_catalog as pc


def test_resolve_known_model_substring():
    pc.load_catalog_titles.cache_clear()
    if not pc._CSV_PATH.is_file():
        return
    resolved = pc.resolve_model_name("solo flex black")
    assert resolved is not None
    assert "solo flex" in resolved.lower()


def test_resolve_empty_returns_none():
    assert pc.resolve_model_name("") is None
    assert pc.resolve_model_name("x") is None


def test_resolve_catalog_price_known_model():
    pc.load_catalog_titles.cache_clear()
    pc.load_catalog_base_prices.cache_clear()
    if not pc._CSV_PATH.is_file():
        return
    price = pc.resolve_catalog_price("solo flex")
    if price is not None:
        assert 500 <= price <= 50000


def test_format_model_display_name_strips_kakaotalk_bundle():
    raw = "Osaki OS-Pro Maestro (Kakaotalk) with FREE Eye Massager"
    assert pc.format_model_display_name(raw) == "Osaki OS-Pro Maestro"


def test_format_model_display_name_strips_warranty_promo():
    raw = "Osaki Trion Flex Duo 4D+3D (Free 2Yrs Ext Warranty 🎁)"
    assert pc.format_model_display_name(raw) == "Osaki Trion Flex Duo 4D+3D"


def test_resolve_maestro_prefers_clean_title_over_kakaotalk_bundle():
    pc.load_catalog_titles.cache_clear()
    if not pc._CSV_PATH.is_file():
        return
    resolved = pc.resolve_model_name("Maestro")
    assert resolved == "Osaki OS-Pro Maestro 4D"
    assert "kakaotalk" not in resolved.lower()
    assert "free eye" not in resolved.lower()


def test_resolve_maestro_le_prefers_clean_le_title():
    pc.load_catalog_titles.cache_clear()
    if not pc._CSV_PATH.is_file():
        return
    resolved = pc.resolve_model_name("Maestro LE")
    assert resolved == "Osaki OS-Pro Maestro LE"
    assert "kakaotalk" not in resolved.lower()

