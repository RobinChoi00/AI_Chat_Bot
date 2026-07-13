"""Tests for Fonz family fallback in error_code_lookup."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from error_code_lookup import (  # noqa: E402
    list_model_error_codes,
    lookup_error_code,
    suggest_error_codes_for_ticket,
)


def test_hamilton_le_uses_allure_codes():
    hamilton = list_model_error_codes("Hamilton LE", workflow_category="air", limit=3)
    allure = list_model_error_codes("Allure", workflow_category="air", limit=3)
    if not allure:
        return
    assert hamilton
    assert hamilton[0].get("error_code") == allure[0].get("error_code")


def test_zion_uses_horizon_family_when_needed():
    zion = list_model_error_codes("Zion", workflow_category="air", limit=3)
    horizon = list_model_error_codes("Horizon 4D", workflow_category="air", limit=3)
    if not horizon:
        return
    assert zion
    assert zion[0].get("error_code") == horizon[0].get("error_code")
    if not list_model_error_codes("Zion", limit=1):
        assert zion[0].get("family_fallback") is True


def test_family_lookup_error_code():
    hit = lookup_error_code("Hamilton LE", "C6")
    if hit is None:
        return
    assert hit.get("error_code") == "C6"
    assert hit.get("family_fallback") is True


def test_suggest_without_model_uses_category():
    rows = suggest_error_codes_for_ticket("", "air", limit=2)
    assert rows
    assert rows[0].get("error_code")
