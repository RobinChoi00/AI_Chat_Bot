"""Tests for Tidio button flattening + label/number resolution."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_tidio_buttons import (  # noqa: E402
    append_numbered_menu,
    flatten_buttons_for_flow,
    prioritize_quick_replies,
    resolve_button_choice,
)


def test_prioritize_caps_and_prefers_shop():
    raw = [
        {"label": "Talk to a human", "payload": "human"},
        {"label": "Specs", "payload": "specs:abc"},
        {"label": "Shop this chair", "payload": "open:https://osakiusa.com/products/x"},
        {"label": "Email me this pick", "payload": "lead:save_pick"},
        {"label": "Prefer stronger", "payload": "recommend:intensity:strong"},
        {"label": "Extra", "payload": "menu"},
    ]
    out = prioritize_quick_replies(raw, limit=5)
    assert len(out) == 5
    assert out[0]["payload"].startswith("open:")
    assert any(b["payload"] == "lead:save_pick" for b in out)
    assert out[-1]["payload"] != "menu" or len(out) < 6


def test_flatten_and_numbered_menu():
    buttons = prioritize_quick_replies(
        [
            {"label": "Shop this chair", "payload": "open:https://osakiusa.com/products/x"},
            {"label": "Email me this pick", "payload": "lead:save_pick"},
        ]
    )
    flat = flatten_buttons_for_flow(buttons)
    assert flat["button_count"] == 2
    assert flat["button_1_label"] == "Shop this chair"
    assert flat["button_1_url"].startswith("https://")
    assert flat["button_3_label"] == ""

    plain = append_numbered_menu("Here is your pick.", buttons)
    assert "reply with the number:" in plain.lower()
    assert "1) Shop this chair" in plain
    assert "2) Email me this pick" in plain


def test_resolve_number_and_label():
    buttons = [
        {"label": "Shop this chair", "payload": "open:https://osakiusa.com/products/x"},
        {"label": "Email me this pick", "payload": "lead:save_pick"},
        {"label": "Talk to a human", "payload": "human"},
    ]
    assert resolve_button_choice("1", buttons) == buttons[0]["payload"]
    assert resolve_button_choice("1)", buttons) == buttons[0]["payload"]
    assert resolve_button_choice("1.", buttons) == buttons[0]["payload"]
    assert resolve_button_choice("2", buttons) == "lead:save_pick"
    assert resolve_button_choice("Email me this pick", buttons) == "lead:save_pick"
    assert resolve_button_choice("3) Talk to a human", buttons) == "human"
    assert resolve_button_choice("something else", buttons) is None


def test_resolve_falls_back_to_default_menu_when_session_lost():
    assert resolve_button_choice("1", None) == "recommend"
    assert resolve_button_choice("1.", []) == "recommend"
    assert resolve_button_choice("2", None) == "stock"
    assert resolve_button_choice("recommend", None) == "recommend"
    assert resolve_button_choice("Recommend a chair", None) == "recommend"
