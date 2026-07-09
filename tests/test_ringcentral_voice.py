"""
tests/test_ringcentral_voice.py
===============================
Unit tests for RingCentral voice adapter (no live RC API).
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_voice import (  # noqa: E402
    REPEAT_DTMF,
    build_after_hours_closure_script,
    build_after_hours_welcome_script,
    build_business_hours_connect_script,
    build_menu_script,
    build_sales_transfer_script,
    build_terminal_script,
    menu_dtmf_patterns,
    post_diy_dtmf_patterns,
)


def test_build_menu_script_includes_dtmf_options():
    node = {
        "prompt": "What type of warranty issue?",
        "options": [
            {"label": "Installation Issue", "answer_key": "installation"},
            {"label": "Delivery Issue", "answer_key": "delivery"},
            {"label": "Defect", "answer_key": "defect"},
        ],
    }
    script = build_menu_script(node)
    assert "Press 1 for Installation Issue" in script
    assert "Press 2 for Delivery Issue" in script
    assert "Press 3 for Defect" in script
    assert f"Press {REPEAT_DTMF} to hear these options again" in script


def test_menu_dtmf_patterns_includes_repeat():
    node = {"options": [{"label": "A"}, {"label": "B"}]}
    assert menu_dtmf_patterns(node) == ["1", "2", REPEAT_DTMF]


def test_build_terminal_script_includes_post_diy_prompt():
    node = {"prompt": "Try reconnecting the air hose."}
    script = build_terminal_script(node, None)
    assert "Press 1 if that fixed the issue" in script
    assert f"Press {REPEAT_DTMF} to hear these steps again" in script
    assert "specialist" not in script.lower()


def test_build_after_hours_closure_script_mentions_business_hours():
    script = build_after_hours_closure_script()
    assert "Press 1 to end this call" in script
    assert f"Press {REPEAT_DTMF} to hear this message again" in script
    assert "call back" in script.lower()
    assert "text" in script.lower()


def test_build_after_hours_welcome_mentions_closed_and_docs():
    script = build_after_hours_welcome_script()
    assert "closed" in script.lower()
    assert "warranty" in script.lower()
    assert "invoice" in script.lower() or "order number" in script.lower()
    assert "text message" in script.lower()


def test_build_business_hours_connect_script_mentions_connecting():
    script = build_business_hours_connect_script()
    assert "connecting" in script.lower()
    assert "invoice" in script.lower() or "order number" in script.lower()


def test_build_sales_transfer_script_announces_sales():
    script = build_sales_transfer_script()
    assert "sales" in script.lower()
    assert "transfer" in script.lower()


def test_post_diy_patterns_are_repeat_or_hangup_only():
    assert post_diy_dtmf_patterns() == ["1", REPEAT_DTMF]
