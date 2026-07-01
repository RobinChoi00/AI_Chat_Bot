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
    AGENT_DTMF,
    build_menu_script,
    build_terminal_script,
    menu_dtmf_patterns,
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
    assert f"Press {AGENT_DTMF} to speak with a warranty specialist" in script


def test_menu_dtmf_patterns_includes_agent_escape():
    node = {"options": [{"label": "A"}, {"label": "B"}]}
    assert menu_dtmf_patterns(node) == ["1", "2", AGENT_DTMF]


def test_build_terminal_script_includes_post_diy_prompt():
    node = {"prompt": "Try reconnecting the air hose."}
    script = build_terminal_script(node, None)
    assert "Press 1 if that fixed the issue" in script
    assert "Press 2 to speak with a warranty specialist" in script
