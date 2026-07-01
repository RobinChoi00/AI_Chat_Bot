"""
tests/test_ringcentral_ivr.py
=============================
Unit tests for RingCentral IVR orchestration (call enter → defect menu).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_ivr import handle_call_enter  # noqa: E402
from ringcentral_voice import get_call_context  # noqa: E402
from warranty_workflow import WarrantyEngine  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_call_contexts():
    from ringcentral_voice import _call_contexts  # noqa: WPS433

    _call_contexts.clear()
    yield
    _call_contexts.clear()


def test_handle_call_enter_starts_at_defect_problem_type():
    payload = {
        "sessionId": "rc-session-1",
        "inParty": {
            "id": "party-1",
            "from": {"phoneNumber": "+18888482630"},
        },
    }
    with (
        patch("ringcentral_ivr.play_prompt") as mock_play,
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/menu.wav"),
    ):
        handle_call_enter(payload)

    ctx = get_call_context("rc-session-1")
    assert ctx is not None
    assert ctx.phase.value == "menu"

    node = WarrantyEngine.get_current_node(ctx.ticket_id)
    assert node is not None
    assert node["node_id"] == "defect_problem_type"

    ticket = WarrantyEngine.get_ticket(ctx.ticket_id)
    assert ticket is not None
    assert str(ticket.issue_type) == "defect"
    assert ticket.get_collected().get("channel") == "phone"

    mock_play.assert_called_once()
    play_uri = mock_play.call_args.kwargs.get("audio_uri") or mock_play.call_args[1].get(
        "audio_uri"
    )
    assert play_uri  # TTS URI generated
