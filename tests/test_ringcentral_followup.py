"""
tests/test_ringcentral_followup.py
==================================
Unit tests for post-call SMS follow-up.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_followup import (  # noqa: E402
    DEFAULT_FOLLOWUP_SMS,
    build_followup_sms_message,
    send_call_followup_sms,
)
from ringcentral_ivr import handle_call_enter, handle_call_exit  # noqa: E402
from ringcentral_voice import get_call_context  # noqa: E402


def test_build_followup_sms_message_default():
    with patch.dict("os.environ", {"RC_SMS_FOLLOWUP_MESSAGE": ""}, clear=False):
        msg = build_followup_sms_message()
    assert msg == DEFAULT_FOLLOWUP_SMS
    assert "service@osakititan.com" in msg
    assert "24 hours" in msg


def test_build_followup_sms_message_custom_override():
    custom = "Custom follow-up text."
    with patch.dict("os.environ", {"RC_SMS_FOLLOWUP_MESSAGE": custom}, clear=False):
        assert build_followup_sms_message() == custom


def test_send_call_followup_sms_disabled_without_from_number():
    with patch.dict("os.environ", {"RC_SMS_FROM_NUMBER": ""}, clear=False):
        assert send_call_followup_sms("+15551234567", ticket_id="t-1") is False


def test_send_call_followup_sms_skips_invalid_phone():
    with (
        patch.dict("os.environ", {"RC_SMS_FROM_NUMBER": "+12149602952"}, clear=False),
        patch("ringcentral_followup.send_sms") as mock_send,
    ):
        assert send_call_followup_sms("5551234567", ticket_id="t-2") is False
    mock_send.assert_not_called()


def test_send_call_followup_sms_sends_when_configured():
    with (
        patch.dict("os.environ", {"RC_SMS_FROM_NUMBER": "+12149602952"}, clear=False),
        patch("ringcentral_followup.send_sms") as mock_send,
    ):
        ok = send_call_followup_sms("+15551234567", ticket_id="t-3")
    assert ok is True
    mock_send.assert_called_once()
    assert "service@osakititan.com" in mock_send.call_args.kwargs["text"]


@pytest.fixture(autouse=True)
def _clear_call_contexts():
    from ringcentral_voice import _call_contexts  # noqa: WPS433

    _call_contexts.clear()
    yield
    _call_contexts.clear()


def test_handle_call_exit_sends_followup_sms():
    payload_enter = {
        "sessionId": "rc-session-exit",
        "inParty": {
            "id": "party-exit",
            "from": {"phoneNumber": "+15559876543"},
        },
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt"),
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/a.wav"),
    ):
        handle_call_enter(payload_enter)

    assert get_call_context("rc-session-exit") is not None

    with patch("ringcentral_ivr.send_call_followup_sms") as mock_sms:
        handle_call_exit({"sessionId": "rc-session-exit"})

    mock_sms.assert_called_once()
    assert mock_sms.call_args.args[0] == "+15559876543"
    assert mock_sms.call_args.kwargs["ticket_id"]
    assert get_call_context("rc-session-exit") is None
