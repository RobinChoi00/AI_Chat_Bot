"""
tests/test_ringcentral_followup.py
==================================
Unit tests for post-call SMS follow-up.
"""

import sys
import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_followup import (  # noqa: E402
    build_followup_sms_message,
    send_call_followup_sms,
    send_call_followup_team_email,
    send_phone_call_followups,
    try_claim_phone_followup,
)
from ringcentral_ivr import handle_call_enter, handle_call_exit  # noqa: E402
from ringcentral_voice import get_call_context  # noqa: E402


def _create_followup_ticket() -> str:
    from warranty_models import WarrantyTicket, warranty_db_session

    ticket_id = f"followup-{uuid.uuid4().hex}"
    with warranty_db_session() as db:
        db.add(
            WarrantyTicket(
                ticket_id=ticket_id,
                session_id=f"session-{uuid.uuid4().hex}",
                domain="phone",
                current_node_id="test",
                status="in_progress",
                collected_data=json.dumps({"channel": "phone"}),
            )
        )
    return ticket_id


def test_build_followup_sms_message_default():
    with patch.dict("os.environ", {"RC_SMS_FOLLOWUP_MESSAGE": ""}, clear=False):
        msg = build_followup_sms_message()
    assert "service@osakititan.com" in msg
    assert "24 hours" in msg


def test_build_followup_sms_message_includes_case_ref_and_resume():
    msg = build_followup_sms_message(
        case_ref="WR-20260701-ABC123",
        resume_url="https://titanchair.com/warranty?resume=abc",
    )
    assert "WR-20260701-ABC123" in msg
    assert "https://titanchair.com/warranty?resume=abc" in msg


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
        patch("ringcentral_followup._followup_context", return_value=("WR-20260701-T3", "")),
        patch("ringcentral_followup.send_sms") as mock_send,
    ):
        ok = send_call_followup_sms(
            "+15551234567",
            ticket_id="t-3",
            session_id="sess-3",
        )
    assert ok is True
    mock_send.assert_called_once()
    assert "WR-20260701-T3" in mock_send.call_args.kwargs["text"]
    assert "service@osakititan.com" in mock_send.call_args.kwargs["text"]


def test_send_call_followup_team_email_disabled_without_sender():
    with patch.dict("os.environ", {"RC_IVR_TEAM_EMAIL_ENABLED": "true"}, clear=False):
        with patch("ringcentral_followup._team_email_enabled", return_value=False):
            assert (
                send_call_followup_team_email(
                    caller_phone="+15551234567",
                    ticket_id="t-4",
                    session_id="sess-4",
                )
                is False
            )


def test_send_call_followup_team_email_sends_when_ticket_exists():
    fake_ticket = type(
        "Ticket",
        (),
        {
            "status": "in_progress",
            "issue_type": "defect",
            "model_name": "OS-4000T",
            "current_node_id": "defect_problem_type",
            "ticket_id": "t-5",
            "created_at": None,
        },
    )()

    with (
        patch("ringcentral_followup._team_email_enabled", return_value=True),
        patch("warranty_workflow.WarrantyEngine.get_ticket", return_value=fake_ticket),
        patch("warranty_workflow.WarrantyEngine.get_turns", return_value=[]),
        patch("warranty_case_ref.case_reference_for_ticket", return_value="WR-20260701-T5"),
        patch("warranty_email.send_phone_ivr_team_email", return_value=True) as mock_email,
    ):
        ok = send_call_followup_team_email(
            caller_phone="+15551234567",
            ticket_id="t-5",
            session_id="sess-5",
            sms_sent=True,
        )

    assert ok is True
    mock_email.assert_called_once()
    kwargs = mock_email.call_args.kwargs
    assert kwargs["caller_phone"] == "+15551234567"
    assert kwargs["ticket_id"] == "t-5"
    assert kwargs["session_id"] == "sess-5"
    assert kwargs["case_reference"] == "WR-20260701-T5"
    assert kwargs["sms_sent"] is True


def test_try_claim_phone_followup_is_idempotent():
    payload_enter = {
        "sessionId": "rc-session-claim",
        "inParty": {
            "id": "party-claim",
            "from": {"phoneNumber": "+15559876543"},
        },
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt"),
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/a.wav"),
    ):
        handle_call_enter(payload_enter)

    ctx = get_call_context("rc-session-claim")
    assert ctx is not None
    ticket_id = ctx.ticket_id

    assert try_claim_phone_followup(ticket_id) is True
    assert try_claim_phone_followup(ticket_id) is False


def test_send_phone_call_followups_skips_after_first_claim():
    with (
        patch("ringcentral_followup.try_claim_phone_followup", return_value=False),
        patch("ringcentral_followup.send_call_followup_sms") as mock_sms,
        patch("ringcentral_followup.send_call_followup_team_email") as mock_email,
    ):
        result = send_phone_call_followups(
            caller_phone="+15551234567",
            ticket_id="t-dup",
            session_id="sess-dup",
        )

    assert result["skipped"] is True
    mock_sms.assert_not_called()
    mock_email.assert_not_called()


@pytest.fixture(autouse=True)
def _clear_call_contexts():
    from ringcentral_voice import _call_contexts  # noqa: WPS433
    from warranty_models import RingCentralCallState, warranty_db_session  # noqa: WPS433

    _call_contexts.clear()
    with warranty_db_session() as db:
        db.query(RingCentralCallState).delete(synchronize_session=False)
    yield
    _call_contexts.clear()
    with warranty_db_session() as db:
        db.query(RingCentralCallState).delete(synchronize_session=False)


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

    with (
        patch("ringcentral_ivr.send_phone_call_followups") as mock_followups,
    ):
        handle_call_exit({"sessionId": "rc-session-exit"})

    mock_followups.assert_called_once()
    assert mock_followups.call_args.kwargs["caller_phone"] == "+15559876543"
    assert mock_followups.call_args.kwargs["ticket_id"]
    assert mock_followups.call_args.kwargs["session_id"] == "rc-session-exit"
    assert get_call_context("rc-session-exit") is None


def test_failed_followups_remain_retryable():
    ticket_id = _create_followup_ticket()
    with (
        patch("ringcentral_followup._followup_enabled", return_value=True),
        patch("ringcentral_followup._team_email_enabled", return_value=True),
        patch("ringcentral_followup.send_call_followup_sms", return_value=False) as mock_sms,
        patch("ringcentral_followup.send_call_followup_team_email", return_value=False) as mock_email,
    ):
        first = send_phone_call_followups(
            caller_phone="+15551234567", ticket_id=ticket_id, session_id="s-retry"
        )
        second = send_phone_call_followups(
            caller_phone="+15551234567", ticket_id=ticket_id, session_id="s-retry"
        )

    assert first["sms_sent"] is False
    assert second["skipped"] is False
    assert mock_sms.call_count == 2
    assert mock_email.call_count == 2


def test_successful_channel_is_not_resent_when_other_channel_retries():
    ticket_id = _create_followup_ticket()
    with (
        patch("ringcentral_followup._followup_enabled", return_value=True),
        patch("ringcentral_followup._team_email_enabled", return_value=True),
        patch("ringcentral_followup.send_call_followup_sms", return_value=True) as mock_sms,
        patch(
            "ringcentral_followup.send_call_followup_team_email",
            side_effect=[False, True],
        ) as mock_email,
    ):
        send_phone_call_followups(
            caller_phone="+15551234567", ticket_id=ticket_id, session_id="s-partial"
        )
        result = send_phone_call_followups(
            caller_phone="+15551234567", ticket_id=ticket_id, session_id="s-partial"
        )

    assert mock_sms.call_count == 1
    assert mock_email.call_count == 2
    assert result["sms_sent"] is True
    assert result["email_sent"] is True
