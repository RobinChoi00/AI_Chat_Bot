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

from ringcentral_ivr import (  # noqa: E402
    _present_node,
    _present_terminal,
    handle_call_enter,
    handle_command_update,
)
from ringcentral_voice import IvrPhase, VoiceCallContext, get_call_context  # noqa: E402
from warranty_workflow import WarrantyEngine  # noqa: E402


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


def test_handle_call_enter_starts_at_issue_type_menu():
    payload = {
        "sessionId": "rc-session-1",
        "inParty": {
            "id": "party-1",
            "from": {"phoneNumber": "+18888482630"},
        },
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt") as mock_play,
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/menu.wav"),
    ):
        handle_call_enter(payload)

    ctx = get_call_context("rc-session-1")
    assert ctx is not None
    assert ctx.phase.value == "menu"

    node = WarrantyEngine.get_current_node(ctx.ticket_id)
    assert node is not None
    assert node["node_id"] == "issue_type"

    ticket = WarrantyEngine.get_ticket(ctx.ticket_id)
    assert ticket is not None
    assert ticket.issue_type is None
    assert ticket.get_collected().get("channel") == "phone"

    mock_play.assert_called_once()
    play_uri = mock_play.call_args.kwargs.get("audio_uri") or mock_play.call_args[1].get(
        "audio_uri"
    )
    assert play_uri  # TTS URI generated


def test_issue_type_digit_three_advances_to_defect_menu():
    payload = {
        "sessionId": "rc-session-defect",
        "inParty": {
            "id": "party-defect",
            "from": {"phoneNumber": "+15551234567"},
        },
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt"),
        patch("ringcentral_ivr.collect_digits"),
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/menu.wav"),
    ):
        handle_call_enter(payload)
        handle_command_update(
            {
                "sessionId": "rc-session-defect",
                "status": "Completed",
                "command": "Play",
                "partyId": "party-defect",
            }
        )
        handle_command_update(
            {
                "sessionId": "rc-session-defect",
                "status": "Completed",
                "command": "Collect",
                "partyId": "party-defect",
                "parameters": {"digits": "3"},
            }
        )

    ctx = get_call_context("rc-session-defect")
    assert ctx is not None
    node = WarrantyEngine.get_current_node(ctx.ticket_id)
    assert node is not None
    assert node["node_id"] == "defect_problem_type"
    ticket = WarrantyEngine.get_ticket(ctx.ticket_id)
    assert ticket is not None
    assert str(ticket.issue_type) == "defect"


def test_stale_duplicate_play_completion_does_not_start_collect_twice():
    payload = {
        "sessionId": "rc-session-stale-play",
        "inParty": {"id": "party-stale-play"},
    }
    completed = {
        "sessionId": "rc-session-stale-play",
        "status": "Completed",
        "command": "Play",
        "partyId": "party-stale-play",
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt"),
        patch("ringcentral_ivr.collect_digits") as mock_collect,
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/menu.wav"),
    ):
        handle_call_enter(payload)
        handle_command_update(completed)
        handle_command_update(completed)

    mock_collect.assert_called_once()


def test_handle_call_enter_during_business_hours_plays_connect_then_forwards():
    payload = {
        "sessionId": "rc-session-hours",
        "inParty": {
            "id": "party-hours",
            "from": {"phoneNumber": "+18888482630"},
        },
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=True),
        patch("ringcentral_ivr.forward_call") as mock_forward,
        patch("ringcentral_ivr.play_prompt") as mock_play,
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/connect.wav"),
        patch("ringcentral_ivr._lazy_engine") as mock_engine,
    ):
        handle_call_enter(payload)

    mock_forward.assert_not_called()
    mock_engine.assert_not_called()
    ctx = get_call_context("rc-session-hours")
    assert ctx is not None
    assert ctx.phase.value == "connecting"
    assert ctx.ticket_id
    ticket = WarrantyEngine.get_ticket(ctx.ticket_id)
    assert ticket is not None
    collected = ticket.get_collected()
    assert collected.get("channel") == "phone"
    assert collected.get("caller_phone") == "+18888482630"
    assert collected.get("ivr_path") == "business_hours_live_forward"
    mock_play.assert_called_once()

    with patch("ringcentral_ivr.forward_call") as mock_forward:
        handle_command_update(
            {
                "sessionId": "rc-session-hours",
                "status": "Completed",
                "command": "Play",
                "partyId": "party-hours",
            }
        )

    mock_forward.assert_called_once_with(
        session_id="rc-session-hours",
        party_id="party-hours",
    )
    assert get_call_context("rc-session-hours") is None


def test_handle_call_enter_after_hours_welcome_mentions_closed():
    payload = {
        "sessionId": "rc-session-welcome",
        "inParty": {
            "id": "party-welcome",
            "from": {"phoneNumber": "+15551234567"},
        },
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt"),
        patch("ringcentral_ivr.resolve_play_uri") as mock_uri,
    ):
        handle_call_enter(payload)

    assert mock_uri.called
    script = mock_uri.call_args[0][0]
    assert "closed" in script.lower()
    assert "invoice" in script.lower() or "order number" in script.lower()
    assert "text message" in script.lower()


def test_menu_repeat_zero_replays_current_prompt():
    payload = {
        "sessionId": "rc-session-repeat",
        "inParty": {
            "id": "party-repeat",
            "from": {"phoneNumber": "+15551234567"},
        },
    }
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt") as mock_play,
        patch("ringcentral_ivr.collect_digits") as mock_collect,
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/menu.wav"),
        patch("ringcentral_ivr.forward_call") as mock_forward,
    ):
        handle_call_enter(payload)
        handle_command_update(
            {
                "sessionId": "rc-session-repeat",
                "status": "Completed",
                "command": "Play",
                "partyId": "party-repeat",
            }
        )
        handle_command_update(
            {
                "sessionId": "rc-session-repeat",
                "status": "Completed",
                "command": "Collect",
                "partyId": "party-repeat",
                "parameters": {"digits": "0"},
            }
        )

    ctx = get_call_context("rc-session-repeat")
    assert ctx is not None
    assert mock_play.call_count >= 2
    mock_forward.assert_not_called()
    mock_collect.assert_called()


def test_sales_handoff_after_hours_does_not_transfer():
    ctx = VoiceCallContext(
        session_id="rc-sales-closed",
        party_id="party-sales",
        ticket_id="ticket-sales",
        caller_phone="+15551234567",
    )
    node = {"type": "terminal", "action": "sales_handoff", "prompt": "Sales"}
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt") as mock_play,
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/x.wav"),
        patch("ringcentral_ivr.forward_call") as mock_forward,
    ):
        _present_terminal(ctx, node)

    assert ctx.phase == IvrPhase.POST_DIY
    mock_play.assert_called_once()
    mock_forward.assert_not_called()


def test_sales_handoff_during_open_hours_announces_then_transfers_on_play_done():
    ctx = VoiceCallContext(
        session_id="rc-sales-open",
        party_id="party-sales-open",
        ticket_id="ticket-sales-open",
        caller_phone="+15551234567",
    )
    from ringcentral_voice import set_call_context  # noqa: WPS433

    set_call_context(ctx)
    node = {"type": "terminal", "action": "sales_handoff", "prompt": "Sales"}
    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=True),
        patch("ringcentral_ivr.play_prompt"),
        patch("ringcentral_ivr.resolve_play_uri", return_value="https://example.com/x.wav"),
        patch("ringcentral_ivr.forward_call") as mock_forward,
    ):
        _present_terminal(ctx, node)
        handle_command_update(
            {
                "sessionId": "rc-sales-open",
                "status": "Completed",
                "command": "Play",
                "partyId": "party-sales-open",
            }
        )

    assert ctx.phase == IvrPhase.DONE
    mock_forward.assert_called_once()


def test_question_text_node_plays_web_handoff_script():
    ctx = VoiceCallContext(
        session_id="rc-qtext",
        party_id="party-qtext",
        ticket_id="ticket-qtext",
        caller_phone="+15551234567",
    )
    node = {
        "type": "question_text",
        "prompt": "Enter your order number",
        "next": "delivery_lookup",
    }
    with (
        patch("ringcentral_ivr.play_prompt") as mock_play,
        patch("ringcentral_ivr.resolve_play_uri") as mock_uri,
    ):
        _present_node(ctx, node)

    assert ctx.phase == IvrPhase.MENU
    script = mock_uri.call_args[0][0]
    assert "website" in script.lower() or "text" in script.lower()
    mock_play.assert_called_once()
