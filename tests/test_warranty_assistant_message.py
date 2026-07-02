"""Tests for shared warranty assistant-message enrichment."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


def test_build_assistant_message_bundle_uses_terminal_enrichment(monkeypatch):
    monkeypatch.setattr(
        "warranty_terminal_enrichment.build_terminal_enrichment",
        lambda *_args, **_kwargs: {"message": "Terminal enriched message"},
    )

    from warranty_assistant_message import build_assistant_message_bundle  # noqa: E402

    bundle = build_assistant_message_bundle(
        engine=SimpleNamespace(),
        ticket=SimpleNamespace(ticket_id="t1"),
        node={"node_id": "terminal", "type": "terminal", "prompt": "Done"},
    )
    assert bundle["assistant_message"] == "Terminal enriched message"


def test_format_warranty_result_includes_customer_message(monkeypatch):
    monkeypatch.setattr(
        "warranty_assistant_message.build_assistant_message_bundle",
        lambda **_kwargs: {
            "assistant_message": "Freshdesk-backed step message with next question?",
            "terminal_enrichment": None,
            "step_enrichment": {"message": "Freshdesk-backed step message with next question?"},
        },
    )

    from agent_tools import _format_warranty_result  # noqa: E402

    result = _format_warranty_result(
        "ticket-1",
        {
            "is_terminal": False,
            "next_node": {
                "node_id": "defect_power_outlet",
                "type": "question",
                "prompt": "Is it plugged in?",
                "options": [{"answer_key": "yes", "label": "Yes"}],
            },
        },
        engine=SimpleNamespace(),
        ticket=SimpleNamespace(ticket_id="ticket-1"),
    )
    assert "CUSTOMER_MESSAGE:" in result
    assert "Freshdesk-backed step message" in result
    assert "Deliver CUSTOMER_MESSAGE" in result
