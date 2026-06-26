"""Tests for AI-assisted warranty case summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_email import build_transcript_body  # noqa: E402
from warranty_summary import (  # noqa: E402
    build_deterministic_case_summary,
    summarize_warranty_case,
)


class _Turn:
    def __init__(
        self,
        answer_key: str = "",
        customer_answer: str = "",
        node_prompt: str = "",
        node_id: str = "",
    ):
        self.answer_key = answer_key
        self.customer_answer = customer_answer
        self.node_prompt = node_prompt
        self.node_id = node_id


def test_build_deterministic_case_summary_includes_path():
    turns = [
        _Turn("defect"),
        _Turn("footrest"),
        _Turn("air_not_inflating"),
        _Turn("air_blowing", "Yes air is blowing"),
    ]
    summary = build_deterministic_case_summary(
        issue_type="defect",
        model_name="OS-4000T",
        turns=turns,
        terminal_node_id="defect_air_footrest_wg_tech_terminal",
    )
    assert "OS-4000T" in summary
    assert "footrest" in summary
    assert "defect_air_footrest_wg_tech_terminal" in summary


def test_summarize_without_llm_uses_deterministic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = summarize_warranty_case(
        issue_type="defect",
        model_name="OS-4000T",
        turns=[_Turn("air"), _Turn("feet_calves"), _Turn("never_worked")],
        terminal_node_id="defect_air_pump_terminal",
        use_llm=True,
    )
    assert result["source"] == "deterministic"
    assert "feet_calves" in result["summary"] or "Path:" in result["summary"]
    assert result["suggested_subject"]


def test_summarize_llm_high_confidence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeMessage:
        content = json.dumps(
            {
                "summary": (
                    "Customer reports footrest airbags not inflating on an OS-4000T. "
                    "They confirmed air blows through the hose when disconnected."
                ),
                "suggested_subject": "OS-4000T footrest air not inflating",
                "confidence": "high",
            }
        )

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("warranty_summary._openai_client", lambda: FakeClient)

    result = summarize_warranty_case(
        issue_type="defect",
        model_name="OS-4000T",
        turns=[
            _Turn(
                "defect_footrest_which",
                "Airbags are NOT inflating in the footrest",
                node_prompt="What footrest issue?",
            )
        ],
        terminal_node_id="defect_air_footrest_wg_tech_terminal",
    )
    assert result["source"] == "llm"
    assert "footrest airbags" in result["summary"].lower()
    assert "OS-4000T footrest" in result["suggested_subject"]


def test_build_transcript_body_includes_case_summary():
    body = build_transcript_body(
        ticket_id="T-1",
        session_id="sess-1",
        customer_email="buyer@example.com",
        domain="osaki.com",
        issue_type="defect",
        model_name="OS-4000T",
        turns=[],
        case_summary="Footrest air issue on OS-4000T; customer checked hose connection.",
    )
    assert "Case summary (AI-assisted)" in body
    assert "Footrest air issue" in body
