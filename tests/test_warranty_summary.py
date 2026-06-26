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
    contains_promise_language,
    format_case_summary_for_email,
    format_case_summary_section_header,
    sanitize_email_subject,
    summarize_warranty_case,
    validate_llm_summary_against_transcript,
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


def _footrest_air_turns() -> list[_Turn]:
    return [
        _Turn("defect", node_id="defect_root"),
        _Turn("footrest", node_id="defect_footrest_which"),
        _Turn(
            "air_not_inflating",
            "Airbags are NOT inflating in the footrest",
            node_prompt="What footrest issue?",
            node_id="defect_footrest_which",
        ),
        _Turn(
            "air_blowing",
            "Yes, air blows through the hose when disconnected",
            node_prompt="Is air coming out when you disconnect the hose?",
            node_id="defect_air_footrest_raise",
        ),
    ]


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
                    "Footrest airbags are not inflating on an OS-4000T. "
                    "Customer confirmed air blows through the hose when disconnected."
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
        turns=_footrest_air_turns(),
        terminal_node_id="defect_air_footrest_wg_tech_terminal",
    )
    assert result["source"] == "llm"
    assert "footrest airbags" in result["summary"].lower()
    assert "OS-4000T footrest" in result["suggested_subject"]


def test_validate_rejects_promise_in_summary():
    ok, reason = validate_llm_summary_against_transcript(
        summary="We will dispatch a technician to replace the footrest.",
        suggested_subject="OS-4000T footrest",
        issue_type="defect",
        model_name="OS-4000T",
        turns=_footrest_air_turns(),
    )
    assert ok is False
    assert reason == "promise_in_summary"


def test_validate_rejects_promise_in_subject():
    ok, reason = validate_llm_summary_against_transcript(
        summary="Footrest airbags not inflating on OS-4000T after hose check.",
        suggested_subject="Approved — will ship replacement footrest",
        issue_type="defect",
        model_name="OS-4000T",
        turns=_footrest_air_turns(),
    )
    assert ok is False
    assert reason == "promise_in_subject"


def test_validate_rejects_wrong_model():
    ok, reason = validate_llm_summary_against_transcript(
        summary="Epic 4D footrest airbags not inflating after hose check.",
        suggested_subject="Epic 4D footrest air",
        issue_type="defect",
        model_name="OS-4000T",
        turns=_footrest_air_turns(),
    )
    assert ok is False
    assert reason == "model_not_in_output"


def test_validate_rejects_hallucinated_symptoms():
    ok, reason = validate_llm_summary_against_transcript(
        summary=(
            "OS-4000T compressor failure prevents shoulder airbags from inflating "
            "during zero gravity mode."
        ),
        suggested_subject="OS-4000T compressor failure",
        issue_type="defect",
        model_name="OS-4000T",
        turns=_footrest_air_turns(),
    )
    assert ok is False
    assert reason == "facts_not_in_transcript"


def test_llm_output_rejected_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeMessage:
        content = json.dumps(
            {
                "summary": (
                    "OS-4000T compressor failure prevents shoulder airbags from inflating."
                ),
                "suggested_subject": "OS-4000T compressor failure",
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
        turns=_footrest_air_turns(),
        terminal_node_id="defect_air_footrest_wg_tech_terminal",
    )
    assert result["source"] == "deterministic"
    assert "Path:" in result["summary"]


def test_contains_promise_language_and_subject_sanitize():
    assert contains_promise_language("We will dispatch a technician tomorrow.")
    fallback = "OS-4000T — defect case"
    assert sanitize_email_subject("Will ship replacement part", fallback=fallback) == fallback


def test_format_case_summary_disclaimer_for_llm_only():
    assert "AI-generated — verify workflow below" in format_case_summary_section_header("llm")
    body = format_case_summary_for_email("Footrest air issue.", "llm")
    assert "AI-generated — verify workflow below" in body
    assert "from workflow" in format_case_summary_section_header("deterministic")


def test_build_transcript_body_includes_llm_disclaimer():
    body = build_transcript_body(
        ticket_id="T-1",
        session_id="sess-1",
        customer_email="buyer@example.com",
        domain="osaki.com",
        issue_type="defect",
        model_name="OS-4000T",
        turns=[],
        case_summary="Footrest air issue on OS-4000T; customer checked hose connection.",
        case_summary_source="llm",
    )
    assert "AI-generated — verify workflow below" in body
    assert "Footrest air issue" in body
