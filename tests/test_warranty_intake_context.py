"""Tests for warranty intake context and chat guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_intake_context import (  # noqa: E402
    enrich_path_text,
    get_intake_summary,
    intake_aware_step_summary,
    persist_intake_summary,
)


class _CollectedTicket:
    def __init__(self):
        self._data: dict[str, str] = {}

    def get_collected(self) -> dict:
        return dict(self._data)

    def set_collected(self, key: str, value: str) -> None:
        self._data[key] = value


def test_persist_and_read_intake_summary():
    ticket = _CollectedTicket()
    persist_intake_summary(
        ticket,
        summary="Footrest air not inflating.",
        raw_message="OS-4000T footrest air not inflating after setup",
    )
    assert get_intake_summary(ticket) == "Footrest air not inflating."
    assert ticket.get_collected()["intake_raw_message"].startswith("OS-4000T")


def test_enrich_path_text_prepends_intake_once():
    ticket = _CollectedTicket()
    persist_intake_summary(ticket, summary="Footrest air not inflating.")
    enriched = enrich_path_text("defect air footrest", ticket)
    assert enriched.startswith("Footrest air not inflating.")
    assert "defect air footrest" in enriched


def test_intake_aware_step_summary_only_on_early_turns():
    ticket = _CollectedTicket()
    persist_intake_summary(ticket, summary="Remote screen is blank.")
    early = intake_aware_step_summary(
        ticket=ticket,
        turns=[1, 2, 3],
        summary="This looks like a **remote** issue.",
    )
    assert "Remote screen is blank" in early
    late = intake_aware_step_summary(
        ticket=ticket,
        turns=list(range(8)),
        summary="This looks like a **remote** issue.",
    )
    assert "Remote screen is blank" not in late


def test_tool_answer_side_question_does_not_advance(monkeypatch):
    from agent_tools import tool_answer_warranty_question, tool_start_warranty_workflow

    monkeypatch.setattr(
        "warranty_side_questions.fetch_delivery_spec_answer",
        lambda model, spec: "Box size: 48 x 32 x 30 inches.",
    )

    start = tool_start_warranty_workflow(session_id="side-agent-test", domain="osaki.com")
    ticket_id = None
    for line in start.splitlines():
        if line.startswith("TICKET_ID:"):
            ticket_id = line.split(":", 1)[1].strip()
            break
    assert ticket_id

    from warranty_workflow import WarrantyEngine  # noqa: E402

    tool_answer_warranty_question(ticket_id=ticket_id, answer_key="warranty")
    tool_answer_warranty_question(ticket_id=ticket_id, answer_key="defect")
    WarrantyEngine.set_model_name(ticket_id, "OS-4000T")

    node_before = WarrantyEngine.get_current_node(ticket_id)
    node_id_before = node_before.get("node_id")

    result = tool_answer_warranty_question(
        ticket_id=ticket_id,
        answer_key="power",
        customer_text="Will the shipping box fit through a 32 inch doorway?",
    )
    assert "WARRANTY_SIDE_QUESTION" in result
    assert "Box size" in result
    assert WarrantyEngine.get_current_node(ticket_id).get("node_id") == node_id_before
