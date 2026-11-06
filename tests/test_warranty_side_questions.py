"""Tests for mid-workflow side-question handling."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_side_questions import (  # noqa: E402
    try_answer_side_question,
)


def test_spec_question_on_defect_node_returns_catalog_answer(monkeypatch):
    monkeypatch.setattr(
        "warranty_side_questions.fetch_delivery_spec_answer",
        lambda _model, spec: (
            f"For **Titan Nido 3D**, here is what we have on {spec.title}:\n"
            "- Carton Width: 34 inches"
        ),
    )
    node = {
        "node_id": "defect_power_outlet",
        "type": "question",
        "prompt": "Is the chair plugged into a working outlet?",
        "options": [
            {"answer_key": "yes_outlet", "label": "Yes"},
            {"answer_key": "no_outlet", "label": "No"},
        ],
    }
    msg = try_answer_side_question(
        node=node,
        answer="What are the shipping box dimensions?",
        model_name="Titan Nido 3D",
        issue_type="defect",
        turns=[],
    )
    assert msg is not None
    assert "Carton Width" in msg
    assert "working outlet" in msg


def test_valid_yes_no_answer_is_not_side_question():
    node = {
        "node_id": "defect_power_outlet",
        "type": "question",
        "prompt": "Is the chair plugged into a working outlet?",
        "options": [
            {"answer_key": "yes_outlet", "label": "Yes"},
            {"answer_key": "no_outlet", "label": "No"},
        ],
    }
    assert (
        try_answer_side_question(
            node=node,
            answer="yeah it's plugged in",
            model_name="Titan Nido 3D",
            issue_type="defect",
            turns=[],
        )
        is None
    )


def test_delivery_order_prompt_still_side_answers_spec(monkeypatch):
    monkeypatch.setattr(
        "warranty_side_questions.fetch_delivery_spec_answer",
        lambda _model, spec: f"Spec answer for {spec.topic_id}",
    )
    node = {
        "node_id": "delivery_get_name",
        "type": "question_text",
        "prompt": "Please provide your order number or email.",
        "answer_key": "order_or_email",
    }
    msg = try_answer_side_question(
        node=node,
        answer="give me size of the box",
        model_name="Titan Nido 3D",
        issue_type="delivery",
        turns=[],
    )
    assert msg is not None
    assert "order number" in msg.lower()


def test_delivery_faq_side_question_does_not_show_defect_tips(monkeypatch):
    import warranty_knowledge as wk

    repair = wk.KnowledgeEntry(
        source="freshdesk",
        category="voice",
        title="Voice crackling",
        diagnostic="Replacing the Voice PCB often fixes crackling speaker issues.",
        customer_steps=(
            "Replacing the Voice PCB often fixes crackling speaker issues.",
            "Checking the footrest mechanism can help.",
        ),
    )
    monkeypatch.setattr(wk, "search_knowledge", lambda **kwargs: [repair])

    node = {
        "node_id": "delivery_get_name",
        "type": "question_text",
        "prompt": "Please provide your order number or email.",
    }
    msg = try_answer_side_question(
        node=node,
        answer="How long does shipping usually take?",
        model_name="OS-4000T",
        issue_type="delivery",
        turns=[],
    )
    assert msg is not None
    assert "Voice PCB" not in msg
    assert "order number" in msg.lower()


def test_technician_qa_csv_is_not_shown_as_a_side_answer(monkeypatch):
    import warranty_knowledge as wk

    tech = wk.KnowledgeEntry(
        source="qa_csv",
        category="air",
        title="chair has no air at all",
        diagnostic="The number of laps detected that Hall was damaged",
        customer_steps=("Check and change rolling mechanism",),
    )
    monkeypatch.setattr(wk, "contextual_search_knowledge", lambda **kwargs: [tech])
    monkeypatch.setattr(wk, "search_knowledge", lambda **kwargs: [tech])

    node = {
        "node_id": "defect_air_location",
        "type": "question",
        "prompt": "Which area has no air?",
        "options": [
            {"answer_key": "feet", "label": "Feet / Calves"},
            {"answer_key": "back", "label": "Back"},
        ],
    }
    msg = try_answer_side_question(
        node=node,
        answer="what does this mean?",
        model_name="OS-4000T",
        issue_type="defect",
        turns=[],
    )
    assert msg is not None
    assert "Hall was damaged" not in msg
    assert "chair has no air at all" not in msg
    assert "Check and change rolling mechanism" not in msg


def test_faq_without_knowledge_still_reprompts():
    node = {
        "node_id": "install_concern",
        "type": "question",
        "prompt": "What installation issue are you having?",
        "options": [
            {"answer_key": "general_setup", "label": "General assembly / setup help"},
        ],
    }
    msg = try_answer_side_question(
        node=node,
        answer="What is your return policy for opened boxes?",
        model_name="OS-4000T",
        issue_type="installation",
        turns=[],
    )
    assert msg is not None
    assert "installation issue" in msg.lower()


_ISSUE_TYPE = {
    "node_id": "issue_type",
    "type": "question",
    "prompt": "Which area do you need help with: setup, delivery, or a chair malfunction?",
    "options": [
        {"answer_key": "installation", "label": "Installation Issue"},
        {"answer_key": "delivery", "label": "Delivery Issue"},
        {"answer_key": "defect", "label": "Defect / Malfunction"},
    ],
}


class _Ticket:
    def __init__(self, order_number=""):
        self.collected = {"order_number": order_number} if order_number else {}

    def get_collected(self):
        return dict(self.collected)


def test_order_number_at_area_menu_is_saved_in_the_reply():
    msg = try_answer_side_question(
        node=_ISSUE_TYPE,
        answer="order # OSTI1254",
        model_name="Titan TI Reflex 3D",
        turns=[],
        ticket=_Ticket(),
    )
    assert msg is not None
    assert "OSTI1254" in msg
    assert "not looked it up" in msg
    assert "wasn't fully sure" not in msg
    assert "setup, delivery, or a chair malfunction" in msg


def test_case_progress_question_does_not_use_the_generic_faq_line():
    msg = try_answer_side_question(
        node=_ISSUE_TYPE,
        answer="are you working on it or this is worng number",
        model_name="Titan TI Reflex 3D",
        turns=[],
        ticket=_Ticket("OSTI1254"),
    )
    assert msg is not None
    assert "Nobody is working on this yet" in msg
    assert "OSTI1254" in msg
    assert "has not been verified" in msg
    assert "don't have a specific answer" not in msg
    assert "setup, delivery, or a chair malfunction" in msg
