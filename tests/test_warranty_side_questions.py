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
