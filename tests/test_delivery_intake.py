"""Unit tests for delivery workflow free-text validation."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from delivery_intake import (  # noqa: E402
    detect_delivery_spec_question,
    is_plausible_email,
    is_plausible_order_id,
    is_plausible_tracking_number,
    looks_like_box_size_question,
    validate_delivery_text_answer,
)


def test_is_plausible_order_id_accepts_common_formats():
    assert is_plausible_order_id("12345")
    assert is_plausible_order_id("OSKUS11308")
    assert is_plausible_order_id("#1001".replace("#", ""))


def test_is_plausible_order_id_rejects_sentences():
    assert not is_plausible_order_id("give me size of the box")
    assert not is_plausible_order_id("what is my order")


def test_is_plausible_tracking_number_accepts_carrier_formats():
    assert is_plausible_tracking_number("1Z999AA10123456784")
    assert is_plausible_tracking_number("1234567890")


def test_is_plausible_tracking_number_rejects_questions():
    assert not is_plausible_tracking_number("what is the tracking number?")


def test_looks_like_box_size_question():
    assert looks_like_box_size_question("give me size of the box")
    assert looks_like_box_size_question("What are the carton dimensions?")
    assert not looks_like_box_size_question("buyer@example.com")


def test_detect_delivery_spec_question_doorway():
    spec = detect_delivery_spec_question("What is the minimum doorway for this chair?")
    assert spec is not None
    assert spec.topic_id == "doorway"


def test_validate_delivery_get_name_answers_doorway_and_reprompts(monkeypatch):
    monkeypatch.setattr(
        "warranty_side_questions.fetch_delivery_spec_answer",
        lambda _model, spec: (
            f"For **Titan Nido 3D**, here is what we have on {spec.title}:\n"
            "- Minimum Doorway: 32 inches"
        ),
    )
    node = {
        "node_id": "delivery_get_name",
        "type": "question_text",
        "prompt": "Please provide your order number or email.",
    }
    from warranty_side_questions import try_answer_side_question  # noqa: E402

    msg = try_answer_side_question(
        node=node,
        answer="What is the minimum doorway?",
        model_name="Titan Nido 3D",
        issue_type="delivery",
        turns=[],
    )
    assert msg is not None
    assert "Minimum Doorway" in msg
    assert "order number" in msg.lower()


def test_validate_delivery_get_name_rejects_box_size_question(monkeypatch):
    monkeypatch.setattr(
        "warranty_side_questions.fetch_delivery_spec_answer",
        lambda _model, spec: (
            f"For **Titan Nido 3D**, here is what we have on {spec.title}:\n"
            "- Carton Width: 34 inches"
        ),
    )
    node = {
        "node_id": "delivery_get_name",
        "type": "question_text",
        "prompt": "Please provide your order number or email.",
    }
    from warranty_side_questions import try_answer_side_question  # noqa: E402

    msg = try_answer_side_question(
        node=node,
        answer="give me size of the box",
        model_name="Titan Nido 3D",
        issue_type="delivery",
        turns=[],
    )
    assert msg is not None
    assert "order number" in msg.lower()


def test_validate_delivery_get_name_accepts_email():
    validate_delivery_text_answer("delivery_get_name", "buyer@example.com")


def test_validate_delivery_get_name_accepts_order():
    validate_delivery_text_answer("delivery_get_name", "#OSKUS11308")


def test_validate_delivery_get_tracking_number_rejects_gibberish():
    with pytest.raises(ValueError, match="tracking number"):
        validate_delivery_text_answer(
            "delivery_get_tracking_number",
            "how big is the box",
        )


def test_is_plausible_email():
    assert is_plausible_email("buyer@example.com")
    assert not is_plausible_email("not-an-email")
