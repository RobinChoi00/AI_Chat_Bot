"""Customer phrases that production routed to the wrong template."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import sales_agent as agent  # noqa: E402
from sales_agent import respond  # noqa: E402
from sales_intent import INTENT_DISCOUNT, classify  # noqa: E402
from sales_language import is_medical_scope_question  # noqa: E402
from sales_policy import TOPIC_MECHANISM, detect_topic  # noqa: E402
from sales_tidio_buttons import prioritize_quick_replies, resolve_button_choice  # noqa: E402
from warranty_common_questions import try_factual_common_question  # noqa: E402
from warranty_nlp import interpret_issue_type, interpret_warranty_answer  # noqa: E402
from warranty_terminal_enrichment import _install_message  # noqa: E402

NODE = {"node_id": "issue_type", "prompt": "Which area?", "options": []}


@pytest.fixture(autouse=True)
def _no_live_price(monkeypatch):
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    monkeypatch.setattr(agent, "fetch_live_stock", lambda *a, **k: None)


def test_costco_membership_is_not_the_discount_template():
    reply = respond("I'm a Costco member and I am browsing the chairs")
    assert "can't quote a discount" not in reply.reply
    assert reply.intent != INTENT_DISCOUNT


def test_price_match_costco_stays_discount():
    intent = classify("Can you price match costcos deal")
    assert intent.label == INTENT_DISCOUNT


def test_letter_of_medical_necessity_is_not_a_discount():
    reply = respond("I need a Letter of Medical Necessity for a Costco purchase")
    assert "letter of medical necessity" in reply.reply.lower()
    assert "can't quote a discount" not in reply.reply
    assert reply.intent != INTENT_DISCOUNT


def test_heart_valve_gets_clinician_guidance():
    text = "If you have a heart valve transplant can you use the massage chair"
    assert is_medical_scope_question(text)
    reply = respond(text)
    assert "clinician" in reply.reply.lower()
    assert "how tall" not in reply.reply.lower()


def test_black_friday_is_not_the_shipping_dump():
    reply = respond("when is black friday sales?")
    assert "sale calendar" in reply.reply.lower()
    assert "White Glove" not in reply.reply


def test_stock_question_does_not_become_a_mechanism_essay():
    text = "DUKE XL 4D MASSAGE CHAIR ... IN STOCK ? LEAD TIME?"
    assert detect_topic(text) != TOPIC_MECHANISM
    reply = respond(text)
    assert "2D massages along" not in reply.reply


def test_height_followup_uses_the_focused_chair():
    reply = respond(
        "what height is this chair good up till",
        prefs={"pending_primary": "Osaki Maestro LE 2.0"},
    )
    assert "designed for" in reply.reply
    assert "Which area" not in reply.reply


def test_where_does_it_say_stays_on_the_focused_chair():
    reply = respond(
        "Where does it say that?",
        prefs={"pending_primary": "Osaki Maestro LE 2.0"},
    )
    assert "published catalog" in reply.reply.lower()
    assert "What can I help" not in reply.reply


def test_named_models_compare_without_the_word_compare():
    reply = respond("Pro quest 4d with flexible SL and Titan Malibu synk")
    assert reply.intent == "compare"
    assert "Which model would you like specs for" not in reply.reply
    assert "Malibu" in reply.reply
    assert "Quest" in reply.reply


def test_brand_line_compare_keeps_the_named_sides():
    reply = respond("difference between quest and titan models")
    assert reply.intent == "compare"
    assert "which two models" not in reply.reply.lower()
    assert "Quest" in reply.reply
    assert "Titan" in reply.reply


def test_button_two_after_a_discount_is_price_not_the_greeting():
    reply = respond("Can you price match costcos deal")
    buttons = prioritize_quick_replies(
        [{"label": item.label, "payload": item.payload} for item in reply.quick_replies]
    )
    chosen = resolve_button_choice("2", buttons)
    assert chosen == "price"
    closed = respond("", payload="done")
    assert "shopping assistant" not in closed.reply.lower()


def test_too_little_pressure_is_not_the_air_repair():
    answer = try_factual_common_question(NODE, "I'm small framed and the shoulders dont squeeze enough")
    assert answer
    assert "will not inflate" in answer
    assert "not strong enough" in answer
    assert interpret_warranty_answer(
        {"node_id": "air_area", "options": [{"key": "air", "label": "Air"}]},
        "the air is too intense, can I lower the intensity",
    ) is None


def test_auto_check_and_watts_are_not_the_air_hose():
    auto = try_factual_common_question(NODE, "How do I start the auto-check on the Escape Duo")
    watts = try_factual_common_question(NODE, "how many watts does the Escape Duo use")
    assert auto and "won't guess the remote buttons" in auto
    assert watts and "won't estimate" in watts
    assert "air hose" not in (auto or "").lower()
    assert "air hose" not in (watts or "").lower()


def test_parts_diagram_does_not_send_the_customer_to_sales():
    answer = try_factual_common_question(NODE, "I need replacement parts. Do you have a parts diagram?")
    assert answer
    assert "service@osakititan.com" in answer
    assert "shopping assistant" not in answer.lower()


def test_fault_style_symptoms_leave_the_menu():
    assert interpret_issue_type("the wheel bracket snapped off") == "defect"
    assert interpret_issue_type("the power button is blinking fast") == "defect"


def test_general_install_guide_does_not_teach_the_air_hose():
    body = _install_message("Osaki Pro 4D", "Please watch the setup guide.")
    assert "air hose" not in str(body.get("message") or "").lower()
