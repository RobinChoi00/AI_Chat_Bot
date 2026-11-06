"""Free-text replies on the delivery workflow map to the menu the customer meant."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from delivery_intake import (  # noqa: E402
    delivery_identity_parts,
    delivery_sales_topic,
    detect_delivery_spec_question,
    looks_like_delivery_report,
    map_delivery_free_text,
    tracking_number_explainer,
    tracking_number_in,
)
from warranty_self_help import build_delivery_diagnosis  # noqa: E402


@pytest.mark.parametrize("node, text, key", [
    ("delivery_intent_q", "when will my chair arrive", "status_check"),
    ("delivery_intent_q", "my chair hasn't arrived yet", "status_check"),
    ("delivery_intent_q", "the box came crushed", "damage_issue"),
    ("delivery_intent_q", "it never came", "damage_issue"),
    ("delivery_status_tracking_q", "I don't know", "no_tracking"),
    ("delivery_status_tracking_q", "it's in my email", "no_tracking"),
    ("delivery_tracking_q", "1Z999AA10123456784", "has_tracking"),
    ("delivery_problem_type_q", "it never came", "never_arrived"),
    ("delivery_problem_type_q", "missing the remote", "missing_parts"),
    ("delivery_problem_type_q", "box torn, chair scratched", "damaged_in_transit"),
    ("delivery_problem_type_q", "they sent the wrong color", "wrong_item"),
    ("delivery_problem_type_q", "it was two weeks late", "late_delivery"),
    ("delivery_visible_damage_q", "the box was crushed", "yes_box_damage"),
    ("delivery_visible_damage_q", "box looked fine", "no_box_damage"),
    ("delivery_signed_q", "I didn't sign, it was left at the door", "signed_unknown"),
    ("delivery_signed_q", "I signed and wrote damaged on it", "signed_damaged"),
    ("delivery_damage_when_opened_q", "right away", "visible_at_unboxing"),
    ("delivery_damage_when_opened_q", "a few days later", "noticed_later"),
])
def test_free_text_maps_to_the_intended_option(node, text, key):
    assert map_delivery_free_text(node, text) == key


@pytest.mark.parametrize("text", [
    "can I reschedule the delivery?",
    "I need to change my address",
    "do you offer white glove delivery?",
    "how much is shipping?",
    "how long does shipping take?",
    "which carrier do you use?",
])
def test_sales_topics_are_not_forced_into_a_menu_choice(text):
    assert delivery_sales_topic(text)
    assert map_delivery_free_text("delivery_intent_q", text) is None


def test_wrong_item_is_not_treated_as_an_address_change():
    assert delivery_sales_topic("they sent the wrong chair") is None


def test_tracking_number_question_gets_an_explanation_not_an_answer():
    assert tracking_number_explainer("what is a tracking number?")
    assert map_delivery_free_text("delivery_tracking_q", "what is a tracking number?") is None


def test_how_long_shipping_is_not_a_chair_dimension_question():
    assert detect_delivery_spec_question("how long does it take to deliver?") is None


def test_identity_needs_a_digit_and_keeps_both_parts():
    assert delivery_identity_parts("#12345 buyer@example.com") == ("12345", "buyer@example.com")
    assert delivery_identity_parts("I have it")[0] == ""


def test_tracking_number_in_skips_order_numbers_and_phone_numbers():
    assert tracking_number_in("tracking 1Z999AA10123456784 says delivered") == "1Z999AA10123456784"
    assert tracking_number_in("order OSTI12345678 has not arrived") is None
    assert tracking_number_in("call me at 8885015988") is None


@pytest.mark.parametrize("text, expected", [
    ("Delivery came but the remote and side panel are missing", True),
    ("My order #12345 buyer@example.com hasn't arrived", True),
    ("the remote is missing a button", False),
    ("#12345 buyer@example.com my remote is broken", False),
])
def test_delivery_report_detection(text, expected):
    assert looks_like_delivery_report(text) is expected


def test_flowchart_lets_customers_say_they_did_not_sign():
    chart = json.loads((ROOT / "data" / "warranty_flowchart.json").read_text())
    nodes = chart["nodes"] if isinstance(chart.get("nodes"), dict) else {n["node_id"]: n for n in chart["nodes"]}
    keys = {opt["answer_key"] for opt in nodes["delivery_signed_q"]["options"]}
    assert "signed_unknown" in keys


def test_delivery_prompts_do_not_promise_outcomes():
    text = (ROOT / "data" / "warranty_flowchart.json").read_text().lower()
    for promise in ("at no cost", "goodwill", "we will replace", "we'll replace"):
        assert promise not in text


def test_unknown_receipt_does_not_demand_a_signed_receipt():
    for symptom in ("visible_at_unboxing", "noticed_later"):
        diagnosis = build_delivery_diagnosis(symptom=symptom, path_text="", receipt_unknown=True)
        steps = " ".join(diagnosis["steps"]).lower()
        assert "it is fine if you don't" in steps
        assert "signed delivery receipt" not in steps
        assert "your **your chair**" not in diagnosis["summary"]
