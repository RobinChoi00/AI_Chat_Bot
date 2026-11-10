"""Tests for warranty-only scope gate on the embed chat."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_scope import (  # noqa: E402
    build_warranty_scope_refusal,
    evaluate_warranty_scope,
    filter_warranty_menu_options,
    is_sales_workflow_answer,
)


def test_blocks_sales_pricing_question():
    decision = evaluate_warranty_scope("How much is the Osaki Solo Flex?")
    assert decision.is_blocked
    assert decision.reason == "sales_topic"


def test_blocks_pre_purchase_and_sales_team_questions():
    from warranty_scope import is_pre_purchase_sales, wants_sales_contact

    for text in (
        "Looking for a 4d seat that will give me plenty room, 6ft 245",
        "Hi, I’m interested in buying the Osaki AI 4D Yoga Flex massage chair. Is there an email address or live chat for the sales team?",
        "Does the $275 White Glove Delivery include full assembly and installation in my living room?",
        "again. havent bought anything yet. Want to make sure that the tight hallway I have is ok for delivery.",
        "Want to make sure that the tight hallway I have is ok for delivery.",
        "Will the Osaki Oasis fit through a 32 inch hallway?",
    ):
        assert is_pre_purchase_sales(text), text
        decision = evaluate_warranty_scope(text)
        assert decision.is_blocked, text
        assert decision.reason == "sales_topic", text

    assert wants_sales_contact(
        "Could you please connect me with a sales representative?"
    )
    assert not is_pre_purchase_sales("OS-4000T footrest air not inflating")
    assert evaluate_warranty_scope("OS-4000T footrest air not inflating").in_scope


def test_blocks_off_topic():
    decision = evaluate_warranty_scope("Write me a Python script")
    assert decision.is_blocked


def test_allows_warranty_defect_message():
    decision = evaluate_warranty_scope("OS-4000T footrest air not inflating")
    assert decision.in_scope


def test_blocks_hawaii_free_delivery_policy():
    decision = evaluate_warranty_scope("is it free delivery for hawaii")
    assert decision.is_blocked
    assert decision.reason == "shipping_policy"
    msg = build_warranty_scope_refusal(decision.reason)
    assert "hawaii" in msg.lower()
    assert "alaska" in msg.lower()
    assert "guam" in msg.lower()
    assert "sales" in msg.lower()
    assert "ext. 2" in msg
    assert "you pay the freight" not in msg.lower()
    assert "do not ship" not in msg.lower()


def test_blocks_alaska_and_guam_shipping_questions():
    for text in (
        "Do you ship to Alaska?",
        "Can you deliver to Guam?",
        "shipping to HI available?",
    ):
        decision = evaluate_warranty_scope(text)
        assert decision.is_blocked, text
        assert decision.reason == "shipping_policy", text


def test_allows_how_long_shipping_for_verified_tracking():
    decision = evaluate_warranty_scope("how long does shipping take")
    assert decision.in_scope
    assert decision.reason == "delivery"


def test_allows_post_purchase_delivery_for_verified_lookup():
    decision = evaluate_warranty_scope("Where is my FedEx tracking number?")
    assert decision.in_scope
    assert decision.reason == "delivery"
    msg = build_warranty_scope_refusal(decision.reason)
    assert "ext. 2" in msg
    assert "ext. 3" in msg
    assert "sales" in msg.lower()


def test_allows_damaged_delivery_flow():
    decision = evaluate_warranty_scope(
        "My Hawaii shipment arrived damaged and the box was crushed"
    )
    assert decision.in_scope
    assert decision.reason == "delivery"


def test_allows_shipping_help_for_verified_lookup():
    decision = evaluate_warranty_scope("need help with shipping the chair")
    assert decision.in_scope
    assert decision.reason == "delivery"


def test_blocks_sales_answer_key():
    assert is_sales_workflow_answer("sales")
    decision = evaluate_warranty_scope("sales")
    assert decision.is_blocked


def test_filters_sales_but_keeps_delivery_in_root_menu():
    node = {
        "node_id": "root",
        "options": [
            {"answer_key": "warranty", "label": "Warranty"},
            {"answer_key": "sales", "label": "Sales"},
            {"answer_key": "delivery", "label": "Delivery & tracking"},
        ],
    }
    filtered = filter_warranty_menu_options(node)
    keys = [opt["answer_key"] for opt in filtered]
    assert keys == ["warranty", "delivery"]


def test_greeting_hi_with_white_glove_is_not_hawaii_shipping_policy():
    from warranty_scope import is_pre_purchase_shipping_policy, is_pre_purchase_sales

    text = (
        "Hi, I’m planning to purchase the Titan eCabin 3D massage chair.\n\n"
        "I live on the second floor with no elevator, and the staircase has a turn in the middle.\n\n"
        "Does the $275 White Glove Delivery include carrying the chair up the stairs?"
    )
    assert is_pre_purchase_shipping_policy(text) is False
    assert is_pre_purchase_sales(text) is True
    decision = evaluate_warranty_scope(text)
    assert decision.is_blocked
    assert decision.reason == "sales_topic"


def test_shipping_policy_refusal_lists_both_extensions():
    msg = build_warranty_scope_refusal("shipping_policy")
    assert "ext. 2" in msg
    assert "ext. 3" in msg
    assert "sales" in msg.lower()
    assert "you pay the freight" not in msg.lower()
    msg = build_warranty_scope_refusal()
    assert "warranty support" in msg.lower()
    assert "ext. 2" in msg
    assert "ext. 3" in msg
    assert "sales" in msg.lower()


def test_blocks_order_cancel_as_order_cancel_reason():
    from warranty_scope import is_order_cancel_request

    for text in (
        "Cancel my purchase",
        "Cancel my.purchase",
        "I want to cancel my order",
        "Please refund my order",
        "I need to return my chair",
    ):
        assert is_order_cancel_request(text), text
        decision = evaluate_warranty_scope(text)
        assert decision.is_blocked, text
        assert decision.reason == "order_cancel", text

    msg = build_warranty_scope_refusal("order_cancel")
    assert "warranty team" in msg.lower()
    assert "discount" not in msg.lower()
    assert "saved" in msg.lower()
    assert "has not been confirmed" in msg.lower()


def test_order_cancel_not_confused_with_defect():
    decision = evaluate_warranty_scope("OS-4000T footrest air not inflating")
    assert decision.in_scope
    assert evaluate_warranty_scope("Cancel my purchase").reason == "order_cancel"
