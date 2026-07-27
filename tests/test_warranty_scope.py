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
    assert "do not deliver" in msg.lower()


def test_blocks_alaska_and_guam_shipping_questions():
    for text in (
        "Do you ship to Alaska?",
        "Can you deliver to Guam?",
        "shipping to HI available?",
    ):
        decision = evaluate_warranty_scope(text)
        assert decision.is_blocked, text
        assert decision.reason == "shipping_policy", text


def test_allows_post_purchase_delivery_tracking():
    decision = evaluate_warranty_scope("Where is my FedEx tracking number?")
    assert decision.in_scope


def test_allows_damaged_delivery_even_if_hawaii_mentioned():
    decision = evaluate_warranty_scope(
        "My Hawaii shipment arrived damaged and the box was crushed"
    )
    assert decision.in_scope


def test_blocks_sales_answer_key():
    assert is_sales_workflow_answer("sales")
    decision = evaluate_warranty_scope("sales")
    assert decision.is_blocked


def test_filters_sales_from_root_menu():
    node = {
        "node_id": "root",
        "options": [
            {"answer_key": "warranty", "label": "Warranty"},
            {"answer_key": "sales", "label": "Sales"},
        ],
    }
    filtered = filter_warranty_menu_options(node)
    keys = [opt["answer_key"] for opt in filtered]
    assert keys == ["warranty"]


def test_refusal_mentions_warranty_only():
    msg = build_warranty_scope_refusal()
    assert "warranty support" in msg.lower()
    assert "sales" in msg.lower()
