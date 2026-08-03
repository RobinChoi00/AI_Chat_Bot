"""
tests/test_sales_intent.py
==========================
Guardrails-first contract tests for the Sales AI intent classifier.

These tests are the *safety net*: if a warranty question or a cancel/refund
message ever gets classified as ``price`` or ``recommend`` again, the sales
AI could accidentally answer instead of handing off to the warranty team.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_intent import (  # noqa: E402
    HANDOFF_INTENTS,
    INTENT_CANCEL_REFUND,
    INTENT_COMPARE,
    INTENT_DISCOUNT,
    INTENT_ETA_SHIPPING,
    INTENT_GREETING,
    INTENT_HUMAN,
    INTENT_INTENSITY,
    INTENT_ORDER_STATUS,
    INTENT_PARTS_TECHNICIAN,
    INTENT_PRICE,
    INTENT_RECOMMEND,
    INTENT_SPECS,
    INTENT_STOCK,
    INTENT_UNCLEAR,
    INTENT_WARRANTY_REDIRECT,
    classify,
    handoff_message,
)


# ---------------------------------------------------------------------------
# Guardrails — every message here MUST end up as a handoff.
# ---------------------------------------------------------------------------


def test_warranty_defect_language_redirects_to_warranty():
    for text in [
        "my chair won't power on",
        "footrest is stuck and airbag stopped inflating",
        "remote not working",
        "error code E1 on the display",
        "my chair is broken, please help",
        "installation help — the base wobbles",
        "delivered damaged yesterday",
        "보증 관련 질문이에요",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_WARRANTY_REDIRECT, text
        assert intent.is_handoff, text


def test_cancel_and_refund_route_to_warranty_team():
    for text in [
        "I want to cancel my order",
        "please refund my purchase",
        "return my chair",
        "cancel my subscription",
        "환불해주세요",
        "취소하고 싶어요",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_CANCEL_REFUND, text
        assert intent.is_handoff, text


def test_parts_and_technician_route_to_warranty_team():
    for text in [
        "I need a replacement part for my footrest",
        "I need replacement parts for my chair",
        "can you send a technician to my house",
        "please dispatch a repair tech",
        "send someone to come fix it",
        "I need a spare part",
        "I need parts for my OS-Pro",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_PARTS_TECHNICIAN, text
        assert intent.is_handoff, text
        assert "service@osakititan.com" in (handoff_message(intent) or "").lower()


def test_discount_never_answered_directly():
    for text in [
        "any discount available?",
        "do you have a promo code",
        "can you do better on the price",
        "price match with amazon",
        "coupon code please",
        "할인 되나요",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_DISCOUNT, text
        assert intent.is_handoff, text


def test_eta_and_delivery_promise_route_to_human():
    for text in [
        "when will it arrive",
        "how long until delivery",
        "estimated delivery date?",
        "can you guarantee delivery before Christmas",
        "lead time please",
        "do you offer free shipping",
        "can you ship to Hawaii",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_ETA_SHIPPING, text
        assert intent.is_handoff, text


def test_shipping_and_tracking_handoff_copy_points_to_warranty():
    from sales_intent import SalesIntent, handoff_message

    for label in (INTENT_ETA_SHIPPING, INTENT_ORDER_STATUS, INTENT_WARRANTY_REDIRECT):
        msg = handoff_message(SalesIntent(label=label, confidence="high", handoff=True))
        assert msg
        assert "warranty" in msg.lower()
        assert "zip" not in msg.lower()
        assert "%" not in msg


def test_discount_handoff_does_not_explain_policy():
    from sales_intent import SalesIntent, handoff_message

    msg = handoff_message(
        SalesIntent(label=INTENT_DISCOUNT, confidence="high", handoff=True)
    )
    assert msg
    assert "email" in msg.lower()
    assert "%" not in msg
    assert "promo" not in msg.lower()
    assert "offer" not in msg.lower()
    assert "discount" not in msg.lower()  # don't talk about the policy topic


def test_human_request_is_recognized():
    for text in [
        "talk to a human",
        "connect me to a rep",
        "speak with sales",
        "call me please",
        "상담원 연결",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_HUMAN, text
        assert intent.is_handoff, text


def test_all_handoff_labels_produce_a_handoff_message():
    """Every guardrail intent must have a canned safe reply."""
    for label in HANDOFF_INTENTS:
        from sales_intent import SalesIntent

        intent = SalesIntent(label=label, confidence="high", handoff=True)
        assert handoff_message(intent), f"missing handoff copy for {label}"


# ---------------------------------------------------------------------------
# Priority: mixed messages fall to the *safer* intent.
# ---------------------------------------------------------------------------


def test_mixed_defect_plus_price_routes_to_warranty():
    """
    A message that includes both a warranty problem AND a price question
    must go to warranty — never answered as a price question.
    """
    intent = classify("my chair is broken and how much is the OS-Pro Maestro?")
    assert intent.label == INTENT_WARRANTY_REDIRECT
    assert intent.is_handoff


def test_mixed_cancel_plus_price_routes_to_cancel():
    intent = classify("I want to cancel my order but also how much is the Titan?")
    assert intent.label == INTENT_CANCEL_REFUND
    assert intent.is_handoff


# ---------------------------------------------------------------------------
# Happy-path sales sub-intents.
# ---------------------------------------------------------------------------


def test_price_intent():
    for text in [
        "how much is the Osaki OS-Pro Maestro LE",
        "price of the Titan Jupiter",
        "what's the cost of this chair",
        "가격 알려주세요",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_PRICE, text
        assert not intent.is_handoff


def test_stock_intent():
    for text in [
        "is the OS-Pro Maestro in stock",
        "do you have it available",
        "재고 있나요",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_STOCK, text


def test_recommend_intent():
    for text in [
        "can you recommend a chair for a tall guy",
        "which chair should I buy",
        "best chair for back pain",
        "I am 6'2\" and 220 lb",
        "추천 부탁해요",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_RECOMMEND, text


def test_compare_intent():
    for text in [
        "compare OS-Pro Maestro vs Titan Jupiter",
        "difference between 3D and 4D",
        "which is better, Osaki or Titan",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_COMPARE, text


def test_specs_intent():
    for text in [
        "does it have zero gravity",
        "what's the weight capacity",
        "L-track or S-track?",
        "features of this chair",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_SPECS, text


def test_intensity_intent():
    for text in [
        "is the massage strong enough",
        "how deep does it go",
        "massage intensity please",
        "세기가 얼마나 세나요",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_INTENSITY, text


def test_body_fit_hints_route_to_recommend_not_intensity():
    """Mixed body-fit + intensity language must go to recommend so the AI
    proposes an actual chair for the customer's body (regression: this
    used to be labelled intensity because ``strong`` matched first)."""
    for text in [
        "I'm 5'5\", 200 pounds and prefer strong massage",
        "I want strong hamstring and glute massage",
        "6'2 220 lb, back pain, need a firm massage",
        "my lower back hurts and I like deep massage",
        "petite wife with neck pain — which chair?",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_RECOMMEND, text


def test_cancel_and_shipping_reuse_warranty_department_copy():
    """Every warranty-route intent must reuse the Warranty Department
    contact copy (email / phone / Freshdesk) — never invent its own."""
    from sales_intent import SalesIntent

    warranty_labels = (
        INTENT_WARRANTY_REDIRECT,
        INTENT_CANCEL_REFUND,
        INTENT_PARTS_TECHNICIAN,
        INTENT_ETA_SHIPPING,
        INTENT_ORDER_STATUS,
    )
    baseline = handoff_message(
        SalesIntent(label=INTENT_WARRANTY_REDIRECT, confidence="high", handoff=True)
    )
    assert baseline
    assert "service@osakititan.com" in baseline.lower()
    assert "titanchair.freshdesk.com" in baseline.lower()
    assert "1-888-848-2630" in baseline
    for label in warranty_labels:
        msg = handoff_message(SalesIntent(label=label, confidence="high", handoff=True))
        assert msg == baseline, f"{label} should reuse the shared warranty redirect copy"


def test_greeting_is_greeting():
    for text in ["hi", "hello!", "안녕하세요"]:
        intent = classify(text)
        assert intent.label == INTENT_GREETING, text


def test_empty_text_is_unclear():
    assert classify("").label == INTENT_UNCLEAR
    assert classify("   ").label == INTENT_UNCLEAR


def test_random_off_topic_is_unclear_not_answered():
    """A message we can't classify must NOT be answered — force the menu."""
    intent = classify("purple monkey dishwasher")
    assert intent.label == INTENT_UNCLEAR


def test_order_status_is_recognized():
    for text in [
        "where is my order",
        "tracking number for my chair",
        "FedEx delivery for my order",
        "tracking",
    ]:
        intent = classify(text)
        assert intent.label == INTENT_ORDER_STATUS, text
        assert intent.is_handoff, text


def test_tidio_short_triggers_are_classified():
    """Every short phrase we put in Tidio 'Visitor says' must resolve to a
    real intent — never fall through to unclear."""
    expected = {
        "hello": INTENT_GREETING,
        "hi": INTENT_GREETING,
        "hey": INTENT_GREETING,
        "good morning": INTENT_GREETING,
        "good afternoon": INTENT_GREETING,
        "help": INTENT_GREETING,
        "price": INTENT_PRICE,
        "how much": INTENT_PRICE,
        "cost": INTENT_PRICE,
        "recommend": INTENT_RECOMMEND,
        "which chair": INTENT_RECOMMEND,
        "which model": INTENT_RECOMMEND,
        "best chair": INTENT_RECOMMEND,
        "budget": INTENT_RECOMMEND,
        "under 3000": INTENT_RECOMMEND,
        "under 6000": INTENT_RECOMMEND,
        "tall": INTENT_RECOMMEND,
        "petite": INTENT_RECOMMEND,
        "back pain": INTENT_RECOMMEND,
        "neck pain": INTENT_RECOMMEND,
        "in stock": INTENT_STOCK,
        "available": INTENT_STOCK,
        "out of stock": INTENT_STOCK,
        "compare": INTENT_COMPARE,
        "vs": INTENT_COMPARE,
        "difference": INTENT_COMPARE,
        "specs": INTENT_SPECS,
        "features": INTENT_SPECS,
        "4D": INTENT_SPECS,
        "3D": INTENT_SPECS,
        "SL-Track": INTENT_SPECS,
        "zero gravity": INTENT_SPECS,
        "heating": INTENT_SPECS,
        "massage intensity": INTENT_INTENSITY,
        "discount": INTENT_DISCOUNT,
        "coupon": INTENT_DISCOUNT,
        "sale": INTENT_DISCOUNT,
        "deal": INTENT_DISCOUNT,
        "promo": INTENT_DISCOUNT,
        "financing": INTENT_DISCOUNT,
        "shipping": INTENT_ETA_SHIPPING,
        "delivery": INTENT_ETA_SHIPPING,
        "arrive": INTENT_ETA_SHIPPING,
        "tracking": INTENT_ORDER_STATUS,
        "warranty": INTENT_WARRANTY_REDIRECT,
        "broken": INTENT_WARRANTY_REDIRECT,
        "not working": INTENT_WARRANTY_REDIRECT,
        "repair": INTENT_WARRANTY_REDIRECT,
        "parts": INTENT_PARTS_TECHNICIAN,
        "replacement parts": INTENT_PARTS_TECHNICIAN,
        "technician": INTENT_PARTS_TECHNICIAN,
        "talk to a human": INTENT_HUMAN,
        "agent": INTENT_HUMAN,
        "representative": INTENT_HUMAN,
    }
    for phrase, label in expected.items():
        intent = classify(phrase)
        assert intent.label == label, f"{phrase!r} -> {intent.label}, want {label}"
