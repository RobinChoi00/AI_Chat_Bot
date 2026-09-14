"""
tests/test_sales_policy.py
==========================
Pre-purchase policy answers.

Two properties matter here and both are load-bearing:

1. A shopper's policy question gets a published answer instead of a handoff.
2. The same words from someone who already owns a chair still hand off, and
   no answer ever invents a delivery date, a shipping price, or an APR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_intent import INTENT_PREPURCHASE_POLICY, INTENT_RECOMMEND, classify  # noqa: E402
from sales_policy import (  # noqa: E402
    POLICY_TOPICS,
    TOPIC_FINANCING,
    TOPIC_INTERNATIONAL,
    TOPIC_LEASE,
    TOPIC_LIMITS,
    TOPIC_MECHANISM,
    TOPIC_REFURBISHED,
    TOPIC_RESTRICTED_REGION,
    TOPIC_REMOTE_SHIPPING,
    TOPIC_RETURNS,
    TOPIC_SHIPPING,
    TOPIC_SHOWROOM,
    TOPIC_TRADE_IN,
    TOPIC_WARRANTY_TERMS,
    TOPIC_WHITE_GLOVE,
    detect_topic,
    is_post_purchase,
    policy_answer,
)


# ---------------------------------------------------------------------------
# Topic routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "topic"),
    [
        ("what is your return policy", TOPIC_RETURNS),
        ("can I return it if I don't like it", TOPIC_RETURNS),
        ("is there a restocking fee", TOPIC_RETURNS),
        ("what if I change my mind", TOPIC_RETURNS),
        ("how long is the warranty", TOPIC_WARRANTY_TERMS),
        ("what warranty comes with it", TOPIC_WARRANTY_TERMS),
        ("do you have an extended warranty", TOPIC_WARRANTY_TERMS),
        ("how much is shipping", TOPIC_SHIPPING),
        ("is delivery free", TOPIC_SHIPPING),
        ("how long does shipping take", TOPIC_SHIPPING),
        ("do you ship to Alaska", TOPIC_REMOTE_SHIPPING),
        ("can you deliver to Hawaii", TOPIC_REMOTE_SHIPPING),
        ("shipping to Guam?", TOPIC_RESTRICTED_REGION),
        ("what's the difference between 3D and 4D", TOPIC_MECHANISM),
        ("what is a dual roller", TOPIC_MECHANISM),
        ("do you assemble it", TOPIC_WHITE_GLOVE),
        ("is white glove available", TOPIC_WHITE_GLOVE),
        ("can you carry it upstairs", TOPIC_WHITE_GLOVE),
        ("do you offer financing", TOPIC_FINANCING),
        ("can I pay monthly", TOPIC_FINANCING),
        ("do you take affirm", TOPIC_FINANCING),
        ("what's the APR", TOPIC_FINANCING),
        ("where is your showroom", TOPIC_SHOWROOM),
        ("can I try one in person", TOPIC_SHOWROOM),
        ("what are your hours", TOPIC_SHOWROOM),
        ("book a showroom visit", TOPIC_SHOWROOM),
        ("are you open Sunday", TOPIC_SHOWROOM),
        ("can I pick up today", TOPIC_SHOWROOM),
        ("do you sell refurbished chairs", TOPIC_REFURBISHED),
        ("any open box models", TOPIC_REFURBISHED),
        ("do you take trade-ins", TOPIC_TRADE_IN),
        (
            "do you have exchange programs to replace previous models with newer models?",
            TOPIC_TRADE_IN,
        ),
        ("is this a brand new chair?", TOPIC_REFURBISHED),
        ("is this a new chair", TOPIC_REFURBISHED),
        ("96701 is my zip", TOPIC_REMOTE_SHIPPING),
        ("99501 is my zip", TOPIC_REMOTE_SHIPPING),
        ("96913 is my zip", TOPIC_RESTRICTED_REGION),
        ("75001 is my zip", TOPIC_SHIPPING),
        ("tablet instalations", TOPIC_WHITE_GLOVE),
        ("do you have an affiliate program", TOPIC_LIMITS),
        ("government 889 form", TOPIC_LIMITS),
        ("can tou look up the one my friend has", TOPIC_LIMITS),
        (
            "hello, i am trying to make a purchase but my amex is not going through",
            TOPIC_LIMITS,
        ),
        ("can I lease a chair", TOPIC_LEASE),
        ("ship to Canada", TOPIC_INTERNATIONAL),
        ("international shipping to Mexico", TOPIC_INTERNATIONAL),
        ("is it pet friendly", TOPIC_LIMITS),
        ("do you speak Spanish", TOPIC_LIMITS),
        ("gift wrap", TOPIC_LIMITS),
        ("safe during pregnancy", TOPIC_LIMITS),
        ("do you take PayPal", TOPIC_LIMITS),
        ("can I use HSA", TOPIC_LIMITS),
        ("I live on the third floor", TOPIC_WHITE_GLOVE),
        ("lease to own", TOPIC_FINANCING),
    ],
)
def test_prepurchase_questions_route_to_their_topic(message, topic):
    assert detect_topic(message) == topic
    assert classify(message).label == INTENT_PREPURCHASE_POLICY


@pytest.mark.parametrize(
    "message",
    [
        "my chair is broken",
        "where is my order",
        "I bought a chair last year, is it still under warranty",
        "I want to return my order",
        "when will my order arrive",
        "my order number is 12345, what's the return policy",
        "I received it damaged, can I get a refund",
        "tracking says delivered but nothing arrived",
    ],
)
def test_post_purchase_messages_never_get_the_policy_answer(message):
    """Ownership wording must keep the existing warranty/order handoff."""
    assert is_post_purchase(message) is True
    assert detect_topic(message) is None
    assert classify(message).label != INTENT_PREPURCHASE_POLICY


@pytest.mark.parametrize(
    "message",
    ["how much is the OS-Pro Maestro", "is it in stock", "recommend a chair", "hi"],
)
def test_ordinary_sales_questions_are_untouched(message):
    assert detect_topic(message) is None
    assert classify(message).label != INTENT_PREPURCHASE_POLICY


# ---------------------------------------------------------------------------
# Answer content — published facts only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", POLICY_TOPICS)
def test_every_topic_renders_a_non_empty_answer(topic):
    answer = policy_answer(topic, "osakiusa.com")
    assert answer and len(answer) > 80


@pytest.mark.parametrize(
    ("topic", "path"),
    [
        (TOPIC_RETURNS, "/pages/sales-policy"),
        (TOPIC_WARRANTY_TERMS, "/pages/warranty"),
        (TOPIC_SHIPPING, "/pages/shipping-handling"),
        (TOPIC_WHITE_GLOVE, "/pages/shipping-handling"),
        (TOPIC_REMOTE_SHIPPING, "/pages/shipping-handling"),
        (TOPIC_RESTRICTED_REGION, "/pages/shipping-handling"),
    ],
)
def test_answers_link_the_official_policy_page(topic, path):
    """Every quoted fact stays verifiable by the customer."""
    assert f"https://osakiusa.com{path}" in policy_answer(topic, "osakiusa.com")


def test_policy_links_follow_the_storefront_domain():
    assert "titanchair.com" in policy_answer(TOPIC_RETURNS, "titanchair.com")


def test_guam_is_an_explicit_no():
    answer = policy_answer(TOPIC_RESTRICTED_REGION, "osakiusa.com")
    assert "don't ship to Guam" in answer or "don't ship to guam" in answer.lower()
    assert "guam" in answer.lower()
    assert "hawaii" in answer.lower()


def test_hawaii_and_alaska_ship_with_customer_paid_freight():
    answer = policy_answer(TOPIC_REMOTE_SHIPPING, "osakiusa.com").lower()
    assert "hawaii" in answer and "alaska" in answer
    assert "you pay" in answer or "customer" in answer
    assert "quote" in answer
    assert "don't deliver" not in answer
    assert "don't ship" not in answer


def test_returns_answer_states_the_published_window_and_costs():
    answer = policy_answer(TOPIC_RETURNS, "osakiusa.com")
    assert "30 days" in answer
    assert "outbound" in answer.lower() or "both" in answer.lower()
    assert "white glove" in answer.lower()
    assert "not refundable" in answer.lower()
    assert "20%" in answer
    assert "RMA" in answer
    assert "sales-policy" in answer


def test_warranty_answer_states_the_published_coverage():
    answer = policy_answer(TOPIC_WARRANTY_TERMS, "osakiusa.com")
    assert "three (3) years" in answer
    assert "Year 1" in answer and "Year 2" in answer


def test_shipping_answer_states_lead_times_without_a_calendar_date():
    """Sales published ceiling is up to 2 / 3 weeks — not a promised day."""
    answer = policy_answer(TOPIC_SHIPPING, "osakiusa.com").lower()
    assert "up to 2 weeks" in answer
    assert "up to 3 weeks" in answer
    assert "can't promise a specific calendar date" in answer
    assert "assembly is not included" in answer


def test_mechanism_answer_explains_each_axis():
    answer = policy_answer(TOPIC_MECHANISM, "osakiusa.com").lower()
    assert "2d" in answer and "3d" in answer and "4d" in answer and "5d" in answer
    assert "dual roller" in answer
    assert "x and y" in answer
    assert "in and out" in answer


def test_showroom_answer_lists_hours_and_is_not_a_booking():
    answer = policy_answer(TOPIC_SHOWROOM, "osakiusa.com")
    assert "9:30" in answer
    assert "Carrollton" in answer
    assert "request" in answer.lower()
    assert "lock a calendar slot" in answer.lower()


def test_showroom_sunday_does_not_invent_open_hours():
    answer = policy_answer(
        TOPIC_SHOWROOM, "osakiusa.com", message="are you open Sunday"
    )
    assert "Sunday isn't listed" in answer
    assert "9:30" in answer


def test_showroom_pickup_does_not_confirm_same_day():
    answer = policy_answer(
        TOPIC_SHOWROOM, "osakiusa.com", message="can I pick up today"
    )
    assert "same-day pickup" in answer.lower()


def test_refurbished_trade_in_lease_are_honest_nos():
    assert "new" in policy_answer(TOPIC_REFURBISHED, "osakiusa.com").lower()
    assert "don't take trade-ins" in policy_answer(
        TOPIC_TRADE_IN, "osakiusa.com"
    ).lower()
    lease = policy_answer(TOPIC_LEASE, "osakiusa.com")
    assert "don't lease" in lease.lower()
    assert "Affirm" in lease
    assert "%" not in lease


def test_financing_answer_quotes_no_rate_or_term():
    answer = policy_answer(TOPIC_FINANCING, "osakiusa.com")
    assert "Affirm" in answer
    assert "%" not in answer
    assert "apr" not in answer.lower()


def test_international_does_not_promise_us_curbside():
    answer = policy_answer(TOPIC_INTERNATIONAL, "osakiusa.com").lower()
    assert "canada" in answer and "mexico" in answer
    assert "published" in answer
    assert "included shipping" not in answer


def test_office_is_recommend_not_commercial():
    assert detect_topic("for my office") is None
    assert classify("for my office").label == INTENT_RECOMMEND


def test_limits_kinds_stay_honest(monkeypatch):
    pets = policy_answer(TOPIC_LIMITS, "osakiusa.com", message="is it pet friendly")
    assert "pet-proof" in pets.lower()
    spanish = policy_answer(TOPIC_LIMITS, "osakiusa.com", message="do you speak Spanish")
    assert "english" in spanish.lower()
    paypal = policy_answer(TOPIC_LIMITS, "osakiusa.com", message="do you take PayPal")
    assert "affirm" in paypal.lower()
    assert "paypal" not in paypal.lower()
    hsa = policy_answer(TOPIC_LIMITS, "osakiusa.com", message="can I use HSA")
    assert "medical" in hsa.lower()
    affiliate = policy_answer(
        TOPIC_LIMITS, "osakiusa.com", message="do you have an affiliate program"
    )
    assert "affiliate" in affiliate.lower()
    checkout = policy_answer(
        TOPIC_LIMITS,
        "osakiusa.com",
        message="hello, i am trying to make a purchase but my amex is not going through",
    )
    assert "amex" not in checkout.lower()
    assert "checkout" in checkout.lower() or "card" in checkout.lower()
    friend = policy_answer(
        TOPIC_LIMITS,
        "osakiusa.com",
        message="can tou look up the one my friend has",
    )
    assert "catalog" in friend.lower()
    assert "model name" in friend.lower()


def test_hawaii_zip_is_remote_freight_not_included_shipping():
    answer = policy_answer(
        TOPIC_REMOTE_SHIPPING, "osakiusa.com", message="96701 is my zip"
    ).lower()
    assert "hawaii" in answer or "alaska" in answer or "freight" in answer
    assert "you pay" in answer
    assert "included shipping does not" in answer


def test_tablet_install_does_not_invent_a_tablet_service():
    answer = policy_answer(
        TOPIC_WHITE_GLOVE, "osakiusa.com", message="tablet instalations"
    ).lower()
    assert "white glove" in answer
    assert "tablet-install" in answer or "tablet install" in answer
    assert "3 weeks" in answer


def test_want_a_new_chair_is_not_refurbished():
    assert detect_topic("I want a new chair") is None
    assert classify("I want a new chair").label != INTENT_PREPURCHASE_POLICY


@pytest.mark.parametrize("topic", POLICY_TOPICS)
def test_no_answer_invents_a_delivery_promise(topic):
    answer = policy_answer(topic, "osakiusa.com").lower()
    for banned in ("business days", "arrives in", "guaranteed by", "next day"):
        assert banned not in answer


# ---------------------------------------------------------------------------
# End-to-end through the agent
# ---------------------------------------------------------------------------


def test_agent_answers_return_policy_without_handoff(monkeypatch):
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    from sales_agent import respond

    reply = respond("what is your return policy", domain="osakiusa.com")
    assert reply.intent == INTENT_PREPURCHASE_POLICY
    assert reply.handoff is False
    assert "30 days" in reply.reply
    assert "policy.returns" in reply.tools_used


def test_agent_still_hands_off_a_broken_chair(monkeypatch):
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    from sales_agent import respond

    reply = respond("my chair is broken", domain="osakiusa.com")
    assert reply.handoff is True
    assert reply.intent != INTENT_PREPURCHASE_POLICY
