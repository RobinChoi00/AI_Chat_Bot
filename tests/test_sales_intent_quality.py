"""Deidentified regressions for observed sales conversation routing failures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_intent import (  # noqa: E402
    HANDOFF_INTENTS,
    WARRANTY_ROUTE_INTENTS,
    INTENT_CANCEL_REFUND,
    INTENT_COMPARE,
    INTENT_GREETING,
    INTENT_HUMAN,
    INTENT_ORDER_CHANGE,
    INTENT_ORDER_STATUS,
    INTENT_PARTS_TECHNICIAN,
    INTENT_PREPURCHASE_POLICY,
    INTENT_RECOMMEND,
    INTENT_SPECS,
    INTENT_WARRANTY_REDIRECT,
    classify,
    handoff_message,
)
from sales_policy import (  # noqa: E402
    TOPIC_LIMITS,
    TOPIC_MECHANISM,
    TOPIC_SHOWROOM,
    detect_topic,
    policy_answer,
)


@pytest.mark.parametrize("message", [
    "I have an Osaki 4000XT that I need a replacement remote for, but I also "
    "found a connection that doesn't seem to have a mate to connect to",
    "foot rest replacement for model OS-Hiro LT",
    "Can I replace the remote control for my chair?",
    "I need a spare tablet for an OS-Hiro LT",
])
def test_existing_service_parts_do_not_become_chair_shopping(message):
    intent = classify(message)
    assert intent.label == INTENT_PARTS_TECHNICIAN
    assert intent.is_handoff


def test_positive_purchase_feedback_is_acknowledged_without_service_ticket():
    intent = classify("I already purchased a chair and I love it!")
    assert intent.label == INTENT_GREETING
    assert intent.matched_terms == ("positive_feedback",)
    assert not intent.is_handoff


@pytest.mark.parametrize("message", [
    "I already purchased a chair and I love it, but the remote is broken",
    "I already purchased a chair. Thanks, but I need a refund",
    "I already purchased a chair and love it. The remote stopped working",
    "I already purchased a chair and love it. Please send a user manual",
])
def test_praise_does_not_hide_a_real_service_request(message):
    assert classify(message).is_handoff
    assert classify(message).label != INTENT_GREETING


@pytest.mark.parametrize("message", [
    "Does Titan TP-Epic 4D have foot rollers?",
    "Does the Osaki Maestro 4D massage chair have calf rollers?",
    "How heavy is Osaki titan 4d massage chair?",
    "How much does the Titan 4D massage chair weigh?",
    "What is the weight capacity of the Osaki 4D massage chair?",
    "What are the dimensions of the Titan 4D massage chair?",
])
def test_direct_product_feature_questions_are_specs_not_fit_interviews(message):
    assert detect_topic(message) is None
    intent = classify(message)
    assert intent.label == INTENT_SPECS
    assert not intent.is_handoff


@pytest.mark.parametrize("message", [
    "What is 4D massage?",
    "Explain the difference between 3D and 4D",
    "2D vs 4D massage chairs",
])
def test_actual_mechanism_explanations_are_preserved(message):
    assert detect_topic(message) == TOPIC_MECHANISM


def test_body_preferences_still_enter_recommendation():
    assert classify("I'm 6 feet 2, 350 pounds, and want a deep massage").label == INTENT_RECOMMEND


@pytest.mark.parametrize("message", [
    "Where to try this chair before buying?",
    "Where can I try the Maestro 4D massage chair?",
    "Is there a dealer in Hawaii where I can try it?",
])
def test_try_before_buying_is_showroom_even_with_model_or_shipping_region(message):
    assert detect_topic(message) == TOPIC_SHOWROOM
    assert classify(message).label == INTENT_PREPURCHASE_POLICY
    answer = policy_answer(TOPIC_SHOWROOM, "osakiusa.com", message=message)
    assert "Carrollton" in answer
    if "dealer" in message:
        assert "don't have a verified local dealer list" in answer


@pytest.mark.parametrize("message", [
    "can i use FSA",
    "Can I pay with HSA?",
    "Is this covered by flexible spending?",
    "Will insurance cover the massage chair?",
])
def test_benefit_payment_question_gets_eligibility_answer_without_inventing_policy(message):
    assert detect_topic(message) == TOPIC_LIMITS
    assert classify(message).label == INTENT_PREPURCHASE_POLICY
    answer = policy_answer(TOPIC_LIMITS, "osakiusa.com", message=message).lower()
    assert "don't have verified" in answer
    assert "plan administrator" in answer
    assert "can't promise reimbursement" in answer
    assert "won't give medical advice" not in answer
    assert "clinician" not in answer


@pytest.mark.parametrize("message", [
    "Is it safe with a pacemaker?",
    "Is the chair safe during pregnancy?",
    "I'm pregnant. Can I use FSA and is a massage chair safe for me?",
])
def test_real_medical_safety_questions_keep_clinical_caution(message):
    assert detect_topic(message) == TOPIC_LIMITS
    answer = policy_answer(TOPIC_LIMITS, "osakiusa.com", message=message).lower()
    assert "clinician" in answer


@pytest.mark.parametrize("message", [
    "I just ordered an OS-3D Ai Vito today but forgot to add white glove delivery. "
    "Is it too late to add it?",
    "I've already ordered; can I upgrade to white glove?",
    "I have ordered a chair and need to change my delivery address",
    "I have ordered a chair; can you change my order to add white-glove delivery?",
])
def test_existing_order_upgrade_is_sales_review_not_prepurchase_policy(message):
    intent = classify(message)
    assert intent.label == INTENT_ORDER_CHANGE
    assert intent.is_handoff
    assert INTENT_ORDER_CHANGE in HANDOFF_INTENTS
    assert INTENT_ORDER_CHANGE not in WARRANTY_ROUTE_INTENTS
    answer = handoff_message(intent).lower()
    assert "order number" in answer and "email" in answer
    assert "no delivery upgrade or order change has been confirmed" in answer


def test_order_tracking_remains_in_its_existing_flow():
    assert classify("Where is my order? Please send the tracking number").label == INTENT_ORDER_STATUS


def test_cancellation_is_not_overridden_by_mentioning_an_order_upgrade():
    assert classify(
        "I forgot to add white glove to my order. Please cancel my order instead"
    ).label == INTENT_CANCEL_REFUND


@pytest.mark.parametrize("message", [
    "No thanks, I do not want an agent",
    "I don’t need an agent; just compare chairs",
    "I don't want to talk to a human; compare these models",
    "I do not want to cancel my order, just compare chairs",
    "Please don't cancel my order. Compare the Vito and Epic instead",
])
def test_negated_human_or_cancellation_requests_do_not_trigger_the_opposite(message):
    intent = classify(message)
    assert intent.label not in {INTENT_HUMAN, INTENT_CANCEL_REFUND}
    assert not intent.is_handoff
    if "compare" in message.lower():
        assert intent.label == INTENT_COMPARE


@pytest.mark.parametrize(("message", "label"), [
    ("I do not want an agent, but I do want to cancel my order", INTENT_CANCEL_REFUND),
    ("Do not cancel my order, but please connect me to a human", INTENT_HUMAN),
    ("I don't want to cancel my order, but I need a refund", INTENT_CANCEL_REFUND),
    ("My remote is not working; I do not want an agent", INTENT_WARRANTY_REDIRECT),
])
def test_negation_is_local_and_does_not_suppress_a_separate_positive_request(message, label):
    assert classify(message).label == label
