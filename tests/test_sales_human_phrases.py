"""Customer-requested escalation must work beyond the exact word 'human'."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import sales_agent as agent
from sales_intent import (
    INTENT_CANCEL_REFUND,
    INTENT_HUMAN,
    INTENT_ORDER_STATUS,
    INTENT_PARTS_TECHNICIAN,
    INTENT_PREPURCHASE_POLICY,
    INTENT_WARRANTY_REDIRECT,
    classify,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "fetch_live_stock", lambda *args, **kwargs: None)


@pytest.mark.parametrize("message", [
    "I don't want a questionnaire. I'd like a real sales person.",
    "No more bot answers. Connect me now.",
    "Could you connect me with someone who can confirm?",
    "Please have someone follow up tomorrow rather than promise a delivery date.",
    "Please ask a sales representative to contact me about delivery.",
    "Can you get somebody to call me tomorrow?",
    "A live sales person please.",
])
def test_explicit_person_request_interrupts_a_fit_interview(message):
    reply = agent.respond(message, prefs={
        "awaiting_recommend": True,
        "recommend_prefs": {"budget_max_usd": "4000"},
    })
    assert reply.intent == INTENT_HUMAN
    assert reply.handoff is True
    assert not reply.products
    assert "user height" not in reply.reply.lower()
    assert "not yet confirmed" in reply.reply.lower()
    assert any(quick.payload in {"human", "human:confirm", "lead:email"} for quick in reply.quick_replies)


def test_callback_request_interrupts_email_capture_without_claiming_it_was_sent():
    reply = agent.respond(
        "Could you connect me with someone who can confirm?",
        prefs={"awaiting_email_for_pick": True, "pending_primary": "Maestro 4D"},
    )
    assert reply.intent == INTENT_HUMAN
    assert reply.handoff is True
    assert not reply.lead_capture
    assert reply.prefs_patch["awaiting_email_for_pick"] is False
    assert "not yet confirmed" in reply.reply.lower()


@pytest.mark.parametrize("message", [
    "Please don't connect me now; compare the Maestro and Paragon.",
    "Don't have someone follow up tomorrow. Compare the Maestro and Paragon.",
    "I do not want a real sales person. Just compare the Maestro and Paragon.",
    "Do not call me. What is the price of the Maestro?",
])
def test_local_negation_does_not_create_a_human_request(message):
    assert classify(message).label != INTENT_HUMAN


@pytest.mark.parametrize("message,label", [
    ("Please cancel my order and connect me now.", INTENT_CANCEL_REFUND),
    ("I need a replacement remote; I'd like a real sales person.", INTENT_PARTS_TECHNICIAN),
    ("My remote is not working. Connect me now.", INTENT_WARRANTY_REDIRECT),
    ("Where is my order? Could you connect me with someone?", INTENT_ORDER_STATUS),
])
def test_service_and_existing_order_requests_keep_their_specialist_route(message, label):
    assert classify(message).label == label


@pytest.mark.parametrize("message", [
    "What are your shipping policies?",
    "How long does delivery take?",
    "What are your showroom hours?",
])
def test_policy_questions_without_a_person_request_still_get_an_answer(message):
    reply = agent.respond(message)
    assert reply.intent == INTENT_PREPURCHASE_POLICY
    assert reply.handoff is False


def test_medical_safety_question_preserves_clinician_guidance():
    reply = agent.respond("I have a pacemaker. Could you connect me with someone who can confirm it is safe?")
    assert reply.intent == INTENT_PREPURCHASE_POLICY
    assert "clinician" in reply.reply.lower()
    assert not reply.products
