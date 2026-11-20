from __future__ import annotations

import pytest

from sales_intent import (
    INTENT_CANCEL_REFUND,
    INTENT_PARTS_TECHNICIAN,
    INTENT_PREPURCHASE_POLICY,
    INTENT_STOCK,
    INTENT_WARRANTY_REDIRECT,
    classify,
)
from sales_policy import TOPIC_TECHNICAL_SERVICE, detect_topic, reports_owned_chair_problem


@pytest.mark.parametrize('question', [
    'This chair is already in my house. It is broken. I need technical support, not a new chair.',
    'The chair is already in our home. We need technical support.',
    'I already have it and it stopped working. Where can I get technical support?',
    'This chair is faulty. What technical support can I get?',
    'I own this chair. I need technical help.',
    'The chair is in my living room. Who can fix it?',
    "I've had my Titan TP-Epic 4D in my house for two years. Its foot rollers no longer work. How can I get technical support?",
    'It is broken and I need technical support.',
])
def test_confirmed_fault_and_owner_service_requests_are_not_purchase_policy(question):
    assert reports_owned_chair_problem(question)
    assert detect_topic(question) is None
    intent = classify(question)
    assert intent.label == INTENT_WARRANTY_REDIRECT
    assert intent.is_handoff


@pytest.mark.parametrize('question', [
    'Before I buy, do you offer technical support if my chair breaks?',
    "Before I buy, do you offer technical support if the chair's remote is blank?",
    'If this chair is broken after delivery, what technical support do you offer?',
    'Once the chair is in my house, what technical support is available if it breaks?',
    'I am thinking of buying this chair; what technical service is included?',
    'I have a chair. It is not broken. What technical support comes with a second chair?',
    'My chair is not broken. I am considering a second chair. What technical support comes with it?',
])
def test_hypothetical_or_negated_fault_preserves_prebuy_service_policy(question):
    assert not reports_owned_chair_problem(question)
    assert detect_topic(question) == TOPIC_TECHNICAL_SERVICE
    intent = classify(question)
    assert intent.label == INTENT_PREPURCHASE_POLICY
    assert not intent.is_handoff


def test_negated_support_request_does_not_override_inventory():
    question = "I don't need technical support. Is the Osaki OS-Champ in stock?"
    assert not reports_owned_chair_problem(question)
    assert detect_topic(question) is None
    assert classify(question).label == INTENT_STOCK


@pytest.mark.parametrize('description', ['is not broken', "isn't defective", 'is no longer stuck'])
def test_negated_fault_description_does_not_override_other_chair_inventory(description):
    question = f'My chair {description}. Is the Osaki OS-Champ in stock?'
    assert not reports_owned_chair_problem(question)
    assert classify(question).label == INTENT_STOCK


def test_reported_fault_does_not_override_cancellation_or_parts_request():
    assert classify('This chair is broken. Please cancel my order.').label == INTENT_CANCEL_REFUND
    assert classify('This chair is broken. I need a replacement remote.').label == INTENT_PARTS_TECHNICIAN
