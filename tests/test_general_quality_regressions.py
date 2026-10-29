"""Adversarial customer examples keep routing and factual claims grounded."""
import json

import pytest

from app.answer_guard import sanitize_agent_response
from app.scope_classifier import evaluate_scope


@pytest.mark.parametrize("query", [
    "Write me a poem about cats",
    "What's the weather in Dallas today?",
    "Give me a pasta recipe under $100",
])
def test_unrelated_request_does_not_inherit_chair_scope(query, monkeypatch):
    monkeypatch.setenv("SCOPE_CLASSIFIER_LLM", "0")
    history = [{"role": "user", "content": "Recommend an Osaki massage chair"}]
    assert evaluate_scope(query, history).is_blocked


@pytest.mark.parametrize("query", [
    "Can I watch movies in this chair?",
    "The remote shows error code E12",
    "What does the installation guide say about the power cable?",
    "under $3000",
])
def test_related_requests_and_followups_remain_in_scope(query, monkeypatch):
    monkeypatch.setenv("SCOPE_CLASSIFIER_LLM", "0")
    assert evaluate_scope(query, [{"role": "user", "content": "Osaki chair"}]).in_scope


@pytest.mark.parametrize("price", [
    "USD 99", "99 USD", "99 dollars", "US$99", "99달러", "€99", "99 EUR", "₩99000", "CA$99", "C$99", "A$99",
])
def test_currency_spelling_cannot_bypass_catalog_grounding(price):
    text = f"Fixture Alpha costs {price}."
    output = sanitize_agent_response(text, tools_called=[], user_query="price")
    assert price not in output


@pytest.mark.parametrize("price", ["USD 3,000", "3,000 USD", "3000 dollars", "US$3,000", "3000달러"])
def test_equivalent_verified_usd_spellings_are_preserved(price):
    text = f"Fixture Alpha costs {price}."
    blob = "MODEL_FACTS_JSON: " + json.dumps({
        "model": "Fixture Alpha", "currency": "USD", "price_usd": 3000, "specs": {},
    })
    assert sanitize_agent_response(text, tools_called=["search_chair_specs"], user_query="price", tool_results=[blob]) == text


@pytest.mark.parametrize("price", ["USD 99", "99 USD", "99 dollars", "99달러", "€3000", "₩3000", "CA$3000", "C$3000", "A$3000"])
def test_price_and_currency_remain_bound_to_authoritative_model(price):
    text = f"Fixture Alpha costs {price}."
    blob = "MODEL_FACTS_JSON: " + json.dumps({
        "model": "Fixture Alpha", "currency": "USD", "price_usd": 3000, "specs": {},
    })
    output = sanitize_agent_response(text, tools_called=["search_chair_specs"], user_query="price", tool_results=[blob])
    assert "accurate information" in output


@pytest.mark.parametrize("query,expected", [
    ("영어로 답해주세요", "en"),
    ("스페인어로 답해주세요", "es"),
    ("Responde en inglés", "en"),
    ("한국어 말고 영어로 답해주세요", "en"),
])
def test_explicit_language_preference_applies_to_general_chat_and_followups(query, expected):
    from app.answer_guard import conversation_language
    assert conversation_language(query) == expected
    assert conversation_language("3000", [{"role": "user", "content": query}]) == expected


@pytest.mark.parametrize("query,expected", [
    ("Track order OSKMC1234 customer@example.com; is it near your showroom?", "lookup_order_status"),
    ("My chair is broken; can your showroom help?", "start_warranty_workflow"),
    ("The remote is not working; should I bring it to the showroom?", "get_repair_help"),
    ("Where is your showroom?", "get_showroom_info"),
])
def test_secondary_showroom_reference_does_not_displace_primary_support_intent(query, expected):
    from app.intent_router import infer_forced_tool
    assert infer_forced_tool(query) == expected


@pytest.mark.parametrize("candidate", ["Fixture Alpha costs$99.", "The price is$99."])
def test_symbol_prices_remain_detected_when_customer_copy_omits_space(candidate):
    output = sanitize_agent_response(candidate, tools_called=[], user_query="price")
    assert "$99" not in output


def test_verified_symbol_price_without_space_is_preserved():
    candidate = "Fixture Alpha costs$3,000."
    blob = "MODEL_FACTS_JSON: " + json.dumps({
        "model": "Fixture Alpha", "currency": "USD", "price_usd": 3000, "specs": {},
    })
    assert sanitize_agent_response(candidate, tools_called=["search_chair_specs"], user_query="price", tool_results=[blob]) == candidate
