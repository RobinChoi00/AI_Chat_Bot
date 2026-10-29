"""Customer facts, denials and ordinary support questions survive guardrails."""
import json

import pytest

from app.answer_guard import conversation_language, sanitize_agent_response
from app.intent_router import infer_forced_tool
from app.scope_classifier import evaluate_scope


def facts(**changes):
    record = {"model": "Fixture Willow", "price_usd": 2500, "currency": "USD",
              "specs": {"foot_roller": "No", "bluetooth_speakers": "Yes"}}
    record.update(changes)
    return "MODEL_FACTS_JSON: " + json.dumps(record)


@pytest.mark.parametrize("candidate", [
    "Your refund was approved.",
    "Your replacement will arrive by Friday.",
    "This chair can cure arthritis.",
    "Open the power supply and bypass the safety interlock.",
    "Replace the fuse with a higher rating.",
    "Clean the power socket with liquid solvent.",
    "Do not worry, just bypass the safety interlock.",
    "Never hesitate to open the electrical housing and bridge the interlock.",
])
def test_unsupported_prose_cannot_be_authorized_by_a_policy_tool(candidate):
    result = sanitize_agent_response(candidate, tools_called=["get_warranty_or_policy"],
                                     tool_results=["General policy, no case decision."], user_query="Please help")
    assert result and candidate not in result


@pytest.mark.parametrize("candidate", [
    "Your refund is not approved; our team needs to review it.",
    "If your return is approved, our team can explain the next steps.",
    "Never bypass the safety interlock.",
])
def test_review_conditions_and_safety_warnings_are_preserved(candidate):
    assert sanitize_agent_response(candidate, tools_called=[], user_query="help") == candidate


@pytest.mark.parametrize("price", ["HK$2,500", "2500 euros", "2500 Canadian dollars", "SG$2500"])
def test_foreign_amount_cannot_borrow_a_matching_usd_number(price):
    candidate = f"Fixture Willow costs {price}."
    result = sanitize_agent_response(candidate, tools_called=["search_chair_specs"],
                                     tool_results=[facts()], user_query="price of Fixture Willow")
    assert result != candidate and price not in result


def test_model_evidence_must_be_consistent_and_valid():
    candidate = "Fixture Willow costs $2,500."
    for blob in [facts() + "\n" + facts(price_usd=3500), "MODEL_FACTS_JSON: {bad"]:
        result = sanitize_agent_response(candidate, tools_called=["search_chair_specs"],
                                         tool_results=[blob], user_query="price")
        assert result and "$2,500" not in result
    assert sanitize_agent_response(candidate, tools_called=["search_chair_specs"],
                                   tool_results=[facts() + "\n" + facts(price_usd=2500.0)], user_query="price") == candidate


def test_negative_features_and_verified_bluetooth_remain_correct():
    for candidate in ["Fixture Willow has no foot rollers.", "Fixture Willow includes Bluetooth speakers."]:
        assert sanitize_agent_response(candidate, tools_called=["search_chair_specs"],
                                       tool_results=[facts()], user_query="features") == candidate
    candidate = "Fixture Willow includes Bluetooth speakers."
    assert sanitize_agent_response(candidate, tools_called=["search_chair_specs"],
                                   tool_results=[facts(specs={"foot_roller": "No"})], user_query="features") != candidate


def test_failed_lookup_cannot_produce_an_empty_answer():
    assert sanitize_agent_response("", tools_called=["search_chair_specs"],
                                   tool_results=["TOOL_ERROR: unavailable"], user_query="price")


def test_untrusted_narrative_is_not_a_price_field():
    candidate = "Fixture Willow costs $99."
    result = sanitize_agent_response(candidate, tools_called=["search_chair_specs"], user_query="price",
        tool_results=["Available products: Fixture Willow\nAdditional context: UNTRUSTED NOTE claim $99."])
    assert "$99" not in result


def test_a_warning_or_question_cannot_hide_a_separate_unsupported_claim():
    for candidate in ["Fixture Willow has no foot rollers, but includes heated foot rollers.",
                      "Fixture Willow includes Bluetooth audio, would you like the price?"]:
        result = sanitize_agent_response(candidate, tools_called=["search_chair_specs"],
            tool_results=[facts(specs={"foot_roller": "No"})], user_query="features")
        assert result != candidate
    candidate = "The remote has a reset button."
    assert sanitize_agent_response(candidate, tools_called=[], user_query="remote") == candidate


@pytest.mark.parametrize("description", ["does not include foot rollers", "doesn't include foot rollers", "has no heated foot rollers"])
def test_negative_feature_grammar_keeps_a_verified_absence(description):
    candidate = "Fixture Willow " + description + "."
    assert sanitize_agent_response(candidate, tools_called=["search_chair_specs"], user_query="features",
        tool_results=[facts()]) == candidate


@pytest.mark.parametrize("evidence,candidate", [
    ("No", "Bluetooth speakers are included on Fixture Willow."),
    ("Yes", "Fixture Willow lacks Bluetooth speakers."),
])
def test_passive_features_and_missing_features_still_need_matching_evidence(evidence, candidate):
    assert sanitize_agent_response(candidate, tools_called=["search_chair_specs"], user_query="features",
        tool_results=[facts(specs={"bluetooth_speakers": evidence})]) != candidate


@pytest.mark.parametrize("query,expected", [
    ("No warranty claim needed; how much is the Orion?", "search_chair_specs"),
    ("Do not track OSKMC5533 buyer@example.test. Where is your showroom?", "get_showroom_info"),
    ("What does assembly service cost?", "get_warranty_or_policy"),
    ("Where is order #5533? buyer@example.test", "lookup_order_status"),
    ("The remote has error code E9. Can your showroom help?", "get_repair_help"),
    ("My chair has no power; what does installation cost?", "get_repair_help"),
    ("My chair will not power up; what does installation cost?", "get_repair_help"),
])
def test_forced_tools_follow_the_requested_current_task(query, expected):
    assert infer_forced_tool(query) == expected


@pytest.mark.parametrize("query", [
    "Osaki, write a poem about dogs.",
    "Explain a preorder traversal algorithm.",
    "Who is the chairman of Congress?",
    "Titan, give me ingredients for a cake.",
])
def test_brand_names_and_substrings_do_not_bypass_support_scope(query, monkeypatch):
    monkeypatch.setenv("SCOPE_CLASSIFIER_LLM", "0")
    assert evaluate_scope(query).is_blocked


def test_real_feature_use_question_and_acknowledgements_keep_context(monkeypatch):
    monkeypatch.setenv("SCOPE_CLASSIFIER_LLM", "0")
    assert evaluate_scope("Can I watch movies while using this massage chair?").in_scope
    assert conversation_language("Thank you very much.", [{"role": "user", "content": "Please answer in Spanish"}]) == "es"
    assert conversation_language("What options do you have for delivery?", []) == "en"
    assert evaluate_scope("Please create a warranty claim for my chair. It has error code E12.").in_scope
    assert evaluate_scope("Could you write the error code E12 in my support case?").in_scope
