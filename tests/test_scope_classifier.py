"""Tests for scope_classifier — off-topic pre-filter."""

import sys
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from scope_classifier import build_scope_refusal, evaluate_scope


def test_blocks_obvious_off_topic():
    decision = evaluate_scope("Write me a Python script to scrape websites")
    assert decision.is_blocked
    assert decision.reason == "rule_out_of_scope"


def test_allows_chair_question():
    decision = evaluate_scope("How much is the Osaki Solo Flex?")
    assert decision.in_scope
    assert not decision.is_blocked


def test_allows_greeting():
    decision = evaluate_scope("Hello!")
    assert decision.in_scope


def test_allows_short_follow_up_with_chair_history():
    history = [
        {"role": "user", "content": "Tell me about Hypnos 4D specs"},
        {"role": "assistant", "content": "Here are the dimensions..."},
    ]
    decision = evaluate_scope("price", history)
    assert decision.in_scope


def test_blocks_weather_without_chair_context():
    decision = evaluate_scope("What's the weather in Dallas today?")
    assert decision.is_blocked


def test_refusal_message_mentions_specialization():
    msg = build_scope_refusal()
    assert "massage chair" in msg.lower()
    assert "business hours" in msg.lower()


def test_refusal_message_is_localized_for_korean():
    msg = build_scope_refusal("파이썬 코드를 작성해 주세요")
    assert "마사지 의자" in msg
    assert "운영 시간" in msg


def test_refusal_message_is_localized_for_spanish():
    msg = build_scope_refusal("Escribe un poema, por favor")
    assert "sillas de masaje" in msg
    assert "Horario" in msg


def test_skip_flag_bypasses_classifier():
    decision = evaluate_scope("Write me a poem about cats", skip=True)
    assert decision.in_scope
    assert decision.reason == "disabled"


def test_llm_blocks_when_high_confidence_out_of_scope():
    with patch("scope_classifier._llm_classify", return_value=False):
        decision = evaluate_scope("Something ambiguous maybe")
    assert decision.is_blocked
    assert decision.used_llm


def test_llm_allows_when_high_confidence_in_scope():
    with patch("scope_classifier._llm_classify", return_value=True):
        decision = evaluate_scope("Something ambiguous maybe")
    assert decision.in_scope
    assert decision.used_llm
