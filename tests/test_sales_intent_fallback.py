"""
tests/test_sales_intent_fallback.py
===================================
Second-chance routing for messages the regex classifier calls ``unclear``.

Production showed 27% of turns landing on ``unclear``. These tests lock in the
recovery behaviour and, just as importantly, the guardrail that the fallback
only ever picks a *route* — never customer-facing text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_intent import INTENT_RECOMMEND, INTENT_SPECS, INTENT_UNCLEAR, classify  # noqa: E402
from sales_intent_fallback import (  # noqa: E402
    llm_fallback,
    named_model_in_text,
    rule_fallback,
    resolve_unclear,
)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Keep these tests offline and deterministic."""
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Named-model detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "tell me about the Maestro",
        "is the highpointe any good?",
        "thoughts on OS-Champ",
        "achilles or not",
    ],
)
def test_named_model_routes_to_specs(message):
    assert classify(message).label in {INTENT_UNCLEAR, INTENT_SPECS}
    result = rule_fallback(message)
    assert result is not None, f"{message!r} should recover"
    assert result.label == INTENT_SPECS


def test_bare_maestro_token_resolves_to_the_canonical_4d():
    """The token index used to return Maestro LE because it appeared first."""
    assert named_model_in_text("tell me about the Maestro") == "Osaki OS-Pro Maestro 4D"
    assert named_model_in_text("Maestro LE") == "Osaki OS-Pro Maestro LE"


@pytest.mark.parametrize(
    "message",
    [
        "do you have a cover?",
        "is it a recliner?",
        "made in japan?",
        "what about the premium package",
    ],
)
def test_ambiguous_english_words_are_not_treated_as_models(message):
    """Catalog tokens that are ordinary words must not resolve a product."""
    assert named_model_in_text(message) is None


# ---------------------------------------------------------------------------
# Browse / choose-help phrasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "help me pick one",
        "not sure which one to get",
        "what models do you have",
        "show me your chairs",
        "I'm a first time buyer",
        "where do I start",
        "too many options",
    ],
)
def test_browse_help_routes_to_recommend(message):
    result = rule_fallback(message)
    assert result is not None, f"{message!r} should recover"
    assert result.label == INTENT_RECOMMEND


@pytest.mark.parametrize(
    "message",
    ["tell me more", "how does it work", "more details please"],
)
def test_tell_me_more_routes_to_specs(message):
    result = rule_fallback(message)
    assert result is not None
    assert result.label == INTENT_SPECS


# ---------------------------------------------------------------------------
# Genuine no-ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    ["", "   ", "asdfghjkl", "what's the weather in Dallas"],
)
def test_unroutable_text_stays_unclear(message):
    assert rule_fallback(message) is None
    assert resolve_unclear(message) is None


def test_llm_fallback_is_a_noop_without_a_key():
    assert llm_fallback("something oddly phrased") is None


def test_llm_fallback_disabled_by_env(monkeypatch):
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    assert llm_fallback("something oddly phrased") is None


# ---------------------------------------------------------------------------
# LLM layer contract (stubbed — no network)
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, payload: str):
        self._payload = payload
        self.seen: dict = {}

    def create(self, **kwargs):
        self.seen.update(kwargs)

        class _Msg:
            content = self._payload

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self, payload: str):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(payload)})()


def _patch_client(monkeypatch, payload: str) -> _FakeClient:
    monkeypatch.setenv("SALES_INTENT_LLM", "1")
    client = _FakeClient(payload)
    monkeypatch.setattr("sales_intent_fallback._openai_client", lambda: client)
    return client


def test_llm_high_confidence_label_is_accepted(monkeypatch):
    _patch_client(monkeypatch, '{"label": "price", "confidence": "high"}')
    result = llm_fallback("ballpark on the fancy one")
    assert result is not None
    assert result.label == "price"
    assert result.matched_terms == ("llm_fallback",)


def test_llm_low_confidence_is_rejected(monkeypatch):
    _patch_client(monkeypatch, '{"label": "price", "confidence": "low"}')
    assert llm_fallback("ballpark on the fancy one") is None


def test_llm_cannot_invent_a_label(monkeypatch):
    """A label outside the closed set must be discarded, not passed through."""
    _patch_client(monkeypatch, '{"label": "free_shipping_promise", "confidence": "high"}')
    assert llm_fallback("can you promise it by friday") is None


def test_llm_cannot_return_unclear_as_a_route(monkeypatch):
    _patch_client(monkeypatch, '{"label": "unclear", "confidence": "high"}')
    assert llm_fallback("hmm") is None


def test_llm_handoff_label_keeps_handoff_flag(monkeypatch):
    _patch_client(monkeypatch, '{"label": "warranty_redirect", "confidence": "high"}')
    result = llm_fallback("the thing I bought last year makes a grinding sound")
    assert result is not None
    assert result.is_handoff is True


def test_llm_malformed_json_is_a_noop(monkeypatch):
    _patch_client(monkeypatch, "not json at all")
    assert llm_fallback("hmm") is None


def test_llm_is_only_asked_for_a_route(monkeypatch):
    """The prompt must forbid answering, so no fact can come from the model."""
    client = _patch_client(monkeypatch, '{"label": "specs", "confidence": "high"}')
    llm_fallback("what's it like")
    system = client.chat.completions.seen["messages"][0]["content"].lower()
    assert "never answer the customer" in system
    assert client.chat.completions.seen["temperature"] == 0
