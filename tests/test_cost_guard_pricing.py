"""Usage costs cannot claim that an unpriced model or missing usage is free."""
from types import SimpleNamespace

import pytest

from app import cost_guard


def response(prompt=1000, cached=250, completion=200):
    return SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    ))


def test_known_model_applies_its_configured_cached_input_price():
    # Existing gpt-4.1-mini rates are .40 input, .10 cached and 1.60 output.
    assert cost_guard.estimate_cost_usd("gpt-4.1-mini", 1000, 250, 200) == pytest.approx(.000645)


@pytest.mark.parametrize("model", [
    "gpt-6.1-sol", "gpt-6-luna", "unknown-model",
    "gpt-4.1-mini-2025-04-14", "gpt-4.1-mini-custom",
])
def test_unlisted_model_names_do_not_inherit_a_guessed_price(model):
    assert cost_guard.estimate_cost_usd(model, 1000, 250, 200) is None


def test_unpriced_usage_preserves_tokens_and_unknown_cost_in_json_summary():
    recorder = cost_guard.UsageRecorder("gpt-6.1-sol")
    recorder.record(response())
    summary = recorder.summary_dict()
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 1000
    assert summary["cached_tokens"] == 250
    assert summary["completion_tokens"] == 200
    assert summary["estimated_cost_usd"] is None
    assert summary["estimated_cost_available"] is False


def test_missing_usage_does_not_turn_a_known_model_call_into_a_free_call():
    recorder = cost_guard.UsageRecorder("gpt-4.1-mini")
    recorder.record(response())
    recorder.record(SimpleNamespace(usage=None))
    summary = recorder.summary_dict()
    assert summary["calls"] == 2
    assert summary["prompt_tokens"] == 1000
    assert summary["estimated_cost_usd"] is None
    assert summary["estimated_cost_available"] is False


def test_no_model_calls_have_a_known_zero_cost_even_when_model_is_unpriced():
    recorder = cost_guard.UsageRecorder("gpt-6.1-sol")
    assert recorder.summary_dict()["calls"] == 0
    assert recorder.summary_dict()["estimated_cost_usd"] == 0.0
    assert recorder.summary_dict()["estimated_cost_available"] is True


def test_priced_usage_and_real_zero_token_usage_remain_available():
    recorder = cost_guard.UsageRecorder("gpt-4.1-mini")
    recorder.record(response())
    recorder.record(response(prompt=0, cached=0, completion=0))
    assert recorder.summary_dict()["estimated_cost_usd"] == .000645
    assert recorder.summary_dict()["estimated_cost_available"] is True
