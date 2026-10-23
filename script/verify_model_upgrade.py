#!/usr/bin/env python3
"""Bounded, opt-in provider compatibility probe with synthetic inputs only.

Production deployment runs this in the candidate container before replacing the
running services. No application startup, customer records, email, order APIs,
or customer prompts are used. The report contains only model IDs, usage, timing,
and symbolic failure codes. Passing establishes access/compatibility, not a
measurement of customer-answer quality.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
PROFILE_KEYS = frozenset({"OPENAI_AGENT_MODEL", "OPENAI_ROUTER_MODEL"})
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


class ProbeFailure(RuntimeError):
    """Failure code only; never retain a provider response or credential."""


def load_profile(path: Path) -> dict[str, str]:
    """Read only the checked-in, non-secret two-model release profile."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ProbeFailure("model_profile_unreadable") from None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in PROFILE_KEYS or key in values or not MODEL_ID.fullmatch(value):
            raise ProbeFailure("model_profile_invalid")
        values[key] = value
    if set(values) != PROFILE_KEYS:
        raise ProbeFailure("model_profile_incomplete")
    return values


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _provider_model(response: Any, expected: str) -> str:
    actual = _get(response, "model")
    # Provider aliases may resolve to a dated snapshot of the same named model.
    if not isinstance(actual, str) or not MODEL_ID.fullmatch(actual):
        raise ProbeFailure("provider_model_unreported")
    if actual != expected and not re.fullmatch(re.escape(expected) + r"-\d{4}-\d{2}-\d{2}", actual):
        raise ProbeFailure("provider_model_mismatch")
    return actual


def _observation(response: Any, *, expected: str, task: str, started: float) -> dict[str, Any]:
    usage = _get(response, "usage")
    return {
        "task": task,
        "requested_model": expected,
        "provider_model": _provider_model(response, expected),
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "input_tokens": _get(usage, "input_tokens", _get(usage, "prompt_tokens")),
        "output_tokens": _get(usage, "output_tokens", _get(usage, "completion_tokens")),
    }


def run_probe(client: Any, *, agent: str, router: str) -> dict[str, Any]:
    """At most four provider requests; caller supplies a zero-retry client."""
    from model_runtime import create_completion
    from app.responses_completion import ResponsesCompletionSession

    observations: list[dict[str, Any]] = []
    initial = [{"role": "user", "content": "Look up the price of fixture-chair with the provided lookup tool. This is a synthetic test."}]
    function = {
        "name": "fixture_lookup", "description": "Read synthetic fixture catalog data.",
        # Match the application's existing Chat schemas: omitted strict flag
        # and a genuinely optional field must survive Responses conversion.
        "parameters": {"type": "object", "properties": {"id": {"type": "string", "enum": ["fixture-chair"]}, "topic": {"type": "string", "enum": ["price"]}}, "required": ["id"], "additionalProperties": False},
    }
    tool = {"type": "function", "function": function}
    options = {
        "model": agent, "reasoning_effort": "medium", "verbosity": "low",
        "max_completion_tokens": 2048, "store": False, "parallel_tool_calls": False,
    }
    session = ResponsesCompletionSession(client, observer=lambda **_: None)
    started = time.monotonic()
    response = session.create(
        **options, messages=initial, tools=[tool], tool_choice={"type": "function", "function": {"name": "fixture_lookup"}},
    )
    observations.append(_observation(response, expected=agent, task="agent_tool_call", started=started))
    calls = response.choices[0].message.tool_calls or []
    if len(calls) != 1 or _get(_get(calls[0], "function"), "name") != "fixture_lookup":
        raise ProbeFailure("agent_tool_call_invalid")
    try:
        arguments = json.loads(_get(_get(calls[0], "function"), "arguments", ""))
    except (ValueError, TypeError):
        raise ProbeFailure("agent_tool_arguments_invalid") from None
    call_id = _get(calls[0], "id")
    if (not isinstance(arguments, dict) or arguments.get("id") != "fixture-chair"
        or set(arguments) - {"id", "topic"} or ("topic" in arguments and arguments["topic"] != "price")
        or not isinstance(call_id, str) or not call_id):
        raise ProbeFailure("agent_tool_arguments_invalid")
    continuation = [
        *initial,
        {"role": "assistant", "content": response.choices[0].message.content, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "fixture_lookup", "arguments": _get(_get(calls[0], "function"), "arguments")}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"model": "fixture-chair", "currency": "USD", "price": 1234})},
    ]
    started = time.monotonic()
    answer = session.create(**options, messages=continuation, tools=[tool], tool_choice="none")
    observations.append(_observation(answer, expected=agent, task="agent_grounded_reply", started=started))
    text = answer.choices[0].message.content
    if not isinstance(text, str) or not re.search(r"\b1,?234(?:\.00)?\b", text):
        raise ProbeFailure("agent_grounded_reply_invalid")
    for prompt, expected in (
        ("I want to talk to a person.", "human"),
        ("의자 전원이 켜지지 않아요.", "warranty_redirect"),
    ):
        started = time.monotonic()
        classification = create_completion(
            client, model=router, role="router", max_output_tokens=2048,
            reasoning_effort="low", json_mode=True, observer=lambda **_: None,
            messages=[
                {"role": "system", "content": 'Classify the customer text. Return JSON only with exactly these keys: {"label":"human"|"warranty_redirect","confidence":"high"|"low"}. Explicit requests for a person are human. A broken owned chair is warranty_redirect. Never answer the customer.'},
                {"role": "user", "content": prompt},
            ],
        )
        observations.append(_observation(classification, expected=router, task="router_" + expected, started=started))
        try:
            parsed = json.loads(classification.choices[0].message.content or "")
        except (ValueError, TypeError, IndexError, AttributeError):
            raise ProbeFailure("router_json_invalid") from None
        if parsed != {"label": expected, "confidence": "high"}:
            raise ProbeFailure("router_classification_invalid")
    return {"passed": True, "provider_requests": len(observations), "observations": observations, "customer_quality_evaluated": False, "customer_records_created": False, "customer_emails_sent": False}


def _record_probe_observations(result: dict[str, Any], recorder: Any = None) -> int:
    """Submit safe operational evidence only after the whole probe succeeds."""
    if result.get("passed") is not True:
        raise ProbeFailure("cannot_record_failed_probe")
    if recorder is None:
        from ai_quality import record_model_call
        recorder = record_model_call
    observations = result["observations"]
    for row in observations:
        recorder(role="agent" if row["task"].startswith("agent_") else "router",
                 requested_model=row["requested_model"], provider_model=row["provider_model"],
                 succeeded=True)
    return len(observations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Opt in to at most four paid requests using synthetic inputs.")
    parser.add_argument("--record-observations", action="store_true", help="After full success, submit metadata-only deployment-probe observations to admin health.")
    parser.add_argument("--profile", type=Path, default=ROOT / "deploy/customer-models.env")
    parser.add_argument("--expected-agent")
    parser.add_argument("--expected-router")
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        agent, router = profile["OPENAI_AGENT_MODEL"], profile["OPENAI_ROUTER_MODEL"]
        if args.expected_agent is not None and args.expected_agent != agent:
            raise ProbeFailure("expected_agent_profile_mismatch")
        if args.expected_router is not None and args.expected_router != router:
            raise ProbeFailure("expected_router_profile_mismatch")
        if args.record_observations and not args.live:
            raise ProbeFailure("recording_requires_live_probe")
        if not args.live:
            print(json.dumps({"profile_valid": True, "agent": agent, "router": router, "provider_access_tested": False}))
            return 0
        if os.environ.get("OPENAI_AGENT_MODEL", "").strip() != agent or os.environ.get("OPENAI_ROUTER_MODEL", "").strip() != router:
            raise ProbeFailure("effective_model_profile_mismatch")
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ProbeFailure("provider_key_unavailable")
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=20.0, max_retries=0)
        result = run_probe(client, agent=agent, router=router)
        if args.record_observations:
            result["operational_observations_submitted"] = _record_probe_observations(result)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except ProbeFailure as exc:
        print(json.dumps({"passed": False, "failure_code": str(exc)}))
    except Exception as exc:
        # Provider exceptions can contain request/response data. Never print them.
        print(json.dumps({"passed": False, "failure_code": "provider_probe_failed", "error_type": type(exc).__name__}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
