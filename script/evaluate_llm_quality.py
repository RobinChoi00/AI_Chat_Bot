#!/usr/bin/env python3
"""Evaluate complete conversations through the application's HTTP APIs.

``--dry-run`` validates fixtures only; it never establishes AI quality. Live
runs require ``--live`` and create synthetic ``eval-*`` sessions in the target
server. Run against an isolated development/staging database. Configure model
candidates on that server; this client does not override a server's model.

Examples:
    python script/evaluate_llm_quality.py --dry-run
    AI_EVAL_ADMIN_API_KEY=... python script/evaluate_llm_quality.py --live \
        --base-url http://127.0.0.1:8000 --out data/reports/quality-baseline.json
    python script/evaluate_llm_quality.py --live --suite conversations \
        --baseline data/reports/quality-baseline.json --out data/reports/candidate.json

Never put credentials in a URL or command-line arguments. Remote targets require
an additional explicit ``--allow-remote`` opt-in. No live run occurs by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CASES_PATH = ROOT / "data" / "llm_eval_cases.json"
CONVERSATIONS_PATH = ROOT / "data" / "ai_conversation_eval_cases.json"
LEGACY_SECTIONS = ("tool_selection", "faithfulness", "refusal", "language_consistency")
WARRANTY_ACTIONS = frozenset({"register-model", "confirm-model", "quick-start", "natural-start", "smart-start", "answer", "state"})
CHECK_KEYS = frozenset({"contains", "contains_any", "not_contains", "json_equals", "expected_intent", "expected_handoff", "expected_tool", "expected_tool_sequence", "expected_sources", "expected_language", "expect_refusal", "min_products", "max_products", "max_product_price", "min_product_capacity_lb", "required_product_mechanism", "max_product_doorway_assembled_in", "max_product_doorway_disassembled_in"})
DIALOGUE_CHECK_KEYS = frozenset({"not_contains", "final_contains_any", "max_identical_assistant_replies"})


def _get_model() -> str:
    """Local config reference only; remote runtime identity comes from traces."""
    from config import AGENT_MODEL
    return AGENT_MODEL


def validate_structure(cases: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for section in LEGACY_SECTIONS:
        rows = cases.get(section)
        if not isinstance(rows, list) or not rows:
            errors.append(f"Section '{section}' must be a non-empty list")
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"{section}[{i}] must be an object")
                continue
            cid = row.get("id")
            if not isinstance(cid, str) or not cid or cid in ids:
                errors.append(f"{section}[{i}] has missing or duplicate id")
            ids.add(str(cid))
            if not isinstance(row.get("query"), str) or not row["query"].strip():
                errors.append(f"{section}[{i}] needs a query")
            required = {"tool_selection": "expected_tool", "refusal": "expect_refusal", "language_consistency": "response_language"}.get(section)
            if required and required not in row:
                errors.append(f"{section}[{i}] needs {required}")
            if section == "faithfulness" and not any(row.get(key) for key in ("must_call_tool", "response_must_contain", "response_must_not_contain")):
                errors.append(f"{section}[{i}] needs a tool or a response constraint")
    return errors


def validate_conversations(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return ["Conversation cases must be a non-empty list"]
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            errors.append("Conversation case must be an object")
            continue
        cid = case.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            errors.append("Conversation id is missing or duplicated")
        ids.add(str(cid))
        if case.get("provenance") not in {"synthetic", "reviewed_anonymized"}:
            errors.append(f"{cid}: provenance must be explicit")
        if case.get("surface") not in {"general", "sales", "warranty"}:
            errors.append(f"{cid}: unsupported surface")
        if case.get("split", "regression") not in {"regression", "held_out"}:
            errors.append(f"{cid}: split must be regression or held_out")
        dialogue = case.get("dialogue_expect", {})
        if not isinstance(dialogue, dict) or set(dialogue) - DIALOGUE_CHECK_KEYS:
            errors.append(f"{cid}: unsupported dialogue expectations")
        elif dialogue:
            for key in ("not_contains", "final_contains_any"):
                if key in dialogue and (not isinstance(dialogue[key], list) or not dialogue[key] or not all(isinstance(item, str) and item.strip() for item in dialogue[key])):
                    errors.append(f"{cid}: {key} must contain nonempty strings")
            count = dialogue.get("max_identical_assistant_replies")
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
                errors.append(f"{cid}: max_identical_assistant_replies must be a positive integer")
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            errors.append(f"{cid}: turns must not be empty")
            continue
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                errors.append(f"{cid}/{index}: turn must be an object")
                continue
            checks = turn.get("expect", {})
            if not isinstance(checks, dict) or not checks or set(checks) - CHECK_KEYS:
                errors.append(f"{cid}/{index}: needs supported, nonempty expectations")
            elif checks:
                for key in ("contains", "contains_any", "not_contains"):
                    if key in checks and (not isinstance(checks[key], list) or not checks[key] or not all(isinstance(value, str) and value.strip() for value in checks[key])):
                        errors.append(f"{cid}/{index}: {key} must contain nonempty strings")
                if "json_equals" in checks and (not isinstance(checks["json_equals"], dict) or not checks["json_equals"]):
                    errors.append(f"{cid}/{index}: json_equals must contain field expectations")
                for key in ("expected_handoff", "expect_refusal"):
                    if key in checks and not isinstance(checks[key], bool):
                        errors.append(f"{cid}/{index}: {key} must be boolean")
                for key in ("min_products", "max_products", "max_product_price", "min_product_capacity_lb", "max_product_doorway_assembled_in", "max_product_doorway_disassembled_in"):
                    if key in checks and (isinstance(checks[key], bool) or not isinstance(checks[key], (int, float)) or not math.isfinite(checks[key]) or checks[key] < 0):
                        errors.append(f"{cid}/{index}: {key} must be nonnegative")
                if "required_product_mechanism" in checks and checks["required_product_mechanism"] not in {"2D", "3D", "4D", "5D"}:
                    errors.append(f"{cid}/{index}: required_product_mechanism must name a supported mechanism")
                if "expected_tool_sequence" in checks:
                    sequence = checks["expected_tool_sequence"]
                    if (case.get("surface") != "general" or not isinstance(sequence, list)
                        or any(not isinstance(name, str) or not name.strip() for name in sequence)):
                        errors.append(f"{cid}/{index}: expected_tool_sequence needs tool names on general chat")
                if "expected_sources" in checks:
                    sources = checks["expected_sources"]
                    if (case.get("surface") != "general" or not isinstance(sources, list) or not sources
                        or any(not isinstance(item, dict) or not item.get("source_type") or set(item) - {"source_type", "model", "currency"}
                               or not all(isinstance(value, str) and value.strip() for value in item.values()) for item in sources)):
                        errors.append(f"{cid}/{index}: expected_sources needs typed source identities on general chat")
            if case.get("surface") == "warranty":
                if turn.get("action") not in WARRANTY_ACTIONS:
                    errors.append(f"{cid}/{index}: unsupported warranty action")
                if not isinstance(turn.get("body", {}), dict):
                    errors.append(f"{cid}/{index}: body must be an object")
            elif not isinstance(turn.get("message"), str) or not turn["message"].strip():
                errors.append(f"{cid}/{index}: needs a message")
    return errors


def legacy_conversations(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for section in LEGACY_SECTIONS:
        for row in catalog[section]:
            checks: dict[str, Any] = {}
            for source, dest in (("expected_tool", "expected_tool"), ("must_call_tool", "expected_tool"), ("response_must_contain", "contains"), ("response_must_not_contain", "not_contains"), ("expect_refusal", "expect_refusal"), ("response_language", "expected_language")):
                if source in row:
                    checks[dest] = row[source]
            cases.append({"id": row["id"], "surface": "general", "provenance": "synthetic", "turns": [{"message": row["query"], "expect": checks}]})
    return cases


def validate_target(base_url: str, allow_remote: bool) -> str:
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("base URL must be an HTTP(S) origin")
    if url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
        raise ValueError("base URL must not contain credentials, paths, queries or fragments")
    if url.hostname.lower() not in {"localhost", "127.0.0.1", "::1"} and not allow_remote:
        raise ValueError("Remote evaluation requires --allow-remote and an isolated staging target")
    return base_url.rstrip("/")


def _lookup(payload: Any, path: str, default: Any = None) -> Any:
    current = payload
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment, default)
        elif isinstance(current, list) and segment.isdigit() and int(segment) < len(current):
            current = current[int(segment)]
        else:
            return default
    return current


def _detect_language(text: str) -> str:
    if len(re.findall(r"[\uac00-\ud7af]", text)) > 3:
        return "ko"
    if len(re.findall(r"\b(hola|gracias|silla|precio|pedido|garant[ií]a|cu[aá]nto|nuestro|equipo|por\s+favor)\b", text, re.I)) > 2:
        return "es"
    return "en"


def check_answer(text: str, payload: dict[str, Any], checks: dict[str, Any], trace: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if not text.strip():
        errors.append("empty final response")
    lower = text.lower()
    if any(marker in lower for marker in (
        "an unexpected error occurred. please try again", "we could not complete this lookup",
        "no pudimos completar esta consulta", "현재 조회를 완료하지 못했습니다",
        "authoritative lookup is temporarily unavailable", "fuente oficial no está disponible temporalmente",
        "공식 정보 조회가 일시적으로 불가능",
    )):
        errors.append("application error fallback was returned")
    for needle in checks.get("contains", []):
        if needle.lower() not in lower:
            errors.append(f"missing required text: {needle}")
    if checks.get("contains_any") and not any(needle.lower() in lower for needle in checks["contains_any"]):
        errors.append("none of the required alternatives appeared")
    for needle in checks.get("not_contains", []):
        if needle.lower() in lower:
            errors.append(f"forbidden text appeared: {needle}")
    for path, expected in checks.get("json_equals", {}).items():
        missing = object()
        if _lookup(payload, path, missing) != expected:
            errors.append(f"unexpected JSON field: {path}")
    for check, field in (("expected_intent", "intent"), ("expected_handoff", "handoff")):
        if check in checks and payload.get(field) != checks[check]:
            errors.append(f"unexpected {field}")
    if "expected_language" in checks and _detect_language(text) != checks["expected_language"]:
        errors.append("unexpected response language (heuristic check)")
    if "expect_refusal" in checks:
        markers = ("i can only", "outside my scope", "i'm here to help with", "i am here to help with", "assist with massage chair", "i specialize in", "not able to help with that", "massage chairs only")
        detected = any(marker in lower for marker in markers)
        if detected != checks["expect_refusal"]:
            errors.append("unexpected refusal result (heuristic check)")
    if "expected_tool" in checks:
        tools = payload.get("tools_used", []) if trace is None else trace.get("tools_called", [])
        # The real route intentionally answers showroom questions without an LLM.
        authoritative_shortcut = checks["expected_tool"] == "get_showroom_info" and trace and trace.get("short_circuit") == "showroom"
        if not authoritative_shortcut and checks["expected_tool"] not in tools:
            errors.append("required tool execution not observed")
        matching = [turn for turn in (trace or {}).get("turns", []) if turn.get("tool_name") == checks["expected_tool"]]
        if matching and not any(turn.get("tool_success") is True for turn in matching):
            errors.append("required tool did not execute successfully")
    if "expected_tool_sequence" in checks:
        # Membership alone misses duplicate loops and unauthorized intermediate
        # actions. Use authenticated execution records, including failed calls.
        events = (trace or {}).get("turns")
        valid_events = isinstance(events, list) and all(isinstance(turn, dict) for turn in events)
        turns = [turn for turn in events if turn.get("tool_name")] if valid_events else []
        sequence = [turn["tool_name"] for turn in turns]
        if not valid_events or sequence != checks["expected_tool_sequence"]:
            errors.append("executed tool sequence differs from expected sequence")
        if any(turn.get("tool_success") is not True for turn in turns):
            errors.append("tool sequence contains an unsuccessful execution")
    if "expected_sources" in checks:
        evidence = [item for turn in (trace or {}).get("turns", []) if turn.get("tool_success") is True
                    for item in turn.get("source_evidence", []) if item.get("available") is True]
        for expected in checks["expected_sources"]:
            if not any(all(str(item.get(key, "")).casefold() == value.casefold() for key, value in expected.items()) for item in evidence):
                errors.append("required source/model evidence not observed")
    products = payload.get("products", [])
    if "min_products" in checks and len(products) < checks["min_products"]:
        errors.append("not enough recommended products")
    if "max_products" in checks and len(products) > checks["max_products"]:
        errors.append("too many recommended products")
    if "max_product_price" in checks:
        for product in products:
            value = product.get("price_usd", product.get("price"))
            try:
                if value is None or not math.isfinite(float(value)) or float(value) > float(checks["max_product_price"]):
                    errors.append("product price missing or above the hard budget")
            except (TypeError, ValueError):
                errors.append("product price cannot be checked")
    if "min_product_capacity_lb" in checks:
        for product in products:
            value = _lookup(product, "fit.maximum_user_weight_lb")
            try:
                if value is None or not math.isfinite(float(value)) or float(value) < checks["min_product_capacity_lb"]:
                    errors.append("recommended capacity missing or below the customer's exact weight")
            except (TypeError, ValueError):
                errors.append("recommended capacity cannot be checked")
    for check, field in (
        ("max_product_doorway_assembled_in", "fit.doorway_assembled_in"),
        ("max_product_doorway_disassembled_in", "fit.doorway_disassembled_in"),
    ):
        if check not in checks:
            continue
        for product in products:
            value = _lookup(product, field)
            try:
                if isinstance(value, bool) or value is None or not math.isfinite(float(value)) or float(value) <= 0 or float(value) > checks[check]:
                    errors.append("recommended doorway requirement missing or exceeds the customer's opening")
            except (TypeError, ValueError):
                errors.append("recommended doorway requirement cannot be checked")
    if "required_product_mechanism" in checks:
        for product in products:
            if str(_lookup(product, "specs.massage_mechanism") or "").upper() != checks["required_product_mechanism"]:
                errors.append("recommended massage mechanism missing or violates the requirement")
    return errors


def check_dialogue(replies: list[str], checks: dict[str, Any]) -> list[str]:
    """Judge the complete dialogue; retain no raw replies in the report."""
    errors: list[str] = []
    combined = "\n".join(replies).casefold()
    if any(needle.casefold() in combined for needle in checks.get("not_contains", [])):
        errors.append("dialogue-wide forbidden text appeared")
    if checks.get("final_contains_any") and (not replies or not any(needle.casefold() in replies[-1].casefold() for needle in checks["final_contains_any"])):
        errors.append("final dialogue goal was not demonstrated")
    if "max_identical_assistant_replies" in checks:
        normalized = [re.sub(r"\s+", " ", reply).strip().casefold() for reply in replies if reply.strip()]
        if normalized and max(normalized.count(reply) for reply in set(normalized)) > checks["max_identical_assistant_replies"]:
            errors.append("identical assistant reply loop exceeded limit")
    return errors


def _needs_trace(checks: dict[str, Any], expected_model: str | None = None) -> bool:
    return bool(checks.get("expected_tool") or "expected_tool_sequence" in checks or checks.get("expected_sources") or expected_model)


def _read_stream(response) -> str:
    response.raise_for_status()
    text = "".join(response.iter_text())
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return text
    # The current general-chat API streams raw UTF-8 chunks under this media
    # type. Preserve that wire format as well as genuine SSE data frames.
    if not any(line.startswith("data:") for line in text.splitlines()):
        if any(line.startswith(("event:", "id:", "retry:", ":")) for line in text.splitlines()):
            return ""
        return text
    parts: list[str] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            continue
        try:
            event = json.loads(value)
            if isinstance(event, dict):
                parts.append(str(event.get("text", event.get("content", ""))))
            elif isinstance(event, str):
                parts.append(event)
        except ValueError:
            parts.append(value)
    return "".join(parts)


def _trace(client, base_url: str, request_id: str, admin_key: str) -> dict[str, Any]:
    for attempt in range(3):
        result = client.get(f"{base_url}/admin/traces/{request_id}", headers={"X-Admin-Key": admin_key})
        if result.status_code != 404 or attempt == 2:
            result.raise_for_status()
            return result.json()
        # Trace persistence is a background task after the response stream closes.
        time.sleep(0.05 * (attempt + 1))
    raise RuntimeError("trace unavailable")


def _model_identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) else None


def _provider_model_matches(expected: str, observed: str) -> bool:
    """An alias may resolve to its exact ISO-dated provider snapshot only."""
    if observed == expected:
        return True
    suffix = observed.removeprefix(expected + "-")
    if suffix == observed or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", suffix):
        return False
    try:
        date.fromisoformat(suffix)
    except ValueError:
        return False
    return True


def model_evidence(trace: dict[str, Any] | None, expected_model: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Keep requested configuration separate from successful provider evidence."""
    trace = trace or {}
    requested = _model_identifier(trace.get("model"))
    requested_models = trace.get("requested_models", [])
    providers = trace.get("provider_models", [])
    requested_models = list(dict.fromkeys(item for item in requested_models if _model_identifier(item)))[:16] if isinstance(requested_models, list) else []
    providers = list(dict.fromkeys(item for item in providers if _model_identifier(item)))[:16] if isinstance(providers, list) else []
    successful = trace.get("model_calls_successful", 0)
    successful = successful if isinstance(successful, int) and not isinstance(successful, bool) and successful >= 0 else 0
    unverified = trace.get("model_calls_unverified", 0)
    errors: list[str] = []
    if expected_model:
        if requested != expected_model or (requested_models and any(item != expected_model for item in requested_models)):
            errors.append("observed runtime model differs from expected model")
        if (successful <= 0 or not providers or not requested_models
            or not isinstance(unverified, int) or isinstance(unverified, bool) or unverified != 0):
            errors.append("successful provider model execution was not verified")
        if any(not _provider_model_matches(expected_model, model) for model in providers):
            errors.append("provider-returned model differs from expected model")
    return {
        "requested_model": requested,
        "requested_models": requested_models,
        "provider_models": providers,
        "observed_model": providers[0] if len(providers) == 1 else None,
        "model_calls_successful": successful,
    }, errors


def run_case(case: dict[str, Any], client, *, base_url: str, domain: str, admin_key: str = "", expected_model: str | None = None) -> dict[str, Any]:
    session_id = f"eval-{uuid.uuid4().hex}"
    history: list[dict[str, str]] = []
    ticket_id = ""
    results: list[dict[str, Any]] = []
    replies: list[str] = []
    surface = case["surface"]
    if surface in {"general", "sales", "warranty"}:
        try:
            setup = [("consent", {"domain": domain})]
            if surface == "warranty":
                setup.append(("contact-email", {"customer_email": f"{session_id}@example.invalid"}))
            for action, body in setup:
                response = client.post(f"{base_url}/api/v1/warranty/session/{session_id}/{action}", json=body)
                response.raise_for_status()
        except Exception as exc:
            return {"id": case["id"], "surface": surface, "provenance": case["provenance"], "passed": False, "turns": [], "unrun_turns": len(case["turns"]), "error": f"{surface} consent setup failed: {type(exc).__name__}"}
    for index, turn in enumerate(case["turns"]):
        started = time.perf_counter()
        request_id = f"eval-{uuid.uuid4().hex}"
        trace = None
        try:
            if surface == "general":
                body = {"session_id": session_id, "user_query": turn["message"], "chat_history": history, "current_domain": domain}
                with client.stream("POST", f"{base_url}/api/v1/chat", json=body, headers={"X-Request-ID": request_id}) as response:
                    text = _read_stream(response)
                payload: dict[str, Any] = {}
                if _needs_trace(turn["expect"], expected_model):
                    if not admin_key:
                        raise ValueError("admin trace credential required to verify tool/model evidence")
                    trace = _trace(client, base_url, request_id, admin_key)
                history.extend([{"role": "user", "content": turn["message"]}, {"role": "assistant", "content": text}])
            elif surface == "sales":
                response = client.post(f"{base_url}/api/v1/sales/chat", json={"session_id": session_id, "message": turn["message"], "domain": domain, "channel": "test", **({"payload": turn["payload"]} if turn.get("payload") else {})})
                response.raise_for_status()
                payload = response.json()
                text = str(payload.get("reply", ""))
            else:
                action = turn["action"]
                if action == "answer":
                    if not ticket_id:
                        raise ValueError("answer requires a ticket from an earlier turn")
                    path = f"/api/v1/warranty/{ticket_id}/answer"
                else:
                    path = f"/api/v1/warranty/session/{session_id}" + (f"/{action}" if action != "state" else "")
                response = client.get(base_url + path) if action == "state" else client.post(base_url + path, json={"domain": domain, **turn.get("body", {})})
                response.raise_for_status()
                payload = response.json()
                ticket_id = str(_lookup(payload, "ticket.ticket_id") or ticket_id)
                # Evaluate the complete structured UI state, not just LLM prose.
                text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            replies.append(text)
            errors = check_answer(text, payload, turn["expect"], trace)
            evidence, model_errors = model_evidence(trace, expected_model if surface == "general" else None)
            errors.extend(model_errors)
            results.append({"turn": index + 1, "passed": not errors, "errors": errors, "latency_ms": elapsed_ms, "response_sha256": hashlib.sha256(text.encode()).hexdigest(), "response_chars": len(text), **evidence})
        except Exception as exc:
            # Do not echo provider errors, response bodies or credential-bearing URLs.
            results.append({"turn": index + 1, "passed": False, "errors": [f"request/evidence failed: {type(exc).__name__}"], "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
            break
    dialogue_errors = check_dialogue(replies, case.get("dialogue_expect", {}))
    return {"id": case["id"], "surface": surface, "provenance": case["provenance"], "split": case.get("split", "regression"), "passed": len(results) == len(case["turns"]) and all(row["passed"] for row in results) and not dialogue_errors, "dialogue_errors": dialogue_errors, "unrun_turns": len(case["turns"]) - len(results), "turns": results}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(results: list[dict[str, Any]], baseline: dict[str, Any] | None = None, max_p95_ms: float | None = None) -> dict[str, Any]:
    latencies = [turn["latency_ms"] for case in results for turn in case["turns"]]
    passed = sum(bool(case["passed"]) for case in results)
    regressions: list[str] = []
    comparison_errors: list[str] = []
    if baseline is not None:
        prior = {case["id"]: case for case in baseline.get("cases", [])}
        current = {case["id"]: case for case in results}
        if not prior or set(prior) != set(current):
            comparison_errors.append("baseline and candidate must contain the same nonempty case IDs")
        regressions = [cid for cid, case in current.items() if prior.get(cid, {}).get("passed") and not case["passed"]]
    p95 = percentile(latencies, 0.95)
    latency_failed = max_p95_ms is not None and (p95 is None or p95 > max_p95_ms)
    success = bool(results) and passed == len(results) and not regressions and not comparison_errors and not latency_failed
    surfaces = {}
    for surface in sorted({case.get("surface", "unknown") for case in results}):
        rows = [case for case in results if case.get("surface", "unknown") == surface]
        samples = [turn["latency_ms"] for case in rows for turn in case["turns"]]
        surfaces[surface] = {"cases": len(rows), "passed": sum(bool(case["passed"]) for case in rows), "p50_ms": percentile(samples, 0.5), "p95_ms": percentile(samples, 0.95)}
    all_errors = [error for case in results for turn in case["turns"] for error in turn.get("errors", [])]
    failure_counts = {
        "route_or_handoff": sum(error in {"unexpected intent", "unexpected handoff"} for error in all_errors),
        "source_evidence": sum("source/model evidence" in error or "required tool" in error or "tool sequence" in error for error in all_errors),
        "fit_or_budget": sum(any(word in error for word in ("capacity", "mechanism", "budget", "product price")) for error in all_errors),
        "request_or_availability": sum("request/evidence failed" in error or "application error fallback" in error for error in all_errors) + sum(bool(case.get("error")) for case in results),
        "dialogue_goal_or_loop": sum(len(case.get("dialogue_errors", [])) for case in results),
    }
    observed = sum("response_sha256" in turn for case in results for turn in case["turns"])
    return {"status": "pass" if success else "fail", "mode": "live_api", "quality_evaluated": observed > 0, "evaluated_turns": observed, "total_cases": len(results), "passed": passed, "failed": len(results) - passed, "unrun_turns": sum(case.get("unrun_turns", 0) for case in results), "pass_rate_pct": round(100 * passed / len(results), 1) if results else 0, "assertion_failure_counts": failure_counts, "latency_ms": {"p50": percentile(latencies, 0.5), "p95": p95, "samples": len(latencies), "budget_failed": latency_failed}, "surfaces": surfaces, "regressions": regressions, "comparison_errors": comparison_errors, "cases": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Offline fixture structure validation only")
    mode.add_argument("--live", action="store_true", help="Run real requests that create synthetic staging sessions and can incur API costs")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--domain", default="osakiusa.com")
    parser.add_argument("--suite", choices=("all", "legacy", "conversations"), default="all")
    parser.add_argument("--split", choices=("all", "regression", "held_out"), default="all", help="Keep reviewed regression cases separate from independently curated held-out cases")
    parser.add_argument("--cases", type=Path, default=CONVERSATIONS_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--expected-model", help="Require successful general-chat provider model evidence; exact ID or its ISO-dated snapshot. Configure the isolated server first.")
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        legacy = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        conversations = json.loads(args.cases.read_text(encoding="utf-8"))
        errors = validate_structure(legacy) + validate_conversations(conversations)
        if errors:
            print(json.dumps({"status": "invalid_fixtures", "quality_evaluated": False, "errors": errors}, indent=2))
            return 1
        selected = ([] if args.suite == "conversations" else legacy_conversations(legacy)) + ([] if args.suite == "legacy" else conversations["cases"])
        if args.split != "all":
            selected = [case for case in selected if case.get("split", "regression") == args.split]
        if not selected:
            raise ValueError("selected evaluation split has no cases; no quality measurement is possible")
        if args.dry_run:
            print(json.dumps({"status": "structure_valid", "mode": "dry_run", "quality_evaluated": False, "total_cases": len(selected), "synthetic_cases": sum(case["provenance"] == "synthetic" for case in selected), "reviewed_anonymized_cases": sum(case["provenance"] == "reviewed_anonymized" for case in selected), "local_config_model": _get_model(), "note": "No model, server, retrieval, stock or customer outcomes were evaluated."}, indent=2))
            return 0
        if not args.live:
            parser.error("choose --dry-run or --live; live requests never run implicitly")
        base_url = validate_target(args.base_url, args.allow_remote)
        admin_key = os.environ.get("AI_EVAL_ADMIN_API_KEY", "")
        if any(case["surface"] == "general" and any(_needs_trace(turn["expect"], args.expected_model) for turn in case["turns"]) for case in selected) and not admin_key:
            raise ValueError("AI_EVAL_ADMIN_API_KEY is required for authenticated tool/model trace evidence")
        if args.expected_model and not any(case["surface"] == "general" for case in selected):
            raise ValueError("--expected-model requires general-chat cases with runtime trace evidence")
        if args.timeout <= 0 or (args.max_p95_ms is not None and args.max_p95_ms <= 0):
            raise ValueError("timeouts and latency budgets must be positive")
        fixture_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
        baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
        if baseline is not None:
            if baseline.get("mode") != "live_api" or not baseline.get("quality_evaluated"):
                raise ValueError("baseline must be a completed live API evaluation, not a dry run")
            if baseline.get("fixture_sha256") != fixture_hash or baseline.get("domain") != args.domain:
                raise ValueError("baseline must use the same fixtures and domain as the candidate")
            if {row.get("id") for row in baseline.get("cases", [])} != {row["id"] for row in selected}:
                raise ValueError("baseline must contain the same case IDs as the candidate")
        import httpx
        with httpx.Client(timeout=args.timeout, follow_redirects=False) as client:
            results = [run_case(case, client, base_url=base_url, domain=args.domain, admin_key=admin_key, expected_model=args.expected_model) for case in selected]
        report = summarize(results, baseline, args.max_p95_ms)
        report.update({"local_config_model": _get_model(), "expected_runtime_model": args.expected_model, "domain": args.domain, "split": args.split, "fixture_sha256": fixture_hash, "limitations": ["Regression fixtures are not an independent held-out estimate of customer success or conversion.", "Language/refusal checks are heuristic; expert review is still required.", "The client cannot override the server model; use a separately configured staging server.", "Fixture HTTP calls create staging records and may call paid external services."]})
        output = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0 if report["status"] == "pass" else 1
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "quality_evaluated": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
