#!/usr/bin/env python3
"""Continue local customer conversations after mistakes and measure recovery."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from script.simulate_chatbot_quality import offline_api
from script.simulate_chatbot_broad_quality import source_hash

MODULES = ("warranty", "product", "service")
RUBRIC = ROOT / "docs/quality/continued-journeys-v4-rubric.json"
AREAS = {"warranty", "sales_product", "sales_service"}
DIMENSIONS = {"understanding", "accuracy", "context", "recovery"}
OUTCOMES = {"goal_met", "partially_met", "not_met", "blocked"}


def collect_journeys():
    journeys = []
    for name in MODULES:
        journeys.extend(importlib.import_module(f"script.quality_v4_{name}").journeys())
    if len(journeys) != 24 or len({j.id for j in journeys}) != 24:
        raise ValueError("Require exactly 24 distinct customer goal cards")
    if Counter(j.area for j in journeys) != Counter({area: 8 for area in AREAS}):
        raise ValueError("Require eight journeys for each business area")
    for j in journeys:
        if not j.persona or not j.goal or not callable(j.run) or not 6 <= len(j.planned_inputs) <= 10:
            raise ValueError(f"Incomplete customer goal card: {j.id}")
    return journeys


def validate_result(result):
    if result.get("execution_error"):
        return
    if result.get("final_outcome") not in OUTCOMES:
        raise ValueError("Unknown customer outcome")
    if not isinstance(result.get("customer_turns"), int) or not 6 <= result["customer_turns"] <= 10:
        raise ValueError("Every completed journey requires 6–10 actual customer turns")
    if not result.get("transcript") or not result.get("checks"):
        raise ValueError("Outcomes require actual transcripts and checkpoints")
    for check in result["checks"]:
        if check.get("dimension") not in DIMENSIONS or check.get("status") not in {"pass", "fail", "unmeasured"}:
            raise ValueError("Unknown or missing checkpoint status/dimension")
        if not check.get("name"):
            raise ValueError("A checkpoint must name the customer expectation")
    for attempt in result.get("recovery_attempts", []):
        if attempt.get("recovered") is not None and type(attempt.get("recovered")) is not bool:
            raise ValueError("Recovery must be observed or explicitly unmeasured")


def summarize(results):
    if len({r["id"] for r in results}) != len(results):
        raise ValueError("Duplicate journey evidence")
    for result in results:
        validate_result(result)
    completed = [r for r in results if not r.get("execution_error")]
    area_results = {}
    for area in sorted(AREAS):
        items = [r for r in completed if r["area"] == area]
        area_results[area] = {"executed": len(items), "outcomes": dict(Counter(r["final_outcome"] for r in items)),
                              "with_failed_checkpoints": sum(any(c["status"] == "fail" for c in r["checks"]) for r in items)}
    checks = [c for r in completed for c in r["checks"]]
    recovery = [a for r in completed for a in r.get("recovery_attempts", [])]
    return {
        "journeys": len(results), "execution_errors": len(results) - len(completed),
        "customer_turns": sum(r["customer_turns"] for r in completed),
        "actual_http_requests": sum(r.get("transport_request_count", 0) for r in results),
        "outcomes": dict(Counter(r["final_outcome"] for r in completed)),
        "goal_met_after_earlier_failures": sum(r["final_outcome"] == "goal_met" and any(c["status"] == "fail" for c in r["checks"]) for r in completed),
        "checkpoint_counts": dict(Counter(c["status"] for c in checks)),
        "dimensions": {dimension: dict(Counter(c["status"] for c in checks if c["dimension"] == dimension)) for dimension in sorted(DIMENSIONS)},
        "recovery_attempts": {"observed_success": sum(a.get("recovered") is True for a in recovery),
                              "observed_failure": sum(a.get("recovered") is False for a in recovery),
                              "unmeasured": sum(a.get("recovered") is None for a in recovery)},
        "areas": area_results,
        "live_model_score": None, "universal_score_out_of_100": None,
    }


def fingerprint():
    files = [Path(__file__), RUBRIC, ROOT / "script/quality_v4_support.py",
             ROOT / "script/simulate_chatbot_quality.py", ROOT / "script/simulate_chatbot_broad_quality.py"]
    files.extend(ROOT / f"script/quality_v4_{name}.py" for name in MODULES)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def customer_text(turn):
    payload = turn.get("response") or {}
    if not isinstance(payload, dict):
        return str(payload)
    ticket = payload.get("ticket") or {}
    smart = payload.get("smart_start") or {}
    routing = smart.get("routing_confirmation") or payload.get("routing_confirmation") or {}
    return str(payload.get("reply") or payload.get("assistant_message")
               or ticket.get("customer_message") or routing.get("message")
               or (ticket.get("current_node") or {}).get("prompt") or payload.get("detail") or "")


def execute_journey(journey):
    row = {"id": journey.id, "area": journey.area, "persona": journey.persona,
           "goal": journey.goal, "planned_inputs": journey.planned_inputs}
    transport = []
    try:
        with offline_api() as client:
            original_request = client.request

            def record_request(method, url, **kwargs):
                event = {"method": method, "path": str(url), "status": None}
                transport.append(event)
                try:
                    response = original_request(method, url, **kwargs)
                    event["status"] = response.status_code
                    return response
                except Exception as exc:
                    event["error_type"] = type(exc).__name__
                    raise

            with patch.object(client, "request", side_effect=record_request):
                row.update(journey.run(client))
            row["transport_request_count"] = len(transport)
            row["transport_evidence"] = transport
            row["blocked_external_attempts"] = len(client.attempted_external_delivery)
            if client.attempted_external_delivery:
                raise RuntimeError("Unexpected external delivery attempted during local role-play")
        validate_result(row)
        failed = [c for c in row["checks"] if c["status"] == "fail"]
        row["first_failed_checkpoint"] = failed[0] if failed else None
        if failed:
            index = failed[0].get("turn")
            candidates = [t for t in row["transcript"] if t.get("turn") == index and not t.get("setup") and t.get("customer")]
            if candidates:
                turn = candidates[-1]
                row["first_failed_turn_observation"] = {
                    "turn": index, "customer": turn["customer"], "bot_reply": customer_text(turn),
                    "checkpoint": failed[0]["name"], "expected": failed[0].get("expected")}
    except Exception as exc:
        row.update(execution_error=type(exc).__name__, error=str(exc), transport_evidence=transport,
                   transport_request_count=len(transport))
    return row


def run_simulation():
    app_hash = source_hash()
    journeys = collect_journeys()
    evaluator_hash = fingerprint()
    results = []
    for index, journey in enumerate(journeys, 1):
        result = execute_journey(journey)
        results.append(result)
        print(f"Journey {index}/24: {journey.id}: {result.get('final_outcome', result.get('execution_error'))}", file=sys.stderr, flush=True)
    if source_hash() != app_hash or fingerprint() != evaluator_hash:
        raise RuntimeError("Application or goal-card evaluator changed; rerun the frozen simulation")
    return {"version": "continued-customer-journeys-v4", "mode": "scripted_customer_roleplay_with_actual_local_api_replies",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "app_source_sha256": app_hash, "evaluator_sha256": evaluator_hash,
            "application_unchanged": True, "rubric": json.loads(RUBRIC.read_text()),
            "limitations": [
                "Customers are scripted synthetic personas; bot responses are actual local API responses, not imagined model answers.",
                "OpenAI, live stock/order systems and notifications are unavailable; the tested model-dependent behavior is its deterministic/fallback path.",
                "Final goal met means the stated information or correct service next step was obtained, not that a chair was repaired, an order paid, or a claim approved.",
                "Checkpoint counts are diagnostic observations with unequal numbers per goal; they are not an accuracy percentage or a production quality score."
            ], **summarize(results), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = run_simulation()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"results", "rubric"}}, indent=2, ensure_ascii=False))
    return 1 if report["execution_errors"] or report["checkpoint_counts"].get("fail", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
