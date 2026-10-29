#!/usr/bin/env python3
"""Independent broad offline re-evaluation; failures are findings, not targets to erase.

No model or production quality is inferred. Every scenario gets a fresh
in-memory API/database; conversation turns share state only within that case.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from script.quality_v2_support import CheckFailure
from script.simulate_chatbot_quality import offline_api

MODULES = ("sales", "safety", "grounding", "language", "privacy")
RUBRIC = ROOT / "docs/quality/broad-audit-v2-rubric.json"


def source_hash(root: Path = ROOT) -> str:
    """Use the previous report's app-source identity for an exact comparison."""
    digest = hashlib.sha256()
    for path in sorted((root / "app").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_scenarios():
    scenarios = []
    for name in MODULES:
        module = importlib.import_module(f"script.quality_v2_{name}")
        scenarios.extend(module.scenarios())
    validate_scenarios(scenarios)
    return scenarios


def validate_scenarios(scenarios):
    weights = json.loads(RUBRIC.read_text())["categories"]
    identifiers = [row.id for row in scenarios]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Scenario identifiers must be globally unique")
    if {row.category for row in scenarios} != set(weights):
        raise ValueError("All ten rubric categories must execute")
    for category in weights:
        rows = [row for row in scenarios if row.category == category]
        if len(rows) < 20 or len({row.family for row in rows}) < 5:
            raise ValueError(f"Insufficient breadth in {category}")
    for row in scenarios:
        if row.level not in {"function", "agent", "http_api"}:
            raise ValueError(f"Unsupported execution level: {row.id}")
        if row.severity not in {"critical", "high", "medium", "low"}:
            raise ValueError(f"Unsupported severity: {row.id}")
        if not row.id or not row.family or not row.title or not callable(row.run):
            raise ValueError("Every scenario needs an explicit behavior and executable check")


def score_results(results: list[dict[str, Any]], weights: dict[str, int]):
    if any(row["status"] not in {"pass", "fail", "error"} for row in results):
        raise ValueError("Unknown execution statuses cannot be scored")
    if len({row["id"] for row in results}) != len(results):
        raise ValueError("Duplicate results cannot be scored")
    if set(row["category"] for row in results) != set(weights):
        raise ValueError("Missing or unknown result category")
    categories = {}
    for category, weight in weights.items():
        rows = [row for row in results if row["category"] == category]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["family"]].append(row)
        if len(rows) < 20 or len(grouped) < 5:
            raise ValueError("Incomplete category results cannot be scored")
        families = {}
        for family, items in sorted(grouped.items()):
            passed = sum(row["status"] == "pass" for row in items)
            families[family] = {"passed": passed, "total": len(items), "pass_fraction": passed / len(items)}
        fraction = sum(row["pass_fraction"] for row in families.values()) / len(families)
        categories[category] = {
            "weight": weight, "score": fraction * weight,
            "passed": sum(row["status"] == "pass" for row in rows), "total": len(rows),
            "families": families,
        }
    counts = Counter(row["status"] for row in results)
    return {
        "score_out_of_100": sum(row["score"] for row in categories.values()),
        "passed": counts["pass"], "failed": counts["fail"], "errors": counts["error"],
        "executed": len(results), "categories": categories,
        "levels": dict(Counter(row["level"] for row in results)),
        "failed_by_severity": dict(Counter(row["severity"] for row in results if row["status"] != "pass")),
        "behavior_families": sum(len(row["families"]) for row in categories.values()),
    }


def execute_scenario(case):
    row = {"id": case.id, "category": case.category, "family": case.family,
           "title": case.title, "severity": case.severity, "level": case.level,
           "inputs": case.inputs, "status": "error"}
    started = time.perf_counter()
    client = None
    try:
        with offline_api() as client:
            row["evidence"] = case.run(client)
            attempts = list(client.attempted_external_delivery)
            if attempts:
                raise CheckFailure("Scenario attempted a forbidden external delivery", expected=0, observed=len(attempts))
        row["status"] = "pass"
    except CheckFailure as exc:
        row.update(status="fail", failure=str(exc), expected=exc.expected, observed=exc.observed)
    except AssertionError as exc:
        row.update(status="fail", failure=str(exc), error_type=type(exc).__name__)
    except Exception as exc:
        row.update(status="error", failure=str(exc), error_type=type(exc).__name__)
    finally:
        if client is not None:
            row["blocked_external_attempts"] = len(client.attempted_external_delivery)
    # This is audit harness runtime, not production chat response latency.
    row["harness_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return row


def run_audit():
    before = source_hash()
    rubric = json.loads(RUBRIC.read_text())
    cases = collect_scenarios()
    manifest = [{"id": c.id, "category": c.category, "family": c.family, "title": c.title,
                 "severity": c.severity, "level": c.level, "inputs": c.inputs} for c in cases]
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    evaluator_files = [Path(__file__), ROOT / "script/quality_v2_support.py",
                       ROOT / "script/simulate_chatbot_quality.py", RUBRIC]
    evaluator_files.extend(ROOT / f"script/quality_v2_{name}.py" for name in MODULES)
    def evaluator_fingerprint():
        digest = hashlib.sha256()
        for path in sorted(evaluator_files):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    evaluator_hash = evaluator_fingerprint()
    results = []
    for index, case in enumerate(cases, 1):
        results.append(execute_scenario(case))
        if index % 25 == 0:
            print(f"Executed {index}/{len(cases)} independent scenarios", file=sys.stderr, flush=True)
    after = source_hash()
    if before != after:
        raise RuntimeError("Application changed during the audit; results are not comparable")
    if evaluator_fingerprint() != evaluator_hash:
        raise RuntimeError("Evaluation files changed during the audit; rerun the frozen suite")
    result_ids = [row["id"] for row in results]
    if result_ids != [case.id for case in cases]:
        raise RuntimeError("Missing, duplicated or reordered execution evidence")
    previous_path = ROOT / "docs/quality/chatbot-simulation-after.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    return {
        "version": rubric["version"], "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "broad_offline_audit_of_unchanged_application", "app_source_sha256": before,
        "application_unchanged": True, "manifest_sha256": manifest_hash,
        "same_application_as_previous_v1_report": before == previous.get("app_source_sha256"),
        "evaluator_sha256": evaluator_hash, "live_model_score": None,
        "scoring": rubric["scoring"], "rubric": rubric,
        "limitations": [
            "This broader rubric differs from the earlier 40-case regression suite; scores are not a like-for-like trend.",
            "New synthetic cases were authored after the previous fixes; this measures generalization to new cases, not an independent production benchmark.",
            "Real LLM answers, external store/order data, production latency/availability, browser accessibility and actual customer resolution remain unmeasured.",
            "Safety-critical failures remain material regardless of the average score.",
        ],
        **score_results(results, rubric["categories"]), "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"cases", "rubric", "categories"}}, indent=2, ensure_ascii=False))
    return 1 if report["failed"] or report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
