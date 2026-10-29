#!/usr/bin/env python3
"""Replay everyday warranty and Sales goals through the actual isolated APIs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from script.simulate_chatbot_broad_quality import execute_scenario, source_hash

MODULES = ("warranty", "sales_catalog", "sales_service")
RUBRIC = ROOT / "docs/quality/business-audit-v3-rubric.json"


def summarize(rows, categories):
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case evidence")
    if {row["category"] for row in rows} != set(categories):
        raise ValueError("Missing or unknown business category")
    if any(row["status"] not in {"pass", "fail", "error"} for row in rows):
        raise ValueError("Unexecuted outcomes cannot be reported")
    by_category = {}
    for category in categories:
        cases = [row for row in rows if row["category"] == category]
        if len(cases) < 18 or len({row["family"] for row in cases}) < 5:
            raise ValueError(f"Insufficient everyday inquiry coverage: {category}")
        by_category[category] = {"executed": len(cases),
            **{status: sum(row["status"] == status for row in cases) for status in ("pass", "fail", "error")},
            "behavior_families": len({row["family"] for row in cases})}
    counts = Counter(row["status"] for row in rows)
    return {"executed": len(rows), "passed": counts["pass"], "failed": counts["fail"], "errors": counts["error"],
            "categories": by_category, "levels": dict(Counter(row["level"] for row in rows)),
            "business_score_out_of_100": None,
            "score_reason": "Customer inquiry frequencies are unknown; report observable goals by business category instead of invented weights."}


def collect_scenarios():
    cases = []
    for name in MODULES:
        cases.extend(importlib.import_module(f"script.quality_v3_{name}").scenarios())
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Case identifiers must be unique")
    if any(case.level != "http_api" for case in cases):
        raise ValueError("This audit requires actual isolated HTTP customer journeys")
    return cases


def fingerprint():
    paths = [Path(__file__), RUBRIC, ROOT / "script/quality_v2_support.py",
             ROOT / "script/simulate_chatbot_broad_quality.py", ROOT / "script/simulate_chatbot_quality.py"]
    paths.extend(ROOT / f"script/quality_v3_{name}.py" for name in MODULES)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_audit():
    app_hash = source_hash()
    cases = collect_scenarios()
    evaluator_hash = fingerprint()
    rubric = json.loads(RUBRIC.read_text())
    rows = []
    for index, case in enumerate(cases, 1):
        rows.append(execute_scenario(case))
        if index % 20 == 0:
            print(f"Replayed {index}/{len(cases)} customer scenarios", file=sys.stderr, flush=True)
    if app_hash != source_hash() or evaluator_hash != fingerprint():
        raise RuntimeError("Application or evaluation changed during the run; freeze and replay")
    if [row["id"] for row in rows] != [case.id for case in cases]:
        raise RuntimeError("Incomplete execution evidence")
    previous = json.loads((ROOT / "docs/quality/chatbot-broad-audit-v2.json").read_text())
    return {"version": rubric["version"], "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "app_source_sha256": app_hash, "evaluator_sha256": evaluator_hash,
            "same_application_as_v2": app_hash == previous["app_source_sha256"],
            "application_unchanged": True, "mode": "offline_real_http_customer_journeys",
            "live_model_score": None, "rubric": rubric,
            "limitations": [
                "Warranty natural-language extraction and optional Sales intent models run without OpenAI credentials; this evaluates their deterministic/fallback path.",
                "A clarification or a deliberate confirmation prompt is not a failure by itself.",
                "HTTP payloads and persisted state are observed; browser rendering, live model answers, live stock, order status and actual notifications are not measured.",
                "Cases are selected synthetic business examples, not measured customer traffic proportions or a universal benchmark."
            ],
            **summarize(rows, rubric["categories"]), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"cases", "rubric"}}, ensure_ascii=False, indent=2))
    return 1 if report["failed"] or report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
