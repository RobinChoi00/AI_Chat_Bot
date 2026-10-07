#!/usr/bin/env python3
"""Live LLM evaluation runner.

Sends golden-set queries to the real OpenAI model and scores:
  - tool_selection:  Did the agent pick the right first tool?
  - faithfulness:    Does the response stay within tool-grounded facts?
  - refusal:         Are off-topic queries refused, on-topic ones answered?
  - language:        Does the reply language match the query language?

Usage
-----
  # Dry-run (schema / structure check only, no API calls)
  python script/evaluate_llm_quality.py --dry-run

  # Live run against staging or production (requires OPENAI_API_KEY)
  python script/evaluate_llm_quality.py

  # Live run, save report to file
  python script/evaluate_llm_quality.py --out data/reports/llm_eval_report.json

Exit code 0 = all checks pass, 1 = at least one failure.
CI should run ``--dry-run`` only (no API cost).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

CASES_PATH = ROOT / "data" / "llm_eval_cases.json"


# ---------------------------------------------------------------------------
# Dry-run: structure validation only
# ---------------------------------------------------------------------------

def validate_structure(cases: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty = pass)."""
    errors: List[str] = []

    for section in ("tool_selection", "faithfulness", "refusal", "language_consistency"):
        if section not in cases:
            errors.append(f"Missing top-level section: {section}")
            continue
        if not isinstance(cases[section], list) or len(cases[section]) == 0:
            errors.append(f"Section '{section}' must be a non-empty list")
            continue
        for i, case in enumerate(cases[section]):
            if "id" not in case:
                errors.append(f"{section}[{i}]: missing 'id'")
            if "query" not in case:
                errors.append(f"{section}[{i}]: missing 'query'")

    ids = []
    for section in ("tool_selection", "faithfulness", "refusal", "language_consistency"):
        for case in cases.get(section, []):
            cid = case.get("id", "")
            if cid in ids:
                errors.append(f"Duplicate case id: {cid}")
            ids.append(cid)

    return errors


# ---------------------------------------------------------------------------
# Live evaluation helpers
# ---------------------------------------------------------------------------

def _get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    from config import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT
    return OpenAI(
        api_key=api_key,
        timeout=float(OPENAI_REQUEST_TIMEOUT),
        max_retries=int(OPENAI_MAX_RETRIES),
    )


def _get_model() -> str:
    try:
        from config import OPENAI_AGENT_MODEL
        return OPENAI_AGENT_MODEL
    except ImportError:
        return os.environ.get("OPENAI_AGENT_MODEL", "gpt-4o")


def _load_tool_schemas():
    try:
        from agent_tools import TOOL_SCHEMAS
        return TOOL_SCHEMAS
    except ImportError:
        return []


def _call_agent(client, model: str, query: str, tool_schemas: list) -> Dict[str, Any]:
    """Single agent call. Returns {tool_called, response, error}."""
    from main import build_system_prompt
    system = build_system_prompt("osakichair.com")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            tool_choice="auto" if tool_schemas else None,
            parallel_tool_calls=False,
        )
        elapsed = int((time.time() - t0) * 1000)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        first_tool = tool_calls[0].function.name if tool_calls else None
        return {
            "tool_called": first_tool,
            "response": msg.content or "",
            "elapsed_ms": elapsed,
            "error": None,
        }
    except Exception as exc:
        return {
            "tool_called": None,
            "response": "",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "error": str(exc),
        }


def _detect_language(text: str) -> str:
    ko = len(re.findall(r"[\uac00-\ud7af]", text))
    es = len(re.findall(
        r"\b(hola|gracias|silla|precio|pedido|garant[ií]a|cu[aá]nto|nuestro|equipo|por\s+favor)\b",
        text, re.I,
    ))
    if ko > 3:
        return "ko"
    if es > 2:
        return "es"
    return "en"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_tool_selection(cases: list, client, model: str, schemas: list) -> List[Dict]:
    results = []
    for case in cases:
        r = _call_agent(client, model, case["query"], schemas)
        passed = r["tool_called"] == case["expected_tool"]
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected": case["expected_tool"],
            "actual": r["tool_called"],
            "elapsed_ms": r["elapsed_ms"],
            "error": r["error"],
        })
    return results


def score_faithfulness(cases: list, client, model: str, schemas: list) -> List[Dict]:
    results = []
    for case in cases:
        r = _call_agent(client, model, case["query"], schemas)
        passed = True
        reasons = []

        if case.get("must_call_tool") and r["tool_called"] != case["must_call_tool"]:
            passed = False
            reasons.append(f"expected tool {case['must_call_tool']}, got {r['tool_called']}")

        resp_lower = r["response"].lower()
        for needle in case.get("response_must_contain", []):
            if needle.lower() not in resp_lower:
                passed = False
                reasons.append(f"missing: {needle!r}")
        for needle in case.get("response_must_not_contain", []):
            if needle.lower() in resp_lower:
                passed = False
                reasons.append(f"forbidden: {needle!r}")

        results.append({
            "id": case["id"],
            "passed": passed,
            "reasons": reasons,
            "tool_called": r["tool_called"],
            "response_snippet": r["response"][:300],
            "elapsed_ms": r["elapsed_ms"],
            "error": r["error"],
        })
    return results


def score_refusal(cases: list, client, model: str, schemas: list) -> List[Dict]:
    results = []
    refusal_signals = (
        "i can only", "outside my scope", "i'm here to help with",
        "i am here to help with", "assist with massage chair",
        "i specialize in", "not able to help with that",
    )
    for case in cases:
        r = _call_agent(client, model, case["query"], schemas)
        resp_lower = r["response"].lower()
        looks_refused = any(s in resp_lower for s in refusal_signals)
        expected_refusal = case.get("expect_refusal", False)
        passed = looks_refused == expected_refusal
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected_refusal": expected_refusal,
            "detected_refusal": looks_refused,
            "response_snippet": r["response"][:300],
            "elapsed_ms": r["elapsed_ms"],
            "error": r["error"],
        })
    return results


def score_language(cases: list, client, model: str, schemas: list) -> List[Dict]:
    results = []
    for case in cases:
        r = _call_agent(client, model, case["query"], schemas)
        detected = _detect_language(r["response"])
        expected = case.get("response_language", "en")
        passed = detected == expected
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected_lang": expected,
            "detected_lang": detected,
            "response_snippet": r["response"][:300],
            "elapsed_ms": r["elapsed_ms"],
            "error": r["error"],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="LLM quality evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate structure only (no API calls)")
    parser.add_argument("--out", type=str, default=None, help="Write JSON report to file")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    # Always validate structure
    struct_errors = validate_structure(cases)
    if struct_errors:
        print("STRUCTURE ERRORS:")
        for e in struct_errors:
            print(f"  - {e}")
        return 1

    if args.dry_run:
        total = sum(len(cases.get(s, [])) for s in ("tool_selection", "faithfulness", "refusal", "language_consistency"))
        print(json.dumps({
            "status": "pass",
            "mode": "dry_run",
            "total_cases": total,
            "sections": {
                "tool_selection": len(cases.get("tool_selection", [])),
                "faithfulness": len(cases.get("faithfulness", [])),
                "refusal": len(cases.get("refusal", [])),
                "language_consistency": len(cases.get("language_consistency", [])),
            },
        }, indent=2))
        return 0

    # Live mode
    client = _get_openai_client()
    if client is None:
        print("ERROR: OPENAI_API_KEY not set. Use --dry-run for offline checks.")
        return 1

    model = _get_model()
    schemas = _load_tool_schemas()
    print(f"Model: {model}, Cases: {CASES_PATH.name}")

    report: Dict[str, Any] = {"model": model, "sections": {}}
    total_passed = 0
    total_cases = 0

    # Tool selection
    ts = score_tool_selection(cases.get("tool_selection", []), client, model, schemas)
    p = sum(1 for r in ts if r["passed"])
    report["sections"]["tool_selection"] = {"passed": p, "total": len(ts), "details": ts}
    total_passed += p
    total_cases += len(ts)
    print(f"  tool_selection: {p}/{len(ts)}")

    # Faithfulness
    ff = score_faithfulness(cases.get("faithfulness", []), client, model, schemas)
    p = sum(1 for r in ff if r["passed"])
    report["sections"]["faithfulness"] = {"passed": p, "total": len(ff), "details": ff}
    total_passed += p
    total_cases += len(ff)
    print(f"  faithfulness:   {p}/{len(ff)}")

    # Refusal
    rf = score_refusal(cases.get("refusal", []), client, model, schemas)
    p = sum(1 for r in rf if r["passed"])
    report["sections"]["refusal"] = {"passed": p, "total": len(rf), "details": rf}
    total_passed += p
    total_cases += len(rf)
    print(f"  refusal:        {p}/{len(rf)}")

    # Language
    lg = score_language(cases.get("language_consistency", []), client, model, schemas)
    p = sum(1 for r in lg if r["passed"])
    report["sections"]["language"] = {"passed": p, "total": len(lg), "details": lg}
    total_passed += p
    total_cases += len(lg)
    print(f"  language:       {p}/{len(lg)}")

    all_pass = total_passed == total_cases
    report["status"] = "pass" if all_pass else "fail"
    report["total_passed"] = total_passed
    report["total_cases"] = total_cases
    report["pass_rate_pct"] = round(total_passed / total_cases * 100, 1) if total_cases else 0

    print(f"\nOverall: {total_passed}/{total_cases} ({report['pass_rate_pct']}%) — {'PASS' if all_pass else 'FAIL'}")

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report_json, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report_json)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
