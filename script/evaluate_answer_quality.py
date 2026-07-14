#!/usr/bin/env python3
"""Offline release gate for deterministic customer-answer invariants.

This intentionally makes no live model calls. It protects high-risk routing,
grounding guards, multilingual fallbacks, and non-negotiable prompt policies on
every commit. Live model candidates should be evaluated against the same case
catalog in staging before changing ``OPENAI_AGENT_MODEL``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from answer_guard import sanitize_agent_response  # noqa: E402
from intent_router import infer_forced_tool  # noqa: E402


def main() -> int:
    cases = json.loads((ROOT / "data" / "answer_quality_eval_cases.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    passed = 0

    for case in cases["routing"]:
        actual = infer_forced_tool(case["query"])
        if actual != case["expected_tool"]:
            failures.append(
                f"routing query={case['query']!r} expected={case['expected_tool']!r} actual={actual!r}"
            )
        else:
            passed += 1

    for case in cases["guards"]:
        output = sanitize_agent_response(
            case["response"],
            tools_called=case.get("tools_called", []),
            user_query=case["query"],
            tool_results=case.get("tool_results", []),
        )
        case_failed = False
        for needle in case.get("must_contain", []):
            if needle not in output:
                failures.append(f"guard {case['name']}: missing {needle!r} in {output!r}")
                case_failed = True
        for needle in case.get("must_not_contain", []):
            if needle in output:
                failures.append(f"guard {case['name']}: forbidden {needle!r} in {output!r}")
                case_failed = True
        if not case_failed:
            passed += 1

    prompt_source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    for required in cases["prompt_requirements"]:
        if required not in prompt_source:
            failures.append(f"prompt requirement missing: {required!r}")
        else:
            passed += 1

    summary = {
        "status": "pass" if not failures else "fail",
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
