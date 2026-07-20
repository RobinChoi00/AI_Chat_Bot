#!/usr/bin/env python3
"""
Run live LLM warranty scenarios against a deployed API.

Requires OPENAI on the server (smart-start / natural-start use LLM there).
Usage:
  WARRANTY_VERIFY_BASE_URL=https://api.osakichair.com python script/verify_warranty_llm_live.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Callable

import requests

BASE_URL = os.getenv("WARRANTY_VERIFY_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DOMAIN = os.getenv("WARRANTY_VERIFY_DOMAIN", "osakiusa.com")
TIMEOUT = float(os.getenv("WARRANTY_VERIFY_TIMEOUT", "45"))


def _post(session: requests.Session, path: str, body: dict) -> dict[str, Any]:
    resp = session.post(f"{BASE_URL}{path}", json=body, timeout=TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"{path} -> {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _customer_text(payload: dict[str, Any]) -> str:
    text = str(payload.get("assistant_message") or "").strip()
    if text:
        return text
    node = (payload.get("ticket") or {}).get("current_node") or {}
    return str(node.get("prompt") or "").strip()


def _get_path(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _assert_in(payload: dict[str, Any], dotted: str, expected: str) -> None:
    actual = _get_path(payload, dotted)
    if str(actual) != expected:
        raise AssertionError(f"expected {dotted}={expected!r}, got {actual!r}")


def _assert_node(payload: dict[str, Any], node_id: str) -> None:
    _assert_in(payload, "ticket.current_node.node_id", node_id)


def _assert_contains(payload: dict[str, Any], needle: str, *, case_insensitive: bool = False) -> None:
    hay = _customer_text(payload)
    ok = needle.lower() in hay.lower() if case_insensitive else needle in hay
    if not ok:
        node_id = _get_path(payload, "ticket.current_node.node_id")
        raise AssertionError(
            f"expected {needle!r} in customer text; node={node_id!r}; text={hay[:200]!r}"
        )


def _assert_not_contains(payload: dict[str, Any], needle: str) -> None:
    if needle in _customer_text(payload):
        raise AssertionError(f"did not expect {needle!r} in customer text")


def _register(session: requests.Session, session_id: str, model: str = "OS-4000T") -> dict[str, Any]:
    return _post(
        session,
        f"/api/v1/warranty/session/{session_id}/register-model",
        {"model": model, "domain": DOMAIN},
    )


def _smart_start(session: requests.Session, session_id: str, message: str) -> dict[str, Any]:
    return _post(
        session,
        f"/api/v1/warranty/session/{session_id}/smart-start",
        {"message": message, "domain": DOMAIN},
    )


def _natural_start(session: requests.Session, session_id: str, message: str) -> dict[str, Any]:
    return _post(
        session,
        f"/api/v1/warranty/session/{session_id}/natural-start",
        {"message": message, "domain": DOMAIN},
    )


def _quick_start(session: requests.Session, session_id: str, issue_type: str) -> dict[str, Any]:
    return _post(
        session,
        f"/api/v1/warranty/session/{session_id}/quick-start",
        {"issue_type": issue_type, "domain": DOMAIN},
    )


def _answer(session: requests.Session, ticket_id: str, answer: str) -> dict[str, Any]:
    return _post(session, f"/api/v1/warranty/{ticket_id}/answer", {"answer": answer})


ScenarioFn = Callable[[requests.Session, str], tuple[bool, str]]


def scenario_smart_start_footrest_air(session: requests.Session, session_id: str) -> tuple[bool, str]:
    _register(session, session_id)
    payload = _smart_start(session, session_id, "OS-4000T footrest air not inflating")
    _assert_in(payload, "ticket.model_name", "OS-4000T")
    _assert_contains(payload, "footrest", case_insensitive=True)
    _assert_not_contains(payload, "Red blinking light")
    return True, "ok"


def scenario_smart_start_vague_hello(session: requests.Session, session_id: str) -> tuple[bool, str]:
    _register(session, session_id)
    payload = _smart_start(session, session_id, "hello there")
    _assert_node(payload, "issue_type")
    return True, "ok"


def scenario_natural_warranty_help(session: requests.Session, session_id: str) -> tuple[bool, str]:
    _register(session, session_id)
    payload = _natural_start(session, session_id, "I need warranty help")
    node_id = _get_path(payload, "ticket.current_node.node_id")
    if node_id != "issue_type":
        _assert_in(payload, "ticket.issue_type", "warranty")
    return True, "ok"


def scenario_natural_fedex_tracking(session: requests.Session, session_id: str) -> tuple[bool, str]:
    _register(session, session_id)
    payload = _natural_start(session, session_id, "Where is my FedEx tracking number?")
    _assert_contains(payload, "tracking", case_insensitive=True)
    return True, "ok"


def scenario_natural_korean_power(session: requests.Session, session_id: str) -> tuple[bool, str]:
    _register(session, session_id, "OS-4000T")
    payload = _natural_start(session, session_id, "의자가 켜지지 않아요 OS-4000T")
    _assert_contains(payload, "power", case_insensitive=True)
    return True, "ok"


def scenario_shoulders_free_text(session: requests.Session, session_id: str) -> tuple[bool, str]:
    _register(session, session_id, "3D LTX")
    payload = _quick_start(session, session_id, "defect")
    ticket_id = payload["ticket"]["ticket_id"]
    _answer(session, ticket_id, "air")
    payload = _answer(session, ticket_id, "shoulders")
    _assert_in(payload, "ticket.current_node.node_id", "defect_air_shoulders_hissing_q")
    return True, "ok"


SCENARIOS: list[tuple[str, ScenarioFn]] = [
    ("smart-start footrest air", scenario_smart_start_footrest_air),
    ("smart-start vague hello", scenario_smart_start_vague_hello),
    ("natural-start warranty help", scenario_natural_warranty_help),
    ("natural-start FedEx tracking", scenario_natural_fedex_tracking),
    ("natural-start Korean power", scenario_natural_korean_power),
    ("defect shoulders free text", scenario_shoulders_free_text),
]


def main() -> int:
    print(f"Base URL: {BASE_URL}")
    print(f"Domain:   {DOMAIN}\n")
    passed = 0
    failed = 0
    for name, fn in SCENARIOS:
        session_id = str(uuid.uuid4())
        session = requests.Session()
        try:
            ok, detail = fn(session, session_id)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
