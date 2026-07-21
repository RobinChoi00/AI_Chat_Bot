#!/usr/bin/env python3
"""
Check RingCentral IVR readiness via local /rc/health and email if degraded.

Usage (EC2):
  python3 script/check_rc_ivr_readiness.py
  python3 script/check_rc_ivr_readiness.py --url http://127.0.0.1:8000/rc/health
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _fetch_health(url: str, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.getcode()
        payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Health response was not a JSON object.")
        return status, payload


def _format_health_report(status_code: int, payload: dict) -> str:
    lines = [
        f"HTTP status: {status_code}",
        f"Service status: {payload.get('status', 'unknown')}",
        "",
        "Checks:",
    ]
    checks = payload.get("checks") or {}
    for key, ok in checks.items():
        lines.append(f"  - {key}: {'OK' if ok else 'MISSING/FAILED'}")
    events = payload.get("events") or {}
    calls = payload.get("calls") or {}
    if events:
        lines.extend(["", "Events:", json.dumps(events, indent=2)])
    if calls:
        lines.extend(["", "Calls:", json.dumps(calls, indent=2)])
    lines.extend(
        [
            "",
            "Blocker reminder:",
            "  ApplicationExtension must be enabled by RingCentral platform team",
            "  before after-hours warranty IVR can receive live calls.",
            "",
            "Next steps:",
            "  1) Confirm RC activation email completed",
            "  2) Roman: route after-hours warranty queue to Osaki Warranty IVR",
            "  3) Test call + SMS + email follow-up",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RC IVR health and alert if degraded.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/rc/health",
        help="RC health endpoint (default: local backend)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Email ops when health is degraded",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    _load_env_file(ROOT / ".env")

    try:
        status_code, payload = _fetch_health(args.url, args.timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        report = f"RC health check failed: {exc}"
        print(report)
        if args.notify:
            from ops_notify import send_ops_alert  # noqa: WPS433

            send_ops_alert("[AI Chat Bot] RC IVR health check failed", report)
        return 1

    healthy = payload.get("status") == "ok" and status_code == 200
    report = _format_health_report(status_code, payload)
    print(report)

    if healthy:
        return 0

    if args.notify:
        from ops_notify import send_ops_alert  # noqa: WPS433

        send_ops_alert("[AI Chat Bot] RC IVR degraded", report)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
