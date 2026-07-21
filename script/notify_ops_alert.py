#!/usr/bin/env python3
"""Send an ops alert email (cron failures, RC health, etc.)."""

from __future__ import annotations

import argparse
import os
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Send ops alert email.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", type=Path, default=None)
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="When using --body-file, include only the last N lines.",
    )
    args = parser.parse_args()

    _load_env_file(ROOT / ".env")

    from ops_notify import send_ops_alert  # noqa: WPS433

    body = args.body
    if args.body_file:
        text = args.body_file.read_text(encoding="utf-8", errors="replace")
        if args.tail and args.tail > 0:
            lines = text.splitlines()
            text = "\n".join(lines[-args.tail :])
        body = f"{body}\n{text}".strip() if body else text

    if not body:
        print("No alert body provided.", file=sys.stderr)
        return 2

    ok = send_ops_alert(args.subject, body)
    if not ok:
        print("Alert not sent (missing SMTP or recipients).", file=sys.stderr)
        return 1
    print("Alert sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
