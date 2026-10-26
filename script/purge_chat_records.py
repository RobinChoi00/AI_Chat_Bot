#!/usr/bin/env python3
"""Report expired customer records; deletion requires --apply and hold release."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete expired records only when PRIVACY_RETENTION_LEGAL_HOLD=0")
    args = parser.parse_args()
    from privacy_retention import purge_chat_records

    result = purge_chat_records(apply=args.apply)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
