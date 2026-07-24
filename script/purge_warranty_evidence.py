#!/usr/bin/env python3
"""CLI: purge warranty evidence older than retention window.

Examples:
  python3 script/purge_warranty_evidence.py              # dry-run
  python3 script/purge_warranty_evidence.py --apply
  WARRANTY_EVIDENCE_RETENTION_DAYS=60 python3 script/purge_warranty_evidence.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge old warranty evidence files.")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Retention days (default: WARRANTY_EVIDENCE_RETENTION_DAYS or 90)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files/rows (default is dry-run)",
    )
    args = parser.parse_args()

    from warranty_evidence_purge import purge_old_evidence  # noqa: WPS433

    result = purge_old_evidence(days=args.days, apply=bool(args.apply))
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
