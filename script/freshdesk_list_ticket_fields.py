#!/usr/bin/env python3
"""
Fetch Freshdesk ticket field choices (status + custom dropdowns) and save a snapshot.

Usage:
  python script/freshdesk_list_ticket_fields.py
  python script/freshdesk_list_ticket_fields.py --refresh
  python script/freshdesk_list_ticket_fields.py --print-only

Requires .env:
  FRESHDESK_DOMAIN, FRESHDESK_API_KEY

Output:
  data/freshdesk_field_choices.json

Share this file with Fred's Ticket Queue extension so status/custom-field IDs
resolve to the correct labels (e.g. Pending Parts vs Parts Need Info).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from freshdesk_field_catalog import (  # noqa: E402
    _CATALOG_PATH,
    fetch_ticket_field_catalog,
    format_parts_related_fields,
    format_status_table,
    save_field_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Freshdesk field ID catalog")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force live fetch even if a cached file exists",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print tables to stdout; do not write JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_CATALOG_PATH,
        help=f"Output JSON path (default: {_CATALOG_PATH})",
    )
    args = parser.parse_args()

    load_dotenv()

    try:
        catalog = fetch_ticket_field_catalog()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_status_table(catalog))
    print()
    parts_block = format_parts_related_fields(catalog)
    print("Parts-related custom fields:")
    print(parts_block)
    print()

    if not args.print_only:
        out = save_field_catalog(catalog, args.output)
        print(f"Wrote {out}")
        print(json.dumps({"domain": catalog.get("domain"), "path": str(out)}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
