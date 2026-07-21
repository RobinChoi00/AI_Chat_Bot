#!/usr/bin/env python3
"""
Sync OsakiUSA Shopify catalog → raw_data/products_export.csv

Usage:
  python script/sync_shopify_products.py
  python script/sync_shopify_products.py --dry-run
  python script/sync_shopify_products.py --rebuild-clean-csv
"""

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


_load_env_file(ROOT / ".env")

from shopify_product_sync import DEFAULT_CSV_PATH, sync_osakiusa_products  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync OsakiUSA Shopify products to CSV.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and write report only; do not replace products_export.csv",
    )
    parser.add_argument(
        "--min-handles",
        type=int,
        default=None,
        help="Minimum unique handles required (default: max(50, 50%% of current CSV))",
    )
    parser.add_argument(
        "--rebuild-clean-csv",
        action="store_true",
        help="Regenerate data/cleaned_osaki_products.csv after a successful sync",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Output CSV path (default: {DEFAULT_CSV_PATH})",
    )
    args = parser.parse_args()

    result = sync_osakiusa_products(
        csv_path=args.csv_path,
        dry_run=args.dry_run,
        min_handles=args.min_handles,
    )
    print(result.message)
    if result.report_path:
        print(f"Report: {result.report_path}")
    if result.added_handles:
        print(f"Added handles ({len(result.added_handles)}): {', '.join(result.added_handles[:10])}")
        if len(result.added_handles) > 10:
            print(f"  ... and {len(result.added_handles) - 10} more")
    if result.removed_handles:
        print(f"Removed handles ({len(result.removed_handles)}): {', '.join(result.removed_handles[:10])}")
        if len(result.removed_handles) > 10:
            print(f"  ... and {len(result.removed_handles) - 10} more")
    if result.renamed_products:
        print(f"Renamed by title ({len(result.renamed_products)}): "
              f"{result.renamed_products[0]['old_handle']} -> {result.renamed_products[0]['new_handle']}"
              + (f" (+{len(result.renamed_products) - 1} more)" if len(result.renamed_products) > 1 else ""))
    if result.truly_added_handles:
        print(f"New products ({len(result.truly_added_handles)}): {', '.join(result.truly_added_handles[:8])}")
    if result.truly_removed_handles:
        print(f"Removed products ({len(result.truly_removed_handles)}): {', '.join(result.truly_removed_handles[:8])}")
    if result.unchanged_handles:
        print(f"Unchanged handles: {result.unchanged_handles}")

    if not result.ok:
        return 1

    if args.rebuild_clean_csv and not args.dry_run:
        sys.path.insert(0, str(ROOT / "script"))
        try:
            import clean_shopify_data  # noqa: WPS433

            clean_shopify_data.clean_shopify_for_rag()
        except Exception as exc:
            print(f"WARN: clean_shopify_data skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
