#!/usr/bin/env python3
"""
Ingest Fonz's All-in-one Warranty List.xlsx → JSON for warranty lookup + FAISS.

Usage:
  python script/ingest_fonz_warranty.py
  python script/ingest_fonz_warranty.py --input "/path/to/workbook.xlsx"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from fonz_warranty_data import DEFAULT_XLSX_PATH, ingest_workbook  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Fonz warranty Excel to JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_XLSX_PATH,
        help=f"Path to workbook (default: {DEFAULT_XLSX_PATH})",
    )
    args = parser.parse_args()

    stats = ingest_workbook(args.input)
    print(f"✅ Wrote data/fonz_error_codes.json ({stats['error_code_entries']} entries)")
    print(f"✅ Wrote data/fonz_model_diagnostics.json ({stats['model_diagnostics']} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
