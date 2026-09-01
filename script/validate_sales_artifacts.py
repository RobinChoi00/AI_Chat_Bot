#!/usr/bin/env python3
"""Fail CI when deployed Sales case/spec artifacts are missing or inconsistent."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from sales_cases import (  # noqa: E402
    BUDGETS,
    FOOT_PRIORITIES,
    GOALS,
    HEIGHTS,
    INTENSITIES,
    SPACES,
    WEIGHTS,
)
from sales_spec_index import doorway_ok, lookup_fit_spec, wall_ok, weight_ok  # noqa: E402

EXPECTED_ROWS = (
    len(HEIGHTS)
    * len(WEIGHTS)
    * len(BUDGETS)
    * len(GOALS)
    * len(INTENSITIES)
    * len(FOOT_PRIORITIES)
    * len(SPACES)
)
MODEL_COLUMNS = ("Primary Model", "Alternative Model 1", "Alternative Model 2")


def validate_spec_index() -> None:
    path = ROOT / "data" / "sales" / "spec_index.json"
    if not path.is_file():
        raise SystemExit(f"missing sales spec index: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models") or []
    with_door = sum(
        1
        for row in models
        if row.get("door_asm_in") is not None or row.get("door_dis_in") is not None
    )
    if len(models) < 150 or with_door < 150:
        raise SystemExit(
            f"spec index coverage too low: models={len(models)} doorway={with_door}"
        )
    print(f"spec index: {len(models)} models, {with_door} with doorway")


def validate_casebook(brand: str) -> None:
    path = ROOT / "data" / "sales" / f"practical_cases_{brand}.csv.gz"
    if not path.is_file():
        raise SystemExit(f"missing case book: {path}")

    rows = 0
    keys: set[tuple[str, ...]] = set()
    listed = 0
    known_specs = 0
    failures: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            key = tuple(
                row[name]
                for name in (
                    "Height",
                    "Weight",
                    "Budget",
                    "Primary Goal",
                    "Intensity",
                    "Foot & Calf Priority",
                    "Space Constraint",
                )
            )
            if key in keys:
                failures.append(f"duplicate scenario key: {key}")
            keys.add(key)

            for column in MODEL_COLUMNS:
                model = (row.get(column) or "").strip()
                if not model or model == "NO VERIFIED MATCH":
                    continue
                listed += 1
                if lookup_fit_spec(model) is None:
                    continue
                known_specs += 1
                if not weight_ok(model, row["Weight"]):
                    failures.append(f"{model}: fails weight {row['Weight']}")
                if row["Space Constraint"] == "Narrow Doorway" and not doorway_ok(
                    model, limit_in=32.0, mode="assembled"
                ):
                    failures.append(f"{model}: fails 32in assembled doorway")
                if row["Space Constraint"] == "Small Room" and not wall_ok(
                    model, "Small Room"
                ):
                    failures.append(f"{model}: fails small-room clearance")

    if rows != EXPECTED_ROWS:
        failures.append(f"row count {rows}, expected {EXPECTED_ROWS}")
    if len(keys) != EXPECTED_ROWS:
        failures.append(f"unique scenario count {len(keys)}, expected {EXPECTED_ROWS}")
    if listed < 10_000:
        failures.append(f"too few listed recommendations: {listed}")
    if known_specs < int(listed * 0.95):
        failures.append(
            f"spec coverage below 95%: known={known_specs} listed={listed}"
        )
    if failures:
        raise SystemExit(f"{brand} validation failed:\n  " + "\n  ".join(failures[:20]))
    print(
        f"{brand}: {rows} unique scenarios, {listed} listed picks, "
        f"{known_specs / listed:.1%} spec coverage"
    )


def main() -> None:
    validate_spec_index()
    for brand in ("osaki", "titan"):
        validate_casebook(brand)
    print("Sales artifacts passed deterministic validation.")


if __name__ == "__main__":
    main()
