"""Golden fit checks for Sales recommend quality.

Two layers:

1. **Spec gates** — doorway / weight / wall from ``spec_index.json``.
2. **Case-book goldens** — pinned lookups from the practical-case Excel
   (exported to ``data/sales/practical_cases_*.csv.gz``) plus a full
   sweep that every listed model still passes those spec gates.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_agent import _collect_tier_candidates  # noqa: E402
from sales_cases import TIER_BUDGETS, cases_available, lookup_case  # noqa: E402
from sales_spec_index import (  # noqa: E402
    doorway_inches_for_model,
    doorway_ok,
    lookup_fit_spec,
    wall_ok,
    weight_ok,
)

ROOT = Path(__file__).resolve().parent.parent
NARROW_MAX_IN = 32.0

# Pinned lookups from the Excel case books. Intensity/foot match the
# chat defaults after core four (goal → intensity, foot → Not Important)
# except where the scenario needs a specific goal.
_OSAKI_LOOKUPS = (
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "No Space Constraint",
        "Osaki OS-Champ",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
        "Osaki OS-Champ",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Small Room",
        "Titan Malibu Sync",
    ),
    (
        'Average (5\'4"–5\'11")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Balanced",
        "Not Important",
        "No Space Constraint",
        "Titan Pro Cura 4D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Balanced",
        "Not Important",
        "Narrow Doorway",
        "Titan Pro 4D Astro",
    ),
    (
        'Average (5\'4"–5\'11")',
        "≤180 lb",
        "$5,000–$6,999",
        "Lower Back",
        "Strong",
        "Not Important",
        "No Space Constraint",
        "Osaki 4D Bravo Duo Mech 4D+3D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "261–300 lb",
        "$5,000–$6,999",
        "Lower Back",
        "Strong",
        "Not Important",
        "No Space Constraint",
        "Osaki Duke XL 4D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "301+ lb",
        "$10,000+",
        "Full-Body Relaxation",
        "Gentle",
        "Not Important",
        "No Space Constraint",
        "Osaki OS-Pro 4D+3D DuoMax SE",
    ),
    (
        'Tall (6\'0"–6\'2")',
        "221–260 lb",
        "$7,000–$9,999",
        "Full-Body Relaxation",
        "Gentle",
        "Important",
        "No Space Constraint",
        "Osaki 4D+3D Manhattan Duo",
    ),
    (
        'Extra Tall (6\'3"+)',
        "301+ lb",
        "$10,000+",
        "Stretching & Mobility",
        "Highly Adjustable",
        "Top Priority",
        "No Space Constraint",
        "Osaki OS-Pro 4D+3D DuoMax SE",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "$3,000–$4,999",
        "Foot & Calf",
        "Balanced",
        "Important",
        "Narrow Doorway",
        "Titan TP-Epic 4D",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "$10,000+",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
        "Osaki Platinum - Escape Duo 4D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "181–220 lb",
        "$3,000–$4,999",
        "Upper Back",
        "Balanced",
        "Not Important",
        "Small Room",
        "Osaki OS-Highpointe 4D",
    ),
    (
        'Tall (6\'0"–6\'2")',
        "≤180 lb",
        "Under $3,000",
        "Stretching & Mobility",
        "Highly Adjustable",
        "Not Important",
        "No Space Constraint",
        "Titan Pro 4D Astro",
    ),
    (
        'Petite (<5\'4")',
        "301+ lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "No Space Constraint",
        "Grande XL-Big and Tall",
    ),
)

_TITAN_LOOKUPS = (
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "No Space Constraint",
        "Titan eCabin 3D",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
        "Titan eCabin 3D",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Small Room",
        "Osaki OS-3D Hamilton LE",
    ),
    (
        'Average (5\'4"–5\'11")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Balanced",
        "Not Important",
        "No Space Constraint",
        "Titan eCabin 3D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "≤180 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Balanced",
        "Not Important",
        "Narrow Doorway",
        "Titan Rejuv 4D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "≤180 lb",
        "$5,000–$6,999",
        "Lower Back",
        "Strong",
        "Not Important",
        "No Space Constraint",
        "Osaki 4D Bravo Duo Mech 4D+3D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "261–300 lb",
        "$5,000–$6,999",
        "Lower Back",
        "Strong",
        "Not Important",
        "No Space Constraint",
        "Osaki Duke XL 4D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "301+ lb",
        "$10,000+",
        "Full-Body Relaxation",
        "Gentle",
        "Not Important",
        "No Space Constraint",
        "Osaki OS-Pro 4D+3D DuoMax",
    ),
    (
        'Tall (6\'0"–6\'2")',
        "221–260 lb",
        "$7,000–$9,999",
        "Full-Body Relaxation",
        "Gentle",
        "Important",
        "No Space Constraint",
        "Osaki 4D+3D Manhattan Duo",
    ),
    (
        'Extra Tall (6\'3"+)',
        "301+ lb",
        "$10,000+",
        "Stretching & Mobility",
        "Highly Adjustable",
        "Top Priority",
        "No Space Constraint",
        "Osaki OS-Pro 4D+3D DuoMax",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "$3,000–$4,999",
        "Foot & Calf",
        "Balanced",
        "Important",
        "Narrow Doorway",
        "Titan TP-Epic 4D",
    ),
    (
        'Petite (<5\'4")',
        "≤180 lb",
        "$10,000+",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
        "Osaki Platinum - Escape Duo 4D",
    ),
    (
        'Average (5\'4"–5\'11")',
        "181–220 lb",
        "$3,000–$4,999",
        "Upper Back",
        "Balanced",
        "Not Important",
        "Small Room",
        "Osaki OS-4D Achilles",
    ),
    (
        'Tall (6\'0"–6\'2")',
        "≤180 lb",
        "Under $3,000",
        "Stretching & Mobility",
        "Highly Adjustable",
        "Not Important",
        "No Space Constraint",
        "Titan eCabin 3D",
    ),
    (
        'Petite (<5\'4")',
        "301+ lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "No Space Constraint",
        "Grande XL-Big and Tall",
    ),
)

_EMPTY_LOOKUPS = (
    (
        "osaki",
        'Petite (<5\'4")',
        "301+ lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
    ),
    (
        "titan",
        'Petite (<5\'4")',
        "301+ lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
    ),
    (
        "osaki",
        'Petite (<5\'4")',
        "261–300 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
    ),
    (
        "titan",
        'Petite (<5\'4")',
        "261–300 lb",
        "Under $3,000",
        "Neck & Shoulders",
        "Gentle",
        "Not Important",
        "Narrow Doorway",
    ),
)


def _prefs(row: tuple) -> dict[str, str]:
    height, weight, budget, goal, intensity, foot, space = row[:7]
    return {
        "height": height,
        "weight": weight,
        "budget": budget,
        "goal": goal,
        "intensity": intensity,
        "foot": foot,
        "space": space,
    }


def test_spec_index_highpointe_assembled_override():
    spec = lookup_fit_spec("Osaki OS-Highpointe 4D")
    assert spec is not None
    assert spec.door_asm_in == 36.5
    assert doorway_inches_for_model("Osaki OS-Highpointe 4D", mode="assembled") == 36.5


def test_spec_index_doorway_assembled_hard_filter():
    asm = doorway_inches_for_model("Osaki OS-Champ", mode="assembled")
    assert asm is not None
    assert doorway_ok("Osaki OS-Champ", limit_in=asm, mode="assembled")
    assert not doorway_ok("Osaki OS-Champ", limit_in=asm - 0.5, mode="assembled")


def test_highpointe_fails_narrow_30_assembled():
    assert not doorway_ok(
        "Osaki OS-Highpointe 4D",
        limit_in=30.0,
        mode="assembled",
    )


def test_weight_gate_rejects_under_capacity():
    assert weight_ok("Osaki OS-Champ", "≤180 lb")
    assert not weight_ok("Osaki OS-Champ", "261–300 lb")
    assert not weight_ok("Osaki OS-Champ", "301+ lb")
    assert weight_ok("Osaki Duke XL 4D", "261–300 lb")
    assert weight_ok("Grande XL-Big and Tall", "301+ lb")


def test_wall_gate_small_room():
    assert wall_ok("Osaki OS-Highpointe 4D", "Small Room")
    assert not wall_ok("Osaki OS-Champ", "Small Room")  # 9 in clearance
    assert not wall_ok("Totally Unknown Chair XYZ", "Small Room")


def test_unknown_specs_fail_closed_when_fit_is_constrained():
    unknown = "Totally Unknown Chair XYZ"
    assert not doorway_ok(unknown, limit_in=32, mode="assembled")
    assert not weight_ok(unknown, "≤180 lb")
    assert not wall_ok(unknown, "Small Room")
    assert doorway_ok(unknown, limit_in=None, mode="assembled")
    assert wall_ok(unknown, "No Space Issue")


@pytest.mark.parametrize(
    "height,weight,budget,goal,intensity,foot,space,primary",
    _OSAKI_LOOKUPS,
    ids=[f"osaki-{i}" for i in range(len(_OSAKI_LOOKUPS))],
)
def test_osaki_casebook_primary(
    height, weight, budget, goal, intensity, foot, space, primary
):
    if not cases_available("osaki"):
        pytest.skip("practical_cases_osaki.csv.gz not present")
    match = lookup_case(
        _prefs((height, weight, budget, goal, intensity, foot, space)),
        brand="osaki",
    )
    assert match is not None
    assert match.primary_model == primary


@pytest.mark.parametrize(
    "height,weight,budget,goal,intensity,foot,space,primary",
    _TITAN_LOOKUPS,
    ids=[f"titan-{i}" for i in range(len(_TITAN_LOOKUPS))],
)
def test_titan_casebook_primary(
    height, weight, budget, goal, intensity, foot, space, primary
):
    if not cases_available("titan"):
        pytest.skip("practical_cases_titan.csv.gz not present")
    match = lookup_case(
        _prefs((height, weight, budget, goal, intensity, foot, space)),
        brand="titan",
    )
    assert match is not None
    assert match.primary_model == primary


@pytest.mark.parametrize(
    "brand,height,weight,budget,goal,intensity,foot,space",
    _EMPTY_LOOKUPS,
    ids=[f"empty-{row[0]}-{i}" for i, row in enumerate(_EMPTY_LOOKUPS)],
)
def test_casebook_no_verified_match(
    brand, height, weight, budget, goal, intensity, foot, space
):
    if not cases_available(brand):
        pytest.skip(f"practical_cases_{brand}.csv.gz not present")
    match = lookup_case(
        _prefs((height, weight, budget, goal, intensity, foot, space)),
        brand=brand,
    )
    assert match is not None
    assert match.primary_model == "NO VERIFIED MATCH"


@pytest.mark.parametrize("brand", ("osaki", "titan"))
def test_casebook_models_pass_spec_gates(brand):
    """Every Excel pick with known specs must clear weight / doorway / wall."""
    path = ROOT / "data" / "sales" / f"practical_cases_{brand}.csv.gz"
    if not path.is_file():
        pytest.skip(f"missing {path.name}")
    fails: list[str] = []
    checked = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row in csv_rows(handle):
            weight = row["Weight"]
            space = row["Space Constraint"]
            for col in (
                "Primary Model",
                "Alternative Model 1",
                "Alternative Model 2",
            ):
                name = (row.get(col) or "").strip()
                if not name or name == "NO VERIFIED MATCH":
                    continue
                spec = lookup_fit_spec(name)
                if spec is None:
                    continue
                checked += 1
                if spec.max_user_lb is not None and not weight_ok(name, weight):
                    fails.append(f"{col} {name} fails {weight}")
                if (
                    space == "Narrow Doorway"
                    and spec.door_asm_in is not None
                    and not doorway_ok(
                    name, limit_in=NARROW_MAX_IN, mode="assembled"
                    )
                ):
                    fails.append(f"{col} {name} fails {NARROW_MAX_IN}in doorway")
                if (
                    space == "Small Room"
                    and spec.wall_clearance_in is not None
                    and not wall_ok(name, "Small Room")
                ):
                    fails.append(f"{col} {name} fails small-room wall")
    assert checked > 1000, f"{brand}: expected a populated case book"
    assert fails == [], f"{brand} spec-gate mismatches: {fails[:12]}"


def csv_rows(handle):
    import csv

    return csv.DictReader(handle)


@pytest.mark.parametrize("brand", ("osaki", "titan"))
def test_runtime_30in_assembled_never_picks_highpointe(brand):
    prefs = {
        "height": 'Average (5\'4"–5\'11")',
        "weight": "≤180 lb",
        "space": "Narrow Doorway",
        "goal": "Neck & Shoulders",
        "intensity": "Balanced",
        "foot": "Not Important",
        "doorway_in": "30",
        "doorway_fit": "assembled",
    }
    names: list[str] = []
    used: set[str] = set()
    for _label, budgets in TIER_BUDGETS:
        for name, _reason, _door in _collect_tier_candidates(
            prefs, budgets=budgets, brand=brand, used_models=used
        ):
            names.append(name)
            used.add(name)
    assert names, f"{brand}: expected some 30in candidates"
    assert all("Highpointe" not in n for n in names)
    for name in names:
        assert doorway_ok(name, limit_in=30.0, mode="assembled")


@pytest.mark.parametrize("brand", ("osaki", "titan"))
def test_runtime_heavy_band_never_picks_champ(brand):
    prefs = {
        "height": 'Average (5\'4"–5\'11")',
        "weight": "261–300 lb",
        "space": "No Space Constraint",
        "goal": "Lower Back",
        "intensity": "Strong",
        "foot": "Not Important",
    }
    names: list[str] = []
    used: set[str] = set()
    for _label, budgets in TIER_BUDGETS:
        for name, _reason, _door in _collect_tier_candidates(
            prefs, budgets=budgets, brand=brand, used_models=used
        ):
            names.append(name)
            used.add(name)
    assert names
    assert all("Champ" not in n for n in names)
    for name in names:
        assert weight_ok(name, "261–300 lb")
