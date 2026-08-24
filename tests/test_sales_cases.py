"""
tests/test_sales_cases.py
=========================
Practical-case workbook lookup for Sales AI recommendations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_cases import (  # noqa: E402
    apply_payload_codes,
    brand_for_domain,
    cases_available,
    enrich_implied_prefs,
    height_bucket,
    lookup_case,
    merge_prefs_from_hints,
    missing_required,
    rank_case_models,
    short_do_not_recommend,
    weight_bucket,
    budget_bucket,
)


@pytest.fixture(scope="module")
def osaki_ready():
    if not cases_available("osaki"):
        pytest.skip("practical_cases_osaki.csv.gz not present")
    return True


def test_brand_for_domain():
    assert brand_for_domain("titanchair.com") == "titan"
    assert brand_for_domain("www.titanchair.com") == "titan"
    assert brand_for_domain("osakiusa.com") == "titan"
    assert brand_for_domain("https://osakiusa.com/") == "titan"
    assert brand_for_domain("osakimassagechair.com") == "osaki"
    assert brand_for_domain("www.osakimassagechair.com") == "osaki"
    # missing / unknown → OsakiUSA / Titan & USA book
    assert brand_for_domain(None) == "titan"
    assert brand_for_domain("unknown") == "titan"


def test_bucket_mappers():
    assert height_bucket(62) == 'Petite (<5\'4")'
    assert height_bucket(68) == 'Average (5\'4"–5\'11")'
    assert height_bucket(73) == 'Tall (6\'0"–6\'2")'
    assert height_bucket(76) == 'Extra Tall (6\'3"+)'
    assert weight_bucket(170) == "≤180 lb"
    assert weight_bucket(250) == "221–260 lb"
    assert budget_bucket(2800) == "Under $3,000"
    assert budget_bucket(5500) == "$5,000–$6,999"


def test_payload_codes_and_missing(osaki_ready):
    prefs = apply_payload_codes({}, "recommend:budget:under_3000")
    assert prefs["budget"] == "Under $3,000"
    prefs = apply_payload_codes(prefs, "recommend:height:petite")
    prefs = apply_payload_codes(prefs, "recommend:weight:le180")
    prefs = apply_payload_codes(prefs, "recommend:goal:neck")
    assert missing_required(prefs) == ["intensity", "foot", "space"]
    prefs = apply_payload_codes(prefs, "recommend:intensity:gentle")
    prefs = apply_payload_codes(prefs, "recommend:foot:not_important")
    prefs = apply_payload_codes(prefs, "recommend:space:none")
    assert missing_required(prefs) == []


def test_lookup_case_osaki_primary(osaki_ready):
    prefs = {
        "height": 'Petite (<5\'4")',
        "weight": "≤180 lb",
        "budget": "Under $3,000",
        "goal": "Neck & Shoulders",
        "intensity": "Gentle",
        "foot": "Not Important",
        "space": "No Space Constraint",
    }
    match = lookup_case(prefs, brand="osaki")
    assert match is not None
    assert match.primary_model == "Osaki OS-Champ"
    assert match.reason
    assert match.do_not_recommend_when


def test_rank_promotes_higher_sales_priority(osaki_ready):
    # Ignite Sync is priority 5; OS-Champ is priority 1 on the Osaki active list.
    lead, others, note = rank_case_models(
        "Titan Ignite Sync 3D+2D",
        "Osaki OS-Champ",
        "AmaMedic 3D Astoria",
        brand="osaki",
    )
    assert lead == "Osaki OS-Champ"
    assert "Titan Ignite Sync 3D+2D" in others
    assert note is not None


def test_short_do_not_recommend():
    text = (
        "Do not recommend if user exceeds 260 lb; or the delivery path is under 30 in; "
        "or the room cannot spare ~9 in of wall clearance."
    )
    short = short_do_not_recommend(text)
    assert "260 lb" in short
    assert len(short) < len(text)


def test_enrich_implied_foot_from_goal():
    prefs = enrich_implied_prefs({"goal": "Foot & Calf"})
    assert prefs["foot"] == "Important"


def test_merge_hints_into_prefs():
    prefs = merge_prefs_from_hints(
        {},
        height_in=74,
        weight_lb=230,
        budget_usd=6000,
        focus_areas=["back"],
        free_text="lower back pain",
    )
    assert prefs["height"] == 'Tall (6\'0"–6\'2")'
    assert prefs["weight"] == "221–260 lb"
    assert prefs["budget"] == "$5,000–$6,999"
    assert prefs["goal"] == "Lower Back"
