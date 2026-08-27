"""
tests/test_sales_catalog.py
===========================
Smoke tests for the sales catalog façade — verify that the CSV loads,
resolve_product returns a real title for a common hint, and the
recommendation ranker returns a non-empty list when we hand it a strong
signal like *tall + back*.

These are contract checks, not exhaustive spec assertions — the CSV
changes with every Shopify sync so we only assert *shape* and *presence*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_catalog import (  # noqa: E402
    RecommendationRequest,
    compare,
    list_active_products,
    load_product_index,
    parse_recommendation_hints,
    recommend,
    resolve_product,
)


@pytest.fixture(scope="module")
def catalog():
    products = load_product_index()
    if not products:
        pytest.skip("Shopify catalog CSV not present in this environment.")
    return products


def test_catalog_has_multiple_products(catalog):
    assert len(catalog) >= 5


def test_active_products_include_at_least_one(catalog):
    active = list_active_products()
    assert active, "expected at least one active massage chair"
    for product in active:
        assert product.status.lower() == "active"


def test_resolve_product_returns_shape(catalog):
    """Pick any product from the index and confirm resolve() finds it back."""
    known = catalog[0]
    match = resolve_product(known.display_name)
    assert match is not None
    assert match.handle == known.handle


def test_resolve_product_unknown_returns_none():
    assert resolve_product("blender 9000") is None


def test_resolve_case_workbook_aliases(catalog):
    # Common sales-workbook shorthand should still resolve into the Shopify index.
    champ = resolve_product("Osaki OS-Champ")
    assert champ is not None
    vito = resolve_product("OS-3D AI Vito")
    assert vito is not None


def test_parse_hints_extracts_height_and_weight():
    req = parse_recommendation_hints("I am 6'2\" and 230 lb, back pain")
    assert req.height_in == 74
    assert req.weight_lb == 230
    assert "back" in req.focus_areas


def test_parse_hints_handles_metric():
    req = parse_recommendation_hints("I'm 188cm and 95kg with back pain")
    assert req.height_in is not None
    assert 73 <= req.height_in <= 75
    assert req.weight_lb is not None
    assert 205 <= req.weight_lb <= 215
    assert "back" in req.focus_areas


def test_recommend_returns_ranked_matches_for_tall_back(catalog):
    req = RecommendationRequest(height_in=76, weight_lb=230, focus_areas=["back"])
    picks = recommend(req, limit=3)
    assert 1 <= len(picks) <= 3
    for product in picks:
        assert product.status.lower() == "active"


def test_parse_budget_accepts_trailing_dollar_sign():
    req = parse_recommendation_hints("recommend 6000$ chair")
    assert req.budget_usd == 6000


def test_parse_budget_k_notation_and_under():
    assert parse_recommendation_hints("under $5k").budget_usd == 5000
    assert parse_recommendation_hints("around 6k").budget_usd == 6000
    assert parse_recommendation_hints("budget 5000").budget_usd == 5000


def test_parse_budget_ignores_weight_number():
    req = parse_recommendation_hints("I'm 6 ft 2 around 220 lb budget 5000 lower back")
    assert req.height_in == 74
    assert req.weight_lb == 220
    assert req.budget_usd == 5000
    assert "back" in req.focus_areas


def test_recommend_returns_one_pick_per_price_tier(catalog):
    """Default recommend: Value / Mid / Premium shelves."""
    from sales_catalog import price_tier_label

    req = parse_recommendation_hints("recommend a chair for a tall person")
    picks = recommend(req, limit=3)
    assert picks, "expected tiered recommendations"
    labels = [price_tier_label(p.price_usd) for p in picks]
    assert "Value (under ~$3k)" in labels
    assert "Mid-range (~$5–7k)" in labels
    assert "Premium ($7k+)" in labels
    # One chair per shelf — no two picks in the same band.
    assert len(labels) == len(set(labels))


def test_recommend_empty_request_still_returns_reasonable_set(catalog):
    req = RecommendationRequest()
    picks = recommend(req, limit=3)
    # No hints → ranker returns whatever has a positive tier bonus (3D/4D).
    # The public contract is: never crash, always return active items.
    for product in picks:
        assert product.status.lower() == "active"


def test_compare_returns_none_for_unknown_models():
    assert compare("nonsense-a", "nonsense-b") is None


def test_compare_returns_shape_for_two_known_models(catalog):
    active = list_active_products()
    if len(active) < 2:
        pytest.skip("need at least two active products to compare")
    left, right = active[0], active[1]
    result = compare(left.display_name, right.display_name)
    assert result is not None
    assert set(result.keys()) == {"left", "right", "diff"}
    assert result["left"]["model"] == left.display_name
    assert result["right"]["model"] == right.display_name
