"""Nickname compare: short names, vs/and splits, no silent family guess."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_catalog import list_active_products, load_product_index  # noqa: E402
from sales_compare import (  # noqa: E402
    lookup_shop_models,
    looks_like_compare_pair,
    split_compare_terms,
)


@pytest.fixture(scope="module")
def catalog():
    products = load_product_index()
    if not products:
        pytest.skip("Shopify catalog CSV not present in this environment.")
    return products


def _names(query: str) -> list[str]:
    return [item.display_name for item in lookup_shop_models(query).matches]


def test_maestro_family_is_ambiguous(catalog):
    names = _names("Maestro")
    if len(list_active_products()) < 3:
        pytest.skip("need the Maestro family in the export")
    assert len(names) >= 2
    assert any("Maestro 4D" in name for name in names)
    assert any(name.endswith("Maestro LE") or "Maestro LE" in name for name in names)


def test_short_names_resolve_uniquely(catalog):
    maestro_le = lookup_shop_models("Maestro LE").unique
    maestro_4d = lookup_shop_models("Maestro 4D").unique
    paragon = lookup_shop_models("Paragon").unique
    if not (maestro_le and maestro_4d and paragon):
        pytest.skip("expected Maestro LE / 4D / Paragon in the export")
    assert "LE" in maestro_le.display_name
    assert "4D" in maestro_4d.display_name
    assert "Paragon" in paragon.display_name
    assert maestro_le.handle != maestro_4d.handle


def test_champ_family_is_ambiguous_os_champ_is_not(catalog):
    champ = lookup_shop_models("Champ")
    os_champ = lookup_shop_models("OS-Champ").unique
    if not champ.matches:
        pytest.skip("Champ not in the export")
    assert len(champ.matches) >= 2
    if os_champ is not None:
        assert "Champ II" not in os_champ.display_name


def test_jupiter_is_not_on_store_catalog(catalog):
    looked = lookup_shop_models("Jupiter")
    assert looked.matches == ()
    assert looked.vague is False


def test_brand_or_mechanism_only_is_vague(catalog):
    assert lookup_shop_models("4D").vague is True
    assert lookup_shop_models("Osaki").vague is True
    assert lookup_shop_models("4D").matches == ()


def test_split_accepts_and_or_korean_without_vs():
    assert split_compare_terms("Maestro and Paragon") == ("Maestro", "Paragon")
    assert split_compare_terms("Maestro or Paragon") == ("Maestro", "Paragon")
    assert split_compare_terms("compare Maestro LE vs Champ II") == (
        "Maestro LE",
        "Champ II",
    )
    assert split_compare_terms("비교 Maestro랑 Paragon") == ("Maestro", "Paragon")
    assert split_compare_terms(
        "Maestro vs Paragon and recommend which one"
    ) == ("Maestro", "Paragon")
    assert split_compare_terms("difference between Maestro and Paragon") == (
        "Maestro",
        "Paragon",
    )


def test_two_nicknames_without_vs_still_look_like_a_pair(catalog):
    if lookup_shop_models("Maestro 4D").unique is None:
        pytest.skip("Maestro 4D missing")
    assert looks_like_compare_pair("Maestro 4D and Paragon") is True
    assert looks_like_compare_pair("I am 6'2\" and 230 lb") is False
