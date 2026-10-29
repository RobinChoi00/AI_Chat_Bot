"""General-chat catalog names and prices must use the same exact generation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from product_catalog import resolve_catalog_price, resolve_model_name


@pytest.mark.parametrize("query,canonical", [
    ("OS-Pro Maestro LE 2.0", "Osaki 4D Maestro LE 2.0"),
    ("What is the price of the Maestro LE 2.0?", "Osaki 4D Maestro LE 2.0"),
    ("No, I mean Maestro LE 2.0, not the old OS-Pro Maestro LE.", "Osaki 4D Maestro LE 2.0"),
    ("Please use the old OS-Pro Maestro LE, not Maestro LE 2.0.", "Osaki OS-Pro Maestro LE"),
    ("Can I try Solo Flex on Saturday?", "Osaki Solo Flex 4D"),
    ("Does Solo Flex have Bluetooth speakers?", "Osaki Solo Flex 4D"),
])
def test_general_catalog_keeps_exact_name_and_its_price(query, canonical):
    assert resolve_model_name(query) == canonical
    assert resolve_catalog_price(canonical) is not None
    assert resolve_catalog_price(query) == resolve_catalog_price(canonical)


@pytest.mark.parametrize("query", [
    "Maestro LE 3.0",
    "OS-Pro Maestro LE 2.1",
    "Solo Flex or Bravo Duo Flex",
    "Grande XL or Duke XL",
    "OS-4000T or Hypnos",
])
def test_unknown_or_ambiguous_identity_has_no_price(query):
    assert resolve_model_name(query) is None
    assert resolve_catalog_price(query) is None
