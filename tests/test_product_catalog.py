"""Tests for product_catalog model name resolution."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import product_catalog as pc


def test_resolve_known_model_substring():
    pc.load_catalog_titles.cache_clear()
    if not pc._CSV_PATH.is_file():
        return
    resolved = pc.resolve_model_name("solo flex black")
    assert resolved is not None
    assert "solo flex" in resolved.lower()


def test_resolve_empty_returns_none():
    assert pc.resolve_model_name("") is None
    assert pc.resolve_model_name("x") is None
