"""Tests for model platform family mapping."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from model_families import resolve_family_canonical  # noqa: E402


def test_hamilton_le_maps_to_allure():
    assert resolve_family_canonical("Hamilton LE") == "Allure"


def test_3d_ltx_not_remapped_to_self():
    assert resolve_family_canonical("3D LTX") is None


def test_quantum_maps_to_3d_ltx():
    assert resolve_family_canonical("Quantum") == "3D LTX"
