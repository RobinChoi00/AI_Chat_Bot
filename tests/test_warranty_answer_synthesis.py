"""Tests for unified Fonz / symptom answer synthesis."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_answer_synthesis import (  # noqa: E402
    append_symptom_insights_to_message,
    build_symptom_fonz_block,
    format_fonz_suggestion_entries,
)


def test_format_fonz_suggestion_includes_meaning():
    block = format_fonz_suggestion_entries(
        [
            {
                "error_code": "C6",
                "meaning": "MOS tube of air pump damage.",
                "troubleshooting": "Check hose connections first.",
            }
        ],
        header="**Related codes:**",
    )
    assert "C6" in block
    assert "MOS tube" in block
    assert "hose" in block.lower()


def test_symptom_block_for_3d_ltx_air():
    ticket = SimpleNamespace(model_name="3D LTX", defect_type="air")
    block = build_symptom_fonz_block(ticket)
    assert block
    assert "Related manufacturer error codes" in block


def test_append_symptom_skips_when_code_present():
    ticket = SimpleNamespace(model_name="3D LTX", defect_type="air")
    msg = "Done.\n\n**Error code C6:** meaning here."
    assert append_symptom_insights_to_message(msg, ticket) == msg


def test_category_fallback_without_model():
    ticket = SimpleNamespace(model_name="", defect_type="air")
    block = build_symptom_fonz_block(ticket)
    assert block
    assert "Common manufacturer error codes" in block or "confirm your model" in block.lower()
