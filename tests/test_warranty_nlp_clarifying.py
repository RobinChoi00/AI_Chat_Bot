"""Tests for did-you-mean clarifying helpers."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_nlp as nlp  # noqa: E402


def test_suggest_closest_option_by_label_fragment():
    options = [
        {"answer_key": "air", "label": "Air / Inflation not working"},
        {"answer_key": "power", "label": "Power issue"},
    ]
    hit = nlp.suggest_closest_option(options, "my chair has inflation problem")
    assert hit is not None
    assert hit.get("answer_key") == "air"


def test_clarifying_includes_did_you_mean():
    node = {
        "type": "question",
        "prompt": "What type of problem?",
        "options": [
            {"answer_key": "air", "label": "Air / Inflation not working"},
            {"answer_key": "power", "label": "Power issue"},
        ],
    }
    msg = nlp.build_clarifying_workflow_message(node, "inflation not working")
    assert "Did you mean" in msg
    assert "Air / Inflation" in msg
