"""
tests/test_warranty_nlp.py
===========================
Unit tests for natural-language warranty mapping (Phase 1 hybrid).
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import pytest

import warranty_nlp as nlp


class TestKeywordIssueType:
    def test_delivery_keywords(self):
        assert nlp.interpret_issue_type("My package arrived damaged") == "delivery"

    def test_installation_keywords(self):
        assert nlp.interpret_issue_type("I need help with assembly") == "installation"

    def test_defect_keywords(self):
        assert nlp.interpret_issue_type("The remote is not working") == "defect"

    def test_ambiguous_returns_none_without_llm(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda _prompt, **kwargs: None)
        assert nlp.interpret_issue_type("hello there") is None


class TestHeuristicOptionMatch:
    def test_yes_no_mapping(self):
        options = [
            {"answer_key": "yes_box_damage", "label": "Yes, the box was visibly damaged"},
            {"answer_key": "no_box_damage", "label": "No, the box appeared fine"},
        ]
        node = {"type": "question", "prompt": "Was the box damaged?", "options": options}
        assert nlp.interpret_warranty_answer(node, "yeah the box was crushed") == "yes_box_damage"
        assert nlp.interpret_warranty_answer(node, "no it looked fine") == "no_box_damage"

    def test_body_part_word_maps_to_option(self):
        options = [
            {"answer_key": "feet", "label": "Feet / Calves"},
            {"answer_key": "shoulders_hips", "label": "Shoulders / Hips"},
            {"answer_key": "footrest", "label": "Footrest"},
        ]
        node = {
            "type": "question",
            "prompt": "Which part of the chair is not inflating properly?",
            "options": options,
        }
        assert nlp.interpret_warranty_answer(node, "shoulders") == "shoulders_hips"

    def test_question_text_passthrough(self):
        node = {"type": "question_text", "prompt": "Model name?", "next": "x", "answer_key": "model_name"}
        assert nlp.interpret_warranty_answer(node, "  Maestro V2  ") == "Maestro V2"

    def test_label_substring_match(self):
        options = [
            {"answer_key": "has_tracking", "label": "Yes, I have my tracking number"},
            {"answer_key": "no_tracking", "label": "No, I don't have a tracking number"},
        ]
        node = {"type": "question", "prompt": "Tracking?", "options": options}
        assert nlp.interpret_warranty_answer(node, "I don't have a tracking number") == "no_tracking"

    def test_norm_not_matched_inside_longer_label(self, monkeypatch):
        options = [
            {"answer_key": "repair", "label": "Repair air pump"},
            {"answer_key": "replace", "label": "Replace unit"},
        ]
        node = {"type": "question", "prompt": "Next step?", "options": options}
        monkeypatch.setattr(nlp, "_llm_json", lambda _prompt, **kwargs: None)
        assert nlp.interpret_warranty_answer(node, "air") is None


class TestLlmFallback:
    def test_issue_type_from_llm(self, monkeypatch):
        monkeypatch.setattr(nlp, "_keyword_issue_type", lambda _text: None)
        monkeypatch.setattr(
            nlp,
            "_llm_json",
            lambda _prompt, **kwargs: {"issue_type": "delivery", "confidence": "high"},
        )
        assert nlp.interpret_issue_type("something vague about shipping") == "delivery"

    def test_option_from_llm(self, monkeypatch):
        options = [
            {"answer_key": "air", "label": "Air / Inflation not working"},
            {"answer_key": "power", "label": "Power issue"},
        ]
        node = {"type": "question", "prompt": "What kind of defect?", "options": options}
        monkeypatch.setattr(nlp, "_heuristic_option_match", lambda _opts, _text: None)
        monkeypatch.setattr(
            nlp,
            "_llm_json",
            lambda _prompt, **kwargs: {"answer_key": "power", "confidence": "high"},
        )
        assert nlp.interpret_warranty_answer(node, "it won't power on at all") == "power"

    def test_low_confidence_llm_rejected(self, monkeypatch):
        options = [
            {"answer_key": "yes_box_damage", "label": "Yes, the box was visibly damaged"},
            {"answer_key": "no_box_damage", "label": "No, the box appeared fine"},
        ]
        node = {"type": "question", "prompt": "Was the box damaged?", "options": options}
        monkeypatch.setattr(nlp, "_heuristic_option_match", lambda _opts, _text: None)
        monkeypatch.setattr(
            nlp,
            "_llm_json",
            lambda _prompt, **kwargs: {"answer_key": "yes_box_damage", "confidence": "low"},
        )
        assert nlp.interpret_warranty_answer(node, "maybe sort of") is None
