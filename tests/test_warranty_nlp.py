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


class TestCustomerPhraseGaps:
    """Phrases that used to stick or map to the wrong option without an LLM."""

    def test_wont_inflate_is_air_not_power(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        node = {
            "type": "question",
            "options": [
                {"answer_key": "air", "label": "Air / Inflation not working"},
                {"answer_key": "power", "label": "Power issue (chair won't turn on or has power problems)"},
            ],
        }
        assert nlp.interpret_warranty_answer(node, "won't inflate") == "air"
        assert nlp.interpret_warranty_answer(node, "won't turn on") == "power"
        assert nlp.interpret_warranty_answer(node, "airbags not inflating") == "air"

    def test_air_coming_out_is_not_no_air(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        node = {
            "type": "instruction",
            "options": [
                {"answer_key": "air_blowing", "label": "Yes, air is blowing from the hose"},
                {"answer_key": "no_air", "label": "No, no air coming out"},
            ],
        }
        assert nlp.interpret_warranty_answer(node, "air is coming out") == "air_blowing"
        assert nlp.interpret_warranty_answer(node, "nothing coming out") == "no_air"

    def test_yes_no_on_never_worked_node(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        node = {
            "type": "question",
            "options": [
                {"answer_key": "yes_worked", "label": "Yes, it worked before but stopped"},
                {"answer_key": "never_worked", "label": "No, it has never worked"},
            ],
        }
        assert nlp.interpret_warranty_answer(node, "yes") == "yes_worked"
        assert nlp.interpret_warranty_answer(node, "no") == "never_worked"
        assert nlp.interpret_warranty_answer(node, "it used to work") == "yes_worked"
        assert nlp.interpret_warranty_answer(node, "never worked") == "never_worked"

    def test_common_defect_and_delivery_phrases(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        defect = {
            "type": "question",
            "options": [
                {"answer_key": "air", "label": "Air / Inflation not working"},
                {"answer_key": "cosmetic", "label": "Cosmetic damage (physical damage to appearance)"},
                {"answer_key": "remote", "label": "Remote / controller issue"},
                {"answer_key": "rolling", "label": "Full rolling massage mechanism issue"},
                {"answer_key": "power", "label": "Power issue (chair won't turn on or has power problems)"},
                {"answer_key": "recline", "label": "Recline / position adjustment not working"},
                {"answer_key": "footrest", "label": "Footrest issue"},
                {"answer_key": "heat", "label": "Heating / temperature issue"},
                {"answer_key": "voice", "label": "Voice control issue"},
            ],
        }
        assert nlp.interpret_warranty_answer(defect, "scratched") == "cosmetic"
        assert nlp.interpret_warranty_answer(defect, "rollers not moving") == "rolling"
        assert nlp.interpret_warranty_answer(defect, "won't recline") == "recline"
        assert nlp.interpret_warranty_answer(defect, "heater not working") == "heat"
        assert nlp.interpret_warranty_answer(defect, "alexa") == "voice"
        assert nlp.interpret_warranty_answer(defect, "foot roller not working") == "footrest"
        assert nlp.interpret_warranty_answer(defect, "calf rollers") == "footrest"

        issue = {
            "type": "question",
            "options": [
                {"answer_key": "installation", "label": "Installation Issue"},
                {"answer_key": "delivery", "label": "Delivery Issue"},
                {"answer_key": "defect", "label": "Defect / Malfunction"},
            ],
        }
        assert nlp.interpret_warranty_answer(issue, "chair is broken") == "defect"
        assert nlp.interpret_warranty_answer(issue, "the box is broken") == "delivery"

        location = {
            "type": "question",
            "options": [
                {"answer_key": "feet_calves", "label": "Feet / Calves"},
                {"answer_key": "arms", "label": "Arms"},
                {"answer_key": "shoulders_hips", "label": "Shoulders / Hips"},
                {"answer_key": "footrest", "label": "Footrest"},
                {"answer_key": "side_panel", "label": "Side Panel"},
                {"answer_key": "base", "label": "Base"},
            ],
        }
        assert nlp.interpret_warranty_answer(location, "legs") == "feet_calves"
        assert nlp.interpret_warranty_answer(location, "hips") == "shoulders_hips"
        assert nlp.interpret_warranty_answer(location, "not warm") is None
        assert nlp.interpret_warranty_answer(location, "it's warm in here") is None
        assert nlp.interpret_warranty_answer(issue, "I need to set it up") == "installation"
        assert nlp.interpret_warranty_answer(issue, "help assembling") == "installation"
        assert nlp.interpret_warranty_answer(issue, "where is my package") == "delivery"

    def test_chair_is_not_defect_by_substring(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        issue = {
            "type": "question",
            "options": [
                {"answer_key": "installation", "label": "Installation Issue"},
                {"answer_key": "delivery", "label": "Delivery Issue"},
                {"answer_key": "defect", "label": "Defect / Malfunction"},
            ],
        }
        assert nlp.interpret_warranty_answer(issue, "the chair") is None
        assert nlp.interpret_warranty_answer(issue, "where's my chair") is None
        assert nlp.interpret_warranty_answer(issue, "I need a pair of chairs") is None
        assert nlp.interpret_issue_type("the chair") is None
        assert nlp.interpret_issue_type("fair") is None
        assert nlp.interpret_issue_type("the box is broken") == "delivery"

    def test_yes_no_does_not_steal_unrelated_sentences(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        tracking = {
            "type": "question",
            "options": [
                {"answer_key": "no_tracking", "label": "No, I don't have a tracking number"},
                {"answer_key": "has_tracking", "label": "Yes, I have my tracking number"},
            ],
        }
        assert nlp.interpret_warranty_answer(tracking, "no power") is None
        assert nlp.interpret_warranty_answer(tracking, "never came") is None
        assert nlp.interpret_warranty_answer(tracking, "yes") == "has_tracking"
        assert nlp.interpret_warranty_answer(tracking, "no") == "no_tracking"

        box = {
            "type": "question",
            "options": [
                {"answer_key": "yes_box_damage", "label": "Yes, the box was visibly damaged"},
                {"answer_key": "no_box_damage", "label": "No, the box appeared fine"},
            ],
        }
        assert nlp.interpret_warranty_answer(box, "I have tracking") is None
        assert nlp.interpret_warranty_answer(box, "no air") is None
        assert nlp.interpret_warranty_answer(box, "no it looked fine") == "no_box_damage"

    def test_label_fragments_do_not_unique_match_wrong_option(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        install = {
            "type": "question",
            "options": [
                {"answer_key": "footrest_or_no_air", "label": "Footrest problem or no air anywhere on the chair"},
                {"answer_key": "general_setup", "label": "General assembly / setup help"},
                {"answer_key": "other", "label": "Other installation issue"},
            ],
        }
        assert nlp.interpret_warranty_answer(install, "it's warm in here") is None
        assert nlp.interpret_warranty_answer(install, "install") == "general_setup"

        remote_screen = {
            "type": "question",
            "options": [
                {"answer_key": "blank_screen_commands_ok", "label": "All commands respond, but the screen is blank"},
                {"answer_key": "cable_damaged", "label": "The remote cable appears damaged or cut"},
                {"answer_key": "commands_not_responding", "label": "Certain commands do not respond at all"},
            ],
        }
        assert nlp.interpret_warranty_answer(remote_screen, "arrived damaged") is None
        assert nlp.interpret_warranty_answer(remote_screen, "the remote is not working") is None
        assert nlp.interpret_warranty_answer(remote_screen, "blank screen") == "blank_screen_commands_ok"

        footrest_which = {
            "type": "question",
            "options": [
                {"answer_key": "legrest_not_extend", "label": "Legrest does NOT extend"},
                {"answer_key": "air_not_inflating", "label": "Airbags are NOT inflating in the footrest"},
                {"answer_key": "legrest_not_lowering", "label": "Legrest not lowering or raising"},
                {"answer_key": "foot_rollers", "label": "Foot rollers NOT working"},
                {"answer_key": "calf_roller", "label": "Calf roller NOT working"},
            ],
        }
        assert nlp.interpret_warranty_answer(footrest_which, "foot roller not working") == "foot_rollers"
        assert nlp.interpret_warranty_answer(footrest_which, "won't raise") == "legrest_not_lowering"

        heads = {
            "type": "question",
            "options": [
                {"answer_key": "no_movement", "label": "Heads do not move at all (even when powering off)"},
                {"answer_key": "worked_before_stopped", "label": "Heads worked before but have now stopped"},
                {"answer_key": "power_but_no_move", "label": "Heads seem to have power but barely move"},
            ],
        }
        assert nlp.interpret_warranty_answer(heads, "never worked") is None
        assert nlp.interpret_warranty_answer(heads, "used to work") == "worked_before_stopped"

    def test_send_someone_out_maps_to_team_help(self):
        assert (
            nlp.interpret_troubleshooting_outcome("send someone out")
            == "unable_to_attempt"
        )
        assert nlp.interpret_troubleshooting_outcome("I did that") == "steps_completed"

    def test_short_phrases_do_not_over_map(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        issue = {
            "type": "question",
            "options": [
                {"answer_key": "installation", "label": "Installation Issue"},
                {"answer_key": "delivery", "label": "Delivery Issue"},
                {"answer_key": "defect", "label": "Defect / Malfunction"},
            ],
        }
        assert nlp.interpret_warranty_answer(issue, "the box was fine") is None
        assert nlp.interpret_warranty_answer(issue, "box looked fine") is None
        assert nlp.interpret_warranty_answer(issue, "the box") is None
        assert nlp.interpret_warranty_answer(issue, "the box is broken") == "delivery"
        assert nlp.interpret_issue_type("the box was fine") is None
        assert nlp.interpret_issue_type("the box is broken") == "delivery"

        voice = {
            "type": "question",
            "options": [
                {"answer_key": "voice_no_response", "label": "Voice control does not respond to my commands"},
                {"answer_key": "false_triggers", "label": "Voice picks up random commands / false triggers"},
                {"answer_key": "voice_not_sure", "label": "Not sure"},
            ],
        }
        assert nlp.interpret_warranty_answer(voice, "picks up") is None
        assert nlp.interpret_warranty_answer(voice, "picks up random commands") == "false_triggers"

        remote_screen = {
            "type": "question",
            "options": [
                {"answer_key": "blank_screen_commands_ok", "label": "All commands respond, but the screen is blank"},
                {"answer_key": "cable_damaged", "label": "The remote cable appears damaged or cut"},
                {"answer_key": "commands_not_responding", "label": "Certain commands do not respond at all"},
            ],
        }
        assert nlp.interpret_warranty_answer(remote_screen, "commands work") is None
        assert nlp.interpret_warranty_answer(remote_screen, "blank screen") == "blank_screen_commands_ok"

        delivery_problem = {
            "type": "question",
            "options": [
                {"answer_key": "damaged_in_transit", "label": "Box or chair arrived damaged"},
                {"answer_key": "missing_parts", "label": "Missing parts or incomplete delivery"},
                {"answer_key": "wrong_item", "label": "Wrong item was delivered"},
                {"answer_key": "never_arrived", "label": "Never arrived / marked delivered but missing"},
                {"answer_key": "late_delivery", "label": "Delivery was late"},
                {"answer_key": "other_delivery_problem", "label": "Something else"},
            ],
        }
        assert nlp.interpret_warranty_answer(delivery_problem, "the box is broken") == "damaged_in_transit"

        recline = {
            "type": "question",
            "options": [
                {"answer_key": "backrest", "label": "Backrest recline"},
                {"answer_key": "zero_gravity", "label": "Zero Gravity position"},
                {"answer_key": "footrest_recline", "label": "Footrest recline"},
            ],
        }
        assert nlp.interpret_warranty_answer(recline, "footrest") == "footrest_recline"

    def test_extra_defect_phrases(self, monkeypatch):
        monkeypatch.setattr(nlp, "_llm_json", lambda *_a, **_k: None)
        defect = {
            "type": "question",
            "options": [
                {"answer_key": "air", "label": "Air / Inflation not working"},
                {"answer_key": "remote", "label": "Remote / controller issue"},
                {"answer_key": "rolling", "label": "Full rolling massage mechanism issue"},
                {"answer_key": "power", "label": "Power issue (chair won't turn on or has power problems)"},
            ],
        }
        assert nlp.interpret_warranty_answer(defect, "won't start") == "power"
        assert nlp.interpret_warranty_answer(defect, "no display") == "remote"
        assert nlp.interpret_warranty_answer(defect, "heads stuck") == "rolling"

    def test_append_unmapped_phrase_dedupes_and_caps(self):
        rows = []
        rows = nlp.append_unmapped_phrase(rows, node_id="issue_type", text="hello")
        rows = nlp.append_unmapped_phrase(rows, node_id="issue_type", text="hello")
        assert len(rows) == 1
        for i in range(20):
            rows = nlp.append_unmapped_phrase(
                rows, node_id="issue_type", text=f"phrase {i}"
            )
        assert len(rows) == 12


class TestTroubleshootingOutcome:
    def test_review_stage_maps_tried_steps(self):
        assert (
            nlp.interpret_troubleshooting_outcome("I've tried all the steps")
            == "steps_completed"
        )

    def test_review_stage_maps_send_technician(self):
        assert (
            nlp.interpret_troubleshooting_outcome("please send a technician")
            == "unable_to_attempt"
        )

    def test_bare_yes_is_not_mapped(self):
        assert nlp.interpret_troubleshooting_outcome("yes") is None
        assert (
            nlp.interpret_troubleshooting_outcome(
                "yes",
                previous_outcome="steps_completed",
                issue_type="defect",
            )
            is None
        )

    def test_outcome_stage_maps_still_broken(self):
        assert (
            nlp.interpret_troubleshooting_outcome(
                "it's still not working",
                previous_outcome="steps_completed",
                issue_type="defect",
            )
            == "unresolved"
        )

    def test_outcome_stage_maps_working_now(self):
        assert (
            nlp.interpret_troubleshooting_outcome(
                "it's working now",
                previous_outcome="steps_completed",
                issue_type="defect",
            )
            == "resolved"
        )
