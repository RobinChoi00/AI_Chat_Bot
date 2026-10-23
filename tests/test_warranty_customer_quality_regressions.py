"""Deidentified regressions from reviewed historical Warranty failure patterns.

These are synthetic reproductions, not raw customer conversations or verified
repair procedures. External knowledge/model calls are disabled.
"""
from pathlib import Path
import json
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import warranty_knowledge as knowledge
import warranty_self_help as help
import warranty_terminal_enrichment as terminal
import warranty_step_enrichment as step
import warranty_intake_context as intake
import warranty_error_code_gate as gate
import install_videos


class Ticket:
    ticket_id = "synthetic-quality-case"
    def __init__(self, issue="defect", model="Osaki Hypnos 4D", collected=None):
        self.issue_type, self.model_name = issue, model
        self.collected = collected or {}
    def get_collected(self): return self.collected.copy()
    def set_collected(self, key, value): self.collected[key] = value


class Engine:
    def __init__(self, turns=()): self.turns = list(turns)
    def get_turns(self, _): return self.turns


def turn(key, answer=None):
    return SimpleNamespace(answer_key=key, customer_answer=answer if answer is not None else key, node_prompt="", node_id="synthetic")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "0")
    monkeypatch.setattr(help, "search_knowledge", lambda **kwargs: [])
    monkeypatch.setattr(help, "contextual_search_knowledge", lambda **kwargs: [])
    monkeypatch.setattr(step, "_fonz_match_from_ticket", lambda *args: None)
    monkeypatch.setattr(step, "_fonz_match_from_turns", lambda *args: None)


@pytest.mark.parametrize("text", [
    "Hello, Since the parts are on the way, I will go ahead and close this ticket.",
    "I am glad your chair is back to work! I will go ahead and close this ticket.",
    "Ask customer if they can disconnect and reconnect the power cord.",
    "We will arrange a replacement at no cost to you.",
    "The technician has already been scheduled to inspect your chair.",
])
def test_support_case_status_is_never_a_customer_instruction(text):
    assert not knowledge._is_customer_safe_step(text)
    assert not knowledge._extract_customer_steps(text)
    assert not knowledge.sanitize_customer_answer(text)


def test_raw_case_steps_cannot_bypass_provenance_even_when_plausible(monkeypatch):
    copied = knowledge.KnowledgeEntry("freshdesk", "remote", "Remote issue", "", (
        "Hello, Since the parts are on the way, I will go ahead and close this ticket.",
        "Inspect the visible remote cable for damage while the chair is unplugged.",
    ))
    monkeypatch.setattr(help, "search_knowledge", lambda **kwargs: [copied])
    diagnosis = help.build_remote_diagnosis(symptom="cable_damaged", path_text="remote cable damaged", model_name="Synthetic chair")
    text = " ".join(diagnosis["steps"])
    assert "close this ticket" not in text and "parts are on the way" not in text
    assert "visible remote cable" not in text
    assert "freshdesk" not in diagnosis["sources"]


def test_reviewed_model_scoped_procedure_can_be_presented(monkeypatch):
    reviewed = knowledge.KnowledgeEntry("reviewed_knowledge", "remote", "Reviewed remote check", "", (
        "Inspect the visible remote cable for damage while the chair is unplugged.",
    ), applicable_models=("Synthetic chair",))
    monkeypatch.setattr(help, "search_knowledge", lambda **kwargs: [reviewed])
    diagnosis = help.build_remote_diagnosis(symptom="cable_damaged", path_text="remote cable damaged", model_name="Synthetic chair")
    assert "visible remote cable" in " ".join(diagnosis["steps"])
    assert diagnosis["sources"] == ["reviewed_knowledge"]
    other_model = help.build_remote_diagnosis(symptom="cable_damaged", path_text="remote cable damaged", model_name="Different chair")
    assert "visible remote cable" not in " ".join(other_model["steps"])


def test_raw_cases_remain_internal_but_never_customer_search_results(monkeypatch):
    raw = knowledge.KnowledgeEntry("freshdesk", "remote", "Remote screen flashing", "", (
        "Inspect the visible remote cable for damage while the chair is unplugged.",
    ))
    kb = knowledge.KnowledgeEntry("freshdesk_kb", "remote", "Published remote screen guide", "", (
        "Check the visible remote cable connection while the chair is unplugged.",
    ))
    monkeypatch.setattr(knowledge, "load_knowledge_entries", lambda: (raw, kb))
    matches = knowledge.search_knowledge(path_text="remote screen flashing cable", defect_category="remote")
    assert raw not in matches and kb in matches
    assert raw in knowledge.load_knowledge_entries()
    # Presentation helpers must reject stale/pre-parsed raw results too.
    assert step._pick_step_tips([raw], (), prefer_freshdesk=True) == []
    assert "Remote screen flashing" not in help._friendly_match_summary([raw], defect_category="remote", model_name="Synthetic chair", issue_type="defect")


def test_case_status_cannot_become_a_manual_fallback_instruction():
    assert knowledge._customer_steps_from_manual("The parts are on the way. I will close this ticket.") == ()


def test_fonz_trim_fallback_cannot_rescue_case_correspondence(monkeypatch, tmp_path):
    fixture = tmp_path / "codes.json"
    fixture.write_text(json.dumps([{
        "model": "Synthetic", "error_code": "E9", "meaning": "Remote issue",
        "troubleshooting": "Hello, Since the parts are on the way, I will go ahead and close this ticket.",
    }]))
    monkeypatch.setattr(knowledge, "_FONZ_ERROR_PATH", fixture)
    entries = knowledge._load_fonz_error_entries()
    assert entries and entries[0].customer_steps == ()


def test_overheat_never_asks_customer_to_warm_up_or_retest():
    result = terminal.build_terminal_enrichment(Engine([turn("heat"), turn("too_hot")]), Ticket(), {
        "node_id": "defect_heating_too_hot_terminal", "type": "terminal", "action": "send_info", "prompt": "Heat is too hot.",
    })
    message = result["message"].lower()
    assert "stop using" in message and "keep it off" in message
    assert "give it at least" not in message and "warm-up time" not in message
    assert "before testing again" not in message and "try the steps above" not in message
    assert result["diagnosis"]["safety_stop"] is True
    assert result["interaction_mode"] == "preparation"


def test_confirmed_error_code_cannot_restart_testing_after_safety_stop(monkeypatch):
    monkeypatch.setattr(gate, "fonz_hit_from_ticket", lambda ticket: {
        "model": "Synthetic", "error_code": "E9", "meaning": "Heating fault",
        "troubleshooting": "Power cycle the chair and run a heated program again.",
    })
    result = terminal.build_terminal_enrichment(Engine([turn("heat"), turn("too_hot")]), Ticket(), {
        "node_id": "defect_heating_too_hot_terminal", "type": "terminal", "action": "send_info", "prompt": "Heat is too hot.",
    })
    assert result["diagnosis"]["safety_stop"] is True
    assert all("power cycle" not in step.lower() for step in result["diagnosis"]["steps"])
    assert "run a heated program" not in result["message"].lower()


def test_damage_review_does_not_announce_a_free_replacement():
    result = terminal.build_terminal_enrichment(Engine([turn("visible_at_unboxing")]), Ticket("delivery"), {
        "node_id": "delivery_replace_claim_terminal", "type": "terminal", "action": "awaiting_admin", "prompt": "Review damage.",
    })
    message = result["message"].lower()
    assert "no cost to you" not in message
    assert "must review" in message and "eligibility" in message


def test_white_glove_cosmetic_review_does_not_announce_free_service():
    result = help.build_cosmetic_diagnosis(symptom="wg_install", path_text="White Glove cosmetic damage", model_name="Synthetic chair")
    assert "no cost" not in result["summary"].lower()
    assert "must review" in result["summary"]


def test_recline_observation_does_not_diagnose_replacement_part():
    diagnosis = help.build_recline_diagnosis(symptom="moves_on_off", path_text="recline returns on off", model_name="Synthetic chair")
    text = diagnosis["summary"] + " ".join(diagnosis["steps"])
    assert "likely needs replacement" not in text and "Main PCB" not in text
    assert "do not identify a failed part" in text


@pytest.mark.parametrize("value", ["Return", "refund", "Human please", "I need help", "Unknown", "my chair", "The chair is not working", "/Users/redacted/Downloads/photo.jpg", "Customer provided only the chair model name with no issue description"])
def test_non_models_are_not_valid_model_identity(value):
    assert not intake.is_valid_model_input(value)


def test_model_wrapper_is_normalized_without_changing_identity():
    assert intake.normalize_model_input("The model is the Maxim 3D LE") == "Maxim 3D LE"
    assert intake.is_valid_model_input("The model is the Maxim 3D LE")
    assert intake.is_valid_model_input("Osaki Ai Grand Duo 4D+4D")


@pytest.mark.parametrize("summary", ["Customer provided only the chair model name with no issue description", "/Users/redacted/Downloads/photo.jpg", "Osaki Hypnos 4D"])
def test_internal_or_attachment_intake_is_not_echoed_to_customer(summary):
    ticket = Ticket(collected={"intake_summary": summary})
    result = intake.intake_aware_step_summary(ticket=ticket, turns=[turn("defect")], summary="Which function is affected?")
    assert result == "Which function is affected?"


def test_real_symptom_survives_invalid_extraction_summary():
    ticket = Ticket()
    intake.persist_intake_summary(ticket, summary="Customer provided only model name with no issue description", raw_message="Remote screen flashes and stops responding")
    assert intake.get_intake_summary(ticket) == "Remote screen flashes and stops responding"


def test_already_completed_checks_are_not_repeated_before_next_question():
    ticket = Ticket(collected={"intake_summary": "I already tested the outlet, reseated the power cord and power cycled the chair. None helped."})
    result = step.build_step_enrichment(Engine([turn("defect"), turn("power")]), ticket,
        {"node_id": "synthetic_power_question", "type": "question", "prompt": "Is the remote on?"})
    assert result is None or "Confirm the power cord and outlet" not in result["message"]
    assert result is None or "toggle the back power switch" not in result["message"]


def test_negative_attempt_report_does_not_suppress_needed_check():
    ticket = Ticket(collected={"intake_summary": "I haven't checked the wall outlet yet."})
    checks = ["Test the wall outlet with another device."]
    assert intake.filter_completed_check_steps(checks, ticket=ticket, turns=[]) == checks


def test_completed_heat_warmup_not_prescribed_again():
    result = terminal.build_terminal_enrichment(Engine([turn("heat"), turn("not_heating"), turn("still_no_heat")]), Ticket(), {
        "node_id": "defect_heating_not_heating_terminal", "type": "terminal", "action": "send_info", "prompt": "Heat remains cold.",
    })
    message = result["message"].lower()
    assert "already completed" in message
    assert "give it at least 10 minutes" not in message
    assert "run a heated auto program" not in message


def test_family_diagnostic_buttons_not_assumed_for_variant(monkeypatch):
    import error_code_lookup
    monkeypatch.setattr(error_code_lookup, "get_model_diagnostic", lambda model: {
        "model": "Maxim", "entry_procedure": "DEEP TISSUE + STRETCH + POWER", "entry_method": "Yes",
    })
    assert gate._entry_procedure_hint("Maxim 3D LE") == ""
    assert "DEEP TISSUE" not in gate.build_enter_prompt("Maxim 3D LE")
    assert gate._entry_procedure_hint("Maxim") == "DEEP TISSUE + STRETCH + POWER"


def test_explicit_procedure_applicability_is_supported(monkeypatch):
    import error_code_lookup
    monkeypatch.setattr(error_code_lookup, "get_model_diagnostic", lambda model: {
        "model": "Maxim", "applicable_models": ["Maxim 3D LE"], "entry_procedure": "Verified procedure",
    })
    assert gate._entry_procedure_hint("Maxim 3D LE") == "Verified procedure"


def test_unknown_install_resource_is_not_claimed_exact_video(monkeypatch):
    monkeypatch.setattr(terminal, "lookup_install_video", lambda name: {"url": "https://support.example.com", "label": "General support guides", "match": "default"})
    result = terminal._install_message("Synthetic Unknown Chair", "")
    assert "couldn't find a verified model-specific" in result["message"]
    assert "Here is the installation guide for your" not in result["message"]
    assert "Watch —" not in result["message"]


@pytest.mark.parametrize("text", ["The remote controller has a short", "The chair is too hot", "The chair causes pain and discomfort", "There is a burning smell"])
def test_explicit_hazard_is_recognized_before_more_testing(text):
    assert intake.reported_safety_hazard(text=text)


@pytest.mark.parametrize("text", [
    "The chair is not overheating", "There is no burning smell", "No short circuit", "No electric shock",
    "No pain or discomfort", "There is no smoke or sparks", "It isn't too hot", "It doesn't cause pain",
    "No signs of overheating", "The massage doesn't hurt", "It is not painful", "There is no electrical shock",
])
def test_negated_hazard_is_not_a_stop_use_report(text):
    assert not intake.reported_safety_hazard(text=text)


@pytest.mark.parametrize("text", [
    "No smoke, but the chair sparks", "No pain or discomfort, but there is a burning smell",
    "It isn't overheating but the remote controller has a short", "It doesn't hurt but it is too hot",
])
def test_negation_does_not_hide_a_separate_positive_hazard(text):
    assert intake.reported_safety_hazard(text=text)
