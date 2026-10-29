"""Public API regressions for everyday English repairs and customer corrections."""
from __future__ import annotations

import pytest

from script.simulate_chatbot_quality import offline_api
from script.quality_v3_warranty import WarrantyConversation


@pytest.fixture
def conversation():
    with offline_api() as client:
        chat = WarrantyConversation(client, "customer-correction-regression")
        chat.identity()
        yield chat
        assert not client.attempted_external_delivery


@pytest.mark.parametrize("message,issue", [
    ("My chair will not turn on.", "defect"),
    ("The chair stopped working yesterday.", "defect"),
    ("The chair shuts off after about five minutes.", "defect"),
    ("The heater is not getting warm.", "defect"),
    ("The rollers make a clicking noise.", "defect"),
    ("The leather is torn on the armrest.", "defect"),
    ("I just unpacked it. How do I assemble the footrest?", "installation"),
    ("I need help installing the side panels.", "installation"),
    ("의자가 작동하지 않아요.", "defect"),
    ("Mi silla no enciende.", "defect"),
])
def test_clear_symptom_retains_customer_confirmation(conversation, message, issue):
    data = conversation.request("smart-start", {"message": message, "domain": "osakiusa.com"})
    assert data["smart_start"]["suggested_issue_type"] == issue
    assert data["ticket"]["current_node"]["node_id"] == "issue_type"
    assert data["ticket"]["model_name"] == "OS-4000T"


def test_rejected_airbag_diagnosis_does_not_override_torn_upholstery(conversation):
    conversation.defect()
    conversation.answer("The leather is torn on the armrest, not an airbag problem.")
    assert conversation.node() == "defect_cosmetic_location"
    assert conversation.stored()["model"] == "OS-4000T"


def test_no_clicking_after_completed_checks_retains_negative_result(conversation):
    conversation.defect()
    conversation.answer("power")
    conversation.answer("remote_off")
    data = conversation.answer("I already checked the outlet and reconnected the power cord. There is no clicking sound.")
    assert data["interpreted_option"]["answer_key"] == "no_clicking"
    assert conversation.node() == "defect_power_no_click_terminal"
    assert conversation.stored()["status"] != "self_resolved"


def test_good_fuse_and_secure_cable_are_not_reported_as_a_blown_fuse(conversation):
    conversation.defect()
    conversation.answer("remote")
    conversation.answer("no_power")
    data = conversation.answer("The cable is firmly connected and the fuse is intact. It still will not turn on.")
    assert data["interpreted_option"]["answer_key"] == "all_checked_ok"
    assert conversation.node() == "defect_remote_pcb_check_terminal"
    assert conversation.stored()["status"] != "self_resolved"


def test_fuse_result_alone_does_not_invent_other_completed_checks(conversation):
    conversation.defect()
    conversation.answer("remote")
    conversation.answer("no_power")
    data = conversation.answer("The fuse is intact.")
    assert conversation.node() == "defect_remote_no_power_checks"
    assert (data.get("suggested_option") or {}).get("answer_key") not in {"fuse_broken", "all_checked_ok"}


def test_explicit_part_correction_recovers_existing_case(conversation):
    conversation.defect()
    conversation.answer("air")
    ticket_id = conversation.ticket_id
    data = conversation.answer("No, this is cosmetic damage to the upholstery, not an inflation issue.")
    assert conversation.node() == "defect_cosmetic_location"
    assert conversation.ticket_id == ticket_id
    assert conversation.stored()["model"] == "OS-4000T"
    assert conversation.stored()["collected"]["workflow_corrections"][-1]["corrected_component"] == "cosmetic"
    assert not data.get("handoff")


def test_ambiguous_correction_does_not_guess_a_new_failed_part(conversation):
    conversation.defect()
    conversation.answer("air")
    conversation.answer("No, that is the wrong problem.")
    assert conversation.node() == "defect_air_location"


def test_completed_warmup_is_used_without_requesting_another_run(conversation):
    conversation.defect()
    conversation.answer("heat")
    conversation.answer("not_heating")
    data = conversation.answer("I have run it for twenty minutes with heat switched on, and it is still cold.")
    assert data["interpreted_option"]["answer_key"] == "still_no_heat"
    assert conversation.node() == "defect_heating_not_heating_terminal"


def test_negated_results_are_also_excluded_from_llm_choices(monkeypatch):
    import warranty_nlp
    from warranty_workflow import _NODES

    monkeypatch.setattr(warranty_nlp, "_llm_json", lambda *args, **kwargs: {"answer_key": "fuse_broken", "confidence": "high"})
    assert warranty_nlp.interpret_warranty_answer(_NODES["defect_remote_no_power_checks"], "The fuse is intact.") is None


@pytest.mark.parametrize("message", [
    "I cannot safely remove the side panel and disconnect the hose.",
    "Please have a person review this with me.",
    "담당자와 상담하고 싶어요.",
])
def test_inability_or_person_request_interrupts_current_checks(conversation, message):
    conversation.defect()
    conversation.answer("air")
    conversation.answer("arms")
    original = conversation.ticket_id
    data = conversation.answer(message)
    assert data["handoff"]["requested"] is True
    assert conversation.ticket_id == original
    assert conversation.stored()["model"] == "OS-4000T"
    assert conversation.stored()["collected"]["human_support_requested"] is True
    assert not data["handoff"]["notification_sent"]
    assert "disconnect the hose" not in conversation.text().lower()


@pytest.mark.parametrize("action,body", [
    ("register-model", {"model": "OS-4000T or Hypnos", "domain": "osakiusa.com"}),
    ("confirm-model", {"model": "OS-4000T or Hypnos", "confirmed": True, "domain": "osakiusa.com"}),
])
def test_alternative_model_guesses_do_not_replace_confirmed_identity(conversation, action, body):
    response = conversation.request(action, body, statuses=(422,))
    assert "one chair model" in response["detail"]
    conversation.request("", method="get")
    assert conversation.stored()["model"] == "OS-4000T"


def test_single_unknown_label_remains_eligible_for_manual_support(conversation):
    data = conversation.request("register-model", {"model": "Legacy Comfort ABC123", "domain": "osakiusa.com"})
    assert data["ticket"]["model_confirmed"] is True
    assert data["ticket"]["model_name"] == "Legacy Comfort ABC123"


@pytest.mark.parametrize("node_id,message,expected", [
    ("defect_power_off_checks", "The wall outlet is not working.", "outlet_no_power"),
    ("defect_remote_no_power_checks", "The fuse is intact, but the cable connection is loose.", "bad_connection"),
    ("defect_remote_no_power_checks", "The cable is firmly connected and the fuse is intact.", "all_checked_ok"),
])
def test_positive_and_negative_check_results_keep_their_subject(monkeypatch, node_id, message, expected):
    import warranty_nlp
    from warranty_workflow import _NODES

    monkeypatch.setattr(warranty_nlp, "_llm_json", lambda *args, **kwargs: None)
    assert warranty_nlp.interpret_warranty_answer(_NODES[node_id], message) == expected


@pytest.mark.parametrize("action,input_key", [
    ("natural-start", "message"), ("smart-start", "message"), ("register-model", "model"),
])
def test_model_location_question_preserves_existing_model(conversation, action, input_key):
    original = conversation.ticket_id
    data = conversation.request(action, {input_key: "Where can I find the model number on my chair?", "domain": "osakiusa.com"})
    assert data["model_help"] is True
    assert conversation.ticket_id == original
    assert conversation.stored()["model"] == "OS-4000T"
    assert "serial-number sticker" in data["assistant_message"]
    assert "Do not lift, tip, or open" in data["assistant_message"]
    assert not data.get("purchase_verification_scheduled")


def test_model_location_help_does_not_register_the_question_as_a_model():
    with offline_api() as client:
        chat = WarrantyConversation(client, "model-location-without-model")
        chat.identity(model=None)
        data = chat.request("natural-start", {"message": "Where can I find the model number on my chair?", "domain": "osakiusa.com"})
        assert data["model_help"] is True
        assert data.get("ticket") is None
        assert "visible and easy to reach" in data["assistant_message"]
        assert not client.attempted_external_delivery


@pytest.mark.parametrize("message", [
    "The chair's Bluetooth will not pair with my phone.",
    "My chair's Bluetooth cannot connect to my phone.",
    "I need help with a Bluetooth connection fault.",
])
def test_owned_bluetooth_failure_receives_a_confirmable_malfunction_route(conversation, message):
    data = conversation.request("smart-start", {"message": message, "domain": "osakiusa.com"})
    assert data["smart_start"]["suggested_issue_type"] == "defect"
    assert data["ticket"]["current_node"]["node_id"] == "issue_type"
    assert data["smart_start"]["applied_keys"] in ([], ["warranty"])


@pytest.mark.parametrize("message", [
    "Does this model have Bluetooth speakers?",
    "Can I pair my phone with Bluetooth?",
])
def test_bluetooth_capability_question_does_not_report_a_failed_part(conversation, message):
    data = conversation.request("smart-start", {"message": message, "domain": "osakiusa.com"})
    assert (data.get("smart_start") or {}).get("suggested_issue_type") != "defect"
    assert conversation.stored()["model"] == "OS-4000T"
