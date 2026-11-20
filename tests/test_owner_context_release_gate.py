"""Independent deterministic acceptance cases through the isolated Sales API.

These are synthetic conversations, not live-model judgments or a customer score.
The fixture holds storage in memory and fails if any external delivery is attempted.
"""

from __future__ import annotations

import re

import pytest

from script.quality_v4_product import SalesLog, direct_fact, visible_doorway
from script.simulate_chatbot_quality import offline_api


EPIC = "Titan TP-Epic 4D"
CHAMP = "Osaki OS-Champ"
CORRECTED_OWNER = "OS-4000T"


@pytest.fixture
def sales_api(monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_EMAIL", "0")
    with offline_api() as client:
        yield client
        assert not client.attempted_external_delivery


def conversation(client, name):
    log = SalesLog(client, f"owner-context-gate-{name}")
    assert log.setup_ok
    return log


def answer(log, message):
    response = log.say(message)
    reply = {key: response.get(key) for key in ("reply", "intent", "products", "handoff", "handoff_reason")}
    assert log.transcript[-1]["status"] == 200, (reply, response.get("request_exception"))
    assert str(reply["reply"] or "").strip(), reply
    return reply


def no_false_completion(reply):
    assert not re.search(
        r"\b(?:i|we)(?:['’]ve|\s+have)?\s+(?:already\s+)?"
        r"(?:booked|scheduled|submitted|sent|opened|filed|approved|"
        r"processed|connected|forwarded|transferred|routed)\b",
        reply["reply"], re.I,
    ), reply


def warranty_guidance(reply, *, model=None):
    text = reply["reply"].casefold()
    assert reply["intent"] in {"warranty_redirect", "parts_technician"}, reply
    assert "warranty" in text and "ext. 3" in text, reply
    assert "if you're buying" not in text and not reply.get("products"), reply
    if model is not None:
        assert model.casefold() in text, reply
    no_false_completion(reply)


def start_epic_repair(log):
    warranty_guidance(answer(log, "I own a Titan TP-Epic 4D and its foot rollers stopped working."))


def assert_two_departments(reply, *, owned_model=EPIC, purchase_model=CHAMP):
    lines = reply["reply"].casefold().splitlines()
    assert any(owned_model.casefold() in line and "warranty" in line for line in lines), reply
    assert any(purchase_model.casefold() in line and "sales" in line for line in lines), reply
    no_false_completion(reply)


def test_real_owner_rephrasing_stays_in_service(sales_api):
    log = conversation(sales_api, "production-shape")
    start_epic_repair(log)
    reply = answer(log, "This chair is already in my house. It is broken. I need technical support, not a new chair.")
    warranty_guidance(reply)


def test_owned_chair_fault_is_not_generic_prebuy_support(sales_api):
    log = conversation(sales_api, "direct-owner")
    reply = answer(log, "I've had my Titan TP-Epic 4D in my house for two years. Its foot rollers no longer work. How can I get technical support?")
    warranty_guidance(reply)


@pytest.mark.parametrize("question", [
    "Before I buy the Osaki OS-Champ, what technical support is available if it stops working?",
    "I'm considering the Osaki OS-Champ. What warranty service is available if the chair breaks after I buy it?",
])
def test_hypothetical_breakdown_stays_a_purchase_policy(sales_api, question):
    log = conversation(sales_api, "hypothetical")
    reply = answer(log, question)
    assert reply["intent"] == "prepurchase_policy", reply
    assert "warranty" in reply["reply"].casefold(), reply
    assert not reply.get("handoff"), reply


@pytest.mark.parametrize("question", [
    "Before I buy an Osaki OS-Champ, what warranty service is offered if it breaks?",
    "Can I buy an extended warranty for the new Osaki OS-Champ?",
])
def test_old_repair_does_not_consume_new_purchase_policy(sales_api, question):
    log = conversation(sales_api, "retained-owner-new-policy")
    start_epic_repair(log)
    reply = answer(log, question)
    assert reply["intent"] == "prepurchase_policy", reply
    assert "warranty" in reply["reply"].casefold(), reply


def test_negated_breakdown_does_not_override_a_product_question(sales_api):
    log = conversation(sales_api, "negated")
    reply = answer(log, "The chair I own is not broken. I'm looking to buy an Osaki OS-Champ for my office. Does it have foot rollers?")
    assert reply["intent"] == "specs" and direct_fact(reply, CHAMP, "foot_roller"), reply


def test_existing_owner_can_still_ask_to_buy_replacement_parts(sales_api):
    log = conversation(sales_api, "parts")
    reply = answer(log, "I already own an Osaki OS-4000T, and the remote is broken. Where can I buy the replacement remote?")
    warranty_guidance(reply)
    assert reply["intent"] == "parts_technician", reply


def test_reported_safety_incident_stops_active_shopping(sales_api):
    log = conversation(sales_api, "safety")
    answer(log, "Does the Osaki OS-Champ have foot rollers?")
    reply = answer(log, "My Titan TP-Epic 4D is smoking right now. Can I test it again?")
    assert reply.get("handoff_reason") == "safety_hazard", reply
    text = reply["reply"].casefold()
    assert "stop using" in text and "only if" in text and "do not run another test" in text, reply
    warranty_guidance(reply)


def test_switching_lanes_keeps_both_models_and_purchase_constraints(sales_api):
    log = conversation(sales_api, "distinct-workstreams")
    start_epic_repair(log)
    answer(log, "Separately, I'd like a second chair under $3000. I am 5 foot 8 and weigh 170 pounds, and want lower back massage.")
    answer(log, "My office doorway is 36 inches wide, assembled only.")
    feature = answer(log, "Does the Osaki OS-Champ have foot rollers? This question is about the new purchase.")
    assert direct_fact(feature, CHAMP, "foot_roller"), feature

    repair = answer(log, "Going back to the one that's broken, how do I get technical support for the old chair?")
    warranty_guidance(repair, model=EPIC)
    assert CHAMP.casefold() not in repair["reply"].casefold(), repair

    purchase = answer(log, "And for the new one, how wide should the doorway be?")
    assert visible_doorway(purchase, CHAMP), purchase
    assert not any(p.get("model") == EPIC for p in purchase["products"]), purchase

    summary = answer(log, "Keep both cases separate: Warranty for the repair and Sales for the new purchase.")
    assert_two_departments(summary)
    prefs = log.saved.get("recommend_prefs", {})
    assert float(prefs.get("budget_max_usd", 0)) == 3000, prefs
    assert float(prefs.get("weight_lb", 0)) == 170, prefs
    assert float(prefs.get("doorway_in", 0)) == 36 and prefs.get("doorway_fit") == "assembled", prefs


def test_same_turn_old_model_is_not_used_as_new_purchase(sales_api):
    log = conversation(sales_api, "mixed-first-turn")
    reply = answer(log, "The foot rollers stopped working on the Titan TP-Epic 4D I already own, and I want to buy an Osaki OS-Champ for my office. Please handle these separately.")
    assert_two_departments(reply)


def test_explicit_service_model_correction_does_not_overwrite_new_model(sales_api):
    log = conversation(sales_api, "owner-correction")
    start_epic_repair(log)
    sales = answer(log, "Does the Osaki OS-Champ have foot rollers? This question is about the new purchase.")
    assert direct_fact(sales, CHAMP, "foot_roller"), sales
    answer(log, "Correction: the broken chair is actually an Osaki OS-4000T. The Epic is not broken.")
    reply = answer(log, "Please keep the repair with Warranty and the new purchase with Sales.")
    assert_two_departments(reply, owned_model=CORRECTED_OWNER)
    text = reply["reply"].casefold()
    assert EPIC.casefold() not in text and "os-4000xt" not in text, reply
