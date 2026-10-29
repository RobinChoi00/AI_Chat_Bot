"""Customer goals survive ordinary wording, corrections, and topic changes."""
from __future__ import annotations

import uuid

import pytest

from script.simulate_chatbot_quality import offline_api


@pytest.fixture
def chat():
    with offline_api() as client:
        from sales_models import get_session_collected
        sid = "business-fix-" + uuid.uuid4().hex
        response = client.post(f"/api/v1/warranty/session/{sid}/consent", json={"domain": "osakiusa.com"})
        assert response.status_code == 200

        def turn(message):
            response = client.post("/api/v1/sales/chat", json={
                "session_id": sid, "domain": "osakiusa.com", "message": message,
            })
            assert response.status_code == 200, response.text
            return response.json(), get_session_collected(sid)

        yield turn
        assert not client.attempted_external_delivery


@pytest.mark.parametrize("message", [
    "My remote screen is black and the buttons do not respond.",
    "The foot rollers on my Titan TP-Epic have stopped working.",
    "My remote is now blank.",
    "The air bags are not inflating.",
])
def test_reported_part_failures_reach_warranty_instead_of_shopping(chat, message):
    reply, _ = chat(message)
    assert reply["intent"] == "warranty_redirect"
    assert reply["handoff"]
    assert "service@osakititan.com" in reply["reply"]
    assert "user height" not in reply["reply"].lower()
    assert "approved a repair" in reply["reply"]  # explicitly denies approval


def test_return_policy_then_actual_refund_retains_the_new_order_task(chat):
    first, _ = chat("Before buying, what is your return policy?")
    assert "30 days" in first["reply"] and not first["handoff"]
    shipping, _ = chat("Would I pay shipping both ways if I return it?")
    assert "original outbound shipping" in shipping["reply"]
    assert "return shipping" in shipping["reply"]
    requested, state = chat("Actually I received the chair yesterday and want to return it.")
    assert requested["intent"] == "cancel_refund"
    assert state["sales_support_topic"] == "cancel_refund"
    timing, _ = chat("How soon will the money be back?")
    assert timing["intent"] == "cancel_refund"
    assert "cannot verify the refund processing time" in timing["reply"]
    prepared, _ = chat("What should I provide to Sales so they can review my return request?")
    assert prepared["intent"] == "cancel_refund"
    assert "order number" in prepared["reply"] and "checkout email" in prepared["reply"]
    assert "user height" not in prepared["reply"].lower()


def test_refund_context_yields_to_a_new_product_question(chat):
    chat("I want a refund for my delivered chair.")
    price, state = chat("What is the price of Maestro LE 2.0?")
    assert price["intent"] == "price"
    assert "Maestro LE 2.0" in price["reply"]
    assert state.get("sales_support_topic") == ""


def test_refund_paraphrases_and_explicit_switch_to_a_second_chair(chat):
    request, _ = chat("I need to send back the chair that arrived Friday.")
    assert request['intent'] == 'cancel_refund'
    timing, _ = chat("When would the credit show up on my card?")
    assert timing['intent'] == 'cancel_refund'
    assert 'cannot verify the refund processing time' in timing['reply']
    paperwork, _ = chat("What paperwork should I have ready?")
    assert paperwork['intent'] == 'cancel_refund'
    assert 'order number' in paperwork['reply'] and 'checkout email' in paperwork['reply']
    shopping, state = chat("The return question is finished, now pick a second chair.")
    assert shopping['intent'] == 'recommend'
    assert not state.get('sales_support_topic')


def test_return_freight_paraphrase_uses_the_same_published_policy(chat):
    chat("Before I buy, what is your return policy?")
    reply, _ = chat("Do I pay for the trip here or the trip back?")
    assert reply['intent'] == 'prepurchase_policy'
    assert 'original outbound shipping' in reply['reply'] and 'return shipping' in reply['reply']


def test_order_color_change_and_invoice_stay_with_sales(chat):
    change, _ = chat("I ordered a chair. Can I change the color?")
    assert change["intent"] == "order_change"
    invoice, state = chat("Can I get an invoice?")
    assert invoice["intent"] == "order_status"
    assert state["sales_support_topic"] == "order_status"
    assert "has not retrieved" in invoice["reply"]
    assert "issued an invoice" in invoice["reply"]


def test_policy_followup_preserves_labor_question_then_allows_new_shipping_topic(chat):
    chat("What does the warranty cover?")
    labor, _ = chat("Does that mean free labor for all three years?")
    assert labor["intent"] == "prepurchase_policy"
    assert "Year 1" in labor["reply"] and "Year 2" in labor["reply"]
    shipping, state = chat("How long does shipping normally take?")
    assert "2 weeks" in shipping["reply"]
    assert state["last_policy_topic"] == "shipping"


def test_exact_generation_and_requested_features_survive_pronoun_followups(chat):
    price, state = chat("What is the price of the Maestro LE 2.0?")
    assert "Maestro LE 2.0" in price["reply"]
    assert "OS-Pro Maestro LE**" not in price["reply"]
    bluetooth, _ = chat("Does it have Bluetooth speakers?")
    assert "Maestro LE 2.0" in bluetooth["reply"] and "Bluetooth speakers" in bluetooth["reply"]
    assert "user height" not in bluetooth["reply"].lower()
    colors, _ = chat("What colors can I buy it in?")
    assert "Maestro LE 2.0" in colors["reply"] and "colors" in colors["reply"].lower()
    scanning, _ = chat("Does the LE 2.0 have computerized body scanning?")
    assert "Maestro LE 2.0" in scanning["reply"] and "Body scanning" in scanning["reply"]


def test_general_feature_definition_does_not_require_a_model(chat):
    reply, _ = chat("What does zero gravity mean?")
    assert "reclined position" in reply["reply"]
    assert "Which model" not in reply["reply"]
    assert "user height" not in reply["reply"].lower()


def test_direct_intensity_question_interrupts_the_fit_interview(chat):
    chat("Please recommend a chair for me.")
    reply, _ = chat("How strong is the foot massage intensity? Can I adjust it?")
    assert reply["intent"] == "intensity"
    assert "pressure" in reply["reply"].lower()
    assert "user height" not in reply["reply"].lower()


def test_general_coverage_and_labor_question_can_later_become_an_actual_fault(chat):
    coverage, _ = chat("What does the warranty on a new massage chair cover?")
    assert coverage['intent'] == 'prepurchase_policy'
    assert 'parts' in coverage['reply'] and 'labor' in coverage['reply']
    labor, _ = chat("I mean labor coverage, not a repair claim. Which years cover labor?")
    assert labor['intent'] == 'prepurchase_policy'
    assert 'Year 1' in labor['reply']
    fault, _ = chat("I own an OS-Champ. Its remote screen is black and the buttons do not work.")
    assert fault['intent'] == 'warranty_redirect'
    approved, _ = chat("Is the replacement approved and free already?")
    assert approved['handoff']
    assert 'cannot confirm' in approved['reply'] and 'approved or is free' in approved['reply']


def test_assembled_dimensions_are_specs_and_do_not_start_an_assembly_service_answer(chat):
    from sales_catalog import resolve_product
    from sales_spec_index import lookup_fit_spec
    chair = resolve_product('Titan TP-Epic 4D')
    width = lookup_fit_spec(chair.display_name).door_asm_in
    reply, _ = chat("What assembled doorway width does Titan TP-Epic 4D require?")
    assert reply['intent'] == 'specs'
    assert f'{width:g}' in reply['reply']
    assert not reply['reply'].startswith('**White Glove')
    actual_assembly, _ = chat("Does White Glove delivery include assembly?")
    assert actual_assembly['intent'] == 'prepurchase_policy'
    assert 'assembly' in actual_assembly['reply']


def test_saved_dimensions_can_be_reused_after_an_honest_no_fit_result(chat):
    chat('I am 5 foot 8 and weigh 170 pounds. I want gentle shoulder massage. My budget is $4000. My doorway is 28 inches, assembled only. Recommend a chair.')
    reply, state = chat('Please use the dimensions I gave you.')
    assert reply['intent'] == 'recommend'
    assert state['recommend_prefs']['doorway_in'] == '28'
    assert 'Which model would you like specs' not in reply['reply']
    for product in reply['products']:
        assert product['fit']['doorway_assembled_in'] <= 28


def test_order_context_keeps_color_corrections_and_invoice_requests_with_sales(chat):
    address, _ = chat('I need to change the shipping address on my order.')
    assert address['intent'] == 'order_change'
    color, _ = chat('I also want to change the ordered chair from brown to black.')
    assert color['intent'] == 'order_change' and color['handoff']
    clarification, _ = chat('This is an order change before shipping, not a repair or warranty claim. Can Sales review it?')
    assert clarification['intent'] == 'order_change'
    chat('If the change is impossible, please cancel my order.')
    invoice, _ = chat('Could Sales send me a copy of the invoice for the existing order?')
    assert invoice['intent'] == 'order_status' and 'invoice' in invoice['reply']


def test_declined_card_paraphrase_and_checkout_followup_get_sales_help(chat):
    declined, _ = chat('My credit card keeps getting declined. What should I do?')
    assert declined['intent'] == 'prepurchase_policy'
    assert 'Sales specialist' in declined['reply']
    followup, _ = chat('I am not asking for a different chair. Can Sales help check the checkout problem?')
    assert followup['intent'] == 'prepurchase_policy'
    assert 'Sales specialist' in followup['reply']


def test_return_request_accepts_a_delivered_modifier_after_language_switch(chat):
    chat('Where is my order?')
    reply, _ = chat('Please answer in English. I want to return my delivered chair.')
    assert reply['intent'] == 'cancel_refund'
    assert 'Sales' in reply['reply'] or 'sales' in reply['reply']
    assert 'approved' in reply['reply']
