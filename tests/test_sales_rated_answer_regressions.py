"""Anonymized reconstructions of rated production failures; offline only."""

import pytest

import sales_agent as agent
from sales_shopify_stock import LiveStockSnapshot


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *_a, **_k: None)


@pytest.mark.parametrize('status,expected', [
    (None, "couldn't confirm"), (False, "does not currently list"), (True, "available to order"),
])
def test_black_duke_stock_and_delivery_time_are_answered_not_mechanism(monkeypatch, status, expected):
    colors = () if status is None else (('black', status), ('taupe', True))
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *_a, **_k: LiveStockSnapshot(
        'osaki-duke-xl-4d', 'Duke XL 4D', 'active', True, 5, 'shopify', color_availability=colors))
    r = agent.respond('Osaki Duke XL 4D, Black, one chair. Is it in stock? Lead time?', prefs={'last_policy_topic': 'mechanism'})
    assert r.intent == 'stock'
    assert expected in r.reply and '**Black**' in r.reply and 'confirmed lead time' in r.reply
    assert '2 weeks' not in r.reply and '3 weeks' not in r.reply and 'along the X' not in r.reply


def test_vivant_settings_helps_owner_without_claiming_a_verified_tutorial():
    r = agent.respond('I bought an Osaki Pro Vivant 5D. Where is a tutorial video for the settings?')
    assert 'touchscreen' in r.reply and 'User Manual' in r.reply and 'otasupport.com' in r.reply
    assert "couldn't verify a dedicated settings video" in r.reply
    assert 'Shop:' not in r.reply


def test_parts_questions_keep_support_task_when_model_is_repeated():
    r = agent.respond('Where can I get replacement upholstery for my red OS-3D Cyber Pro?')
    assert r.intent == 'parts_technician' and r.handoff
    followup = agent.respond('Osaki OS-3D Cyber Pro Massage Chair, Red', prefs=r.prefs_patch)
    assert followup.intent == 'parts_technician' and 'upholstery' in followup.reply
    assert 'can check availability' in followup.reply and 'Shop:' not in followup.reply


def test_unidentified_connector_is_not_a_product_pitch_or_wiring_instruction():
    r = agent.respond('My OS-4000XT needs a replacement remote and I found an unmatched connector.')
    assert r.intent == 'parts_technician' and r.handoff
    assert 'Do not connect' in r.reply and 'confirm the exact compatible part' in r.reply
    assert '260 lb' not in r.reply and 'Shop:' not in r.reply


def test_higher_capacity_alternative_does_not_sell_the_same_yoga_flex_again():
    r = agent.respond('I want something similar to the Yoga Flex but less focused on stretching and able to support more weight.')
    assert r.intent == 'recommend' and 'Duke XL 4D' in r.reply and '400 lb' in r.reply
    assert 'not a guarantee of fit' in r.reply
    assert not r.products


def test_higher_capacity_alternative_respects_known_shopper_constraints():
    r = agent.respond('Something similar to Yoga Flex that can handle more weight?', prefs={'recommend_prefs': {'weight_lb': 410}})
    assert 'Duke XL' not in r.reply


def test_orion_full_name_does_not_set_manhattan_as_followup_model():
    start = agent.respond('Osaki AI Orion Duo 4D and 3D Massage Chair')
    assert start.products[0]['catalog_handle'] == 'osaki-4d-3d-orion-duo-mech'
    followup = agent.respond('What is the min person heighth?', prefs=start.prefs_patch)
    assert followup.intent == 'specs' and 'Orion' in followup.reply
    assert 'showroom' not in followup.reply.lower() and 'Manhattan' not in followup.reply


def test_orion_mechanism_names_do_not_become_a_comparison():
    r = agent.respond('Can I adjust speed and pressure on the Osaki AI Orion Duo 4D and 3D Massage Chair?')
    assert r.intent == 'specs' and 'touchscreen' in r.reply and '6 levels' in r.reply
    assert 'catalog.verified_model_specs' in r.tools_used


def test_orion_fit_answers_both_measurements_instead_of_comparing_chairs():
    r = agent.respond('Will Orion Duo 4D + 3D fit someone who is 6ft2 and 220 pounds?')
    assert r.intent == 'specs' and '220 lb' in r.reply and 'within' in r.reply
    assert "6'2\"" in r.reply


def test_orion_source_not_used_when_customer_names_another_chair():
    r = agent.respond('Can I change the pressure on the Manhattan Duo, not the Orion Duo?')
    assert 'catalog.verified_model_specs' not in r.tools_used
    assert '6 levels of 4D roller depth' not in r.reply


def test_verified_answer_preserves_pending_explicit_human_handoff():
    r = agent.respond('How many boxes does a Hamilton LE arrive in?', prefs={'awaiting_pre_human_question': True})
    assert '2 shipping boxes' in r.reply
    assert 'human:confirm' in {q.payload for q in r.quick_replies}
    assert r.prefs_patch['awaiting_pre_human_question'] is False
