"""Hard limits and preference corrections from adversarial shopper simulations."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

import sales_agent as agent
from sales_catalog import parse_recommendation_hints
from sales_constraints import extract_constraints, product_constraint_issues
from sales_catalog import resolve_product
from sales_language import prepare_message


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *a, **k: None)
    monkeypatch.setattr(agent, 'fetch_stock_many', lambda *a, **k: {})


@pytest.mark.parametrize('cap', [0, 100, 300, 450, 499])
def test_small_explicit_budget_is_a_hard_limit(cap):
    message = f"I'm 5 ft 7, 165 pounds and want shoulder massage. My budget is ${cap}."
    reply = agent.respond(message)
    assert reply.prefs_patch['recommend_prefs']['budget_max_usd'] == str(cap)
    assert not reply.products
    assert reply.flow_stage == 'recommend_nofit'


def test_low_budget_correction_replaces_previous_cap_and_survives_followup():
    prefs = {'recommend_prefs': {'height': 'Average (5\'4"–5\'11")', 'height_in': '68', 'weight_lb': '160', 'goal': 'Lower Back', 'budget_max_usd': '4000'}}
    corrected = agent.respond('My budget is now $400, not $4,000.', prefs=prefs)
    assert corrected.prefs_patch['recommend_prefs']['budget_max_usd'] == '400'
    prefs.update(corrected.prefs_patch)
    final = agent.respond('Recommend using the limits I already gave.', prefs=prefs)
    assert not final.products


@pytest.mark.parametrize('unit', ['kilogram', 'kilograms', 'kgs'])
def test_spelled_metric_weight_filters_undersized_chairs(unit):
    message = f"I'm 180 cm tall and weigh 150 {unit}. Recommend a chair for lower back massage."
    reply = agent.respond(message)
    rec = reply.prefs_patch['recommend_prefs']
    assert float(rec['weight_lb']) == pytest.approx(150 * 2.20462)
    for product in reply.products:
        assert product['fit']['maximum_user_weight_lb'] >= float(rec['weight_lb'])
        assert not product_constraint_issues(resolve_product(product['model']), rec, price_usd=product['price_usd'])


@pytest.mark.parametrize('message', ['I need a chair for up to 350 lbs.', 'The user weighs a maximum 200 kilograms.', 'My doorway is at most 74 cm wide.'])
def test_expanded_budget_parser_does_not_capture_fit_dimensions_as_money(message):
    assert 'budget_max_usd' not in extract_constraints(message)
    assert parse_recommendation_hints(message).budget_usd is None


@pytest.mark.parametrize('measurement', ['74 cm', '74 centimeters', '74 centimetres', '0.74 metres'])
def test_metric_doorway_is_saved_without_repeating_the_question(measurement):
    message = f"I'm 180 cm and 80 kg with a $4000 budget. My doorway is {measurement} wide, assembled only. Recommend a chair for shoulder massage."
    reply = agent.respond(message)
    rec = reply.prefs_patch['recommend_prefs']
    assert float(rec['doorway_in']) == pytest.approx(74 / 2.54, abs=0.0001)
    assert rec['height_in'] == '71'
    assert reply.flow_stage != 'ask_doorway'
    for product in reply.products:
        assert product['fit']['doorway_assembled_in'] <= float(rec['doorway_in'])


def test_spanish_wide_door_does_not_replace_saved_body_height():
    translated, language = prepare_message('Mi puerta mide 100 cm, sin desmontar.', {})
    exact = extract_constraints(translated, {'height_in': '70'})
    assert language == 'es'
    assert exact['height_in'] == '70'
    assert float(exact['doorway_in']) == pytest.approx(100 / 2.54, abs=0.0001)
    assert exact['doorway_fit'] == 'assembled'


@pytest.mark.parametrize('height', ['1.82m', '1.82 meters', '1.82 metres', '182 centimeters'])
def test_metric_height_uses_the_stated_value(height):
    assert parse_recommendation_hints(f'I am {height} tall.').height_in == 72
    assert parse_recommendation_hints(f'The doorway is {height} tall.').height_in is None


@pytest.mark.parametrize('message,expected', [
    ("I don't want strong massage", 'Gentle'),
    ('I do not want deep tissue', 'Gentle'),
    ('I want medium massage, not strong', 'Balanced'),
    ('I do not want gentle massage, I prefer strong', 'Strong'),
    ('No strong massage. Medium is best.', 'Balanced'),
    ('I want strong massage, not gentle', 'Strong'),
    ('I do not want strong or firm massage.', 'Gentle'),
    ('I want strong pressure, not gentle and soft.', 'Strong'),
    ('I do not want gentle or soft massage, but I want strong pressure.', 'Strong'),
])
def test_intensity_negation_keeps_the_affirmative_preference(message, expected):
    reply = agent.respond(message, prefs={'recommend_prefs': {'height_in': '68', 'height': 'Average (5\'4"–5\'11")', 'goal': 'Lower Back'}})
    assert reply.prefs_patch['recommend_prefs']['intensity'] == expected


def test_rejected_mechanism_never_becomes_the_requested_one():
    message = "Recommend 3D, not a 4D chair. I'm 5 ft 8, 160 lb and need lower back massage."
    reply = agent.respond(message)
    rec = reply.prefs_patch['recommend_prefs']
    assert rec['mechanism'] == '3D'
    assert reply.products
    assert all('4d' not in p['specs']['massage_mechanism'].lower() for p in reply.products)


def test_negative_mechanism_exclusion_is_retained_until_explicitly_corrected():
    message = "I don't want a 4D chair. I'm 5 ft 8, 160 lb and need lower back massage. Recommend a chair."
    reply = agent.respond(message)
    rec = reply.prefs_patch['recommend_prefs']
    assert rec.get('mechanism') != '4D'
    assert all('4d' not in p['specs']['massage_mechanism'].lower() for p in reply.products)
    allowed = extract_constraints('Actually I want 4D after all.', rec)
    assert allowed['mechanism'] == '4D'
    assert '4D' not in allowed.get('excluded_mechanisms', '').split(',')


@pytest.mark.parametrize('message,chosen,rejected', [
    ('I want a 3D chair, not a 4D or 5D one.', '3D', {'4D', '5D'}),
    ('I do not want 4D or 3D.', None, {'4D', '3D'}),
    ('I do not want 4D or 5D, but I want a 3D chair.', '3D', {'4D', '5D'}),
])
def test_mechanism_negation_extends_across_a_list_but_not_a_new_request(message, chosen, rejected):
    rec = extract_constraints(message)
    assert rec.get('mechanism') == chosen
    assert set(rec['excluded_mechanisms'].split(',')) == rejected


@pytest.mark.parametrize('message', ['I want 3D, not a 4D chair', 'I prefer a 3D chair', 'We need 3D'])
def test_explicit_mechanism_preference_starts_the_fit_interview(message):
    reply = agent.respond(message)
    assert reply.intent == 'recommend'
    assert reply.prefs_patch['recommend_prefs']['mechanism'] == '3D'
    assert reply.flow_stage == 'ask_height'
