"""Customer corrections and household requirements survive real Sales turns."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

import sales_agent as agent
from sales_catalog import parse_recommendation_hints, resolve_product
from sales_constraints import extract_constraints, product_constraint_issues
from sales_language import customer_language, medical_scope_message


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, 'fetch_stock_many', lambda *args, **kwargs: {})


def turn(message, prefs):
    reply = agent.respond(message, prefs=prefs)
    for key, value in (reply.prefs_patch or {}).items():
        prefs[key] = {**prefs.get(key, {}), **value} if isinstance(value, dict) else value
    return reply


@pytest.mark.parametrize('question', [
    'Does the Maestro have a 350 pound capacity?',
    'Is its 350 lb capacity published?',
    'The package is 68 inches high and weighs 300 pounds.',
])
def test_product_measurements_do_not_replace_customer_fit(question):
    saved = {'height_in': '68', 'weight_lb': '170'}
    parsed = extract_constraints(question, saved)
    assert parsed['weight_lb'] == '170'
    assert parsed['height_in'] == '68'
    assert parse_recommendation_hints(question).weight_lb is None
    assert parse_recommendation_hints(question).height_in is None


def test_shared_users_remain_separate_when_only_one_is_corrected():
    saved = extract_constraints('My father is 6 foot 2 and weighs 320 pounds. My mother is 5 foot 2 and weighs 150 pounds. They will share one chair.')
    assert saved['weight_lb'] == '320'
    assert json.loads(saved['user_heights_in']) == {'father': 74, 'mother': 62}
    assert 'weight_capacity' in product_constraint_issues(resolve_product('Osaki OS-Champ'), saved)
    saved = extract_constraints('My mother now weighs 160 pounds.', saved)
    assert saved['weight_lb'] == '320'
    assert json.loads(saved['user_weights_lb']) == {'father': 320, 'mother': 160}
    saved = extract_constraints('Correction: my father weighs 300 pounds, not 320 pounds.', saved)
    assert saved['weight_lb'] == '300'
    assert json.loads(saved['user_weights_lb']) == {'father': 300, 'mother': 160}


def test_a_second_user_supplied_later_cannot_lower_the_shared_capacity_requirement():
    prefs = {}
    turn('Recommend a chair for my father who is 6 foot 2 and weighs 320 pounds. Budget $12000.', prefs)
    turn('My mother will also use it. She is 5 foot 2 and weighs 150 pounds.', prefs)
    reply = turn('They both want gentle full body massage. Recommend a chair for both.', prefs)
    rec = prefs['recommend_prefs']
    assert rec['weight_lb'] == '320'
    assert json.loads(rec['user_weights_lb']) == {'father': 320, 'mother': 150}
    assert json.loads(rec['user_heights_in']) == {'father': 74, 'mother': 62}
    for card in reply.products:
        assert card['fit']['maximum_user_weight_lb'] >= 320


@pytest.mark.parametrize('message,minimum', [
    ('One chair will be shared by my husband at 310 pounds and me at 140 pounds. I am the person typing.', 310),
    ('The two users weigh 320 pounds and 150 pounds.', 320),
    ('We both weigh 315 pounds and 165 pounds and will share a chair.', 315),
])
def test_unidentified_or_self_named_shared_users_do_not_overwrite_a_heavier_person(message, minimum):
    saved = extract_constraints(message)
    assert float(saved['weight_lb']) == minimum
    assert len(json.loads(saved['user_weights_lb'])) == 2
    assert 'weight_capacity' in product_constraint_issues(resolve_product('Osaki OS-Champ'), saved)


def test_alternative_mechanisms_allow_both_until_an_explicit_single_choice():
    saved = extract_constraints('Either 3D or 4D would be fine; I do not have a preference.',
                                {'mechanism': '3D', 'excluded_mechanisms': '4D'})
    candidates = [SimpleNamespace(display_name='Synthetic chair', title='Synthetic chair',
                                  massage_mechanism=mechanism, price_usd=2000)
                  for mechanism in ['3D', '4D']]
    assert all(not product_constraint_issues(candidate, saved) for candidate in candidates)
    changed = extract_constraints('Actually, I want a 4D chair.', saved)
    assert 'mechanism' in product_constraint_issues(candidates[0], changed)
    assert not product_constraint_issues(candidates[1], changed)


@pytest.mark.parametrize('rejection', ['Do not recommend', "Don't recommend"])
def test_rejected_mechanism_list_does_not_become_an_allowed_alternative(rejection):
    prefs = {}
    turn('I am 5 foot 8, weigh 170 pounds, want gentle back massage, and have a $4000 budget. Please recommend a chair.', prefs)
    turn(f'{rejection} 4D or 5D. I want 3D.', prefs)
    reply = turn('Please keep my mechanism preference when choosing.', prefs)
    saved = prefs['recommend_prefs']
    assert saved['mechanism'] == '3D'
    assert set(saved['excluded_mechanisms'].split(',')) == {'4D', '5D'}
    assert not saved.get('allowed_mechanisms')
    for card in reply.products:
        assert card['specs']['massage_mechanism'] == '3D'


def test_removing_a_budget_removes_both_the_hard_cap_and_old_price_bucket():
    prefs = {'recommend_prefs': {'height_in': '68', 'weight_lb': '170',
                                'budget_max_usd': '3000', 'budget': 'Under $3,000'}}
    turn('My budget is no longer a restriction.', prefs)
    saved = prefs['recommend_prefs']
    assert saved.get('budget_max_usd') == ''
    assert saved.get('budget') == ''
    assert 'budget' not in product_constraint_issues(resolve_product('Osaki OS-Pro Maestro LE'), saved)


@pytest.mark.parametrize('message,expected', [
    ('My doorway is not 36 inches but 28 inches.', 28),
    ('Correction: my doorway is 28 inches, not 36. I cannot remove the chair\'s side panels.', 28),
    ('My front doorway is 36 inches wide, but the hallway doorway is 29 inches wide.', 29),
    ('My doorway is 740 millimeters wide.', 740 / 25.4),
])
def test_delivery_route_uses_the_corrected_or_narrowest_real_opening(message, expected):
    parsed = extract_constraints(message, {'doorway_in': '36'})
    assert float(parsed['doorway_in']) == pytest.approx(expected, abs=.0001)


@pytest.mark.parametrize('message,field,value', [
    ('My height is 68 inches.', 'height_in', 68),
    ('I weigh 14 stone.', 'weight_lb', 196),
    ('My budget is USD 4,250.', 'budget_max_usd', 4250),
    ('My budget is three thousand dollars.', 'budget_max_usd', 3000),
    ('My budget is twenty-five hundred dollars.', 'budget_max_usd', 2500),
    ('My budget is three thousand.', 'budget_max_usd', 3000),
    ('My budget is twenty-five hundred.', 'budget_max_usd', 2500),
    ('My budget is $50,001.', 'budget_max_usd', 50001),
])
def test_explicit_unit_and_budget_variants_keep_the_stated_value(message, field, value):
    assert float(extract_constraints(message)[field]) == pytest.approx(value)


@pytest.mark.parametrize('message', [
    'I am in my third trimester. Is it definitely safe to use the heated rollers?',
    'I have an implanted defibrillator. Can you guarantee this is safe?',
    'Me operaron la columna hace una semana. ¿Puedo usar los rodillos sin riesgo?',
])
def test_medical_clearance_questions_get_clinician_guidance(message):
    copy = medical_scope_message(message, 'en')
    assert copy and 'clinician' in copy and 'cannot provide medical clearance' in copy
    reply = agent.respond(message)
    assert 'policy.medical_scope' in reply.tools_used
    assert not reply.products
    assert reply.flow_stage != 'ask_height'


def test_ordinary_heating_feature_question_is_not_medical_clearance():
    assert medical_scope_message('Does the LE 2.0 have heated rollers?') is None


@pytest.mark.parametrize('message,previous,expected', [
    ('Do not answer in English; please use Spanish.', 'en', 'es'),
    ('Please use Korean, not Spanish.', 'es', 'ko'),
    ('I would like information about delivery and assembly.', 'es', 'en'),
    ('180 cm / 80 kg', 'es', 'es'),
    ('Thank you so much.', 'ko', 'ko'),
])
def test_language_changes_follow_customer_requests_and_substantive_questions(message, previous, expected):
    assert customer_language(message, previous) == expected
