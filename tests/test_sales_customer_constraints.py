"""Customer constraints survive natural phrasing and interruptions to a fit interview."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

import sales_agent as agent
from sales_catalog import parse_recommendation_hints, resolve_product
from sales_cases import height_from_shopper_answer, merge_prefs_from_hints
from sales_constraints import extract_constraints, product_constraint_issues


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, 'fetch_stock_many', lambda *args, **kwargs: {})


def turn(message, prefs):
    reply = agent.respond(message, prefs=prefs, domain='osakiusa.com')
    for key, value in (reply.prefs_patch or {}).items():
        if isinstance(value, dict) and isinstance(prefs.get(key), dict):
            prefs[key] = {**prefs[key], **value}
        else:
            prefs[key] = value
    return reply


@pytest.mark.parametrize('message,inches', [
    ("I'm 6 foot 4 and about 290 pounds.", 76),
    ("I'm five feet seven inches tall.", 67),
    ('The buyer is six-foot-five.', 77),
    ('five foot eleven', 71),
    ('I am five feet and ten inches tall.', 70),
    ("I'm 6'2\".", 74),
])
def test_spoken_height_is_measured_without_becoming_a_foot_massage_goal(message, inches):
    hints = parse_recommendation_hints(message)
    assert hints.height_in == inches
    assert extract_constraints(message)['height_in'] == str(inches)
    assert hints.focus_areas == []
    prefs = merge_prefs_from_hints({}, height_in=hints.height_in, focus_areas=hints.focus_areas, free_text=message)
    assert 'goal' not in prefs
    assert 'foot' not in prefs
    assert height_from_shopper_answer(message) == prefs['height']


def test_height_units_do_not_hide_an_explicit_calf_preference_or_product_feature():
    hints = parse_recommendation_hints('I am six foot four, and I want calf and foot massage.')
    assert hints.height_in == 76
    assert hints.focus_areas == ['feet']
    assert parse_recommendation_hints('Does this chair have 3 foot rollers?').height_in is None
    assert parse_recommendation_hints('My height is six feet thirteen inches').height_in is None


@pytest.mark.parametrize('message', [
    'My door opening is 29 inches, assembled only.',
    'The entrance measures 29 inches. No disassembly.',
    'It needs to clear a 29-inch door. Cannot disassemble.',
    'I only have a 29 in wide doorway, assembled only.',
    'The doorway width is 29 inches, without taking the chair apart.',
])
def test_doorway_language_produces_an_exact_assembled_constraint(message):
    constraints = extract_constraints(message)
    assert constraints['doorway_in'] == '29'
    assert constraints['doorway_fit'] == 'assembled'


def test_regression_dialog_never_recommends_a_34_inch_chair_for_a_29_inch_door():
    prefs = {}
    replies = [turn("I have at most $2,500. I'm 6 foot 4 and about 290 pounds. What would you shortlist?", prefs)]
    replies.append(turn('My door opening is 29 inches, assembled only.', prefs))
    # The doorway answer may also trigger a shipping-policy explanation. The
    # customer's measurements must already be retained before that detour.
    assert prefs['recommend_prefs']['doorway_in'] == '29'
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    replies.append(turn('Soft massage, mostly calves. Can you still recommend one?', prefs))
    replies.append(turn("I'm 6'4\". Please use that height and the limits I already gave.", prefs))
    constraints = prefs['recommend_prefs']
    assert constraints['height_in'] == '76'
    assert constraints['weight_lb'] == '290'
    assert constraints['budget_max_usd'] == '2500'
    assert constraints['doorway_in'] == '29'
    assert constraints['doorway_fit'] == 'assembled'
    for reply in replies:
        for product in reply.products:
            assert not product_constraint_issues(resolve_product(product['model']), constraints, price_usd=product['price_usd'])
    assert not replies[-1].products
    assert replies[-1].flow_stage == 'recommend_nofit'
    assert '29' in replies[-1].reply


@pytest.mark.parametrize('interruption', [
    'My doorway is 29 inches, assembled only. How does White Glove delivery work?',
    'My door opening is 29 inches, assembled only. Connect me to a person.',
])
def test_fit_measurements_survive_policy_and_human_interruptions(interruption):
    prefs = {}
    turn('Please recommend a chair with a $4000 maximum budget.', prefs)
    reply = turn(interruption, prefs)
    assert prefs['recommend_prefs']['doorway_in'] == '29'
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    assert prefs['recommend_prefs']['budget_max_usd'] == '4000'
    if 'person' in interruption:
        assert reply.handoff
    reply = turn('I am five feet seven inches, 165 pounds, and want gentle shoulder massage.', prefs)
    for product in reply.products:
        assert product['fit']['doorway_assembled_in'] is not None
        assert product['fit']['doorway_assembled_in'] <= 29
        assert product['price_usd'] <= 4000


def test_corrections_replace_only_the_measurements_that_were_actually_changed():
    prefs = {}
    turn('I am five feet four, 165 pounds, with a $4000 budget and a 36-inch doorway. Assembled only; gentle shoulder massage.', prefs)
    reply = turn('Actually the buyer is six foot five and 320 pounds.', prefs)
    rec = prefs['recommend_prefs']
    assert rec['height_in'] == '77'
    assert rec['weight_lb'] == '320'
    assert rec['budget_max_usd'] == '4000'
    assert rec['doorway_in'] == '36'
    assert rec['doorway_fit'] == 'assembled'
    assert rec['intensity'] == 'Gentle'
    assert rec['goal'] == 'Neck & Shoulders'
    for product in reply.products:
        assert not product_constraint_issues(resolve_product(product['model']), rec, price_usd=product['price_usd'])
    turn('Correction: my doorway opening is 29 inches.', prefs)
    assert prefs['recommend_prefs']['doorway_in'] == '29'
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    assert prefs['recommend_prefs']['height_in'] == '77'


def test_old_measurements_explicitly_negated_in_a_correction_do_not_win():
    message = 'Actually I am five feet eight, not six foot two, and 190 lb, not 220 lb. My budget is $3500, not $7000.'
    exact = extract_constraints(message)
    assert exact['height_in'] == '68'
    assert exact['weight_lb'] == '190'
    assert exact['budget_max_usd'] == '3500'
    hints = parse_recommendation_hints(message)
    assert hints.height_in == 68
    assert hints.weight_lb == 190
    assert hints.budget_usd == 3500
    assert hints.focus_areas == []


@pytest.mark.parametrize('question', [
    'Does the Maestro 4D have a weight capacity of 350 lb?',
    'Is White Glove shipping $1500?',
    'What is the price of the Maestro 4D, $8999?',
])
def test_a_product_fact_or_shipping_fee_is_not_a_new_shopper_limit(question):
    prefs = {}
    turn("I'm 5'7\", 165 lb, with a $4000 budget and need shoulder massage.", prefs)
    before = dict(prefs['recommend_prefs'])
    turn(question, prefs)
    assert prefs['recommend_prefs']['height_in'] == before['height_in']
    assert prefs['recommend_prefs']['weight_lb'] == before['weight_lb']
    assert prefs['recommend_prefs']['budget_max_usd'] == before['budget_max_usd']
    assert prefs['recommend_prefs'].get('mechanism') == before.get('mechanism')


@pytest.mark.parametrize('message', [
    'My doorway is six feet tall and 29 inches wide.',
    'I have six feet of clearance behind the chair.',
    'The doorway height is 185 cm and the chair is 180 cm tall.',
])
def test_doorway_and_space_dimensions_do_not_replace_the_saved_body_height(message):
    assert 'height_in' not in extract_constraints(message)
    assert parse_recommendation_hints(message).height_in is None
    assert parse_recommendation_hints(message).focus_areas == []
    prefs = {}
    turn("I'm 5'7\", 165 lb, with a $4000 budget and need shoulder massage.", prefs)
    turn(message, prefs)
    assert prefs['recommend_prefs']['height_in'] == '67'


@pytest.mark.parametrize('question', [
    'Does my Maestro chair have a maximum weight capacity of 350 lb?',
    'My chair is rated for 350 lb. Is that right?',
    'Does my chair support 350 pounds?',
    'My chair has a maximum 350 lb user weight rating.',
    'I need a chair, but my current chair is rated for 350 lb.',
    'I need a chair, but my current chair is rated for 190 lb.',
])
def test_possessive_my_does_not_make_a_product_capacity_the_customer_weight(question):
    assert 'weight_lb' not in extract_constraints(question)
    assert parse_recommendation_hints(question).weight_lb is None
    prefs = {}
    turn("I'm 5'7\", 165 lb, with a $4000 budget and need shoulder massage.", prefs)
    turn(question, prefs)
    assert prefs['recommend_prefs']['weight_lb'] == '165'


def test_person_measurements_win_over_adjacent_product_dimensions_and_capacity():
    question = 'I weigh 165 lb; the chair is rated for 350 lb. Will it work?'
    assert extract_constraints(question)['weight_lb'] == '165'
    assert parse_recommendation_hints(question).weight_lb == 165
    assert extract_constraints('My doorway is six feet tall, but I am five feet seven inches.')['height_in'] == '67'
    assert extract_constraints('My wife weighs 190 lb, and the chair supports 350 lb.')['weight_lb'] == '190'


@pytest.mark.parametrize('weight', [165, 320])
@pytest.mark.parametrize('object_weight', [
    'The chair is 190 pounds.',
    'The delivery package is 190 pounds.',
    'The 190 lb chair is too heavy for me.',
])
def test_object_weight_cannot_replace_a_saved_customer_weight(weight, object_weight):
    assert 'weight_lb' not in extract_constraints(object_weight)
    assert parse_recommendation_hints(object_weight).weight_lb is None
    prefs = {}
    turn(f"I'm 5'7\", {weight} lb, with a $4000 budget and need shoulder massage.", prefs)
    turn(object_weight, prefs)
    assert prefs['recommend_prefs']['weight_lb'] == str(weight)
    reply = turn('Recommend a chair using the requirements I already gave.', prefs)
    assert prefs['recommend_prefs']['weight_lb'] == str(weight)
    for product in reply.products:
        assert not product_constraint_issues(resolve_product(product['model']), prefs['recommend_prefs'], price_usd=product['price_usd'])


def test_direct_weight_answer_and_personal_weight_after_a_package_remain_supported():
    for message in ['190 lb', 'I weigh 190 pounds.', 'The package is 165 lb, but I weigh 320 lb.']:
        expected = '320' if '320' in message else '190'
        assert extract_constraints(message)['weight_lb'] == expected
        assert parse_recommendation_hints(message).weight_lb == int(expected)
    assert extract_constraints('I need a chair for a 320 lb user.')['weight_lb'] == '320'


@pytest.mark.parametrize('message', [
    'Need a chair for 350 lbs with a 4D massage system.',
    'Recommend a massage chair for up to 350 lbs with a 4D massage system.',
    'I want a chair for at least 350 lb.',
    'I need a chair that can support 350 pounds.',
])
def test_explicit_customer_capacity_requirement_is_preserved(message):
    assert extract_constraints(message)['weight_lb'] == '350'
    assert parse_recommendation_hints(message).weight_lb == 350
