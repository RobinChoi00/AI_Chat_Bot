"""Offline regressions from deidentified customer conversation failures."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import sales_agent as agent
from sales_catalog import RecommendationRequest, recommend, resolve_product
from sales_constraints import extract_constraints, product_constraint_issues
from sales_spec_index import lookup_fit_spec


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *a, **k: None)


def merge(prefs, reply):
    result = dict(prefs)
    for key, value in (reply.prefs_patch or {}).items():
        result[key] = {**result.get(key, {}), **value} if key == 'recommend_prefs' else value
    return result


@pytest.mark.parametrize('question,label', [
    ('I need a replacement remote; the connector does not match.', 'parts_technician'),
    ('I need to replace the footrest on my chair', 'parts_technician'),
    ('I already purchased a chair and I love it!', 'greeting'),
    ('Can I use FSA?', 'prepurchase_policy'),
    ('I already ordered but forgot to upgrade White Glove delivery', 'order_change'),
])
def test_real_failure_routes(question, label):
    reply = agent.respond(question)
    assert reply.intent == label
    assert not reply.products
    assert 'user height' not in reply.reply.lower()


def test_feature_answer_before_interview_even_with_pending_recommendation():
    prefs = merge({}, agent.respond('Recommend a chair'))
    reply = agent.respond('Does Titan TP-Epic 4D have foot rollers?', prefs=prefs)
    assert reply.intent == 'specs'
    assert '3 spinning reflexology massage rollers' in reply.reply
    assert reply.reply.index('Foot/calf roller') < reply.reply.index('Full quick specs')


def test_showroom_question_answers_location_for_saved_model():
    reply = agent.respond('Where to try this chair before buying', prefs={'pending_primary': 'Osaki Maestro LE 2.0'})
    assert 'Carrollton' in reply.reply
    assert reply.intent == 'prepurchase_policy'
    assert 'Full quick specs' not in reply.reply


def test_exact_constraints_retained_across_short_answers():
    first = agent.respond('I need 4D for 350 lb with deep tissue')
    prefs = merge({}, first)
    second = agent.respond('6 ft 2', prefs=prefs)
    prefs = merge(prefs, second)
    third = agent.respond('lower back', prefs=prefs)
    rec = third.prefs_patch['recommend_prefs']
    assert rec['weight_lb'] == '350'
    assert rec['mechanism'] == '4D'
    assert rec['intensity'] == 'Strong'
    assert rec['height_in'] == '74'
    assert third.products
    for product in third.products:
        assert product['specs']['massage_mechanism'] == '4D'
        assert product['fit']['maximum_user_weight_lb'] >= 350
    assert 'height fit before purchase' in third.reply
    assert 'mechanism label alone' in third.reply


def test_budget_is_hard_cap_in_casebook_and_catalog():
    reply = agent.respond('I am 5 ft 10, 220 lb, need lower back relief; budget up to $3500')
    assert reply.products
    assert all(product['price_usd'] <= 3500 for product in reply.products)
    assert 'Value (under ~$3k)' not in reply.reply
    products = recommend(RecommendationRequest(height_in=70, budget_usd=3500, focus_areas=['back']))
    assert products and all(product.price_usd <= 3500 for product in products)


def test_explicit_corrections_replace_saved_fit_and_budget():
    first = agent.respond('I am 6 ft 2, 220 lb, need lower back massage with a $7000 budget')
    prefs = merge({}, first)
    corrected = agent.respond('Actually I am 5 ft 8, 350 lb, my budget is now $3500; require 4D', prefs=prefs)
    rec = corrected.prefs_patch['recommend_prefs']
    assert rec['height_in'] == '68'
    assert rec['weight_lb'] == '350'
    assert rec['budget_max_usd'] == '3500'
    assert rec['height'].startswith('Average')
    assert rec['weight'] == '301+ lb'
    assert all(p['price_usd'] <= 3500 and p['fit']['maximum_user_weight_lb'] >= 350 for p in corrected.products)


def test_live_price_recheck_rejects_over_budget_stock_fallback(monkeypatch):
    product = resolve_product('Titan TP-Epic 4D')
    monkeypatch.setattr(agent, 'fetch_stock_many', lambda *a, **k: {product.handle: SimpleNamespace(in_stock=True, price_usd=8000)})
    assert agent._choose_tier_pick([(product.display_name, '', 999)], domain='osakiusa.com', prefer_compact=False, prefs={'budget_max_usd': '4000'}) is None


def test_unknown_or_insufficient_capacity_is_not_a_safe_match():
    products = recommend(RecommendationRequest(weight_lb=700, free_text='4D'))
    assert products == []
    reply = agent.respond('I am 6 ft 2, 700 lb, need 4D and lower back massage')
    assert reply.products == []
    assert '700 lb' in reply.reply
    assert reply.flow_stage == 'recommend_nofit'


def test_catalog_fallback_keeps_saved_doorway_after_short_answer(monkeypatch):
    monkeypatch.setattr(agent, 'cases_available', lambda *a, **k: False)
    prefs = {'awaiting_recommend': True, 'recommend_prefs': {
        'height': 'Average (5\'4"–5\'11")', 'height_in': '70',
        'weight': '181–220 lb', 'space': 'Narrow Doorway',
        'doorway_in': '25', 'doorway_fit': 'assembled',
        'mechanism': '4D', 'budget_max_usd': '7000',
    }}
    reply = agent.respond('lower back', prefs=prefs)
    assert reply.prefs_patch['recommend_prefs']['doorway_in'] == '25'
    assert all(p['fit']['doorway_assembled_in'] is not None and p['fit']['doorway_assembled_in'] <= 25 for p in reply.products)
    if not reply.products:
        assert reply.flow_stage == 'recommend_nofit'
        assert '25' in reply.reply


def test_old_tier_selection_cannot_bypass_corrected_budget():
    first = agent.respond('I am 5 ft 10 and need lower back massage')
    prefs = merge({}, first)
    prefs['recommend_prefs']['budget_max_usd'] = '1000'
    reply = agent.respond('', payload='tier:1', prefs=prefs)
    assert not reply.products
    assert not reply.prefs_patch['pending_tier_picks']
    assert '1,000' in reply.reply


def test_compare_recommendation_rejects_both_over_budget():
    left = resolve_product('Titan TP-Epic 4D')
    right = resolve_product('Osaki Duke XL 4D')
    prefs = {'pending_compare_pair': [left.handle, right.handle], 'recommend_prefs': {
        'height': 'Average (5\'4"–5\'11")', 'goal': 'Lower Back', 'budget_max_usd': '1000',
    }}
    reply = agent._compare_recommend_reply('which is better for me?', prefs=prefs)
    assert not reply.products
    assert '1,000' in reply.reply


def test_typo_never_silently_switches_to_cheaper_mechanism():
    reply = agent.respond('why is Pinnacle 5d duffle so expensive?', prefs={'pending_primary': 'Osaki Pinnacle 5D DuoFlex'})
    assert not any('LE 2D' in p['model'] for p in reply.products)
    assert '3,999' not in reply.reply
    if not reply.products:
        assert 'which model' in reply.reply.lower()


def test_three_models_clarify_instead_of_false_missing_catalog():
    reply = agent.respond('Compare Escape Duo, Pinnacle 5D and Vanguard')
    assert 'more than two' in reply.reply
    assert 'not on' not in reply.reply
    assert not reply.products


@pytest.mark.parametrize('message', ['I need a human', 'My remote is not working', 'Where is my order?'])
def test_showroom_wait_cannot_consume_task_interrupt_as_time(message):
    reply = agent.respond(message, prefs={'awaiting_showroom_window': True})
    assert not reply.prefs_patch.get('awaiting_showroom_window')
    assert not reply.prefs_patch.get('awaiting_showroom_email')
    assert reply.prefs_patch.get('pending_showroom_window') == ''
    assert 'best email' not in reply.reply.lower()


def test_menu_and_cancel_clear_showroom_capture():
    prefs = {'awaiting_showroom_window': True, 'awaiting_showroom_email': True}
    reply = agent.respond('', payload='menu', prefs=prefs)
    assert not reply.prefs_patch['awaiting_showroom_window']
    assert not reply.prefs_patch['awaiting_showroom_email']
    reply = agent.respond('cancel that', prefs=prefs)
    assert not reply.prefs_patch['awaiting_showroom_window']


def test_actual_day_time_still_advances_showroom_request():
    reply = agent.respond('next Friday at 3 pm', prefs={'awaiting_showroom_window': True})
    assert reply.prefs_patch['awaiting_showroom_email'] is True
    assert reply.prefs_patch['pending_showroom_window'] == 'next Friday at 3 pm'


def test_repeated_unresolved_question_stops_interview_loop():
    first = agent.respond('recommend a chair')
    prefs = merge({}, first)
    second = agent.respond('recommend a chair', prefs=prefs)
    assert second.handoff
    assert second.handoff_reason == 'repeated_unresolved'
    assert 'user height' not in second.reply.lower()


def test_intensity_settings_are_not_invented():
    reply = agent.respond('How strong is Titan TP-Epic 4D?')
    assert '3 of 5' not in reply.reply
    assert 'typically 5' not in reply.reply
    assert 'push deeper' not in reply.reply


def test_constraint_parser_correction_negation_and_decimal_budget():
    parsed = extract_constraints('actually 5 ft 8, 350 lb, budget $3.5k, 4D, not strong', {'weight_lb': '220'})
    assert parsed['height_in'] == '68'
    assert parsed['weight_lb'] == '350'
    assert parsed['budget_max_usd'] == '3500'
    assert parsed['intensity'] == 'Gentle'
    assert not extract_constraints('I am 5 ft 18').get('height_in')
    assert extract_constraints('I have a 30 inch doorway')['doorway_in'] == '30'
    assert extract_constraints('My budget is $3500, but this chair costs $8499')['budget_max_usd'] == '3500'
    assert extract_constraints('3D instead of 4D', {'mechanism': '4D'})['mechanism'] == '3D'
    assert not product_constraint_issues(resolve_product('Titan TP-Epic 4D'), {'doorway_in': 'skip'})
