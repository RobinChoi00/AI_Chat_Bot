"""Persisted Sales follow-ups using the real API and repository catalog.

Only unavailable external boundaries are replaced. No assistant replies or
catalog facts are mocked, and these reviewed cases are regression checks.
"""
from __future__ import annotations

import uuid
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))


@pytest.fixture
def isolated_sales_api(monkeypatch):
    import requests
    import smtplib
    import sales_agent
    import sales_router
    import warranty_models
    import warranty_router
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.setenv('WARRANTY_REQUIRE_CHAT_PRIVACY', '1')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(requests.sessions.Session, 'request', lambda *a, **k: pytest.fail('No external requests'))
    monkeypatch.setattr(smtplib, 'SMTP', lambda *a, **k: pytest.fail('No actual email'))
    monkeypatch.setattr(smtplib, 'SMTP_SSL', lambda *a, **k: pytest.fail('No actual email'))
    monkeypatch.setattr(sales_agent, 'fetch_live_stock', lambda *a, **k: None)
    monkeypatch.setattr(sales_agent, 'fetch_stock_many', lambda *a, **k: {})
    database = create_engine('sqlite:///:memory:', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    warranty_models.Base.metadata.create_all(bind=database)
    monkeypatch.setattr(warranty_models, '_engine', database)
    monkeypatch.setattr(warranty_models, '_SessionFactory', sessionmaker(bind=database, expire_on_commit=False))
    warranty_router.limiter.reset()
    app = FastAPI()
    app.include_router(sales_router.router)
    app.include_router(warranty_router.router)
    with TestClient(app) as client:
        yield client
    database.dispose()


@pytest.fixture
def chat(isolated_sales_api):
    from sales_models import get_session_collected
    sid = f'context-{uuid.uuid4().hex}'
    consent = isolated_sales_api.post(f'/api/v1/warranty/session/{sid}/consent', json={'domain': 'osakiusa.com'})
    assert consent.status_code == 200

    def turn(message, *, payload=None):
        response = isolated_sales_api.post('/api/v1/sales/chat', json={
            'session_id': sid, 'message': message, 'domain': 'osakiusa.com',
            **({'payload': payload} if payload else {}),
        })
        assert response.status_code == 200, response.text
        return response.json(), get_session_collected(sid)
    return turn


@pytest.mark.parametrize('opening', [
    'Can you put Maesto 4D next to Paragun?',
    'Please show Maestro 4D alongside Paragon.',
    'Could you place Maestro 4D beside Paragon?',
    'Compare the Maestro 4D with Paragon. My maximum is $7000.',
])
def test_natural_pair_and_relative_price_use_catalog_without_interview(chat, opening):
    from sales_catalog import resolve_product
    first, prefs = chat(opening)
    assert first['intent'] == 'compare'
    assert len(first['products']) == 2
    second, prefs = chat('Between those two, which one is cheaper?')
    assert second['intent'] == 'compare'
    assert 'height' not in second['reply'].lower()
    assert 'Paragon' in second['reply'] and 'Maestro' in second['reply']
    cheaper = resolve_product('Paragon')
    assert f'**{cheaper.display_name}** has the lower published price' in second['reply']
    assert f'${cheaper.price_usd:,.0f}' in second['reply']
    if 'maximum' in opening:
        assert float(prefs['recommend_prefs']['budget_max_usd']) == 7000
        assert all(card['price_usd'] <= 7000 for card in second['products'])


def test_corrected_pair_and_objective_doorway_have_a_positive_no_fit_answer(chat):
    chat('Can you put Maesto 4D next to Paragun?')
    corrected, prefs = chat('I mean Maestro 4D and Paragon, not 3D.')
    assert corrected['intent'] == 'compare'
    assert len(corrected['products']) == 2
    assert all(card['specs']['massage_mechanism'] == '4D' for card in corrected['products'])
    final, prefs = chat('Which is less money and fits a 31-inch doorway?')
    assert final['intent'] == 'compare'
    assert final['products'] == []
    assert 'Neither of these two passes' in final['reply']
    assert '31.9 in' in final['reply'] and '33 in' in final['reply']
    assert '31 in doorway' in final['reply']
    assert float(prefs['recommend_prefs']['doorway_in']) == 31
    assert len(prefs['pending_compare_pair']) == 2
    assert 'mechanism' not in prefs['recommend_prefs']


def test_pair_budget_survives_assembled_only_doorway_and_correction(chat):
    chat('Compare the Maestro 4D with Paragon. My maximum is $7000.')
    final, prefs = chat('Which of those can pass a 31-inch doorway fully assembled? Stay under that same budget.')
    assert final['intent'] == 'compare'
    assert final['products'] == []
    assert 'Neither of these two passes' in final['reply']
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    assert float(prefs['recommend_prefs']['budget_max_usd']) == 7000
    revised, prefs = chat('Which of these fits my 34-inch doorway instead? No disassembly.')
    assert revised['intent'] == 'compare'
    assert len(revised['products']) == 1
    assert 'Paragon' in revised['products'][0]['model']
    assert float(prefs['recommend_prefs']['doorway_in']) == 34
    assert float(prefs['recommend_prefs']['budget_max_usd']) == 7000


@pytest.mark.parametrize('replacement', [
    'Keep the Paragon, but replace Maestro with Grande XL in that comparison.',
    'Swap Maestro for Grande XL.',
    'Change the Maestro to Grande XL.',
])
def test_replace_only_named_side_of_saved_pair(chat, replacement):
    from sales_compare import product_by_handle
    chat('Compare Maestro 4D versus Paragon.')
    final, prefs = chat(replacement)
    assert final['intent'] == 'compare'
    names = [product_by_handle(handle).display_name for handle in prefs['pending_compare_pair']]
    assert any('Paragon' in name for name in names)
    assert any('Grande' in name for name in names)
    assert not any('Maestro' in name for name in names)
    assert len(final['products']) == 2


def test_negated_replacement_retains_original_pair(chat):
    chat('Compare Maestro 4D vs Paragon.')
    _, before = chat('Which is cheaper?')
    final, after = chat('Do not replace Maestro with Grande XL. Which of those two is cheaper?')
    assert final['intent'] == 'compare'
    assert 'Maestro' in final['reply'] and 'Paragon' in final['reply']
    assert 'Grande' not in final['reply']
    assert after['pending_compare_pair'] == before['pending_compare_pair']


@pytest.mark.parametrize('followup', ['What about the Paragon?', 'How about Paragon?', 'And the Paragon?'])
def test_second_model_inherits_price_task_and_keeps_both_referents(chat, followup):
    chat('What does a Maestro 4D cost?')
    second, prefs = chat(followup)
    assert second['intent'] == 'price'
    assert len(second['products']) == 1 and 'Paragon' in second['products'][0]['model']
    final, prefs = chat('Which of these two is less expensive?')
    assert final['intent'] == 'compare'
    assert 'Maestro' in final['reply'] and 'Paragon' in final['reply']
    assert 'lower published price' in final['reply']
    assert len(prefs['pending_compare_pair']) == 2


@pytest.mark.parametrize('followup', [
    'Is it actually in the warehouse, or are you quoting the catalog?',
    'Are you reading actual warehouse stock, or just a product listing?',
    'Is that live inventory or only the catalog?',
])
def test_warehouse_question_keeps_model_color_and_truthful_unavailable_boundary(chat, followup):
    chat('Can I still buy a Grande XL in ivory?')
    final, prefs = chat(followup)
    assert final['intent'] == 'stock'
    assert 'Grande' in final['reply'] and 'Ivory' in final['reply']
    assert 'cannot verify a physical warehouse' in final['reply']
    assert "couldn't reach live inventory" in final['reply']
    assert 'not among the published options' in final['reply']
    assert 'shopify.inventory' not in final['tools_used']
    assert 'height' not in final['reply'].lower()


def test_warehouse_reference_to_pair_requests_model_instead_of_stale_focus(chat):
    chat('What does Maestro 4D cost?')
    chat('Compare Maestro 4D vs Paragon.')
    final, _ = chat('Is it in the warehouse?')
    assert final['intent'] == 'stock'
    assert 'Which of the two models' in final['reply']
    assert final['products'] == []


@pytest.mark.parametrize('question,facet', [
    ('Which of those two has foot rollers?', 'Foot roller'),
    ('Do both have heating?', 'Heating'),
    ('Which of those has airbags?', 'Airbags'),
    ('What features do these two have?', 'Auto programs'),
])
def test_objective_comparison_answers_requested_catalog_facet(chat, question, facet):
    from sales_catalog import resolve_product
    chat('Compare Maestro 4D vs Paragon.')
    final, _ = chat(question)
    assert final['intent'] == 'compare'
    assert len(final['products']) == 2
    assert facet in final['reply']
    if facet == 'Foot roller':
        published = resolve_product('Maestro 4D').as_public_dict()['specs']['foot_roller']
        assert published in final['reply']
        assert '- Foot roller: Yes.' in final['reply']


def test_direct_fit_pronoun_and_saved_constraints_recap(chat):
    chat('Give me the verified supported user-height range for Maestro 4D.')
    fit, prefs = chat('Will it fit through a 28.5-inch doorway without taking anything apart?')
    assert fit['intent'] == 'specs'
    assert 'Maestro' in fit['reply'] and '31.9 in' in fit['reply']
    assert 'wider than the doorway' in fit['reply']
    assert 'excluded disassembly' in fit['reply']
    assert 'height' not in fit['reply'].lower()
    recap, prefs = chat('Can you confirm which constraints you are applying?')
    assert '28.5 in' in recap['reply'] and 'assembled' in recap['reply']
    assert float(prefs['recommend_prefs']['doorway_in']) == 28.5
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    repeated, _ = chat('Will it fit through my doorway?')
    assert repeated['intent'] == 'specs'
    assert '28.5' in repeated['reply'] and '31.9' in repeated['reply']
    assert 'wider than the doorway' in repeated['reply']
    unknown, _ = chat('Will the Quantum Deluxe fit through my doorway?')
    assert unknown['products'] == []
    assert 'Which exact model' in unknown['reply']


def test_height_fit_pressure_gets_explicit_unknown_evidence(chat):
    chat('Give me the verified supported user-height range for Maestro 4D.')
    final, prefs = chat('I am 6 foot 8. Please just say it definitely fits me.')
    assert final['intent'] == 'specs'
    assert '80 in' in final['reply']
    assert 'cannot confirm personal height fit' in final['reply']
    assert 'not verified' in final['reply'] and 'specialist' in final['reply']
    assert float(prefs['recommend_prefs']['height_in']) == 80


def test_personal_fit_question_after_pair_keeps_both_unknown_ranges(chat):
    chat('Compare Maestro 4D vs Paragon.')
    final, _ = chat('Will it fit me? I am six feet four.')
    assert final['intent'] == 'specs'
    assert 'Maestro' in final['reply'] and 'Paragon' in final['reply']
    assert 'cannot confirm personal height fit' in final['reply']


@pytest.mark.parametrize('promise', [
    'Can you promise it will reach Dallas by next Thursday?',
    'Can you guarantee it will get to Boston before Friday?',
    'Could you promise it will be here by this weekend?',
])
def test_named_stock_followup_gets_delivery_limitation_without_promise(chat, promise):
    chat('Is Paragon available in black?')
    final, _ = chat(promise)
    assert final['intent'] == 'eta_shipping' and final['handoff']
    assert "can't" in final['reply'] and 'delivery date' in final['reply']


def test_product_weight_does_not_override_buyer_or_hide_actual_capacity_question(chat):
    chat('I am 6 feet tall and weigh 320 lb. My maximum is $5000. I want gentle full-body massage.')
    final, prefs = chat('The shipping box weighs only 210 pounds according to the listing. Which option can support me?')
    assert final['intent'] == 'recommend'
    assert '320 lb user weight' in final['reply']
    assert 'Grande' in final['reply'] and '375 lb maximum user capacity' in final['reply']
    assert len(final['products']) >= 1
    assert float(prefs['recommend_prefs']['weight_lb']) == 320
    assert float(prefs['recommend_prefs']['budget_max_usd']) == 5000
    chat('Compare Maestro 4D vs Paragon.')
    pair, prefs = chat('The chair is 190 pounds. Which of those two is cheaper?')
    assert pair['intent'] == 'compare'
    assert pair['products'] == []
    assert 'user-weight capacity' in pair['reply']
    assert float(prefs['recommend_prefs']['weight_lb']) == 320


@pytest.mark.parametrize('phrase', [
    'We are going in circles. I would rather speak with a member of your sales team.',
    'I prefer to chat with your sales staff.',
    'Could I talk to a member of the customer service team?',
])
def test_natural_staff_request_interrupts_interview(chat, phrase):
    chat('Help me find a chair for my narrow doorway.')
    final, _ = chat(phrase)
    assert final['intent'] == 'human' and final['handoff']
    assert 'has not yet confirmed' in final['reply']


def test_negated_staff_request_does_not_handoff(chat):
    final, _ = chat('Do not speak with a member of your sales team. Just show me the Paragon price.')
    assert final['intent'] == 'price' and not final['handoff']


@pytest.mark.parametrize('fact', [
    "My own door is still 30 inches wide; the Paragon's published doorway width is 33 inches. Which can fit my door?",
    "The Paragon's published doorway width is 33 inches. Which of those fits my door?",
    'The chair requires a 33-inch doorway according to its listing. Which one fits my 30-inch doorway?',
])
def test_product_doorway_fact_never_relaxes_saved_home_doorway(chat, fact):
    chat('Compare Maestro 4D versus Paragon.')
    chat('My doorway is 30 inches wide, assembled only. Which one fits?')
    final, prefs = chat(fact)
    assert float(prefs['recommend_prefs']['doorway_in']) == 30
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    assert final['intent'] == 'compare'
    assert final['products'] == []
    assert 'Neither of these two passes' in final['reply']
    assert '30 in doorway' in final['reply']


@pytest.mark.parametrize('fact', [
    "Paragon's listed width is 33 inches. Will it fit?",
    'The chair is listed as 33 inches wide. Will it fit?',
])
def test_incidental_product_width_cannot_bypass_filtered_doorway_parser(chat, fact):
    chat('What is the Paragon price?')
    chat('My doorway is 30 inches wide, assembled only. Will it fit my doorway?')
    final, prefs = chat(fact)
    assert float(prefs['recommend_prefs']['doorway_in']) == 30
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    assert 'wider than the doorway' in final['reply']
    assert 'figure fits the width' not in final['reply']


def test_product_width_cannot_reenter_through_recommendation_preference_merging(chat):
    chat('I am five feet nine and weigh 180 lb. My budget is $5000. I want gentle full-body relaxation. My doorway is 30 inches wide, assembled only.')
    final, prefs = chat('The chair is listed as 33 inches wide. Recommend something using my saved limits.')
    assert float(prefs['recommend_prefs']['doorway_in']) == 30
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    assert all(card['fit']['doorway_assembled_in'] is not None and card['fit']['doorway_assembled_in'] <= 30 for card in final['products'])


def test_product_disassembly_capability_does_not_grant_customer_permission(chat):
    chat('Compare Maestro 4D versus Paragon.')
    chat('My doorway is 30 inches wide, assembled only. Which one fits?')
    final, prefs = chat('The Paragon can be disassembled. Which one fits my doorway?')
    assert prefs['recommend_prefs']['doorway_fit'] == 'assembled'
    assert float(prefs['recommend_prefs']['doorway_in']) == 30
    assert final['products'] == []
    assert 'Neither of these two passes' in final['reply']
    allowed, prefs = chat('I am okay with disassembly. Which of these fits my doorway?')
    assert prefs['recommend_prefs']['doorway_fit'] == 'disassembled'
    assert len(allowed['products']) == 2


def test_negated_replacement_and_price_gap_keep_pair_and_answer_difference(chat):
    chat('Compare Maestro 4D versus Paragon.')
    final, prefs = chat('Do not replace Paragon with Grande XL. Keep comparing the original pair and show that price gap again.')
    assert final['intent'] == 'compare' and not final['handoff']
    assert len(final['products']) == 2
    assert 'published price gap is **$6,000**' in final['reply']
    assert not any('Grande' in card['model'] for card in final['products'])


@pytest.mark.parametrize('utterance', [
    'I do not want the cheaper one. Which is the more expensive model?',
    'Which of those two costs more? Show me the most expensive one.',
    "Don't pick the least expensive chair. Which of those is pricier?",
])
def test_explicit_higher_price_comparison_respects_polarity(chat, utterance):
    chat('Compare Maestro 4D versus Paragon.')
    final, _ = chat(utterance)
    assert final['intent'] == 'compare' and not final['handoff']
    assert '**Osaki OS-Pro Maestro 4D** has the higher published price' in final['reply']
    assert '$8,999' in final['reply']
    assert 'has the lower published price' not in final['reply']


@pytest.mark.parametrize('utterance', [
    'Can you give me a discount on the more expensive model?',
    'Can I get the Paragon cheaper than on Amazon?',
    'Show that price gap and offer me a coupon.',
])
def test_actual_discount_or_competitor_request_retains_specialist_path(chat, utterance):
    chat('Compare Maestro 4D versus Paragon.')
    final, _ = chat(utterance)
    assert final['intent'] == 'discount'
    assert 'specialist' in final['reply']
    assert any(q['payload'] == 'human:confirm' for q in final['quick_replies'])


@pytest.mark.parametrize('question', [
    'Which chair is heavier to move? I mean the weight of the chair itself, not the person sitting in it.',
    "Which of those has the lower chair's own weight?",
    'Which one weighs more? I mean the chair itself, not user capacity.',
])
def test_chair_mass_question_answers_unknown_instead_of_user_capacity(chat, question):
    chat('Compare Maestro 4D versus Paragon.')
    final, _ = chat(question)
    assert final['intent'] == 'compare'
    assert "chair's own weight is not verified" in final['reply']
    assert 'cannot determine which chair is heavier' in final['reply']
    assert "it is not the chair's mass" in final['reply']
    assert '- Chair weight: not verified.' in final['reply']
    assert 'Maximum user weight: 260 lb' not in final['reply']
