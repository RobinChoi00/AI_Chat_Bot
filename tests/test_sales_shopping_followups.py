"""Shopping follow-ups answered from saved context and published catalog facts.

Conversations run through the real Sales API and repository catalog. Only
external boundaries (HTTP, email, live stock) are replaced.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))


@pytest.fixture
def chat(monkeypatch):
    import requests
    import smtplib
    import sales_agent
    import sales_router
    import warranty_models
    import warranty_router
    from sales_models import get_session_collected

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
        sid = f'followups-{uuid.uuid4().hex}'
        assert client.post(f'/api/v1/warranty/session/{sid}/consent', json={'domain': 'osakiusa.com'}).status_code == 200

        def turn(message):
            response = client.post('/api/v1/sales/chat', json={'session_id': sid, 'message': message, 'domain': 'osakiusa.com'})
            assert response.status_code == 200, response.text
            return response.json(), get_session_collected(sid)

        yield turn
    database.dispose()


def _capacity(name):
    from sales_spec_index import lookup_fit_spec
    return lookup_fit_spec(name).max_user_lb


@pytest.mark.parametrize('text', [
    'What assembled doorway width does Titan TP-Epic 4D require?',
    'Is the Osaki 4D Maestro LE 2.0 good for me?',
    'The OP-4D Ultima looks nice',
    'Osaki 4D + 3D Manhattan Duo price',
])
def test_model_name_mechanism_is_not_a_shopper_requirement(text):
    from sales_constraints import extract_constraints
    out = extract_constraints(text)
    assert not out.get('mechanism') and not out.get('allowed_mechanisms')


@pytest.mark.parametrize('text, mechanism', [
    ('I want a 4D chair under $5000', '4D'),
    ('not 4D, I want 3D', '3D'),
    ('I prefer 4D. Does the Titan TP-Epic 4D fit 30 inches?', '4D'),
])
def test_genuine_mechanism_preference_is_still_saved(text, mechanism):
    from sales_constraints import extract_constraints
    assert extract_constraints(text)['mechanism'] == mechanism


def test_doorway_with_assembled_is_assembled_only():
    from sales_constraints import extract_constraints
    assert extract_constraints('Our front door is 35 inches and it has to go in assembled.')['doorway_fit'] == 'assembled'
    assert extract_constraints('doorway 30 inches assembled, back focus')['doorway_fit'] == 'assembled'
    assert extract_constraints('doorway 30 inches, assembled or disassembled is fine').get('doorway_fit') != 'assembled'


def test_heavy_couple_gets_catalog_verified_chairs_and_named_capacity(chat):
    chat('My father is 6 foot 2 and weighs 320 pounds. My mother is 5 foot 2 and weighs 150 pounds. They will share one chair. Budget $12000.')
    chat('They both want gentle full body massage. Please recommend something suitable for both.')
    shortlist, prefs = chat('The doorway is 36 inches, assembled only. Please check the capacity of every recommendation.')
    assert shortlist['products'], shortlist['reply']
    assert 'none clears' not in shortlist['reply']
    for card in shortlist['products']:
        assert _capacity(card['model']) >= 320
        assert card['price_usd'] <= 12000

    listing, _ = chat('Which chairs actually have a published capacity of at least 320 pounds?')
    assert 'Grande XL' in listing['reply'] and 'Duke XL' in listing['reply']
    assert 'left out rather than guessed' in listing['reply']

    named, _ = chat('Does the Grande XL support a 320 pound user? Please check the published maximum weight.')
    assert f"{_capacity('Grande XL'):g} lb maximum user weight" in named['reply']
    assert '✅' in named['reply'] and 'How tall' not in named['reply']

    height, _ = chat("If both users' height fit is not published, please tell me it needs confirmation from Sales.")
    assert height['intent'] != 'unclear'
    assert "father (6'2\")" in height['reply'] and "mother (5'2\")" in height['reply']
    assert 'needs to confirm height fit' in height['reply']


def test_owner_label_question_gets_label_guidance_not_the_menu(chat):
    label, _ = chat('Where should I look for the serial or model label?')
    assert label['intent'] == 'specs' and 'serial-number sticker on the chair base' in label['reply']
    assert 'Do not lift, tip, or open' in label['reply'] and 'service@osakititan.com' in label['reply']
    showroom, _ = chat('Where can I see the Duke XL model in person?')
    assert 'serial' not in showroom['reply']


def test_showroom_visit_window_is_acknowledged_but_not_booked(chat):
    chat("I'd like to visit the showroom.")
    visit, _ = chat('Saturday afternoon is my preferred visit window.')
    assert 'Noted: your preferred visit window is **Saturday afternoon**' in visit['reply']
    assert 'not booked yet' in visit['reply'] and '888-501-5988' in visit['reply']
    sat, _ = chat('I sat in one at the showroom and want to visit again.')
    assert 'Noted:' not in sat['reply']


def test_impossible_hard_limits_answer_before_more_interview_questions(chat):
    reply, prefs = chat("I'm 5'8 and 200 lb, budget $1500, my doorway is 24 inches assembled. Recommend one.")
    assert reply['reply'].startswith('I checked every active chair, and none clears all of these published limits')
    assert 'main focus' not in reply['reply'] and not reply['products']
    assert prefs['awaiting_recommend'] is False
    unknown_fit, _ = chat("I'm 5'8 and 200 lb, budget $1500, my doorway is 22 inches. Recommend one.")
    assert unknown_fit['reply'].startswith('I checked every active chair')


def test_feasible_hard_limits_still_ask_the_main_focus(chat):
    reply, _ = chat("I'm 5'8 and 200 lb, budget $4000, my doorway is 30 inches assembled. Recommend one.")
    assert 'main focus' in reply['reply']


def test_assumed_strong_pressure_tells_the_customer_how_to_change_it(chat):
    reply, _ = chat("I'm 5'9, 170 lb, budget $5000, back focus. Recommend a chair.")
    assert 'strong pressure is only a starting guess' in reply['reply']
    gentle, prefs = chat('gentle')
    assert prefs['recommend_prefs']['intensity'] == 'Gentle'
    assert 'starting guess' not in gentle['reply']


@pytest.mark.parametrize('title, spec_name', [
    ('Titan TI Reflex 3D', 'Titan TI Reflex'),
    ('Osaki Solo Flex 4D', 'Osaki Solo Flex'),
    ('Osaki OS-4000XT', 'Osaki 4000XT'),
    ('Osaki 3D 4000LE', 'Osaki 4000LE'),
    ('Osaki OS-3D Champ II', 'Osaki 3D Champ 2'),
    ('Osaki OS-Pro 4D+4D DuoMax SE', 'Osaki OS-Pro 4D+3D DuoMax SE'),
    ('Zena Vending Chair', 'Titan Zena'),
    ('Titan V03 Vending Massage chair', 'Titan V03'),
    ('Osaki Aeris 4D Duo', 'Osaki Aeris Duo'),
    ('Osaki OS-Champ', 'Osaki OS-Champ'),
])
def test_shop_titles_resolve_to_the_most_specific_published_spec(title, spec_name):
    from sales_spec_index import lookup_fit_spec
    assert lookup_fit_spec(title).name == spec_name


def test_box_or_chair_weight_is_not_the_users_weight():
    from sales_followups import _weights_from_message
    assert _weights_from_message('The shipping box weighs only 210 pounds. Which chairs support 300 lb?') == [300.0]
    assert _weights_from_message('Does the Grande XL support a 320 pound user?') == [320.0]


def test_no_fit_names_closest_chairs_with_real_limit(chat):
    from sales_agent import _no_fit_recommend_reply
    reply = _no_fit_recommend_reply({'budget_max_usd': '1500', 'weight_lb': '200', 'doorway_in': '24', 'doorway_fit': 'assembled'})
    assert '** **' not in reply.reply and '****' not in reply.reply
    assert 'I checked every active chair' in reply.reply


def test_shortlist_feature_and_single_pick_use_every_saved_requirement(chat):
    chat("I'm buying a massage chair for my father, not for myself. He is 5 foot 8 and weighs 195 pounds. Budget is $5000 and he likes gentle shoulder massage. Recommend a chair.")
    _, prefs = chat('The doorway in his home is 32 inches wide. Assembled only.')
    picks = prefs['pending_tier_picks']
    assert len(picks) >= 2
    features, _ = chat('Which of your options has foot rollers as well?')
    for pick in picks:
        assert pick['display'] in features['reply']

    one, _ = chat("Please show one suitable option using my father's requirements, or say which fit information still needs confirmation.")
    assert one['intent'] != 'unclear'
    assert "your father's requirements" in one['reply']
    assert 'Still needs confirmation' in one['reply'] and 'Height fit for your father' in one['reply']
    assert len(one['products']) == 1 and one['products'][0]['price_usd'] <= 5000


def test_pair_followups_give_direct_verdicts_and_keep_the_pair(chat):
    chat('I want to compare Osaki OS-Champ and Titan TP-Epic 4D.')
    rollers, _ = chat('Which of the two has more foot rollers?')
    assert rollers['reply'].startswith('**Titan TP-Epic 4D** has more foot rollers')
    door, _ = chat('Which needs the smaller assembled doorway?')
    assert door['reply'].startswith('**Titan TP-Epic 4D** needs the smaller assembled doorway')
    keep, prefs = chat('Please keep the same two models rather than choosing another chair.')
    assert keep['intent'] == 'compare' and "won't swap in another chair" in keep['reply']
    assert prefs['pending_compare_pair'] == ['osaki-os-champ', 'titan-tp-epic-4d']
    weight, _ = chat('I weigh 250 pounds. Which of these two has a published capacity that supports that weight?')
    assert weight['reply'].startswith('Both publish a capacity that supports **250 lb**')
    heavier, prefs = chat("I'm 262 lb. Which of the two supports me?")
    assert heavier['intent'] == 'compare'
    assert heavier['reply'].startswith('Only **Titan TP-Epic 4D** publishes a capacity that supports **262 lb**')
    assert float(prefs['recommend_prefs']['weight_lb']) == 262
    table, _ = chat('Please put their prices and assembled doorway widths together before I decide.')
    assert table['intent'] == 'compare' and 'height' not in table['reply'].lower()
    assert '- **Price**: $1,299 vs $3,999' in table['reply']
    assert {card['model'] for card in table['products']} == {'Osaki OS-Champ', 'Titan TP-Epic 4D'}


def test_general_concepts_are_explained_and_contrasted(chat):
    general, _ = chat('I mean the general term, not a particular model. What is the reclined position for?')
    assert general['intent'] != 'unclear' and 'recline' in general['reply']
    contrast, prefs = chat('Is zero gravity the same thing as an SL-Track, or are those different features?')
    assert contrast['reply'].startswith('No, they are different features.')
    assert prefs['last_concepts'][:2] == ['zero_gravity', 'track']
    followup, _ = chat('Please explain the difference, or clearly say if you need a Sales specialist to confirm it.')
    assert 'positions you' in followup['reply'] and "couldn't match" not in followup['reply']
    named, _ = chat('Does the Osaki Hypnos 4D Pro AI have zero gravity?')
    assert 'Hypnos' in named['reply'] and 'general meaning' not in named['reply']


def test_repair_and_new_purchase_stay_separate(chat):
    first, prefs = chat('I own a Titan TP-Epic 4D and its foot rollers stopped working.')
    assert first['intent'] == 'warranty_redirect'
    assert prefs['owner_service_case'] == {'model': 'Titan TP-Epic 4D', 'issue': 'foot rollers stopped working'}
    support, _ = chat('This chair is already in my house. It is broken. I need technical support, not a new chair.')
    assert 'two separate requests' not in support['reply']
    note, _ = chat('Separately, I also want to buy a second chair for my office. That is a new purchase.')
    assert note['reply'].startswith('Your **Titan TP-Epic 4D** repair stays with Warranty')
    chat('Recommend a new office chair under $3000. I am 5 foot 8, 170 pounds and prefer gentle back massage.')
    chat('Does the Osaki OS-Champ have foot rollers? This question is about the new purchase.')
    dual, _ = chat('Please route the old Epic repair to Warranty and the new-purchase question to Sales.')
    assert '**1. Repair — Titan TP-Epic 4D (foot rollers stopped working) → Warranty Department**' in dual['reply']
    assert '**2. New purchase — Osaki OS-Champ → Sales**' in dual['reply']
    assert 'service@osakititan.com' in dual['reply'] and 'ext. 2' in dual['reply']


@pytest.mark.parametrize('message', [
    'Does the warranty cover the new chair I buy?',
    'Can I buy an extended warranty with a new chair?',
])
def test_warranty_policy_question_is_not_a_dual_task(chat, message):
    reply, _ = chat(message)
    assert 'two separate requests' not in reply['reply']


def test_named_model_follow_ups_stay_on_that_model(chat):
    chat('Tell me about the Titan TP-Epic 4D.')
    keep, prefs = chat('I am 5 foot 7 and 165 pounds, but please keep discussing the same Titan TP-Epic 4D.')
    assert keep['reply'].startswith("I'll stay on **Titan TP-Epic 4D**")
    assert float(prefs['recommend_prefs']['weight_lb']) == 165
    assert not prefs['recommend_prefs'].get('mechanism')
    zone, _ = chat('Can I choose different massage strength for my calves than for my back? If not verified, please say so.')
    assert zone['reply'].startswith('Not verified')
    adjustable, _ = chat('Does Titan TP-Epic 4D have adjustable massage intensity?')
    assert adjustable['reply'].startswith('Adjustable intensity is not listed')
    width, _ = chat('Is the Osaki 4D Maestro LE 2.0 narrower than 30 inches assembled?')
    assert 'wider than 30"' in width['reply']


def test_korean_shopping_copy_and_input(chat):
    first, prefs = chat('안녕하세요, 한국어로 상담 부탁드려요. 허리 마사지 의자를 찾고 있어요.')
    assert first['intent'] == 'recommend' and '키' in first['reply']
    second, prefs = chat('키는 170cm이고 몸무게는 70kg입니다.')
    assert prefs['conversation_language'] == 'ko'
    assert float(prefs['recommend_prefs']['height_in']) == pytest.approx(67, abs=1)
    assert any('가' <= ch <= '힣' for ch in second['reply'])


def test_korean_handoff_and_white_glove_are_localized(chat):
    human, _ = chat('한국어로 상담하고 싶어요. 담당자에게 연결해 주세요.')
    assert '상담원' in human['reply'] and '+1-888-848-2630' in human['reply']
    shipping, _ = chat('한국어로 알려 주세요. 화이트 글러브 배송에 조립이 포함되나요?')
    assert '조립' in shipping['reply'] and 'White Glove' in shipping['reply']
