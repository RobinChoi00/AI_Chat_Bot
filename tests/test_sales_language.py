"""Real customer copy, exact input facts, and clinical-scope regressions."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

import sales_agent as agent
from sales_language import is_medical_scope_question, prepare_message
from sales_constraints import extract_constraints
from tests.test_evaluation_sales_dialogues import isolated_sales_api


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM', '0')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(agent, 'fetch_live_stock', lambda *a, **kw: None)
    monkeypatch.setattr(agent, 'fetch_stock_many', lambda *a, **kw: {})


@pytest.mark.parametrize('message,required', [
    ('Will the Maestro 4D cure my sciatica if I use it every day?', 'clinician'),
    ('Will using a Maestro every evening get rid of my sciatica permanently?', 'clinician'),
    ('Can you guarantee pain relief after a month?', "can't promise"),
    ('I do not have a pacemaker, but I have recently had spinal surgery. Can I safely try it?', 'clinician'),
    ('¿La silla Maestro me va a curar la hernia de disco?', 'profesional de salud'),
    ('Necesito saber si será segura después de mi operación.', 'profesional de salud'),
    ('마사지 의자로 허리 디스크를 치료할 수 있나요?', '담당 의료진'),
])
def test_medical_claim_gets_a_direct_answer_without_sales_interview(message, required):
    result = agent.respond(message, prefs={'awaiting_recommend': True, 'recommend_prefs': {'budget_max_usd': '3000'}})
    assert required in result.reply
    assert not result.products
    assert not result.flow_stage.startswith('ask_')
    assert result.prefs_patch['awaiting_recommend'] is False
    assert 'your pacemaker' not in result.reply
    assert result.prefs_patch['medical_scope_pending'] is True


@pytest.mark.parametrize('message', [
    'I want a chair for lower back massage.',
    'How do I care for the leather upholstery?',
    'Is the shipping label correct?',
])
def test_ordinary_fit_and_product_questions_are_not_medical_questions(message):
    assert not is_medical_scope_question(message)


def _chat(client, sid, **body):
    response = client.post('/api/v1/sales/chat', json={'session_id': sid, 'domain': 'osakiusa.com', **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_original_spanish_dialogue_saves_exact_facts_and_human_request(isolated_sales_api):
    from sales_models import get_session_collected, SalesMessage
    from warranty_models import warranty_db_session

    client, sid = isolated_sales_api, 'synthetic-spanish-customer'
    assert client.post(f'/api/v1/warranty/session/{sid}/consent', json={'domain': 'osakiusa.com'}).status_code == 200
    original = 'Busco un sillón de masaje por menos de 4000 dólares.'
    first = _chat(client, sid, message=original)
    assert 'estatura' in first['reply']
    second = _chat(client, sid, message='Mido 1,80 metros y peso 90 kg.')
    assert 'zona' in second['reply']
    stored = get_session_collected(sid)
    assert stored['conversation_language'] == 'es'
    assert stored['recommend_prefs']['budget_max_usd'] == '4000'
    assert stored['recommend_prefs']['height_in'] == '71'
    assert float(stored['recommend_prefs']['weight_lb']) == pytest.approx(198.416, abs=0.01)
    third = _chat(client, sid, message='¿Puedo hablar con una persona?')
    assert third['handoff'] is True
    assert 'Todavía no se ha confirmado' in third['reply']
    assert '+1-888-848-2630' in third['reply']
    assert any(quick['label'] == 'Compartir contacto para seguimiento' for quick in third['quick_replies'])
    with warranty_db_session() as db:
        saved = db.query(SalesMessage).filter_by(session_id=sid, role='user').order_by(SalesMessage.id).all()
        assert saved[0].content == original
    assert get_session_collected(sid)['last_shopper_question'] == 'Mido 1,80 metros y peso 90 kg.'


def test_spanish_policy_detour_keeps_correction_and_recommendation_facts():
    prefs = {}
    for text in [
        'Quiero una silla por menos de 5000 dólares. Mido 178 cm y peso 85 kg.',
        'En realidad, mi tope es de 2800 dólares, no 5000. ¿El envío incluye montaje?',
        'Prefiero un masaje suave de cuerpo completo. Recomiéndame algo con ese límite.',
    ]:
        result = agent.respond(text, prefs=prefs)
        prefs.update(result.prefs_patch or {})
        if 'montaje' in text:
            assert 'no incluye montaje' in result.reply
            assert 'cargo adicional' in result.reply
            assert 'no confirman una fecha concreta' in result.reply
            assert prefs['recommend_prefs']['budget_max_usd'] == '2800'
    assert prefs['recommend_prefs']['height_in'] == '70'
    assert float(prefs['recommend_prefs']['weight_lb']) == pytest.approx(187.393, abs=0.01)
    assert result.products
    for card in result.products:
        assert card['model'] in result.reply
        assert f"${card['price_usd']:,.2f} USD" in result.reply
        assert card['price_usd'] <= 2800
    assert 'no reserva existencias' in result.reply


@pytest.mark.parametrize('text', [
    'No quiero hablar con una persona. Recomiéndame una silla.',
    'No necesito hablar con un asesor; solo quiero comparar modelos.',
])
def test_spanish_negated_human_requests_do_not_request_a_human(text):
    assert agent.respond(text).handoff is False


@pytest.mark.parametrize('text', [
    'Mi silla pesa 190 libras.',
    'El paquete pesa 190 libras.',
    'El sillón mide 180 cm.',
])
def test_spanish_product_facts_are_not_normalized_as_person_measurements(text):
    canonical, language = prepare_message(text, {})
    assert language == 'es'
    measured = extract_constraints(canonical)
    assert 'weight_lb' not in measured
    assert 'height_in' not in measured


@pytest.mark.parametrize('amount', ['4.000', '4,000'])
def test_spanish_explicit_usd_thousands_preserve_the_numeric_budget(amount):
    result = agent.respond(f'Busco una silla por menos de {amount} dólares.')
    assert result.prefs_patch['recommend_prefs']['budget_max_usd'] == '4000'


def test_foreign_currency_is_not_silently_treated_as_usd():
    result = agent.respond('Busco una silla con un presupuesto de 4000 euros.')
    assert 'tipo de cambio' in result.reply
    assert not result.products
    assert 'recommend_prefs' not in result.prefs_patch


def test_medical_human_request_does_not_imply_sales_can_give_clearance():
    prefs = {}
    medical = agent.respond('¿La silla puede curar mi ciática?', prefs=prefs)
    prefs.update(medical.prefs_patch or {})
    handoff = agent.respond('¿Puedo hablar con una persona?', prefs=prefs)
    assert handoff.handoff is True
    assert 'autorización médica' in handoff.reply


def test_contact_answers_and_buttons_do_not_overwrite_the_customer_question():
    prefs = {'last_shopper_question': '¿Qué sillón cabe por mi puerta de 74 cm?', 'conversation_language': 'es'}
    for kwargs in [{'message': 'buyer@example.invalid'}, {'message': '2'}, {'message': '', 'payload': 'human'}]:
        result = agent.respond(prefs=prefs, **kwargs)
        assert 'last_shopper_question' not in (result.prefs_patch or {})


def test_before_handoff_replay_does_not_write_a_canonical_question():
    original = 'Busco un sillón por menos de 4000 dólares.'
    result = agent.respond(original, prefs={'last_shopper_question': original}, before_handoff=True)
    assert 'last_shopper_question' not in (result.prefs_patch or {}) or result.prefs_patch['last_shopper_question'] == original


def test_spanish_labels_do_not_mutate_shared_english_quick_replies():
    spanish = agent.respond('Mido 1,80 metros y peso 90 kg.')
    assert any(q.label == 'Cuello y hombros' for q in spanish.quick_replies)
    english = agent.respond("I am 5'10 and 190 lb.")
    assert any(q.label == 'Neck & shoulders' for q in english.quick_replies)
    assert all(q.label != 'Cuello y hombros' for q in english.quick_replies)


def test_a_localized_unresolved_answer_does_not_claim_a_human_request_was_saved():
    result = agent.respond('Just answer my question.', prefs={'conversation_language': 'es'})
    assert result.handoff is True
    assert 'Se ha guardado su solicitud' not in result.reply
