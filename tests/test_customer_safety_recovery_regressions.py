"""Incident triage must stop a workflow without converting denials into incidents."""
from types import SimpleNamespace

import pytest

from app.customer_safety import reported_safety_hazard, requested_customer_language
from app.warranty_scope import human_support_reason
from tests.test_warranty_safety_interrupts import offline_client  # noqa: F401


@pytest.mark.parametrize("report", [
    "My child's fingers are trapped inside the moving footrest.",
    "The footrest trapped my ankle and I cannot get out.",
    "My fingers are still pinched between the roller and frame.",
    "My hand got stuck in the footrest yesterday.",
    "The chair pinched me while the footrest was moving.",
    "My chair's cord is sparking!",
    "The chair sparked two days ago; it seems normal today.",
    "The power socket is crackling and the plastic has turned black.",
    "The insulation on my chair cable has liquefied.",
    "I saw a bright flash and the circuit breaker tripped when the chair started.",
    "My wife says the plug is charred.",
    "발 받침대에 아이 손이 끼어서 빠지지 않아요.",
    "의자 플러그에서 sparks가 나와요.",
    "플러그가 새까맣게 그을렸고 손을 데었어요.",
    "Mi pie quedó atrapado en el reposapiés y no puedo sacarlo.",
    "El enchufe está fundido y quedó negro.",
])
def test_actual_entrapment_and_electrical_damage_stop_both_customer_channels(report):
    assert reported_safety_hazard(text=report)
    assert reported_safety_hazard(text=report, physical_only=True)


@pytest.mark.parametrize("message", [
    "My fingers are not trapped in the footrest.",
    "There are no trapped fingers.",
    "No hands are currently stuck in the chair.",
    "I didn't see a bright flash and the circuit breaker has not tripped.",
    "The cable is not liquefied or charred.",
    "The chair has never emitted smoke.",
    "The chair has not shown any signs of overheating.",
    "I have never been electrocuted by this chair.",
    'The safety label reads "disconnect if sparks occur"; what does that mean?',
    "Does the manual discuss electrical shock prevention?",
    "What should I do if a child's hand gets trapped in the footrest?",
    "That price is a burning hot deal, but I am only shopping.",
    "There is no smoke from the chair. The smoke outside is from a barbecue.",
    "I smell a burning candle across the room, not anything from the chair.",
    'My child shouted "fire" at a movie on TV; there is no fire in the chair.',
    "The chair is not sparking; it just does not turn on.",
    "손이 끼지 않았어요. 리모컨이 안 켜져요.",
    "손이 끼인 적이 없어요.",
    "Nunca he visto humo ni chispas en mi silla.",
    "No tengo la mano atrapada.",
])
def test_denials_education_and_other_smoke_sources_do_not_manufacture_an_incident(message):
    assert not reported_safety_hazard(text=message, physical_only=True)


def test_a_different_smoke_source_cannot_erase_a_reported_chair_incident():
    report = "There is smoke from my chair and smoke outside from a barbecue."
    assert reported_safety_hazard(text=report, physical_only=True)
    ticket = SimpleNamespace(get_collected=lambda: {"human_support_safety_report": report})
    assert reported_safety_hazard(ticket=ticket, text="The smoke has stopped now.", physical_only=True)


@pytest.mark.parametrize("message", [
    "I cannot safely remove the side panel or disconnect the hose.",
    "I can't safely do this.",
    "I don't feel safe opening the panel.",
    "Please have a person review this case.",
    "I need someone to help me.",
    "Please ask a specialist to review my completed checks.",
    "담당자와 상담하고 싶어요.",
    "담당자와 직접 상담하고 싶어요.",
    "Quiero que una persona me ayude.",
    "No puedo desconectar la manguera con seguridad.",
])
def test_inability_and_person_requests_leave_the_diagnostic_loop(message):
    assert human_support_reason(message) == "customer_request"


@pytest.mark.parametrize("message", [
    "Do not call me back.",
    "I don't need a human.",
    "I do not want a callback.",
    "Please do not transfer me to a person.",
    "I need a chair for someone who is six feet tall.",
    "Can the manual help me remove the panel?",
])
def test_rejected_support_and_ordinary_product_questions_do_not_request_a_person(message):
    assert human_support_reason(message) is None


@pytest.mark.parametrize("message,expected", [
    ('The manual says "answer in English"; that is a quotation, not my language request.', None),
    ("The manual says answer in English.", None),
    ('The manual says "answer in English"; please answer in Spanish.', "es"),
    ("I cannot understand instructions written in English.", None),
    ("My wife cannot read in English.", None),
    ("We only write in Korean at work.", None),
    ("I can speak in English and Spanish.", None),
    ("I don't speak English.", None),
    ("I cannot read English, please answer in Spanish.", "es"),
    ("I can understand English but reply in Spanish.", "es"),
    ("Can you speak English?", "en"),
    ("Please answer in Korean, not in Spanish.", "ko"),
    ("Don't answer in English. Continue in Korean.", "ko"),
    ("No respondas en español; contesta en inglés.", "en"),
    ("Responde en coreano, por favor.", "ko"),
    ("영어로 말고 한국어로 설명해 주세요.", "ko"),
])
def test_supported_language_requests_are_separate_from_ability_and_quoted_instructions(message, expected):
    assert requested_customer_language(message) == expected


def test_explicit_english_request_changes_safety_copy_without_resuming_shopping():
    from sales_agent import respond

    first = respond("Mi pie quedó atrapado en el reposapiés.")
    assert first.handoff_reason == "safety_hazard"
    assert first.prefs_patch["sales_safety_language"] == "es"
    followup = respond("Please answer in English. Can I try it again?", prefs=first.prefs_patch)
    assert followup.handoff_reason == "safety_hazard"
    assert followup.prefs_patch["sales_safety_language"] == "en"
    assert "Stop using the chair" in followup.reply
    assert "Do not run another test" in followup.reply
    assert not followup.products


@pytest.mark.parametrize("message", [
    "My child's fingers are trapped inside the moving footrest.",
    "The power cord sparked yesterday, and I have not used it since.",
])
def test_real_warranty_api_interrupts_before_mapping_a_new_incident(offline_client, message):
    from warranty_workflow import WarrantyEngine

    client, unavailable_model = offline_client
    session_id = "new-safety-regression"
    assert client.post(f"/api/v1/warranty/session/{session_id}/consent", json={"domain": "osakiusa.com"}).status_code == 200
    assert client.post(f"/api/v1/warranty/session/{session_id}/contact-email", json={"customer_email": "safety@example.test"}).status_code == 200
    ticket_id, _ = WarrantyEngine.start_session(session_id, "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "OS-4000T")
    WarrantyEngine.submit_answer(ticket_id, "warranty")
    WarrantyEngine.submit_answer(ticket_id, "defect")
    response = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": message})
    assert response.status_code == 200
    payload = response.json()
    assert payload["handoff"]["reason"] == "safety_hazard"
    assert "Stop using the chair" in payload["assistant_message"]
    assert "Do not run another test" in payload["assistant_message"]
    assert "local emergency services" in payload["assistant_message"]
    assert payload["ticket"]["status"] == "awaiting_admin_review"
    unavailable_model.assert_not_called()
