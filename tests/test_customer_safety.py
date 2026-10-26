"""Synthetic paraphrases exercise safety reports separately from workflow routing."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from customer_safety import detect_customer_language, reported_safety_hazard, safety_handoff_message, support_handoff_message


@pytest.mark.parametrize("text", [
    "The chair caught fire.", "My chair is on fire.", "My chair is burning.", "There are flames coming from the chair.",
    "The chair is smoking.", "Smoke came out from underneath.", "I got an electrical shock from the chair.",
    "The chair electrocuted me.", "Touching the frame gives me a mild shock.", "The chair shocked my hand.",
    "I got zapped when I touched the chair.", "There is a burnt plastic smell from the base.",
    "It smells like melting plastic.", "No smoke, but there are sparks.", "No smoke, flames are coming out of the chair.",
    "There is no fire or smoke, but the chair hurts.", 'My chair is "smoking" right now.',
    'My wife said "the chair is on fire".', 'The manual says "stop if smoke appears", but I see flames now.',
    "My chair is 'smoking' as described in the manual.",
    "Mi silla está 'en llamas', como dice el manual.",
    "의자에서 '연기'가 나오는데 설명서에도 그런 문구가 있어요.",
    'The manual says "stop if there is smoke" and my chair is smoking.',
    '설명서에 "연기가 나면 멈추세요"라고 써 있고 의자에서 연기가 나요.',
    'El manual dice "deténgase si hay humo" y mi silla está echando humo.',
    "Could this be smoke coming from my chair?", "The chair smoked yesterday. It has stopped now.",
    "의자에서 연기가 나요.", "배송 온 의자에서 연기가 나요.", "배송 중 연기", "배송 중 불꽃이 보여서 배송 날짜를 연기하고 싶어요.",
    "의자에 불이 붙었어요.", "불꽃이 보여요.", "의자를 쓰다가 감전됐어요.", "탄 냄새가 나요.",
    "연기는 없지만 불꽃이 보여요.", "통증은 없고 감전됐어요.", "쇠 부분을 만지니 찌릿해요.",
    "너무 뜨거워서 손을 못 대겠어요.", "연기가 나서 오늘 설치는 연기해주세요.",
    "의자에서 연기해요.", "배송 온 의자에서 연기해요.",
    "Mi silla está echando humo.", "La silla está en llamas.", "Hay fuego en mi silla.",
    "La silla me dio una descarga eléctrica.", "No hay humo, pero hay chispas.", "No hay humo, fuego sale de la silla.",
    "Se ven llamas debajo de mi silla.", "Llamas salen por debajo del asiento.", "Vemos llamas en la base.",
    "La silla está ardiendo.",
    "La llama sale de debajo de la silla.", "Una llama naranja apareció debajo del asiento.",
    "La silla humea al encenderla.", "El motor está chispeando.", "Siento dolor al usar la silla.",
    "Mi hija se llama Ana y veo llamas debajo de la silla.", "¿Eso es humo que sale de la silla?",
])
def test_reported_hazards_stop_troubleshooting(text):
    assert reported_safety_hazard(text=text), text


@pytest.mark.parametrize("text", [
    "No smoke or sparks.", "There is no fire or smoke.", "The chair isn't smoking.", "The chair is not on fire.",
    "Smoke is not present.", "Smoke isn't visible.", "No pain, no discomfort, no overheating.",
    "There is no evidence of smoke.", "I don't feel any pain.", "It doesn't cause any physical discomfort.",
    "I don't smell any smoke.", "We never saw any sparks.",
    "It is not too hot.", "The heater does not get warm enough.", "Fire resistant fabric.", "I chose the smoke gray chair.",
    'The manual says "stop using the chair if there is smoke".', '"The chair caught fire" is an example in the instructions.',
    "The manual mentions smoke and sparks as warnings.", "What should I do if the chair starts smoking?",
    'What does "smoke" mean in the manual?', '"Smoke" is an example in the guide.',
    'The manual says "stop if there is smoke" and my chair is not smoking.',
    'El manual dice "pare si hay humo" y mi silla no está echando humo.',
    "연기가 없어요.", "불꽃이 보이지 않아요.", "연기가 나지 않고 통증도 없어요.", "연기와 불꽃이 없어요.",
    "불이 안 나요.", "과열되지 않습니다.", "감전된 적이 없어요.", "불이 난 적 없어요.", "불이 안 났어요.",
    "너무 뜨겁지 않아요.", "온열 기능이 안 뜨거워요.", "배송이 연기됐어요.", "배송 날짜를 연기하고 싶어요.",
    "과열되는 건 아니에요.", "뜨거운 건 아니고 따뜻하기만 해요.", "연기가 나오는 것은 아니에요.",
    "배송을 다음 주로 연기해 주세요.", "주문 배송 일정을 연기하려고요.", "설치 방문을 두 시간 뒤로 연기할 수 있나요?",
    "배송 날짜 연기 문의", "예약을 연기하고 싶어요.", "오늘은 연기하자.", "설명서에 연기가 나면 사용을 중단하라고 적혀 있어요.",
    "불이 나면 어떻게 해야 하나요?", "No hay humo ni chispas.", "La silla no está en llamas.", "Sin dolor ni molestias.",
    "No tengo dolor.", "No me duele.", "No se ven llamas debajo de la silla.", "No sale nada de humo.",
    "El modelo se llama Hypnos.", "Mi perro se llama Fuego.", 'Mi perro se llama "Fuego".', "¿Me llamas cuando esté listo?", "Por favor llama a soporte.",
    "Llama al técnico para la entrega.", "Llámame mañana.", "En el manual dice que puede haber humo si se usa mal.",
    "¿Qué debo hacer si salen llamas de una silla?", "Descarga el manual de instalación.",
])
def test_denials_references_postponements_and_call_requests_do_not_report_a_hazard(text):
    assert not reported_safety_hazard(text=text), text


def test_saved_incident_remains_sticky_across_a_short_followup():
    ticket = SimpleNamespace(get_collected=lambda: {"intake_raw_message": "Se ven llamas debajo de mi silla."})
    assert reported_safety_hazard(ticket=ticket, text="Okay, thank you.")
    assert reported_safety_hazard(turns=[SimpleNamespace(customer_answer="의자에서 연기가 나요")], text="지금은 멈췄어요")
    assert not reported_safety_hazard(ticket=SimpleNamespace(get_collected=lambda: {"intake_summary": "No smoke or sparks"}), text="No pain.")


@pytest.mark.parametrize("text,default,expected", [
    ("Se ven llamas debajo de mi silla.", "en", "es"), ("Sale humo.", "en", "es"),
    ("의자에서 연기가 나요", "en", "ko"), ("Okay", "es", "es"), ("OS-4000T", "ko", "ko"),
    ("Please answer in English", "ko", "en"), ("The chair has caught fire", "es", "en"),
])
def test_customer_language_keeps_short_acknowledgements_and_detects_short_safety_reports(text, default, expected):
    assert detect_customer_language(text, default=default) == expected


@pytest.mark.parametrize("language,stop,safe,panels,pending", [
    ("en", "Stop using the chair", "only if you can do so safely", "open panels", "queued"),
    ("ko", "의자 사용을 즉시 중단", "안전하게 할 수 있을 때만", "패널", "발송 대기"),
    ("es", "Deje de usar la silla", "solo si puede hacerlo de forma segura", "paneles", "en cola"),
])
def test_localized_safety_copy_has_safe_actions_and_truthful_pending_status(language, stop, safe, panels, pending):
    phone = "+1-888-848-2630 ext. 3"
    text = safety_handoff_message(language, phone, notification_status="pending")
    assert all(phrase in text for phrase in (stop, safe, panels, pending, phone))
    assert "submitted to our support inbox" not in text
    if language == "en":
        assert "Do not run another test" in text
        assert "haven't sent a notification" in text


@pytest.mark.parametrize("language,submitted,unbooked", [
    ("en", "submitted to our support inbox", "callback has not been booked"),
    ("ko", "알림을 제출했습니다", "콜백은 아직 예약되지 않았습니다"),
    ("es", "enviado una notificación", "no se ha reservado una llamada"),
])
def test_confirmed_notification_does_not_claim_a_staff_reply_or_callback_booking(language, submitted, unbooked):
    text = safety_handoff_message(language, "PHONE", True)
    assert submitted in text and unbooked in text
    assert "queued" not in text and "en cola" not in text and "발송 대기" not in text


def test_english_default_copy_remains_compatible_and_generic_requests_localize():
    text = safety_handoff_message("en", "PHONE")
    assert text == ("Stop using the chair. Switch it off and unplug it only if you can do so safely. "
                    "Do not run another test, resume use, or open panels. "
                    "If there is an active fire or immediate danger, move away and contact local emergency services. "
                    "I've saved the reported safety concern and conversation context with this case for team review. "
                    "I haven't sent a notification or booked a callback. Please call **PHONE** for direct support.")
    assert "담당자 상담 요청" in support_handoff_message("ko", "PHONE")
    assert "solicitud de llamada" in support_handoff_message("es", "PHONE", "callback_request")


@pytest.mark.parametrize("language,conditional,emergency", [
    ("en", "If there is an active fire or immediate danger", "move away and contact local emergency services"),
    ("ko", "불이 나고 있거나 즉각적인 위험이 있으면", "안전한 곳으로 이동하고 현지 긴급 구조 기관에 연락"),
    ("es", "Si hay fuego activo o peligro inmediato", "aléjese y contacte con los servicios de emergencia locales"),
])
def test_active_fire_or_immediate_danger_gets_conditional_local_emergency_guidance(language, conditional, emergency):
    text = safety_handoff_message(language, "SUPPORT PHONE")
    assert conditional in text and emergency in text
    assert "911" not in text
