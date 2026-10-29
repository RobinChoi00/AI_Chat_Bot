"""Customer paraphrases keep safety stops and language requests precise."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import customer_safety as safety


@pytest.mark.parametrize("text", [
    "The power cable is melting.",
    "The cord melted during use.",
    "The adapter melted.",
    "The power cord is getting very hot.",
    "The plug is scalding hot.",
    "The plug is hot to the touch.",
    "My chair gave me an electric jolt.",
    "La silla huele a plástico quemado.",
    "Me dio un toque eléctrico.",
    "El cable se está derritiendo.",
    "El cable se derritió.",
    "El enchufe quema al tocarlo.",
    "전원 코드가 녹고 있어요.",
    "케이블 피복이 녹았어요.",
    "No smoke, but the power cable is melting.",
    "연기는 나지 않지만 전선이 녹고 있어요.",
])
def test_electrical_damage_reports_stop_use_even_without_smoke(text):
    assert safety.reported_safety_hazard(text=text)
    assert safety.reported_safety_hazard(text=text, physical_only=True)


@pytest.mark.parametrize("text", [
    "The power cable is not melting.",
    "The adapter has not melted.",
    "The plug is not hot to the touch.",
    "The plug does not feel very hot.",
    "The cable has never melted.",
    "The adapter isn't actually melting.",
    "El cable no se está derritiendo.",
    "El cable no se derritió.",
    "El cable no se ha derretido.",
    "전원 코드가 녹지는 않았어요.",
    "케이블 피복이 녹은 건 아니에요.",
    "코드가 녹은 적은 없어요.",
    "케이블이 녹은 적이 없어요.",
    "전선이 안 녹았어요.",
    "I have not seen sparks.",
    "연기가 나지는 않아요.",
    "연기가 나오지는 않아요.",
    "타는 냄새가 나지는 않아요.",
    "불꽃이 보이지는 않아요.",
    "감전되지는 않았어요.",
    "아프지는 않아요.",
    "My stress melted away in the chair.",
    "The plug fits fine and the massage melts my stress away.",
    "Why does the manual mention smoke?",
    "What does overheating mean?",
    "What does the term burning smell mean?",
])
def test_denied_damage_and_shopping_metaphors_do_not_report_an_incident(text):
    assert not safety.reported_safety_hazard(text=text)
    assert not safety.reported_safety_hazard(text=text, physical_only=True)


@pytest.mark.parametrize("text,expected", [
    ("영어로 답해주세요", "en"),
    ("한국어 말고 영어로 설명해 주세요", "en"),
    ("스페인어로 안내해 주세요", "es"),
    ("Please respond in español", "es"),
    ("Responde en inglés", "en"),
    ("No quiero español, responde en inglés", "en"),
    ("Responde en coreano, por favor", "ko"),
    ("Do not reply in Spanish; answer in English", "en"),
    ("Please answer in Korean", "ko"),
    ("Can you speak in English?", "en"),
    ("Please speak in Spanish", "es"),
    ("I cannot speak in English; please use Korean", "ko"),
])
def test_customer_can_explicitly_switch_the_safety_instruction_language(text, expected):
    assert safety.requested_customer_language(text) == expected
    assert safety.detect_customer_language(text, default="es") == expected


@pytest.mark.parametrize("text", [
    "Okay, thank you.", "OS-4000T", "No smoke.", "의자에서 연기가 나요.",
    "I cannot speak in English", "I can speak in Spanish and English", "I speak Spanish and English",
    "I cannot read in English", "I don't understand English",
])
def test_ordinary_followups_are_not_explicit_language_switches(text):
    assert safety.requested_customer_language(text) is None


@pytest.mark.parametrize("text", [
    "No smoke, but the adapter melted.",
    "The cable is not melting, but the plug is scalding hot.",
    "연기가 나지는 않지만 감전됐어요.",
    "Why does the manual mention smoke? My plug melted yesterday.",
    "Why does the manual mention smoke? The power cable melted.",
    "Why does the manual mention smoke and my plug is hot to the touch?",
    "The cable melted yesterday, but now it feels cool.",
    "The plug does not feel very hot, but the cord melted yesterday.",
    "El cable se derritió ayer, aunque ya no está caliente.",
    "코드가 녹은 적은 없지만 플러그가 녹고 있어요.",
])
def test_denied_issue_cannot_hide_a_separate_electrical_hazard(text):
    assert safety.reported_safety_hazard(text=text, physical_only=True)
