"""Unsafe-use reports must stop real workflows without a model connection."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import warranty_intake as intake
import warranty_intake_context as context
import warranty_models as models
import warranty_nlp as nlp
import warranty_workflow as workflow


@pytest.mark.parametrize("text", [
    "The chair caught fire.",
    "My chair is on fire.",
    "There are flames coming from the chair.",
    "The chair is smoking.",
    "I got an electrical shock from the chair.",
    "The chair electrocuted me.",
    "No smoke, but there are sparks.",
    "No smoke, flames are coming out of the chair.",
    "There is no fire or smoke, but the chair hurts.",
    "의자에서 연기가 나요.",
    "배송 온 의자에서 연기가 나요.",
    "배송 중 연기",
    "배송 중 불꽃이 보여서 배송 날짜를 연기하고 싶어요.",
    "의자에 불이 붙었어요.",
    "불꽃이 보여요.",
    "의자를 쓰다가 감전됐어요.",
    "탄 냄새가 나요.",
    "연기는 없지만 불꽃이 보여요.",
    "통증은 없고 감전됐어요.",
    "Mi silla está echando humo.",
    "La silla está en llamas.",
    "Hay fuego en mi silla.",
    "La silla me dio una descarga eléctrica.",
    "No hay humo, pero hay chispas.",
    "No hay humo, fuego sale de la silla.",
])
def test_explicit_hazards_in_supported_languages(text):
    assert context.reported_safety_hazard(text=text)


@pytest.mark.parametrize("text", [
    "No smoke or sparks.",
    "There is no fire or smoke.",
    "The chair isn't smoking.",
    "The chair is not on fire.",
    "Smoke is not present.",
    "No pain, no discomfort, no overheating.",
    "연기가 없어요.",
    "불꽃이 보이지 않아요.",
    "연기가 나지 않고 통증도 없어요.",
    "연기와 불꽃이 없어요.",
    "불이 안 나요.",
    "과열되지 않습니다.",
    "감전된 적이 없어요.",
    "불이 난 적 없어요.",
    "불이 안 났어요.",
    "너무 뜨겁지 않아요.",
    "배송이 연기됐어요.",
    "배송 날짜를 연기하고 싶어요.",
    "배송을 다음 주로 연기해 주세요.",
    "주문 배송 일정을 연기하려고요.",
    "배송 일정을 다음 달로 연기해주세요.",
    "설치 방문을 두 시간 뒤로 연기할 수 있나요?",
    "배송 날짜 연기 문의",
    "예약을 연기하고 싶어요.",
    "No hay humo ni chispas.",
    "La silla no está en llamas.",
    "Sin dolor ni molestias.",
    "No tengo dolor.",
    "No me duele.",
    "El modelo se llama Hypnos.",
])
def test_negated_hazards_and_unrelated_terms_do_not_interrupt(text):
    assert not context.reported_safety_hazard(text=text)


def test_prior_reported_hazard_still_prevents_retesting():
    ticket = SimpleNamespace(get_collected=lambda: {"intake_raw_message": "The chair caught fire."})
    assert context.reported_safety_hazard(ticket=ticket, text="It has cooled down now.")
    assert context.reported_safety_hazard(turns=[SimpleNamespace(customer_answer="의자에서 연기가 나요")])


@pytest.fixture
def offline_client(monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1")
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "0")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine, expire_on_commit=False, autocommit=False, autoflush=False)
    models.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(models, "_engine", engine)
    monkeypatch.setattr(models, "_SessionFactory", factory)
    monkeypatch.setattr(workflow, "_SessionFactory", factory)
    unavailable_model = Mock(side_effect=AssertionError("Safety interruption must precede a model call"))
    monkeypatch.setattr(nlp, "_openai_client", unavailable_model)
    monkeypatch.setattr(intake, "_openai_client", unavailable_model)
    from warranty_router import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client, unavailable_model
    engine.dispose()


@pytest.mark.parametrize("action", ["answer", "natural-start", "smart-start"])
@pytest.mark.parametrize("message", [
    "The chair caught fire.",
    "의자에서 연기가 나요.",
    "배송 온 의자에서 연기가 나요.",
    "Mi silla está en llamas.",
])
def test_real_api_stops_for_hazard_without_requiring_model(offline_client, action, message):
    client, unavailable_model = offline_client
    session_id = "synthetic-safety-assessment"
    assert client.post(f"/api/v1/warranty/session/{session_id}/consent", json={"domain": "osaki.com"}).status_code == 200
    assert client.post(f"/api/v1/warranty/session/{session_id}/contact-email", json={"customer_email": "synthetic@example.invalid"}).status_code == 200
    ticket_id, _ = workflow.WarrantyEngine.start_session(session_id, "osaki.com")
    workflow.WarrantyEngine.set_model_name(ticket_id, "OS-4000T")
    workflow.WarrantyEngine.submit_answer(ticket_id, "warranty")
    workflow.WarrantyEngine.submit_answer(ticket_id, "defect")
    path = f"/api/v1/warranty/{ticket_id}/answer" if action == "answer" else f"/api/v1/warranty/session/{session_id}/{action}"
    response = client.post(path, json={"answer" if action == "answer" else "message": message})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["handoff"]["reason"] == "safety_hazard"
    assert payload["handoff"]["requested"] is True
    assert "Stop using the chair" in payload["assistant_message"]
    assert "Do not run another test" in payload["assistant_message"]
    assert "tap one of the options" not in payload["assistant_message"]
    ticket = workflow.WarrantyEngine.get_ticket(ticket_id)
    assert ticket.get_collected()["human_support_reason"] == "safety_hazard"
    unavailable_model.assert_not_called()


@pytest.mark.parametrize("message", [
    "배송 날짜를 연기하고 싶어요.",
    "배송 일정을 다음 달로 연기해주세요.",
    "배송을 다음 주로 연기해 주세요.",
])
def test_postponement_request_does_not_become_a_smoke_report(offline_client, message):
    client, unavailable_model = offline_client
    unavailable_model.side_effect = None
    unavailable_model.return_value = None
    session_id = "synthetic-schedule-assessment"
    assert client.post(f"/api/v1/warranty/session/{session_id}/consent", json={"domain": "osaki.com"}).status_code == 200
    assert client.post(f"/api/v1/warranty/session/{session_id}/contact-email", json={"customer_email": "synthetic@example.invalid"}).status_code == 200
    ticket_id, _ = workflow.WarrantyEngine.start_session(session_id, "osaki.com")
    workflow.WarrantyEngine.set_model_name(ticket_id, "OS-4000T")
    workflow.WarrantyEngine.submit_answer(ticket_id, "warranty")
    workflow.WarrantyEngine.submit_answer(ticket_id, "defect")
    response = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": message})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert (payload.get("handoff") or {}).get("reason") != "safety_hazard"
    assert "Stop using the chair" not in payload.get("assistant_message", "")
    ticket = workflow.WarrantyEngine.get_ticket(ticket_id)
    assert ticket.get_collected().get("human_support_reason") != "safety_hazard"
