"""Regression dialogs found by the synthetic, provider-unavailable API run."""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import warranty_intake as intake
import warranty_models as models
import warranty_nlp as nlp
import warranty_workflow as workflow

HEAT_REPORT = "My OS-4000T heat stays cold. I already ran it for twenty minutes with heat ON and checked the remote setting."


@pytest.mark.parametrize("text", ["The heat is ON and I checked the remote setting.", "error code 4000", "My order number is 10001"])
def test_incidental_words_and_order_or_error_numbers_are_not_models(text):
    assert intake._extract_model_from_intake(text) == ""


def test_completed_warmup_duration_retains_decimal_minutes():
    from warranty_intake_context import _has_completed_heat_warmup

    assert _has_completed_heat_warmup("I already ran heat for 15.5 minutes with heat ON.")
    assert _has_completed_heat_warmup("I already ran heat for twenty minutes with heat ON while on hold.")
    assert not _has_completed_heat_warmup("I already ran heat for 5.5 minutes with heat ON.")


@pytest.mark.parametrize("text", [
    "I already ran heat for five minutes with heat ON and waited twenty minutes on hold.",
    "I waited twenty minutes for support, then tried heat for five minutes with heat ON.",
    "I spent twenty minutes on hold and already used heat for five minutes with heat ON.",
    "I already checked the heat setting for twenty minutes, then ran heat for five minutes with heat ON.",
])
def test_non_heating_durations_cannot_establish_completed_warmup(text):
    from warranty_intake_context import _has_completed_heat_warmup

    assert not _has_completed_heat_warmup(text)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1")
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "0")
    monkeypatch.setattr(nlp, "_openai_client", lambda: None)
    monkeypatch.setattr(intake, "_openai_client", lambda: None)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine, expire_on_commit=False, autocommit=False, autoflush=False)
    models.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(models, "_engine", engine)
    monkeypatch.setattr(models, "_SessionFactory", factory)
    monkeypatch.setattr(workflow, "_SessionFactory", factory)
    from warranty_router import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as api:
        yield api
    engine.dispose()


def setup(client, *, model="OS-4000T"):
    session = "simulation-regression"
    assert client.post(f"/api/v1/warranty/session/{session}/consent", json={"domain": "osaki.com"}).status_code == 200
    assert client.post(f"/api/v1/warranty/session/{session}/contact-email", json={"customer_email": "synthetic@example.invalid"}).status_code == 200
    if model:
        response = client.post(f"/api/v1/warranty/session/{session}/register-model", json={"model": model})
        assert response.status_code == 200, response.text
        return session, response.json()["ticket"]["ticket_id"]
    return session, None


def answer(client, ticket, text):
    response = client.post(f"/api/v1/warranty/{ticket}/answer", json={"answer": text})
    assert response.status_code == 200, response.text
    return response.json()


def test_real_keyword_intake_keeps_confirmed_model_and_heating_complaint(client):
    session, ticket = setup(client)
    response = client.post(f"/api/v1/warranty/session/{session}/smart-start", json={"message": HEAT_REPORT})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticket"]["model_name"] == "OS-4000T"
    assert payload["ticket"]["model_confirmed"] is True
    assert "heat" in payload["smart_start"]["summary"].lower()
    assert "remote" not in payload["smart_start"]["summary"].lower()
    assert payload["smart_start"]["model_name_hint"] == "OS-4000T"
    client.post(f"/api/v1/warranty/session/{session}/quick-start", json={"issue_type": "defect"})
    response = answer(client, ticket, "heat")
    assert "Manhattan" not in response.get("assistant_message", "")
    assert "10 minutes" not in response.get("assistant_message", "")
    assert "10-minute" not in response.get("assistant_message", "")
    response = answer(client, ticket, "not_heating")
    assert response["ticket"]["current_node"]["node_id"] != "defect_heating_warmup_q"
    assert response["ticket"]["current_node"]["is_terminal"] is True
    assert "already completed" in response["assistant_message"]
    assert "10 minutes" not in response["assistant_message"]
    assert "Confirm the heating function is turned ON" not in response["assistant_message"]
    saved = workflow.WarrantyEngine.get_turns(ticket)
    recorded = [turn for turn in saved if turn.answer_key == "still_no_heat"]
    assert len(recorded) == 1 and recorded[0].customer_answer == HEAT_REPORT


def test_inferred_model_cannot_replace_registered_model_but_explicit_correction_can(client, monkeypatch):
    session, _ = setup(client)
    monkeypatch.setattr(intake, "extract_workflow_prefill", lambda **kwargs: {
        "answer_keys": ["warranty", "defect", "heat"], "model_name": "Osaki Manhattan Duo",
        "summary": "Heating issue.", "source": "llm", "confidence": "high",
    })
    payload = client.post(f"/api/v1/warranty/session/{session}/smart-start", json={"message": "The heat stays cold."}).json()
    assert payload["ticket"]["model_name"] == "OS-4000T"
    assert payload["ticket"]["model_confirmed"] is True
    corrected = client.post(f"/api/v1/warranty/session/{session}/confirm-model", json={"model": "OS-4000XT", "confirmed": True})
    assert corrected.status_code == 200
    assert "4000XT" in corrected.json()["ticket"]["model_name"]


@pytest.mark.parametrize("report", [
    "I haven't run heat for twenty minutes with heat ON yet.",
    "I will run heat for twenty minutes with heat ON.",
    "I tried heat for five minutes with heat ON.",
    "I already ran the chair for twenty minutes with heat OFF.",
    "I already ran heat for 5-10 minutes with heat ON.",
    "I already ran heat for almost ten minutes with heat ON.",
    "I already ran heat for twenty minutes and the remote was turned ON.",
    "I already ran heat for five minutes with heat ON and waited twenty minutes on hold.",
    "I waited twenty minutes for support, then tried heat for five minutes with heat ON.",
])
def test_incomplete_or_future_warmup_does_not_skip_confirmation(client, report):
    session, ticket = setup(client)
    client.post(f"/api/v1/warranty/session/{session}/smart-start", json={"message": report})
    client.post(f"/api/v1/warranty/session/{session}/quick-start", json={"issue_type": "defect"})
    answer(client, ticket, "heat")
    payload = answer(client, ticket, "not_heating")
    assert payload["ticket"]["current_node"]["node_id"] == "defect_heating_warmup_q"


def test_three_natural_confusion_phrases_escape_side_question_loop(client):
    session, ticket = setup(client)
    client.post(f"/api/v1/warranty/session/{session}/quick-start", json={"issue_type": "defect"})
    first = answer(client, ticket, "It just feels odd somehow.")
    assert not (first.get("handoff") or {}).get("requested")
    second = answer(client, ticket, "I really cannot tell which of those fits.")
    assert second["handoff"]["offered"] is True
    third = answer(client, ticket, "None of those options makes sense to me.")
    assert third["handoff"]["reason"] == "clarification_loop"


def test_real_spec_side_questions_do_not_increment_confusion(client, monkeypatch):
    session, ticket = setup(client)
    client.post(f"/api/v1/warranty/session/{session}/quick-start", json={"issue_type": "defect"})
    monkeypatch.setattr("warranty_side_questions.fetch_delivery_spec_answer", lambda *args: "Chair width: 32 inches.")
    for text in ["What is the chair width?", "How wide is the shipping box?", "Will the box fit through a 32 inch doorway?"]:
        payload = answer(client, ticket, text)
        assert not (payload.get("handoff") or {}).get("requested")
    assert workflow.WarrantyEngine.get_ticket(ticket).get_collected().get("clarification_count", 0) == 0


@pytest.mark.parametrize("action", ["smart-start", "natural-start"])
@pytest.mark.parametrize("message", [
    "My chair hasn't arrived. Can you check my order?",
    "Order 10001 is late. Don't ask me for my chair model.",
    "배송이 아직 안 왔어요. 주문 상태를 확인해 주세요.",
    "배송을 다음 금요일로 연기할 수 있을까요?",
    "Mi pedido no ha llegado. ¿Puede comprobar mi pedido?",
])
def test_obvious_delivery_language_opens_lookup_without_chair_model(client, action, message):
    session, _ = setup(client, model=None)
    response = client.post(f"/api/v1/warranty/session/{session}/{action}", json={"message": message})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticket"]["issue_type"] == "delivery"
    assert payload["delivery_lookup_required"] is True
    assert "order number" in payload["assistant_message"]
    assert "setup" not in payload["assistant_message"].lower()
