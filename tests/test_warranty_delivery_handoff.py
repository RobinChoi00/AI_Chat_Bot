"""Verified delivery refresh and human interrupts use real workflow persistence."""
import json
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import warranty_models as wm
import warranty_router as wr
import warranty_workflow as wf


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    wm.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(wm, "_engine", engine)
    monkeypatch.setattr(wm, "_SessionFactory", factory)
    monkeypatch.setattr(wf, "_SessionFactory", factory)
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "0")
    monkeypatch.setattr("warranty_assistant_message.build_assistant_message_bundle", lambda **kw: {"assistant_message": kw["node"].get("prompt", "")} if kw.get("node") else {})
    app = FastAPI()
    app.include_router(wr.router)
    with TestClient(app) as http:
        yield http
    engine.dispose()


def _ticket(session="handoff"):
    ticket_id, _ = wf.WarrantyEngine.start_session(session, "osakiusa.com")
    wf.WarrantyEngine.set_model_name(ticket_id, "OS-4000T")
    wf.WarrantyEngine.submit_answer(ticket_id, "warranty")
    wf.WarrantyEngine.submit_answer(ticket_id, "defect")
    return ticket_id


def test_repeated_human_requests_interrupt_before_node_mapping_and_preserve_context(client, monkeypatch):
    ticket_id = _ticket()
    previous = wf.WarrantyEngine.get_ticket(ticket_id).current_node_id
    monkeypatch.setattr("warranty_nlp.interpret_warranty_answer", lambda *_a, **_k: pytest.fail("Human request reached node mapping"))
    for phrase in ("I want to speak to a human", "human please", "a real person", "please call me back", "talk to an agent"):
        response = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": phrase})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ticket"]["ticket_id"] == ticket_id
        assert body["ticket"]["status"] == "awaiting_admin_review"
        assert body["ticket"]["current_node"]["options"] == []
        assert body["handoff"]["requested"] is True
        assert body["handoff"]["notification_sent"] is False
        assert "haven't sent a notification" in body["assistant_message"]
    stored = wf.WarrantyEngine.get_ticket(ticket_id)
    assert stored.current_node_id == previous
    assert stored.model_name == "OS-4000T"
    assert stored.get_collected()["human_support_previous_node"] == previous
    hydrated = client.get("/api/v1/warranty/session/handoff").json()
    assert hydrated["handoff"]["requested"] is True
    with wm.warranty_db_session() as db:
        assert db.query(wm.WarrantyTicket).count() == 1


@pytest.mark.parametrize("route,body", [
    ("register-model", {"model": "human"}),
    ("confirm-model", {"model": "Please call me back"}),
    ("natural-start", {"message": "speak to a representative"}),
    ("smart-start", {"message": "I need a human"}),
    ("handoff", {"reason": "customer_request"}),
])
def test_human_request_before_model_gate(client, monkeypatch, route, body):
    monkeypatch.setattr("warranty_order_verification.verify_ticket_purchase", lambda **_kw: pytest.fail("Human request scheduled a model purchase verification"))
    response = client.post(f"/api/v1/warranty/session/first-{route}/{route}", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["handoff"]["requested"] is True
    assert not response.json()["ticket"]["model_name"]


@pytest.mark.parametrize("route,body", [
    ("troubleshooting-outcome", {"outcome": "Please call me back"}),
    ("customer-note", {"note": "I want a human"}),
])
def test_terminal_message_channels_also_interrupt_before_nlp_or_email(client, route, body):
    ticket_id = _ticket(route)
    response = client.post(f"/api/v1/warranty/{ticket_id}/{route}", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["handoff"]["requested"] is True


def test_repeated_unmapped_replies_offer_then_save_handoff(client, monkeypatch):
    ticket_id = _ticket("clarification")
    monkeypatch.setattr(wr, "_maybe_side_question_message", lambda *_a: None)
    monkeypatch.setattr("warranty_nlp.interpret_warranty_answer", lambda *_a, **_kw: None)
    monkeypatch.setattr("warranty_nlp.suggest_closest_option", lambda *_a, **_kw: None)
    results = [client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "unmatched explanation"}) for _ in range(3)]
    assert all(response.status_code == 200 for response in results)
    assert results[1].json()["handoff"]["offered"] is True
    assert results[2].json()["handoff"]["reason"] == "clarification_loop"
    assert results[2].json()["ticket"]["status"] == "awaiting_admin_review"


def test_delivery_starts_without_model_and_refresh_always_calls_provider(client, monkeypatch):
    calls = []
    class Provider:
        @staticmethod
        def fetch_shopify_order_status(order, email, domain, *, allow_order_only):
            calls.append((order, email, domain, allow_order_only))
            return {"order_number": "#12345", "status": "IN_TRANSIT" if len(calls) == 1 else "DELIVERED",
                "tracking_number": "1Z999AA10123456784", "company": "UPS", "eta": "2026-10-27",
                "tracking_url": "https://www.ups.com/track?tracknum=1Z999AA10123456784",
                "product_names": ["PRIVATE PRODUCT"], "total_amount": "9999", "purchase_date": "2020-01-01"}
    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: Provider())
    started = client.post("/api/v1/warranty/session/delivery/quick-start", json={"issue_type": "delivery", "domain": "osakiusa.com"})
    assert started.status_code == 200, started.text
    assert started.json()["delivery_lookup_required"] is True
    assert not started.json()["ticket"]["model_name"]
    for expected in ("IN_TRANSIT", "DELIVERED"):
        response = client.post("/api/v1/warranty/session/delivery/delivery-tracking", json={"order_number": "12345", "email": "buyer@example.com", "domain": "titanchair.com"})
        assert response.status_code == 200, response.text
        summary = response.json()["tracking_summary"]
        assert summary["available"] is True and summary["status"] == expected
        assert summary["fetched_at"] and summary["source"] == "shopify"
        assert "PRIVATE PRODUCT" not in response.text and "9999" not in response.text
        assert "buyer@example.com" not in response.text
    assert calls == [("12345", "buyer@example.com", "osakiusa.com", False)] * 2


def test_delivery_mismatch_and_provider_failure_do_not_leak_previous_snapshot(client, monkeypatch):
    calls = []
    class Provider:
        @staticmethod
        def fetch_shopify_order_status(order, email, domain, *, allow_order_only):
            calls.append(order)
            if len(calls) == 1:
                return {"order_number": "#12345", "status": "IN_TRANSIT", "company": "UPS", "tracking_number": "SECRET-TRACKING"}
            if len(calls) == 2:
                return {"order_number": "#99999", "status": "DELIVERED", "tracking_number": "OTHER-CUSTOMER", "product_names": ["OTHER PRIVATE"]}
            raise RuntimeError("provider unavailable")
    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: Provider())
    endpoint = "/api/v1/warranty/session/failure/delivery-tracking"
    body = {"order_number": "12345", "email": "buyer@example.com"}
    assert client.post(endpoint, json=body).json()["tracking_summary"]["available"] is True
    for _ in range(2):
        response = client.post(endpoint, json=body)
        assert response.status_code == 200, response.text
        summary = response.json()["tracking_summary"]
        assert summary["available"] is False and summary["tracking_url"] == ""
        assert summary["carrier"] == "" and summary["eta"] == ""
        assert "OTHER-CUSTOMER" not in response.text and "SECRET-TRACKING" not in response.text
        assert "OTHER PRIVATE" not in response.text


def test_delivery_requires_both_identifiers_and_consent_before_provider(client, monkeypatch):
    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: pytest.fail("Unverified request reached provider"))
    from delivery_lookup import lookup_by_order_or_email
    assert lookup_by_order_or_email("buyer@example.com", "osakiusa.com").available is False
    assert lookup_by_order_or_email("#12345", "osakiusa.com").available is False
    endpoint = "/api/v1/warranty/session/privacy/delivery-tracking"
    assert client.post(endpoint, json={"email": "buyer@example.com"}).status_code == 422
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1")
    assert client.post(endpoint, json={"order_number": "12345", "email": "buyer@example.com"}).status_code == 403


def test_delivery_url_never_exposes_script_or_malformed_provider_url():
    from delivery_lookup import TrackingSnapshot, public_delivery_tracking_summary
    for url in ("javascript:alert(1)", "data:text/html,test", "https://[broken", "https://user:secret@example.com"):
        summary = public_delivery_tracking_summary(TrackingSnapshot(source="shopify", available=True, status="PROCESSING", tracking_url=url), "osakiusa.com")
        assert summary["tracking_url"] == ""


@pytest.mark.parametrize("model", ["/Users/customer/Downloads/photo.jpg", "customer provided model name only", "my chair is not working"])
def test_model_registration_rejects_paths_internal_meta_and_symptoms(client, model):
    response = client.post("/api/v1/warranty/session/invalid-model/register-model", json={"model": model})
    assert response.status_code == 422
    with wm.warranty_db_session() as db:
        assert db.query(wm.WarrantyTicket).count() == 0


def test_smart_start_does_not_store_extracted_return_as_model(client, monkeypatch):
    monkeypatch.setattr("warranty_intake.extract_workflow_prefill", lambda **_kw: {
        "answer_keys": ["warranty"], "model_name": "Return", "summary": "customer provided model name only", "source": "llm",
    })
    response = client.post("/api/v1/warranty/session/no-return-model/smart-start", json={"message": "My chair is not working"})
    assert response.status_code == 200, response.text
    assert response.json()["ticket"]["model_name"] == ""


def test_tracking_scan_time_is_not_invented_as_delivery_estimate():
    from track123_client import parse_track123_record
    result = parse_track123_record({"lastTrackingTime": "2020-01-01", "nextUpdateTime": "2026-10-23", "deliveredTime": "2025-01-01"})
    assert result["eta"] == "Pending carrier update"


@pytest.mark.parametrize("text", ["The remote controller has a short", "The chair is overheating", "I see smoke and sparks", "The massage causes pain", "There is a burning smell; please call me"])
def test_reported_safety_hazard_stops_before_diagnostic_mapping(client, monkeypatch, text):
    ticket_id = _ticket("safety")
    node_id = wf.WarrantyEngine.get_ticket(ticket_id).current_node_id
    monkeypatch.setattr(wr, "_submit_answer_with_nlp", lambda *_a, **_kw: pytest.fail("Unsafe report reached diagnostic testing"))
    response = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": text})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["handoff"]["reason"] == "safety_hazard"
    assert body["handoff"]["notification_sent"] is False
    assert body["ticket"]["status"] == "awaiting_admin_review"
    assert body["ticket"]["current_node"]["options"] == []
    message = body["assistant_message"]
    assert "Stop using" in message and "unplug it only if" in message
    assert "Do not run another test" in message and "open panels" in message
    assert "warm-up" not in message and "reset cycle" not in message
    assert wf.WarrantyEngine.get_ticket(ticket_id).current_node_id == node_id
    again = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "human please"}).json()
    assert again["handoff"]["reason"] == "safety_hazard"


@pytest.mark.parametrize("text", ["No pain or discomfort", "The chair is not overheating", "No burning smell or smoke", "No electric shock and no short circuit"])
def test_negated_hazard_does_not_pause_ordinary_workflow(client, monkeypatch, text):
    ticket_id = _ticket("safe")
    monkeypatch.setattr(wr, "_maybe_side_question_message", lambda *_a: None)
    monkeypatch.setattr(wr, "_submit_answer_with_nlp", lambda *_a, **_kw: (None, False, "Please choose an option.", None))
    response = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": text})
    assert response.status_code == 200, response.text
    assert response.json()["ticket"]["status"] == "in_progress"
    assert not response.json().get("handoff", {}).get("requested")


def test_delivery_handoff_uses_sales_contact_but_hazard_uses_warranty(client):
    response = client.post("/api/v1/warranty/session/contact-store/quick-start", json={"issue_type": "delivery"})
    assert response.status_code == 200
    ticket_id = response.json()["ticket"]["ticket_id"]
    handoff = client.post("/api/v1/warranty/session/contact-store/handoff", json={}).json()["handoff"]
    assert handoff["department"] == "sales"
    assert handoff["phone"].endswith("ext. 2") and handoff["phone_href"].endswith(";ext=2")
    assert handoff["hours"]
    hazard = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "The remote has a short"}).json()["handoff"]
    assert hazard["department"] == "warranty"
    assert hazard["phone"].endswith("ext. 3") and hazard["phone_href"].endswith(";ext=3")
    from delivery_lookup import TrackingSnapshot, public_delivery_tracking_summary
    support = public_delivery_tracking_summary(TrackingSnapshot(source="unavailable", available=False), "osakiusa.com")["support_contact"]
    assert support["department"] == "sales" and support["phone_href"].endswith(";ext=2")


def test_explicit_handoff_button_preserves_reported_hazard_priority(client):
    response = client.post("/api/v1/warranty/session/button-safety/handoff", json={"message": "Please call me, there is smoke"})
    assert response.status_code == 200, response.text
    assert response.json()["handoff"]["reason"] == "safety_hazard"
    assert "Stop using" in response.json()["assistant_message"]
