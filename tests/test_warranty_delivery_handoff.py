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


def _setup_ticket(session, model="OS-4000XT"):
    ticket_id, _ = wf.WarrantyEngine.start_session(session, "osakiusa.com")
    wf.WarrantyEngine.set_model_name(ticket_id, model)
    wf.WarrantyEngine.submit_answer(ticket_id, "warranty")
    wf.WarrantyEngine.submit_answer(ticket_id, "installation")
    wf.WarrantyEngine.submit_answer(ticket_id, model)
    assert wf.WarrantyEngine.get_current_node(ticket_id)["node_id"] == "install_concern"
    return ticket_id


@pytest.mark.parametrize("entry", ("answer", "natural-start", "smart-start"))
def test_purchase_media_answers_exact_saved_bliss_case_without_install_reprompt(client, monkeypatch, entry):
    session = f"bliss-media-{entry}"
    monkeypatch.setenv("WARRANTY_CHAT_EMAIL_REQUIRED", "0")
    assert client.post(f"/api/v1/warranty/session/{session}/consent", json={"domain": "osakiusa.com"}).status_code == 200
    ticket_id = _setup_ticket(session, "osaki bliss vl")
    turns_before = len(wf.WarrantyEngine.get_turns(ticket_id))
    message = "i want buy osaki bliss VL but i need photos and videos about this product"
    route = f"{ticket_id}/answer" if entry == "answer" else f"session/{session}/{entry}"
    response = client.post(f"/api/v1/warranty/{route}", json={"answer" if entry == "answer" else "message": message})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["side_question"] is True
    assert body["ticket"]["ticket_id"] == ticket_id
    assert body["ticket"]["issue_type"] == "installation"
    assert body["ticket"]["current_node"]["node_id"] == "install_concern"
    assert not body.get("help_offer_options")  # Photos do not count as repair resolution.
    answer = body["assistant_message"]
    assert "https://osakiusa.com/products/os-bliss-vl" in answer
    assert "don't have a verified official video link" in answer
    assert "installation issue" not in answer.lower()
    assert len(wf.WarrantyEngine.get_turns(ticket_id)) == turns_before
    with wm.warranty_db_session() as db:
        rows = db.query(wm.WarrantyConversationMessage).filter_by(ticket_id=ticket_id).order_by(wm.WarrantyConversationMessage.id).all()
        assert rows[-2].content == message and rows[-2].origin == "customer_input"
        assert rows[-1].content == answer and rows[-1].origin == "api_response"
        assert rows[-1].exchange_id == rows[-2].exchange_id


def test_unknown_shopping_model_never_borrows_saved_bliss_model(client):
    tid = _setup_ticket("unknown-shopping-model", "Osaki Bliss VL")
    response = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": "I want to buy the Osaki Unlisted XX model. Can I see product photos?"})
    assert response.status_code == 200, response.text
    reply = response.json()["assistant_message"]
    assert "couldn't verify the exact product" in reply
    assert "os-bliss-vl" not in reply


@pytest.mark.parametrize("entry", ("answer", "natural-start", "smart-start"))
def test_order_status_in_setup_offers_secure_lookup_without_advancing(client, monkeypatch, entry):
    session = f"delivery-side-{entry}"
    ticket_id = _setup_ticket(session)
    turns_before = len(wf.WarrantyEngine.get_turns(ticket_id))
    # Looking up is explicit, not a side effect of writing a chat question.
    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: pytest.fail("Side question reached provider"))
    route = f"{ticket_id}/answer" if entry == "answer" else f"session/{session}/{entry}"
    response = client.post(f"/api/v1/warranty/{route}", json={"answer" if entry == "answer" else "message": "Where is my order?"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["delivery_lookup_required"] is True
    assert payload["ticket"]["ticket_id"] == ticket_id
    assert payload["ticket"]["issue_type"] == "installation"
    assert payload["ticket"]["current_node"]["node_id"] == "install_concern"
    assert payload.get("tracking_summary") is None
    assert "I haven't looked up the order yet" in payload["assistant_message"]
    assert "email used at checkout" in payload["assistant_message"]
    assert not payload.get("help_offer_options")
    assert len(wf.WarrantyEngine.get_turns(ticket_id)) == turns_before


def test_side_tracking_checks_checkout_identity_and_does_not_put_order_data_on_repair_ticket(client, monkeypatch):
    tid = _setup_ticket("separate-delivery-lookup")
    before = wf.WarrantyEngine.get_ticket(tid).get_collected()
    calls = []

    class Provider:
        @staticmethod
        def fetch_shopify_order_status(order, email, domain, *, allow_order_only):
            calls.append((order, email, domain, allow_order_only))
            return {"order_number": "#12345", "status": "DELIVERED", "company": "UPS"}

    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: Provider())
    endpoint = "/api/v1/warranty/session/separate-delivery-lookup/delivery-tracking"
    assert client.post(endpoint, json={"order_number": "12345"}).status_code == 422
    response = client.post(endpoint, json={"order_number": "12345", "email": "buyer@example.invalid", "domain": "titanchair.com"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["delivery_lookup_required"] is True and payload["tracking_summary"]["available"] is True
    assert calls == [("12345", "buyer@example.invalid", "osakiusa.com", False)]
    assert payload["ticket"]["ticket_id"] == tid and payload["ticket"]["issue_type"] == "installation"
    assert payload["ticket"]["current_node"]["node_id"] == "install_concern"
    assert wf.WarrantyEngine.get_ticket(tid).get_collected() == before
    assert "buyer@example.invalid" not in response.text


@pytest.mark.parametrize("start,expected_node", (
    (_setup_ticket, "install_concern"),
    (_ticket, "defect_problem_type"),
))
def test_delivery_damage_inside_another_case_does_not_promise_a_new_claim(client, start, expected_node):
    tid = start("independent-damage-question")
    response = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": "My new chair arrived with a cracked armrest and a crushed box."})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "haven't opened a new claim" in payload["assistant_message"]
    assert "ext. 2" in payload["assistant_message"]
    assert payload["ticket"]["current_node"]["node_id"] == expected_node


@pytest.mark.parametrize("symptom", ("the screen", "touch screen"))
def test_blank_remote_screen_is_not_assumed_to_mean_controls_work_or_power_failure(client, symptom):
    tid = _ticket("remote-screen-details")
    first = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": symptom})
    assert first.status_code == 200, first.text
    assert first.json()["ticket"]["current_node"]["node_id"] == "defect_remote_power_q"
    before = len(wf.WarrantyEngine.get_turns(tid))
    response = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": "The screen is blank."})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["side_question"] is True
    assert "buttons still move the chair" in payload["assistant_message"]
    assert payload["ticket"]["current_node"]["node_id"] == "defect_remote_power_q"
    assert len(wf.WarrantyEngine.get_turns(tid)) == before
    exact = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": "no_power"})
    assert exact.status_code == 200
    assert exact.json()["ticket"]["current_node"]["node_id"] == "defect_remote_no_power_checks"


@pytest.mark.parametrize("statement,node", (
    ("The screen is blank and the buttons still move the chair.", "defect_remote_screen_q"),
    ("The screen is not blank and everything works.", "defect_remote_screen_q"),
    ("The screen is blank and no buttons respond.", "defect_remote_no_power_checks"),
))
def test_remote_power_step_uses_stated_control_result(client, statement, node):
    tid = _ticket("remote-power-explicit")
    wf.WarrantyEngine.submit_answer(tid, "remote")
    response = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": statement})
    assert response.status_code == 200, response.text
    assert response.json()["ticket"]["current_node"]["node_id"] == node


def test_no_buttons_respond_does_not_become_blank_screen_with_working_commands(client):
    tid = _ticket("remote-screen-no-buttons")
    wf.WarrantyEngine.submit_answer(tid, "remote")
    wf.WarrantyEngine.submit_answer(tid, "has_power")
    response = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": "The screen is blank and no buttons respond."})
    assert response.status_code == 200, response.text
    assert response.json()["ticket"]["current_node"]["node_id"] == "defect_remote_partial_terminal"


@pytest.mark.parametrize("hazard", (
    "My chair is smoking. I want to buy the Osaki Bliss VL and see a video.",
    "I want to buy a replacement chair because the rollers hurt me while sitting in my chair.",
))
def test_shopping_pain_does_not_interrupt_but_real_chair_hazard_does(client, hazard):
    tid = _setup_ticket("shopping-safety-precedence")
    plain = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": "I want to buy the Osaki Bliss VL for my existing back pain. Can I see a demo video?"})
    assert plain.status_code == 200, plain.text
    assert not plain.json().get("handoff")
    assert "os-bliss-vl" in plain.json()["assistant_message"]
    danger = client.post(f"/api/v1/warranty/{tid}/answer", json={"answer": hazard})
    assert danger.status_code == 200, danger.text
    assert danger.json()["handoff"]["reason"] == "safety_hazard"
    assert "Stop using the chair" in danger.json()["assistant_message"]


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
        assert "callback has not been booked" in body["assistant_message"]
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
    body_json = response.json()
    # Opening human-only asks get phone numbers — not a Freshdesk queue entry.
    assert not (body_json.get("handoff") or {}).get("requested")
    assert "ext. 3" in body_json["assistant_message"] or "888-848-2630" in body_json["assistant_message"]
    assert body_json["ticket"]["status"] != "awaiting_admin_review"
    assert not body_json["ticket"]["model_name"]


def test_opening_human_request_does_not_enqueue_handoff(client, monkeypatch):
    from warranty_notification_jobs import WarrantyNotificationJob

    monkeypatch.setenv("WARRANTY_CHAT_EMAIL_REQUIRED", "0")
    assert client.post("/api/v1/warranty/session/bare-human/consent", json={"domain": "osakiusa.com"}).status_code == 200
    response = client.post(
        "/api/v1/warranty/session/bare-human/smart-start",
        json={"message": "I need a human"},
    )
    assert response.status_code == 200, response.text
    assert not (response.json().get("handoff") or {}).get("requested")
    assert "has not opened a warranty case" in response.json()["assistant_message"]
    with wm.warranty_db_session() as db:
        assert db.query(WarrantyNotificationJob).count() == 0


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
    first = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "unmatched explanation"})
    second = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "unmatched explanation"})
    assert first.status_code == 200 and second.status_code == 200
    assert not (first.json().get("handoff") or {}).get("requested")
    assert second.json()["handoff"]["reason"] == "clarification_loop"
    assert second.json()["ticket"]["status"] == "awaiting_admin_review"


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


def _delivery_ticket(session: str) -> str:
    ticket_id, _ = wf.WarrantyEngine.start_session(session, "osakiusa.com")
    wf.WarrantyEngine.set_model_name(ticket_id, "OS-4000T")
    wf.WarrantyEngine.submit_answer(ticket_id, "warranty")
    wf.WarrantyEngine.submit_answer(ticket_id, "delivery")
    assert wf.WarrantyEngine.get_current_node(ticket_id)["node_id"] == "delivery_intent_q"
    return ticket_id


def test_address_change_asks_to_submit_to_sales_before_phone_dump(client):
    ticket_id = _delivery_ticket("addr-offer")
    response = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "I need to change delivery address. I have a tracking number"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("side_question") is True
    assert body["ticket"]["current_node"]["node_id"] == "delivery_intent_q"
    assert not body.get("handoff", {}).get("requested")
    msg = body["assistant_message"].lower()
    assert "submit" in msg and "sales" in msg
    assert "does **not** change" in body["assistant_message"].lower() or "does not change" in msg
    assert "ext. 2" in body["assistant_message"]
    collected = wf.WarrantyEngine.get_ticket(ticket_id).get_collected()
    assert collected.get("awaiting_sales_submit_consent") is True
    assert "change delivery address" in (collected.get("sales_submit_pending_text") or "").lower()


def test_address_change_yes_saves_sales_review_without_changing_order(client):
    ticket_id = _delivery_ticket("addr-yes")
    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "I need to change delivery address. I have a tracking number"},
    )
    yes = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "yes"})
    assert yes.status_code == 200, yes.text
    body = yes.json()
    assert body.get("handoff", {}).get("requested") is True
    assert body["handoff"]["department"] == "sales"
    assert body["handoff"]["reason"] == "sales_submit"
    assert "Sales review request" in body["assistant_message"]
    assert "ext. 2" in body["assistant_message"]
    collected = wf.WarrantyEngine.get_ticket(ticket_id).get_collected()
    assert collected.get("awaiting_sales_submit_consent") is False
    assert "change delivery address" in (collected.get("human_support_request") or "").lower()


def test_address_change_no_returns_sales_phone_and_keeps_delivery_menu(client):
    ticket_id = _delivery_ticket("addr-no")
    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "Please update my shipping address"},
    )
    declined = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "no"})
    assert declined.status_code == 200, declined.text
    body = declined.json()
    assert body.get("side_question") is True
    assert not body.get("handoff", {}).get("requested")
    assert "Sales handles that" in body["assistant_message"]
    assert "ext. 2" in body["assistant_message"]
    assert body["ticket"]["current_node"]["node_id"] == "delivery_intent_q"
    assert body["ticket"]["status"] == "in_progress"
    assert wf.WarrantyEngine.get_ticket(ticket_id).get_collected().get(
        "awaiting_sales_submit_consent"
    ) is False
