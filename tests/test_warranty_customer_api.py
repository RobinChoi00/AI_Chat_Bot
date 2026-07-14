"""
tests/test_warranty_customer_api.py
====================================
Customer-facing warranty HTTP endpoints (no LLM, no admin key).

Covers:
  - POST /api/v1/warranty/session/{id}/quick-start
  - POST /api/v1/warranty/{ticket_id}/answer
  - GET  /api/v1/warranty/session/{id} after terminal transition
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    import warranty_workflow as wf

    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    mem_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=mem_engine,
        expire_on_commit=False,
    )
    wm.Base.metadata.create_all(bind=mem_engine)

    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", mem_session_factory)
    monkeypatch.setattr(wf, "_SessionFactory", mem_session_factory)

    yield


@pytest.fixture
def client():
    from fastapi import FastAPI
    from warranty_router import router  # noqa: WPS433

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _register_model(client, session_id: str, model: str = "OS-4000T"):
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": model, "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    return resp.json()


def test_register_model_then_ready_for_issue_type(client):
    session_id = "cust-api-model"
    data = _register_model(client, session_id)
    ticket = data["ticket"]
    assert ticket["model_name"] == "OS-4000T"
    assert ticket["model_confirmed"] is True
    assert ticket.get("needs_model_confirmation") is not True
    assert ticket["ready_for_issue_type"] is True
    assert ticket["current_node"]["node_id"] == "issue_type"


def test_confirm_model_after_smart_start(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_intake.extract_workflow_prefill",
        lambda **kwargs: {
            "answer_keys": ["warranty"],
            "model_name": "3D LTX",
            "summary": "3D LTX chair",
            "source": "llm",
        },
    )

    session_id = "cust-api-confirm-model"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/smart-start",
        json={"message": "3D LTX footrest air issue", "domain": "osaki.com"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ticket"]["model_name"] == "3D LTX"
    assert data["ticket"]["needs_model_confirmation"] is True
    assert data.get("model_confirmation", {}).get("message")

    confirm = client.post(
        f"/api/v1/warranty/session/{session_id}/confirm-model",
        json={"confirmed": True, "domain": "osaki.com"},
    )
    assert confirm.status_code == 200
    assert confirm.json()["ticket"]["model_confirmed"] is True
    assert confirm.json()["ticket"]["needs_model_confirmation"] is False


def test_natural_start_corrects_model_while_confirmation_pending(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_intake.extract_workflow_prefill",
        lambda **kwargs: {
            "answer_keys": ["warranty"],
            "model_name": "3D LTX",
            "summary": "3D LTX chair",
            "source": "llm",
        },
    )

    session_id = "cust-api-natural-model-fix"
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/smart-start",
        json={"message": "3D LTX footrest air issue", "domain": "osaki.com"},
    )
    assert start.status_code == 200, start.text
    assert start.json()["ticket"]["needs_model_confirmation"] is True

    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/natural-start",
        json={"message": "Hypnos", "domain": "osaki.com"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("model_corrected") is True
    assert data["ticket"]["model_confirmed"] is True
    assert data["ticket"]["needs_model_confirmation"] is False
    model_name = data["ticket"]["model_name"].lower()
    assert "hypnos" in model_name


def test_quick_start_requires_model_first(client):
    session_id = "cust-api-no-model"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "installation", "domain": "osaki.com"},
    )
    assert resp.status_code == 422


def test_quick_start_installation_skips_model_question_when_registered(client):
    session_id = "cust-api-install"
    _register_model(client, session_id)
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "installation", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    data = resp.json()
    ticket = data["ticket"]
    assert ticket["issue_type"] == "installation"
    assert ticket["current_node"]["node_id"] == "install_concern"
    assert ticket["current_node"]["is_terminal"] is False


def test_submit_answer_advances_without_llm(client):
    session_id = "cust-api-answer"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "air"},
    )
    assert resp.status_code == 200
    node = resp.json()["ticket"]["current_node"]
    assert node["node_id"] == "defect_air_location"
    assert len(node["options"]) >= 4


def test_submit_answer_side_questions_box_size_at_delivery_lookup(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_side_questions.fetch_delivery_spec_answer",
        lambda _model, spec: (
            f"For **Titan Nido 3D**, here is what we have on {spec.title}:\n"
            "- Carton Width: 34 inches"
        ),
    )
    session_id = "cust-api-box-size"
    _register_model(client, session_id, model="Titan Nido 3D")
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "delivery", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "no_tracking"},
    )
    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "give me size of the box"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("side_question") is True
    assert "Carton Width" in data.get("assistant_message", "")
    assert "order number" in data.get("assistant_message", "").lower()

    session = client.get(f"/api/v1/warranty/session/{session_id}")
    assert session.json()["ticket"]["current_node"]["node_id"] == "delivery_get_name"


def test_submit_answer_side_questions_box_size_on_defect_node(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_side_questions.fetch_delivery_spec_answer",
        lambda _model, spec: (
            f"For **Titan Nido 3D**, here is what we have on {spec.title}:\n"
            "- Carton Width: 34 inches"
        ),
    )
    session_id = "cust-api-defect-box-size"
    _register_model(client, session_id, model="Titan Nido 3D")
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]
    before_node = start.json()["ticket"]["current_node"]["node_id"]

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "What are the shipping box dimensions?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("side_question") is True
    assert "Carton Width" in data.get("assistant_message", "")
    assert data["ticket"]["current_node"]["node_id"] == before_node


def test_submit_answer_returns_tracking_summary(client, monkeypatch):
    from delivery_lookup import TrackingSnapshot

    def fake_lookup_tracking(tn, domain):
        return TrackingSnapshot(
            source="track123",
            available=True,
            status="IN_TRANSIT",
            tracking_number=tn,
        )

    monkeypatch.setattr("delivery_lookup.lookup_by_tracking_number", fake_lookup_tracking)
    monkeypatch.setattr("delivery_lookup.lookup_by_order_or_email", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "delivery_lookup.format_warranty_tracking_message",
        lambda snap, **_k: f"Status: {snap.status}",
    )
    monkeypatch.setattr("delivery_lookup.persist_snapshot", lambda *_a, **_k: None)

    session_id = "cust-api-tracking"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "delivery", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "has_tracking"},
    )
    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "1Z999AA10123456784"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["tracking_summary"]["available"] is True
    assert "IN_TRANSIT" in data["tracking_summary"]["message"]
    assert data["ticket"]["current_node"]["node_id"] == "delivery_visible_damage_q"


def test_get_session_returns_ticket_after_admin_terminal(client, monkeypatch):
    from delivery_lookup import TrackingSnapshot

    unavailable = TrackingSnapshot(source="unavailable", available=False)

    monkeypatch.setattr("delivery_lookup.lookup_by_order_or_email", lambda *_a, **_k: unavailable)
    monkeypatch.setattr(
        "delivery_lookup.lookup_by_tracking_number",
        lambda *_a, **_k: unavailable,
    )
    monkeypatch.setattr(
        "delivery_lookup.format_warranty_tracking_message",
        lambda snap, **_k: "lookup pending",
    )
    monkeypatch.setattr("delivery_lookup.persist_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "warranty_email.send_warranty_transcript_email",
        lambda **_k: True,
    )

    session_id = "cust-api-terminal"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "delivery", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "no_tracking"},
    )
    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "customer@example.com"},
    )
    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "yes_box_damage"},
    )
    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "signed_cleared"},
    )

    session = client.get(f"/api/v1/warranty/session/{session_id}")
    assert session.status_code == 200
    ticket = session.json()["ticket"]
    assert ticket is not None
    assert ticket["status"] == "awaiting_admin_review"
    assert ticket["current_node"]["is_terminal"] is True


def test_notify_email_endpoint_sends_transcript(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_email.send_warranty_transcript_email",
        lambda **_k: True,
    )

    resp = client.post(
        "/api/v1/warranty/session/sess-notify/notify-email",
        json={
            "message": "Please contact me at buyer@example.com",
            "chat_messages": [
                {"role": "user", "content": "My chair is broken"},
                {"role": "assistant", "content": "Sorry to hear that."},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sent"] is True
    assert data["customer_email"] == "buyer@example.com"


def test_submit_answer_notifies_on_email_in_text(client, monkeypatch):
    from delivery_lookup import TrackingSnapshot

    monkeypatch.setattr(
        "warranty_email.send_warranty_transcript_email",
        lambda **_k: True,
    )
    unavailable = TrackingSnapshot(source="unavailable", available=False)
    monkeypatch.setattr("delivery_lookup.lookup_by_order_or_email", lambda *_a, **_k: unavailable)
    monkeypatch.setattr(
        "delivery_lookup.format_warranty_tracking_message",
        lambda snap, **_k: "lookup pending",
    )
    monkeypatch.setattr("delivery_lookup.persist_snapshot", lambda *_a, **_k: None)

    session_id = "cust-api-email-notify"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "delivery", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]
    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "no_tracking"},
    )

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "follow up at buyer@example.com"},
    )
    assert resp.status_code == 200
    assert resp.json().get("email_notified") is True


def test_natural_start_clarifying_when_unclear(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_nlp.interpret_issue_type",
        lambda _text: None,
    )

    session_id = "cust-api-natural-clarify"
    _register_model(client, session_id, model="3D LTX")
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/natural-start",
        json={"message": "hello there", "domain": "osaki.com"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("side_question") is True
    assert "installation" in data["assistant_message"].lower()
    assert data["ticket"]["current_node"]["node_id"] == "issue_type"
    assert data["ticket"]["issue_type"] == ""


def test_model_then_issue_via_natural_start(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_nlp.interpret_issue_type",
        lambda text: "delivery" if "tracking" in text.lower() else None,
    )

    session_id = "cust-api-model-then-issue"
    _register_model(client, session_id, model="3D LTX")
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/natural-start",
        json={"message": "Where is my tracking number?", "domain": "osaki.com"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ticket"]["issue_type"] == "delivery"
    assert data["ticket"]["current_node"]["node_id"] == "delivery_tracking_q"


def test_submit_answer_clarifying_on_ambiguous(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_nlp.interpret_warranty_answer",
        lambda _node, _text: None,
    )

    session_id = "cust-api-clarify-answer"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "delivery", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "maybe something vague xyz"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("side_question") is True
    assert "choices" in data["assistant_message"].lower() or "tap one" in data["assistant_message"].lower()
    assert data["ticket"]["current_node"]["node_id"] == "delivery_tracking_q"


def test_natural_start_maps_issue_type(client, monkeypatch):
    monkeypatch.setattr(
        "warranty_nlp.interpret_issue_type",
        lambda _text: "defect",
    )

    session_id = "cust-api-natural-start"
    _register_model(client, session_id)
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/natural-start",
        json={"message": "my massage chair won't turn on", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["nlp_interpreted"] is True
    assert data["interpreted_issue_type"] == "defect"
    assert data["ticket"]["issue_type"] == "defect"
    assert data["ticket"]["current_node"]["node_id"] == "defect_problem_type"


def test_submit_answer_nlp_maps_natural_language(client):
    session_id = "cust-api-nlp-answer"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "delivery", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "I don't have a tracking number"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("nlp_interpreted") is True
    assert data["ticket"]["current_node"]["node_id"] == "delivery_get_name"


def test_register_model_rejects_issue_description(client):
    session_id = "cust-api-symptom-as-model"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": "footrest air not inflating", "domain": "osaki.com"},
    )
    assert resp.status_code == 422
    assert "problem description" in resp.json()["detail"].lower()


def test_restart_session_abandons_active_ticket(client):
    """Restart should close the in-progress ticket so the next call sees no
    active session, allowing a clean restart without stale model/issue data."""
    session_id = "cust-api-restart"
    _register_model(client, session_id)

    # Before restart: an active in_progress ticket exists.
    status_before = client.get(f"/api/v1/warranty/session/{session_id}").json()
    assert status_before["ticket"] is not None
    assert status_before["ticket"]["status"] == "in_progress"

    restart = client.post(
        f"/api/v1/warranty/session/{session_id}/restart",
        json={"domain": "osaki.com"},
    )
    assert restart.status_code == 200
    body = restart.json()
    assert body["restarted"] is True
    assert body["ticket"] is None
    assert body["closed_ticket_count"] >= 1

    # After restart: GET session shows no active ticket.
    status_after = client.get(f"/api/v1/warranty/session/{session_id}").json()
    assert status_after["ticket"] is None


def test_restart_is_idempotent_when_no_active_ticket(client):
    """Calling restart with no open ticket should still succeed with count=0."""
    session_id = "cust-api-restart-empty"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/restart",
        json={"domain": "osaki.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["restarted"] is True
    assert body["ticket"] is None


def test_go_back_api_restores_issue_type(client):
    session_id = "cust-api-back"
    reg = _register_model(client, session_id)
    ticket_id = reg["ticket"]["ticket_id"]

    qs = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    assert qs.status_code == 200
    assert qs.json()["ticket"]["current_node"]["node_id"] == "defect_problem_type"

    back = client.post(f"/api/v1/warranty/{ticket_id}/back")
    assert back.status_code == 200
    body = back.json()
    assert body.get("went_back") is True
    ticket = body["ticket"]
    assert ticket["current_node"]["node_id"] == "issue_type"
    assert ticket["issue_type"] == ""
    assert ticket["model_name"] == "OS-4000T"
    assert ticket["can_go_back"] is True
    assert ticket["ready_for_issue_type"] is True


def test_go_back_api_rejects_terminal_ticket(client):
    session_id = "cust-api-back-terminal"
    reg = _register_model(client, session_id)
    ticket_id = reg["ticket"]["ticket_id"]

    client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "installation", "domain": "osaki.com"},
    )
    client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "general_setup"},
    )
    status = client.get(f"/api/v1/warranty/session/{session_id}").json()
    assert status["ticket"]["current_node"]["is_terminal"] is True

    back = client.post(f"/api/v1/warranty/{ticket_id}/back")
    assert back.status_code == 422


def _start_installation_terminal(client, session_id: str) -> str:
    reg = _register_model(client, session_id)
    ticket_id = reg["ticket"]["ticket_id"]
    client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "installation", "domain": "osaki.com"},
    )
    terminal = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "general_setup"},
    )
    assert terminal.status_code == 200
    assert terminal.json()["ticket"]["current_node"]["is_terminal"] is True
    return ticket_id


def test_troubleshooting_progress_persists_before_team_review(client):
    session_id = "cust-api-troubleshooting-unresolved"
    ticket_id = _start_installation_terminal(client, session_id)

    completed = client.post(
        f"/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
        json={"outcome": "steps_completed"},
    )
    assert completed.status_code == 200
    assert completed.json()["self_service_resolved"] is False

    hydrated = client.get(f"/api/v1/warranty/session/{session_id}").json()
    assert hydrated["ticket"]["troubleshooting_outcome"] == "steps_completed"

    unresolved = client.post(
        f"/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
        json={"outcome": "unresolved"},
    )
    assert unresolved.status_code == 200

    hydrated = client.get(f"/api/v1/warranty/session/{session_id}").json()
    assert hydrated["ticket"]["troubleshooting_outcome"] == "unresolved"
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        history = ticket.get_collected()["troubleshooting_history"]
        assert [entry["outcome"] for entry in history] == [
            "steps_completed",
            "unresolved",
        ]


def test_resolved_troubleshooting_closes_shipping_review_ticket(client):
    session_id = "cust-api-troubleshooting-resolved"
    ticket_id = _start_installation_terminal(client, session_id)

    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.admin_note = "Existing admin context must be preserved."

    client.post(
        f"/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
        json={"outcome": "steps_completed"},
    )
    resolved = client.post(
        f"/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
        json={"outcome": "resolved"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "resolved"
    assert resolved.json()["self_service_resolved"] is True
    duplicate = client.post(
        f"/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
        json={"outcome": "resolved"},
    )
    assert duplicate.status_code == 200

    hydrated = client.get(f"/api/v1/warranty/session/{session_id}").json()
    assert hydrated["ticket"] is None
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        assert ticket.status == "resolved"
        assert ticket.admin_decision == "self_resolved"
        assert "Existing admin context" in ticket.admin_note
        assert "[system] Customer confirmed" in ticket.admin_note


def test_troubleshooting_outcome_requires_terminal_and_valid_value(client):
    session_id = "cust-api-troubleshooting-invalid"
    reg = _register_model(client, session_id)
    ticket_id = reg["ticket"]["ticket_id"]

    too_early = client.post(
        f"/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
        json={"outcome": "resolved"},
    )
    assert too_early.status_code == 409

    invalid = client.post(
        f"/api/v1/warranty/{ticket_id}/troubleshooting-outcome",
        json={"outcome": "ship_replacement"},
    )
    assert invalid.status_code == 422

    terminal_ticket_id = _start_installation_terminal(
        client,
        "cust-api-troubleshooting-sequence",
    )
    skipped_steps = client.post(
        f"/api/v1/warranty/{terminal_ticket_id}/troubleshooting-outcome",
        json={"outcome": "resolved"},
    )
    assert skipped_steps.status_code == 409


def test_customer_note_appends_to_collected_data(client):
    """Customer follow-up notes should append to collected_data.customer_notes."""
    session_id = "cust-api-note"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    resp1 = client.post(
        f"/api/v1/warranty/{ticket_id}/customer-note",
        json={"note": "I also noticed a rattling sound."},
    )
    assert resp1.status_code == 200
    notes1 = resp1.json()["customer_notes"]
    assert len(notes1) == 1
    assert notes1[0]["text"] == "I also noticed a rattling sound."
    assert notes1[0]["created_at"]

    resp2 = client.post(
        f"/api/v1/warranty/{ticket_id}/customer-note",
        json={"note": "It happens only on recline."},
    )
    assert resp2.status_code == 200
    notes2 = resp2.json()["customer_notes"]
    assert len(notes2) == 2
    assert notes2[1]["text"] == "It happens only on recline."


def test_customer_note_rejects_empty_and_too_long(client):
    session_id = "cust-api-note-invalid"
    _register_model(client, session_id)
    start = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    ticket_id = start.json()["ticket"]["ticket_id"]

    empty = client.post(
        f"/api/v1/warranty/{ticket_id}/customer-note",
        json={"note": "   "},
    )
    assert empty.status_code == 422

    long_text = "x" * 1500
    too_long = client.post(
        f"/api/v1/warranty/{ticket_id}/customer-note",
        json={"note": long_text},
    )
    assert too_long.status_code == 422


def test_customer_note_unknown_ticket_returns_404(client):
    resp = client.post(
        "/api/v1/warranty/does-not-exist/customer-note",
        json={"note": "hello"},
    )
    assert resp.status_code == 404


def test_smart_start_sets_model_from_hint(client, monkeypatch):
    import warranty_intake as wi  # noqa: WPS433

    monkeypatch.setattr(
        wi,
        "extract_workflow_prefill",
        lambda **kwargs: {
            "answer_keys": ["warranty", "defect", "air", "footrest"],
            "model_name": "OS-4000T",
            "confidence": "high",
            "summary": "Footrest air not inflating on OS-4000T.",
            "source": "llm",
        },
    )

    session_id = "cust-api-smart-model-hint"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/smart-start",
        json={"message": "OS-4000T footrest air not inflating", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticket"]["model_name"] == "OS-4000T"
    assert data["smart_start"]["model_name_hint"] == "OS-4000T"
    assert len(data["smart_start"]["applied_keys"]) >= 3


def test_smart_start_model_only_stays_on_issue_type(client, monkeypatch):
    import warranty_intake as wi  # noqa: WPS433

    monkeypatch.setattr(
        wi,
        "extract_workflow_prefill",
        lambda **kwargs: {
            "answer_keys": ["warranty"],
            "model_name": "Maestro",
            "confidence": "high",
            "summary": "Chair model: Maestro.",
            "source": "model_only",
        },
    )

    session_id = "cust-api-smart-model-only"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/smart-start",
        json={"message": "Maestro", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Maestro" in data["ticket"]["model_name"]
    assert data["ticket"]["current_node"]["node_id"] == "issue_type"
    assert not data["ticket"].get("issue_type")
    assert data["smart_start"]["applied_keys"] == ["warranty"]
    assert data.get("step_enrichment") is None


def test_smart_start_empty_prefill_does_not_default_to_defect(client, monkeypatch):
    import warranty_intake as wi  # noqa: WPS433

    monkeypatch.setattr(
        wi,
        "extract_workflow_prefill",
        lambda **kwargs: {
            "answer_keys": [],
            "model_name": "",
            "confidence": "low",
            "summary": "",
            "source": "empty",
        },
    )

    session_id = "cust-api-smart-no-defect-fallback"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/smart-start",
        json={"message": "Where is my order?", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticket"]["current_node"]["node_id"] == "issue_type"
    assert not data["ticket"].get("issue_type")
    assert "defect" not in data["smart_start"]["applied_keys"]


def test_smart_start_routing_confirmation_when_issue_inferred(client, monkeypatch):
    import warranty_intake as wi  # noqa: WPS433

    monkeypatch.setattr(
        wi,
        "extract_workflow_prefill",
        lambda **kwargs: {
            "answer_keys": ["warranty", "delivery"],
            "model_name": "",
            "confidence": "high",
            "summary": "Customer asking about shipping status.",
            "source": "llm",
        },
    )

    session_id = "cust-api-smart-routing-confirm"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/smart-start",
        json={"message": "Where is my FedEx shipment?", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    data = resp.json()
    confirm = data["smart_start"]["routing_confirmation"]
    assert confirm["inferred_issue_type"] == "delivery"
    assert "delivery" in confirm["message"].lower()


def _start_defect_air_feet(client, session_id: str, model: str = "3D LTX") -> str:
    _register_model(client, session_id, model=model)
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    ticket_id = resp.json()["ticket"]["ticket_id"]
    for answer in ("air", "feet_calves", "never_worked"):
        resp = client.post(
            f"/api/v1/warranty/{ticket_id}/answer",
            json={"answer": answer},
        )
        assert resp.status_code == 200, resp.text
    return ticket_id


def test_midflow_does_not_hijack_workflow_answer_keys(client):
    session_id = "cust-api-gate-no-hijack"
    _register_model(client, session_id, model="3D LTX")
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    ticket_id = resp.json()["ticket"]["ticket_id"]

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "air"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("side_question") is not True
    assert data["ticket"]["current_node"]["node_id"] == "defect_air_location"


def test_error_code_gate_intercepts_before_terminal(client):
    session_id = "cust-api-gate-intercept"
    ticket_id = _start_defect_air_feet(client, session_id)

    data = client.get(f"/api/v1/warranty/session/{session_id}").json()
    node = data["ticket"]["current_node"]
    assert node["node_id"] == "defect_error_code_visible_q"
    assert node["is_terminal"] is False
    assert "error code" in (data.get("assistant_message") or node["prompt"]).lower()


def test_error_code_gate_accepts_typed_c6(client):
    session_id = "cust-api-gate-type-c6"
    ticket_id = _start_defect_air_feet(client, session_id)

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "C6"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ticket"]["current_node"]["node_id"] == "defect_air_pump_terminal"
    assert data["ticket"]["current_node"]["is_terminal"] is True
    assert data.get("nlp_interpreted") is True
    assert "C6" in (data.get("assistant_message") or "")


def test_error_code_gate_midflow_side_question(client):
    session_id = "cust-api-gate-midflow"
    _register_model(client, session_id, model="3D LTX")
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    ticket_id = resp.json()["ticket"]["ticket_id"]

    resp = client.post(
        f"/api/v1/warranty/{ticket_id}/answer",
        json={"answer": "My display shows error code C6"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("side_question") is True
    assert "C6" in data["assistant_message"]
    assert data["ticket"]["current_node"]["node_id"] == "defect_problem_type"
