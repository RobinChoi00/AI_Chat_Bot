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
    assert ticket["ready_for_issue_type"] is True
    assert ticket["current_node"]["node_id"] == "issue_type"


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


def test_submit_answer_rejects_box_size_question_at_delivery_lookup(client):
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
    assert resp.status_code == 422
    assert "order number" in resp.json()["detail"].lower()

    session = client.get(f"/api/v1/warranty/session/{session_id}")
    assert session.json()["ticket"]["current_node"]["node_id"] == "delivery_get_name"


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
    assert body["closed_ticket_count"] == 0


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
