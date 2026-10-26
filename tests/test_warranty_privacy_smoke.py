"""
Deploy safety net: privacy contract with WARRANTY_REQUIRE_CHAT_PRIVACY=1.

Most unit tests disable privacy via tests/conftest.py. This module opts back
on and asserts browser chat requires consent and a saved email before help,
and that public responses mask customer email.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402
import warranty_workflow as wf  # noqa: E402


@pytest.fixture
def privacy_client(monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1")

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

    from warranty_router import router  # noqa: WPS433

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.parametrize("action,body", [
    ("register-model", {"model": "OS-4000T"}),
    ("confirm-model", {"confirmed": True}),
    ("quick-start", {"issue_type": "installation"}),
    ("natural-start", {"message": "Help with setup"}),
    ("smart-start", {"message": "Help with setup"}),
    ("handoff", {"message": "Talk to a person"}),
    ("delivery-tracking", {"order_number": "12345", "email": "buyer@example.com"}),
])
def test_every_chat_entry_requires_consent_and_email(privacy_client, action, body):
    session_id = f"privacy-smoke-{action}"
    endpoint = f"/api/v1/warranty/session/{session_id}/{action}"
    client = privacy_client
    assert client.post(endpoint, json=body).status_code == 403
    consent = client.post(
        f"/api/v1/warranty/session/{session_id}/consent",
        json={"domain": "osaki.com"},
    )
    assert consent.status_code == 200
    assert consent.json()["privacy"]["email_required_before_help"] is True
    blocked = client.post(endpoint, json=body)
    assert blocked.status_code == 403, blocked.text
    assert "email" in blocked.json()["detail"]
    assert client.get(f"/api/v1/warranty/session/{session_id}").json()["ticket"] is None


def test_saved_email_unlocks_chat_and_is_copied_to_ticket(privacy_client):
    session_id = "privacy-smoke-register"
    client = privacy_client

    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": "OS-4000T", "domain": "osaki.com"},
    )
    assert resp.status_code == 403

    client.post(
        f"/api/v1/warranty/session/{session_id}/consent",
        json={"domain": "osaki.com", "policy_store": "osakiusa.com"},
    )
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": "OS-4000T", "domain": "osaki.com"},
    )
    assert resp.status_code == 403, resp.text

    email_resp = client.post(
        f"/api/v1/warranty/session/{session_id}/contact-email",
        json={"customer_email": " Buyer@Example.com ", "skipped": False},
    )
    assert email_resp.status_code == 200
    body = email_resp.json()
    assert body["email_saved"] is True
    assert body["customer_email"] == "b***r@example.com"
    assert "buyer@example.com" not in (body.get("customer_email") or "")

    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": "OS-4000T", "domain": "osaki.com"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ticket"]["model_name"] == "OS-4000T"
    assert resp.json()["privacy"]["email_required_before_help"] is True
    assert resp.json()["privacy"]["consent_recorded"] is True
    assert resp.json()["privacy"]["email_gate_status"] == "provided"
    assert resp.json()["privacy"]["contact_email_available"] is True
    ticket = wf.WarrantyEngine.get_ticket(resp.json()["ticket"]["ticket_id"])
    assert ticket.get_collected()["customer_contact_email"] == "buyer@example.com"


def test_skip_is_rejected_and_cannot_clear_a_saved_email(privacy_client):
    client = privacy_client
    assert client.post("/api/v1/warranty/session/skip/contact-email", json={"skipped": True}).status_code == 403
    consent = client.post("/api/v1/warranty/session/skip/consent", json={"domain": "osakiusa.com"})
    assert consent.status_code == 200
    assert consent.json()["privacy"]["email_gate_status"] == "pending"
    skipped = client.post("/api/v1/warranty/session/skip/contact-email", json={"skipped": True})
    assert skipped.status_code == 422, skipped.text
    state = client.get("/api/v1/warranty/session/skip").json()
    assert state["privacy"]["email_gate_status"] == "pending"
    assert state["privacy"]["contact_email_available"] is False
    assert client.post("/api/v1/warranty/session/skip/contact-email", json={"customer_email": "buyer@example.com"}).status_code == 200
    assert client.post("/api/v1/warranty/session/skip/contact-email", json={"skipped": True}).status_code == 422
    state = client.get("/api/v1/warranty/session/skip").json()
    assert state["privacy"]["email_gate_status"] == "provided"
    assert state["privacy"]["contact_email_available"] is True


@pytest.mark.parametrize("email", ["", "not-an-email", "Email buyer@example.com", "buyer@example.com extra"])
def test_invalid_email_cannot_unlock_chat(privacy_client, email):
    client = privacy_client
    client.post("/api/v1/warranty/session/invalid/consent", json={"domain": "osaki.com"})
    response = client.post("/api/v1/warranty/session/invalid/contact-email", json={"customer_email": email})
    assert response.status_code == 422
    state = client.get("/api/v1/warranty/session/invalid").json()
    assert state["privacy"]["email_gate_status"] == "pending"
    assert state["privacy"]["contact_email_available"] is False
    assert client.post("/api/v1/warranty/session/invalid/quick-start", json={"issue_type": "installation"}).status_code == 403


@pytest.mark.parametrize("email,status,expected", [
    ("buyer@example.com", None, "provided"),
    (None, "provided", "pending"),
    ("malformed", "provided", "pending"),
    (None, "skipped", "skipped"),
])
def test_existing_session_needs_a_valid_saved_email(privacy_client, email, status, expected):
    client = privacy_client
    client.post("/api/v1/warranty/session/existing/consent", json={"domain": "osaki.com"})
    with wm.warranty_db_session() as db:
        row = db.query(wm.WarrantyChatConsent).filter_by(session_id="existing").one()
        row.contact_email = email
        row.email_gate_status = status
    state = client.get("/api/v1/warranty/session/existing").json()
    assert state["privacy"]["email_gate_status"] == expected
    response = client.post("/api/v1/warranty/session/existing/register-model", json={"model": "OS-4000T"})
    assert response.status_code == (200 if expected == "provided" else 403), response.text


def test_disabled_privacy_mode_retains_its_explicit_opt_out(privacy_client, monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")
    client = privacy_client
    client.post("/api/v1/warranty/session/disabled/consent", json={"domain": "osaki.com"})
    skipped = client.post("/api/v1/warranty/session/disabled/contact-email", json={"skipped": True})
    assert skipped.status_code == 200
    assert skipped.json()["privacy"]["email_required_before_help"] is False
    assert skipped.json()["privacy"]["contact_email_available"] is False
    response = client.post("/api/v1/warranty/session/disabled/register-model", json={"model": "OS-4000T"})
    assert response.status_code == 200, response.text


def test_phone_ticket_does_not_require_browser_email(privacy_client):
    from types import SimpleNamespace
    from warranty_router import _require_web_chat_privacy_for_ticket

    _require_web_chat_privacy_for_ticket(SimpleNamespace(session_id="phone-call", get_collected=lambda: {"channel": "phone"}))
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as blocked:
        _require_web_chat_privacy_for_ticket(SimpleNamespace(session_id="web-chat", get_collected=lambda: {}))
    assert blocked.value.status_code == 403


def test_legacy_ticket_cannot_rewind_before_email_is_saved(privacy_client, monkeypatch):
    client = privacy_client
    session_id = "legacy-rewind"
    client.post(f"/api/v1/warranty/session/{session_id}/consent", json={"domain": "osaki.com"})
    # Create the saved workflow using the former email-optional behavior.
    with monkeypatch.context() as optional:
        optional.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")
        registered = client.post(
            f"/api/v1/warranty/session/{session_id}/register-model",
            json={"model": "OS-4000T", "domain": "osaki.com"},
        )
        assert registered.status_code == 200, registered.text
        ticket_id = registered.json()["ticket"]["ticket_id"]
        started = client.post(
            f"/api/v1/warranty/session/{session_id}/quick-start",
            json={"issue_type": "defect", "domain": "osaki.com"},
        )
        assert started.status_code == 200, started.text

    original_turns = [turn.to_dict() for turn in wf.WarrantyEngine.get_turns(ticket_id)]
    assert original_turns
    blocked = client.post(f"/api/v1/warranty/{ticket_id}/back")
    assert blocked.status_code == 403, blocked.text
    assert [turn.to_dict() for turn in wf.WarrantyEngine.get_turns(ticket_id)] == original_turns
    assert str(wf.WarrantyEngine.get_ticket(ticket_id).current_node_id) == "defect_problem_type"

    saved = client.post(
        f"/api/v1/warranty/session/{session_id}/contact-email",
        json={"customer_email": "buyer@example.com"},
    )
    assert saved.status_code == 200
    rewound = client.post(f"/api/v1/warranty/{ticket_id}/back")
    assert rewound.status_code == 200, rewound.text
    assert rewound.json()["went_back"] is True
    assert rewound.json()["ticket"]["current_node"]["node_id"] == "issue_type"
    assert len(wf.WarrantyEngine.get_turns(ticket_id)) < len(original_turns)
