"""
Deploy safety net: privacy contract with WARRANTY_REQUIRE_CHAT_PRIVACY=1.

Most unit tests disable privacy via tests/conftest.py. This module opts back
on and asserts basic help requires consent while contact email is optional,
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


def test_privacy_gate_requires_consent_but_not_email_for_basic_help(privacy_client):
    """Recording consent remains mandatory; basic help does not require email."""
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
    assert resp.status_code == 200, resp.text
    assert resp.json()["privacy"]["email_required_before_help"] is False
    assert resp.json()["privacy"]["consent_recorded"] is True
    assert resp.json()["privacy"]["contact_email_available"] is False

    email_resp = client.post(
        f"/api/v1/warranty/session/{session_id}/contact-email",
        json={"customer_email": "buyer@example.com", "skipped": False},
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


def test_optional_email_skip_is_acknowledged_without_claiming_identity(privacy_client):
    client = privacy_client
    assert client.post("/api/v1/warranty/session/skip/contact-email", json={"skipped": True}).status_code == 403
    consent = client.post("/api/v1/warranty/session/skip/consent", json={"domain": "osakiusa.com"})
    assert consent.status_code == 200
    assert consent.json()["privacy"]["email_gate_status"] == "pending"
    skipped = client.post("/api/v1/warranty/session/skip/contact-email", json={"skipped": True})
    assert skipped.status_code == 200, skipped.text
    assert skipped.json()["skipped"] is True
    assert skipped.json()["email_saved"] is False
    assert skipped.json()["privacy"]["contact_email_available"] is False
    state = client.get("/api/v1/warranty/session/skip").json()
    assert state["privacy"]["email_gate_status"] == "skipped"
    assert state["privacy"]["email_required_before_help"] is False
    # Tracking still requires both identifiers and a syntactically valid email.
    assert client.post("/api/v1/warranty/session/skip/delivery-tracking", json={"order_number": "12345", "email": ""}).status_code == 422
