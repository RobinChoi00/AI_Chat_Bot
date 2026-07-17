"""
tests/test_warranty_consent.py
================================
Live-chat privacy consent recording and ticket attachment.
"""

import sys
import uuid
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
from warranty_consent import record_chat_consent  # noqa: E402


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
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
    from warranty_router import router as warranty_router  # noqa: WPS433

    app = FastAPI()
    app.include_router(warranty_router)
    return TestClient(app)


def test_record_consent_endpoint(client):
    session_id = f"sess-consent-{uuid.uuid4().hex[:8]}"
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/consent",
        json={"domain": "www.osakiusa.com", "policy_store": "www.osakiusa.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["consent_recorded"] is True
    assert body["accepted_at"]
    assert body["policy_store"] == "www.osakiusa.com"


def test_consent_copied_to_ticket_collected_data():
    session_id = f"sess-consent-ticket-{uuid.uuid4().hex[:8]}"
    accepted_at = record_chat_consent(
        session_id,
        domain="www.osakiusa.com",
        policy_store="www.osakiusa.com",
    )

    ticket_id, _node = wf.WarrantyEngine.start_session(session_id, "www.osakiusa.com")
    ticket = wf.WarrantyEngine.get_ticket(ticket_id)
    assert ticket is not None
    collected = ticket.get_collected()
    stored_at = collected.get("chat_consent_accepted_at")
    assert stored_at
    assert stored_at.startswith(accepted_at.isoformat()[:19])
    assert collected.get("chat_consent_policy_store") == "www.osakiusa.com"
    assert collected.get("chat_consent_domain") == "www.osakiusa.com"
