"""
tests/test_sales_router.py
==========================
HTTP-level tests for the Sales AI (Tidio) router.

We cover the guardrail contract end-to-end plus a couple of happy-path
turns. The in-memory SQLite fixture keeps every test isolated.
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


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    """Redirect the shared warranty engine at an in-memory SQLite instance."""
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

    import sales_router as sr  # noqa: WPS433

    monkeypatch.setattr(sr, "_ADMIN_API_KEY", "test-admin-key")
    yield


@pytest.fixture
def client():
    from sales_router import router  # noqa: WPS433

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Guardrails (must never answer directly, always handoff=True)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        ("my chair won't power on", "warranty_redirect"),
        ("I want to cancel my order", "cancel_refund"),
        ("please refund my purchase", "cancel_refund"),
        ("send a technician to my house", "parts_technician"),
        ("any discount available?", "discount"),
        ("when will it arrive?", "eta_shipping"),
        ("where is my order", "order_status"),
        ("do you ship to Alaska", "eta_shipping"),
        ("talk to a human", "human"),
    ],
)
def test_guardrail_intents_return_handoff(client, message, expected_intent):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-guard", "message": message, "domain": "osakiusa.com"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] == expected_intent
    assert body["handoff"] is True
    assert body["handoff_reason"] == expected_intent
    assert body["reply"], "handoff replies must not be empty"
    if expected_intent in ("eta_shipping", "order_status", "warranty_redirect"):
        assert "warranty" in body["reply"].lower()
        assert "zip" not in body["reply"].lower()
    if expected_intent == "discount":
        assert "%" not in body["reply"]
        assert "promo" not in body["reply"].lower()


# ---------------------------------------------------------------------------
# Happy paths (deterministic — no LLM)
# ---------------------------------------------------------------------------


def test_greeting_returns_menu(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-g", "message": "hello"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "greeting"
    assert body["handoff"] is False
    labels = [q["label"] for q in body["quick_replies"]]
    assert "Recommend a chair" in labels
    assert "Talk to a human" in labels


def test_unclear_returns_menu_not_a_guess(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-u", "message": "purple monkey dishwasher"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "unclear"
    assert body["handoff"] is False
    assert any(q["payload"] == "human" for q in body["quick_replies"])


def test_recommend_without_hints_offers_budget_bands(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-r", "message": "can you recommend a chair"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "recommend"
    assert body["handoff"] is False
    assert "budget" in body["reply"].lower()
    payloads = {q["payload"] for q in body["quick_replies"]}
    assert "recommend:budget:6000" in payloads
    assert "recommend:budget:2000" in payloads


def test_recommend_budget_band_payload_returns_near_target_picks(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "s-budget-band",
            "message": "",
            "payload": "recommend:budget:6000",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "recommend"
    assert "around $6,000" in body["reply"]
    # Should list priced picks, not the "what's your budget" prompt.
    assert "budget range" not in body["reply"].lower()
    assert "$" in body["reply"]


def test_menu_button_payload_returns_menu(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-p", "message": "", "payload": "menu"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "greeting"


def test_price_without_model_asks_for_it(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-price", "message": "how much is it?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "price"
    assert body["handoff"] is False
    reply = body["reply"].lower()
    assert "which model" in reply or "model name" in reply


def test_chat_requires_message_or_payload(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-empty", "message": ""},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Session resume + admin dashboards
# ---------------------------------------------------------------------------


def test_session_resume_returns_messages(client):
    client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-resume", "message": "hi"},
    )
    client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-resume", "message": "recommend a chair"},
    )

    resp = client.get("/api/v1/sales/session/s-resume")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session"]["session_id"] == "s-resume"
    roles = [m["role"] for m in body["messages"]]
    assert "user" in roles and "assistant" in roles


def test_session_resume_404_for_unknown(client):
    resp = client.get("/api/v1/sales/session/unknown")
    assert resp.status_code == 404


def test_admin_list_sessions_requires_key(client):
    resp = client.get("/admin/sales/sessions")
    assert resp.status_code in (401, 403)


def test_admin_list_sessions_returns_row_after_chat(client):
    client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-admin", "message": "hello"},
    )
    resp = client.get(
        "/admin/sales/sessions",
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    session_ids = [row["session_id"] for row in body["rows"]]
    assert "s-admin" in session_ids


# ---------------------------------------------------------------------------
# Lead capture
# ---------------------------------------------------------------------------


def test_lead_requires_email_or_phone(client):
    resp = client.post(
        "/api/v1/sales/lead",
        json={"session_id": "s-lead"},
    )
    assert resp.status_code == 422


def test_lead_rejects_invalid_email(client):
    resp = client.post(
        "/api/v1/sales/lead",
        json={"session_id": "s-lead", "email": "not-an-email"},
    )
    assert resp.status_code == 422


def test_lead_captures_and_masks_email(client, monkeypatch):
    import sales_router as sr

    called = {}

    def _fake_fire(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(sr, "_fire_lead_email", _fake_fire)

    resp = client.post(
        "/api/v1/sales/lead",
        json={
            "session_id": "s-lead",
            "email": "jane.doe@example.com",
            "interest_summary": "Interested in OS-Pro Maestro LE",
            "reason": "discount",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    # Public response never leaks a full email.
    assert "@example.com" in body["email"]
    assert body["email"] != "jane.doe@example.com"


def test_admin_leads_lists_captured_lead(client, monkeypatch):
    import sales_router as sr

    monkeypatch.setattr(sr, "_fire_lead_email", lambda **_: None)

    client.post(
        "/api/v1/sales/lead",
        json={
            "session_id": "s-admin-lead",
            "email": "hello@example.com",
            "reason": "human",
        },
    )
    resp = client.get(
        "/admin/sales/leads",
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    reasons = [row["reason"] for row in body["rows"]]
    assert "human" in reasons


# ---------------------------------------------------------------------------
# Mixed / real-world scenario
# ---------------------------------------------------------------------------


def test_mixed_defect_and_price_still_routes_to_warranty(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "s-mix",
            "message": "my chair is broken and how much is the OS-Pro Maestro?",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "warranty_redirect"
    assert body["handoff"] is True


def test_warranty_button_appears_when_relevant(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "s-w",
            "message": "my chair delivered damaged",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "warranty" in body["reply"].lower()
    labels = [q["label"].lower() for q in body["quick_replies"]]
    assert any("email" in label or "menu" in label for label in labels)
