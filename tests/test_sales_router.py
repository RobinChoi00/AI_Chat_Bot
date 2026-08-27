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


def test_recommend_without_hints_asks_height(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={"session_id": "s-r", "message": "can you recommend a chair"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "recommend"
    assert body["handoff"] is False
    assert "height" in body["reply"].lower()
    assert "budget range" not in body["reply"].lower()
    payloads = {q["payload"] for q in body["quick_replies"]}
    assert any(p.startswith("recommend:height:") for p in payloads)


def test_recommend_budget_payload_ignored_still_asks_height(client):
    """Legacy budget payloads no longer gate the flow — still fit-first."""
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
    assert "height" in body["reply"].lower()
    payloads = {q["payload"] for q in body["quick_replies"]}
    assert any(p.startswith("recommend:height:") for p in payloads)
    assert not any(p.startswith("recommend:budget:") for p in payloads)


def test_recommend_guided_flow_returns_tiered_picks(client):
    sid = "s-case-flow"
    body = None
    for payload in (
        "recommend:height:petite",
        "recommend:weight:le180",
        "recommend:space:none",
        "recommend:goal:neck",
    ):
        resp = client.post(
            "/api/v1/sales/chat",
            json={
                "session_id": sid,
                "message": "",
                "payload": payload,
                "domain": "osakimassagechair.com",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
    assert body["intent"] == "recommend"
    reply = body["reply"].lower()
    assert "value" in reply
    assert "mid-range" in reply or "mid" in reply
    assert "premium" in reply
    assert "titan pro cura" in reply or "osaki" in reply
    assert "primary pick" not in reply
    payloads = {q["payload"] for q in body["quick_replies"]}
    assert not any(p.startswith("recommend:budget:") for p in payloads)


def test_recommend_budget_after_fit_still_returns_tiers(client):
    """Budget refine payloads are ignored — always Value/Mid/Premium list."""
    sid = "s-budget-ignored"
    body = None
    for payload in (
        "recommend:height:petite",
        "recommend:weight:le180",
        "recommend:space:none",
        "recommend:goal:neck",
        "recommend:budget:under_3000",
    ):
        resp = client.post(
            "/api/v1/sales/chat",
            json={
                "session_id": sid,
                "message": "",
                "payload": payload,
                "domain": "osakimassagechair.com",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
    reply = body["reply"].lower()
    assert "value" in reply
    assert "premium" in reply
    assert "primary pick" not in reply


def test_recommend_osakiusa_uses_titan_usa_book(client):
    sid = "s-usa-titan-book"
    body = None
    for payload in (
        "recommend:height:petite",
        "recommend:weight:le180",
        "recommend:space:none",
        "recommend:goal:neck",
    ):
        resp = client.post(
            "/api/v1/sales/chat",
            json={
                "session_id": sid,
                "message": "",
                "payload": payload,
                "domain": "osakiusa.com",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
    assert "Titan eCabin" in body["reply"] or "value" in body["reply"].lower()
    assert "Osaki OS-Champ" not in body["reply"]
    assert "primary pick" not in body["reply"].lower()


def test_recommend_demotes_oos_with_live_stock(client, monkeypatch):
    """When primary is OOS and an alt is in stock, lead with the in-stock model."""

    class _Snap:
        def __init__(self, handle, in_stock):
            self.handle = handle
            self.title = handle
            self.status = "active"
            self.available_for_sale = in_stock
            self.total_inventory = 5 if in_stock else 0
            self.source = "shopify"

        @property
        def in_stock(self):
            return self.available_for_sale and self.total_inventory > 0

        @property
        def is_low(self):
            return False

    def _fake_stock(handle, domain=None, title=None, timeout=8.0):
        # Champ OOS; force alts in stock when resolved.
        if "champ" in (handle or "").lower():
            return _Snap(handle, False)
        return _Snap(handle, True)

    monkeypatch.setattr("sales_agent.fetch_live_stock", _fake_stock)
    monkeypatch.setattr("sales_shopify_stock.fetch_live_stock", _fake_stock)

    sid = "s-stock-demote"
    body = None
    for payload in (
        "recommend:height:petite",
        "recommend:weight:le180",
        "recommend:space:none",
        "recommend:goal:neck",
    ):
        resp = client.post(
            "/api/v1/sales/chat",
            json={
                "session_id": sid,
                "message": "",
                "payload": payload,
                "domain": "osakimassagechair.com",
            },
        )
        assert resp.status_code == 200
        body = resp.json()

    assert body is not None
    assert "in stock" in body["reply"].lower() or "out of stock" in body["reply"].lower()
    assert "shopify.inventory" in body.get("tools_used", [])
    payloads = {q["payload"] for q in body["quick_replies"]}
    assert "lead:save_pick" in payloads
    assert "products/" in body["reply"] or "https://" in body["reply"]
    assert "value" in body["reply"].lower()
    assert "https://" in body["reply"]
    assert "Open these links" not in body["reply"]


def test_email_me_this_pick_captures_lead_with_summary(client, monkeypatch):
    monkeypatch.setattr("sales_agent.fetch_live_stock", lambda *a, **k: None)
    monkeypatch.setattr("sales_shopify_stock.fetch_live_stock", lambda *a, **k: None)

    sid = "s-email-pick"
    for payload in (
        "recommend:height:petite",
        "recommend:weight:le180",
        "recommend:space:none",
        "recommend:goal:neck",
    ):
        resp = client.post(
            "/api/v1/sales/chat",
            json={
                "session_id": sid,
                "message": "",
                "payload": payload,
                "domain": "osakiusa.com",
            },
        )
        assert resp.status_code == 200

    ask = client.post(
        "/api/v1/sales/chat",
        json={"session_id": sid, "message": "", "payload": "lead:save_pick", "domain": "osakiusa.com"},
    )
    assert ask.status_code == 200
    assert "email" in ask.json()["reply"].lower()

    done = client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": sid,
            "message": "buyer@example.com",
            "domain": "osakiusa.com",
        },
    )
    assert done.status_code == 200
    body = done.json()
    assert body["handoff"] is True
    assert "buyer@example.com" in body["reply"]
    assert "lead.capture" in body.get("tools_used", [])


def test_free_text_skips_secondary_questions(client, monkeypatch):
    """Core fit + space in one message → recommend without asking intensity/foot."""
    monkeypatch.setattr("sales_agent.fetch_live_stock", lambda *a, **k: None)

    resp = client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "s-skip-secondary",
            "message": "I'm 6'2, 220 lb, lower back pain, no space issue, under $5k",
            "domain": "osakiusa.com",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "recommend"
    reply = body["reply"].lower()
    assert "value" in reply
    assert "premium" in reply
    assert "primary pick" not in reply
    assert "intensity" not in body["reply"].lower() or "defaults" in body["reply"].lower()
    assert "preferred **massage intensity**" not in body["reply"].lower()
    payloads = {q["payload"] for q in body["quick_replies"]}
    assert "lead:save_pick" in payloads


def test_free_text_partial_still_asks_weight(client):
    resp = client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "s-ask-weight",
            "message": "6'2, back pain, under $5k",
            "domain": "osakiusa.com",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "recommend"
    assert "weight" in body["reply"].lower()
    assert "primary pick" not in body["reply"].lower()


def test_tiered_recommend_includes_product_links_when_in_stock(client, monkeypatch):
    class _Snap:
        def __init__(self, handle):
            self.handle = handle
            self.title = handle
            self.status = "active"
            self.available_for_sale = True
            self.total_inventory = 8
            self.source = "shopify"

        @property
        def in_stock(self):
            return True

        @property
        def is_low(self):
            return False

    monkeypatch.setattr(
        "sales_agent.fetch_live_stock",
        lambda handle, domain=None, title=None, timeout=8.0: _Snap(handle or "x"),
    )

    sid = "s-convert"
    body = None
    for payload in (
        "recommend:height:petite",
        "recommend:weight:le180",
        "recommend:space:none",
        "recommend:goal:neck",
    ):
        resp = client.post(
            "/api/v1/sales/chat",
            json={
                "session_id": sid,
                "message": "",
                "payload": payload,
                "domain": "osakiusa.com",
            },
        )
        assert resp.status_code == 200
        body = resp.json()

    assert body is not None
    assert "value" in body["reply"].lower()
    assert "https://" in body["reply"]
    labels = {q["label"] for q in body["quick_replies"]}
    assert any("Email me" in label for label in labels)
    assert "shopify.inventory" in body.get("tools_used", [])

    showroom = client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": sid,
            "message": "",
            "payload": "cta:showroom",
            "domain": "osakiusa.com",
        },
    )
    assert showroom.status_code == 200
    assert "Carrollton" in showroom.json()["reply"]


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
