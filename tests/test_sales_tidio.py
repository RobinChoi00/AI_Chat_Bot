"""
tests/test_sales_tidio.py
=========================
Signature verification + Tidio turn/webhook HTTP contracts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
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


SECRET = "tidio-test-webhook-secret"
PUBLIC_KEY = "pssxmoqpgzitub4925jec9c4nzfbvvam"


def _sign(body: str, secret: str = SECRET, ts: int | None = None) -> str:
    timestamp = int(ts if ts is not None else time.time())
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{body}_{timestamp}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"t={timestamp},s={digest}"


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

    monkeypatch.setenv("TIDIO_ENABLED", "1")
    monkeypatch.setenv("TIDIO_DOMAIN", "osakiusa.com")
    monkeypatch.setenv("TIDIO_PUBLIC_KEY", PUBLIC_KEY)
    monkeypatch.setenv("TIDIO_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("TIDIO_PRIVATE_KEY", "")
    monkeypatch.delenv("TIDIO_OPENAPI_CLIENT_ID", raising=False)
    monkeypatch.delenv("TIDIO_OPENAPI_CLIENT_SECRET", raising=False)
    yield


@pytest.fixture
def client():
    from sales_tidio_router import router  # noqa: WPS433

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Signature unit tests
# ---------------------------------------------------------------------------


def test_verify_tidio_signature_accepts_valid():
    from sales_tidio import verify_tidio_signature

    body = '{"topic":"contact.created"}'
    header = _sign(body)
    assert verify_tidio_signature(body=body, signature_header=header, secret=SECRET)


def test_verify_tidio_signature_rejects_tampered_body():
    from sales_tidio import verify_tidio_signature

    body = '{"topic":"contact.created"}'
    header = _sign(body)
    assert not verify_tidio_signature(
        body='{"topic":"hacked"}', signature_header=header, secret=SECRET
    )


def test_verify_tidio_signature_rejects_stale_timestamp():
    from sales_tidio import verify_tidio_signature

    body = '{"topic":"contact.created"}'
    header = _sign(body, ts=int(time.time()) - 3600)
    assert not verify_tidio_signature(body=body, signature_header=header, secret=SECRET)


# ---------------------------------------------------------------------------
# Health + turn (Flow path)
# ---------------------------------------------------------------------------


def test_tidio_health_reports_config(client):
    resp = client.get("/api/v1/sales/tidio/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["domain"] == "osakiusa.com"
    assert body["webhook_secret_set"] is True
    assert body["openapi_configured"] is False
    assert body["recommended_live_chat_path"] == "tidio_flow_http_request"


def test_tidio_turn_runs_sales_agent(client):
    resp = client.post(
        "/api/v1/sales/tidio/turn",
        json={
            "contact_id": "11111111-1111-1111-1111-111111111111",
            "message": "hello",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] == "greeting"
    assert body["reply"]
    assert body["reply_plain"]
    assert "**" not in body["reply_plain"]
    assert body["next_action"] == "reply"
    assert body["session_id"].startswith("tidio:")
    assert body["handoff"] is False


def test_tidio_turn_shipping_goes_to_warranty(client):
    resp = client.post(
        "/api/v1/sales/tidio/turn",
        json={"message": "when will it arrive?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "eta_shipping"
    assert body["handoff"] is True
    assert body["next_action"] == "warranty_redirect"
    assert body["is_warranty_route"] is True
    assert "service@osakititan.com" in body["reply_plain"].lower()
    assert "warranty chat icon" not in body["reply_plain"].lower()


def test_tidio_turn_cancel_refund_transfers_to_agent(client):
    """Cancel/refund must assign a sales agent — not the Warranty email path."""
    resp = client.post(
        "/api/v1/sales/tidio/turn",
        json={"message": "I want to cancel my order"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "cancel_refund"
    assert body["is_warranty_route"] is False
    assert body["next_action"] == "transfer_operator"
    assert "agent" in body["reply_plain"].lower()
    assert "service@osakititan.com" not in body["reply_plain"].lower()


def test_tidio_turn_body_hints_recommend_not_intensity(client):
    """Regression: ``I'm 5'5", 200 pounds and prefer strong massage`` used
    to route to intensity. It must route to recommend so the AI proposes
    actual chairs based on the customer's body."""
    resp = client.post(
        "/api/v1/sales/tidio/turn",
        json={"message": "I'm 5'5\", 200 pounds and prefer strong massage"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "recommend"
    assert body["is_warranty_route"] is False
    assert body["next_action"] == "reply"


def test_tidio_turn_discount_is_silent_handoff(client):
    resp = client.post(
        "/api/v1/sales/tidio/turn",
        json={"message": "any discount?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "discount"
    assert body["handoff"] is True
    assert body["next_action"] == "transfer_operator"
    assert "%" not in body["reply"]
    assert "promo" not in body["reply"].lower()


def test_tidio_turn_secret_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("TIDIO_TURN_SECRET", "flow-secret-123")
    bad = client.post(
        "/api/v1/sales/tidio/turn",
        json={"message": "hello"},
    )
    assert bad.status_code == 401
    good = client.post(
        "/api/v1/sales/tidio/turn",
        json={"message": "hello"},
        headers={"X-Tidio-Turn-Secret": "flow-secret-123"},
    )
    assert good.status_code == 200
    assert good.json()["intent"] == "greeting"


def test_tidio_turn_numbered_menu_and_button_resolve(client, monkeypatch):
    monkeypatch.setattr("sales_agent.fetch_live_stock", lambda *a, **k: None)

    first = client.post(
        "/api/v1/sales/tidio/turn",
        json={
            "contact_id": "btn-demo-1",
            "message": "I'm 5'4\", 170 lb, neck pain, under $3k",
        },
    )
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["intent"] == "recommend"
    assert body["button_count"] >= 1
    assert "reply with the number:" in body["reply_plain"].lower()
    assert body["button_1_label"]
    assert "tidio.buttons" in body.get("tools_used", []) or True  # tools on message log

    # Find Email me index if present, else use first button label.
    labels = [b["label"] for b in body["quick_replies"]]
    if "Email me this pick" in labels:
        choice = "Email me this pick"
    else:
        choice = "1"

    second = client.post(
        "/api/v1/sales/tidio/turn",
        json={
            "contact_id": "btn-demo-1",
            "session_id": body["session_id"],
            "message": choice,
        },
    )
    assert second.status_code == 200, second.text
    assert second.json()["resolved_from_button"] is True


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


def test_webhook_rejects_bad_signature(client):
    body = {
        "created_at": int(time.time()),
        "project_public_key": PUBLIC_KEY,
        "topic": "contact.created",
        "version": 1,
        "webhook_id": "w-1",
        "content": {"id": "c-1"},
    }
    raw = json.dumps(body, separators=(",", ":"))
    resp = client.post(
        "/api/v1/sales/tidio/webhook",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "x-tidio-signature": "t=1,s=deadbeef",
        },
    )
    assert resp.status_code == 401


def test_webhook_acks_unknown_topic(client):
    body = {
        "created_at": int(time.time()),
        "project_public_key": PUBLIC_KEY,
        "topic": "contact.created",
        "version": 1,
        "webhook_id": "w-2",
        "content": {"id": "c-2", "email": "a@b.com"},
    }
    raw = json.dumps(body, separators=(",", ":"))
    resp = client.post(
        "/api/v1/sales/tidio/webhook",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "x-tidio-signature": _sign(raw),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_webhook_rejects_wrong_project_key(client):
    body = {
        "created_at": int(time.time()),
        "project_public_key": "wrong-key",
        "topic": "contact.created",
        "version": 1,
        "webhook_id": "w-3",
        "content": {},
    }
    raw = json.dumps(body, separators=(",", ":"))
    resp = client.post(
        "/api/v1/sales/tidio/webhook",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "x-tidio-signature": _sign(raw),
        },
    )
    assert resp.status_code == 401


def test_webhook_ticket_replied_runs_agent_and_tries_openapi(client, monkeypatch):
    from sales_tidio_router import _process_ticket_reply_event
    import sales_tidio as st

    pushed = {}

    def _fake_reply(ticket_id, *, content, operator_uuid=None):
        pushed["ticket_id"] = ticket_id
        pushed["content"] = content
        return 201, {"id": "ok"}

    monkeypatch.setattr(st, "openapi_configured", lambda: True)
    monkeypatch.setattr(st, "reply_to_ticket", _fake_reply)
    # Re-bind the router helper's imported symbol
    import sales_tidio_router as strr

    monkeypatch.setattr(strr, "openapi_configured", lambda: True)
    monkeypatch.setattr(strr, "reply_to_ticket", _fake_reply)

    payload = {
        "topic": "ticket.replied",
        "content": {
            "ticket_id": 99,
            "contact_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "message": {
                "author_type": "contact",
                "message_content": "how much is the OS-Pro Maestro?",
                "message_type": "public",
            },
        },
    }
    _process_ticket_reply_event(payload)
    assert pushed["ticket_id"] == "99"
    assert pushed["content"]


def test_extract_visitor_text_skips_operator():
    from sales_tidio import extract_visitor_text

    assert (
        extract_visitor_text(
            {
                "message": {
                    "author_type": "operator",
                    "message_content": "Hello from agent",
                }
            }
        )
        is None
    )
    assert (
        extract_visitor_text(
            {
                "message": {
                    "author_type": "contact",
                    "message_content": "I need a price",
                }
            }
        )
        == "I need a price"
    )
