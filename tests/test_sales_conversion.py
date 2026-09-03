"""
tests/test_sales_conversion.py
==============================
Shopify order attribution — the step that lets the chat prove it made money.

The properties worth protecting: only Shopify can post orders, the same order
can never be counted twice, and an order we cannot tie to a session is
recorded as unattributed rather than quietly credited to someone.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import sales_models as sm  # noqa: E402
import warranty_models as wm  # noqa: E402

SECRET = "shopify-test-secret"


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    wm.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(wm, "_engine", engine)
    monkeypatch.setattr(wm, "_SessionFactory", factory)
    monkeypatch.setenv("SHOPIFY_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("APP_ENV", "test")
    yield


@pytest.fixture
def client():
    from sales_router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _sign(body: bytes, secret: str = SECRET) -> str:
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("utf-8")


def _post_order(client, payload: dict, *, secret: str = SECRET, topic: str = "orders/create"):
    raw = json.dumps(payload).encode("utf-8")
    return client.post(
        "/api/v1/sales/shopify/webhook",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Hmac-Sha256": _sign(raw, secret),
            "X-Shopify-Shop-Domain": "osakiusa.com",
            "X-Shopify-Topic": topic,
        },
    )


def _order(order_id: str = "5001", email: str = "buyer@example.com", total: str = "6499.00"):
    return {
        "id": order_id,
        "order_number": 1042,
        "email": email,
        "total_price": total,
        "currency": "USD",
        "created_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Authenticity
# ---------------------------------------------------------------------------


def test_webhook_rejects_a_bad_signature(client):
    raw = json.dumps(_order()).encode("utf-8")
    resp = client.post(
        "/api/v1/sales/shopify/webhook",
        content=raw,
        headers={"X-Shopify-Hmac-Sha256": "not-the-signature"},
    )
    assert resp.status_code == 401


def test_webhook_rejects_a_signature_from_the_wrong_secret(client):
    assert _post_order(client, _order(), secret="attacker-secret").status_code == 401


def test_webhook_accepts_a_valid_signature(client):
    assert _post_order(client, _order()).status_code == 200


def test_non_order_topics_are_ignored(client):
    resp = _post_order(client, {"id": "1"}, topic="products/update")
    assert resp.status_code == 200
    assert resp.json()["ignored"] == "products/update"


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_order_is_attributed_to_the_session_that_captured_the_email(client):
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesSession(
                session_id="tidio:abc",
                domain="osakiusa.com",
                contact_email="buyer@example.com",
                created_at=datetime.now(),
            )
        )

    resp = _post_order(client, _order())
    assert resp.status_code == 200
    body = resp.json()
    assert body["attributed"] is True
    assert body["matched_by"] == "session_email"

    with wm.warranty_db_session() as db:
        row = db.query(sm.SalesConversion).one()
        assert row.session_id == "tidio:abc"
        assert row.total_usd == 6499.00


def test_order_is_attributed_through_a_lead_row(client):
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesLead(
                session_id="tidio:lead",
                email="buyer@example.com",
                domain="osakiusa.com",
                created_at=datetime.now(),
            )
        )

    assert _post_order(client, _order()).json()["matched_by"] == "lead_email"


def test_email_match_is_case_insensitive(client):
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesSession(
                session_id="tidio:case",
                domain="osakiusa.com",
                contact_email="Buyer@Example.com",
                created_at=datetime.now(),
            )
        )

    assert _post_order(client, _order(email="buyer@example.com")).json()["attributed"] is True


def test_unmatched_order_is_recorded_but_not_credited(client):
    """An honest denominator matters more than a flattering attribution rate."""
    resp = _post_order(client, _order(email="stranger@example.com"))
    assert resp.status_code == 200
    assert resp.json()["attributed"] is False

    with wm.warranty_db_session() as db:
        row = db.query(sm.SalesConversion).one()
        assert row.session_id is None
        assert row.matched_by is None


def test_session_outside_the_attribution_window_is_not_credited(client, monkeypatch):
    monkeypatch.setenv("SALES_ATTRIBUTION_WINDOW_DAYS", "30")
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesSession(
                session_id="tidio:stale",
                domain="osakiusa.com",
                contact_email="buyer@example.com",
                created_at=datetime.now() - timedelta(days=120),
            )
        )

    assert _post_order(client, _order()).json()["attributed"] is False


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_the_same_order_is_never_counted_twice(client):
    """orders/create and orders/paid both fire for one purchase."""
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesSession(
                session_id="tidio:dup",
                domain="osakiusa.com",
                contact_email="buyer@example.com",
                created_at=datetime.now(),
            )
        )

    first = _post_order(client, _order(), topic="orders/create")
    second = _post_order(client, _order(), topic="orders/paid")

    assert first.json()["attributed"] is True
    assert second.json()["duplicate"] is True

    with wm.warranty_db_session() as db:
        assert db.query(sm.SalesConversion).count() == 1


def test_malformed_body_is_rejected(client):
    raw = b"{not json"
    resp = client.post(
        "/api/v1/sales/shopify/webhook",
        content=raw,
        headers={
            "X-Shopify-Hmac-Sha256": _sign(raw),
            "X-Shopify-Topic": "orders/create",
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Revenue surfaces in metrics
# ---------------------------------------------------------------------------


def test_metrics_report_attributed_revenue(client, monkeypatch):
    import sales_metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "_ADMIN_API_KEY", "k")
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesSession(
                session_id="tidio:rev",
                domain="osakiusa.com",
                contact_email="buyer@example.com",
                created_at=datetime.now(),
            )
        )
    _post_order(client, _order(total="7000.00"))
    _post_order(client, _order(order_id="5002", email="stranger@example.com", total="3000.00"))

    app = FastAPI()
    app.include_router(metrics_mod.router)
    totals = (
        TestClient(app)
        .get("/admin/sales/metrics?days=7", headers={"X-Admin-Key": "k"})
        .json()["totals"]
    )
    assert totals["orders_seen"] == 2
    assert totals["orders_attributed"] == 1
    assert totals["attributed_revenue_usd"] == 7000.00
    assert totals["attribution_rate_pct"] == 50.0
