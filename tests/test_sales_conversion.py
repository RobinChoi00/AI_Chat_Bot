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
from datetime import datetime, timedelta, timezone

import pytz
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


def _chat_now():
    return datetime.now(pytz.timezone("America/Chicago")).replace(tzinfo=None)


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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "financial_status": "paid",
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
                created_at=_chat_now(),
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
                created_at=_chat_now(),
            )
        )

    assert _post_order(client, _order()).json()["matched_by"] == "lead_email"


def test_order_is_attributed_to_the_session_that_captured_the_phone(client):
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesSession(
                session_id="tidio:phone",
                domain="osakiusa.com",
                contact_phone="+1 (555) 010-0199",
                created_at=_chat_now(),
            )
        )

    payload = _order(email="anon-checkout@example.com")
    payload["phone"] = "15550100199"
    resp = _post_order(client, payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["attributed"] is True
    assert body["matched_by"] == "session_phone"


def test_email_match_is_case_insensitive(client):
    with wm.warranty_db_session() as db:
        db.add(
            sm.SalesSession(
                session_id="tidio:case",
                domain="osakiusa.com",
                contact_email="Buyer@Example.com",
                created_at=_chat_now(),
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
                created_at=_chat_now() - timedelta(days=120),
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
                created_at=_chat_now(),
            )
        )

    payload = _order()
    first = _post_order(client, payload, topic="orders/create")
    second = _post_order(client, payload, topic="orders/paid")

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
            "X-Shopify-Shop-Domain": "osakiusa.com",
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
                created_at=_chat_now(),
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


def _touch(session_id, at, *, domain="osakiusa.com", email="buyer@example.com", lead=False):
    local_at = at.astimezone(pytz.timezone("America/Chicago")).replace(tzinfo=None)
    with wm.warranty_db_session() as db:
        if lead:
            db.add(sm.SalesLead(session_id=session_id, domain=domain, email=email, created_at=local_at))
        else:
            db.add(sm.SalesSession(session_id=session_id, domain=domain, contact_email=email, created_at=local_at))


def test_attribution_is_same_store_last_touch_across_sessions_and_leads():
    from sales_conversion import find_attributed_session

    ordered = datetime(2026, 7, 10, 20, tzinfo=timezone.utc)
    _touch("older-session", ordered - timedelta(hours=3))
    _touch("newer-lead", ordered - timedelta(hours=2), lead=True)
    _touch("other-store", ordered - timedelta(hours=1), domain="titanchair.com")
    _touch("after-purchase", ordered + timedelta(hours=1))
    assert find_attributed_session("buyer@example.com", ordered_at=ordered, domain="osakiusa.com") == ("newer-lead", "lead_email")
    assert find_attributed_session("buyer@example.com", ordered_at=ordered, domain="www.osakichair.com") == ("other-store", "session_email")


def test_attribution_excludes_future_unknown_store_and_missing_order_time():
    from sales_conversion import find_attributed_session

    ordered = datetime(2026, 7, 10, 20, tzinfo=timezone.utc)
    _touch("future", ordered + timedelta(minutes=1))
    assert find_attributed_session("buyer@example.com", ordered_at=ordered, domain="osakiusa.com") == (None, None)
    assert find_attributed_session("buyer@example.com", ordered_at=ordered, domain="unknown") == (None, None)
    assert find_attributed_session("buyer@example.com", domain="osakiusa.com") == (None, None)


def test_shopify_offset_time_is_converted_before_attribution(monkeypatch):
    from sales_conversion import record_order

    ordered = datetime(2026, 7, 10, 20, tzinfo=timezone.utc)
    _touch("correct", ordered - timedelta(minutes=10))
    _touch("too-late", ordered + timedelta(minutes=1))
    monkeypatch.setenv("OSAKI_SHOP_DOMAIN", "actual-osaki.myshopify.com")
    payload = {**_order(), "created_at": "2026-07-10T15:00:00-05:00"}
    row = record_order(payload, domain="actual-osaki.myshopify.com")
    assert row["session_id"] == "correct"
    assert row["domain"] == "osakiusa.com"


def test_unknown_or_ambiguous_shop_does_not_inherit_a_default(monkeypatch):
    from sales_conversion import conversion_storefront_domain, record_order

    assert record_order(_order(), domain="unrecognized.myshopify.com") is None
    monkeypatch.setenv("OSAKI_SHOP_DOMAIN", "shared.myshopify.com")
    monkeypatch.setenv("TITAN_SHOP_DOMAIN", "shared.myshopify.com")
    assert conversion_storefront_domain("shared.myshopify.com") is None


def test_pending_order_becomes_paid_and_cannot_be_downgraded_by_redelivery(client):
    from sales_conversion import record_order

    pending = {**_order(), "financial_status": "pending", "updated_at": "2026-07-10T20:00:00Z"}
    pending_row = record_order(pending, domain="osakiusa.com")
    assert pending_row["net_paid_amount"] == 0
    paid = {**pending, "updated_at": "2026-07-10T20:01:00Z"}
    paid_row = record_order(paid, domain="osakiusa.com", topic="orders/paid")
    assert paid_row["financial_status"] == "paid"
    assert paid_row["net_paid_amount"] == 6499
    assert record_order(paid, domain="osakiusa.com", topic="orders/paid") is None
    assert record_order(pending, domain="osakiusa.com") is None
    assert record_order({**pending, "financial_status": "voided"}, domain="osakiusa.com", topic="orders/updated") is None
    with wm.warranty_db_session() as db:
        assert db.query(sm.SalesConversion).one().financial_status == "paid"


def _refund(refund_id, amount="100.00", order_id="5001", currency="USD"):
    return {
        "id": refund_id, "order_id": order_id, "created_at": "2026-07-11T20:00:00Z",
        "transactions": [{"kind": "refund", "status": "success", "amount": amount, "currency": currency}],
    }


def test_partial_refunds_are_idempotent_and_reduce_only_net_revenue():
    from sales_conversion import record_order

    record_order(_order(total="1000"), domain="osakiusa.com")
    first = record_order(_refund("r1", "100"), domain="osakiusa.com", topic="refunds/create")
    assert first["financial_status"] == "partially_refunded"
    assert first["refunded_amount"] == 100
    assert first["net_paid_amount"] == 900
    assert record_order(_refund("r1", "100"), domain="osakiusa.com", topic="refunds/create") is None
    last = record_order(_refund("r2", "900"), domain="osakiusa.com", topic="refunds/create")
    assert last["financial_status"] == "refunded"
    assert last["net_paid_amount"] == 0
    # A late paid payload cannot erase either refund.
    replay = record_order(_order(total="1000"), domain="osakiusa.com", topic="orders/paid")
    assert replay is None or replay["financial_status"] == "refunded"


def test_refund_arriving_before_order_preserves_amount_until_payment_arrives():
    from sales_conversion import record_order

    refund = record_order(_refund("r1", "100"), domain="osakiusa.com", topic="refunds/create")
    assert refund["net_paid_amount"] is None
    paid = record_order(_order(total="1000"), domain="osakiusa.com", topic="orders/paid")
    assert paid["financial_status"] == "partially_refunded"
    assert paid["refunded_amount"] == 100
    assert paid["net_paid_amount"] == 900


def test_cancelled_order_stays_excluded_after_a_late_paid_event():
    from sales_conversion import conversion_revenue_summary, record_order

    paid = _order(total="1000")
    record_order(paid, domain="osakiusa.com")
    record_order({**paid, "cancelled_at": "2026-07-11T20:00:00Z"}, domain="osakiusa.com", topic="orders/cancelled")
    record_order(paid, domain="osakiusa.com", topic="orders/paid")
    with wm.warranty_db_session() as db:
        row = db.query(sm.SalesConversion).one()
        assert row.net_paid_amount == 0
        assert conversion_revenue_summary([row])["paid_orders_seen"] == 0


def test_revenue_requires_payment_evidence_and_keeps_currencies_separate():
    from sales_conversion import conversion_revenue_summary

    rows = [
        sm.SalesConversion(session_id="s", currency="USD", financial_status="paid", net_paid_amount=100, total_usd=100),
        sm.SalesConversion(session_id="s", currency="CAD", financial_status="paid", net_paid_amount=200, total_usd=None),
        sm.SalesConversion(session_id="s", currency="USD", financial_status="unknown", total_usd=9999),
        sm.SalesConversion(session_id="s", currency="USD", financial_status="pending", total_usd=7777),
    ]
    summary = conversion_revenue_summary(rows)
    assert summary["paid_orders_seen"] == 2
    assert summary["payment_unknown_orders"] == 1
    assert summary["attributed_revenue_usd"] == 100
    assert summary["attributed_revenue_by_currency"] == {"USD": 100, "CAD": 200}


def test_non_usd_paid_order_is_not_stored_as_usd():
    from sales_conversion import record_order

    row = record_order({**_order(total="1000"), "currency": "CAD"}, domain="osakiusa.com")
    assert row["total_amount"] == 1000
    assert row["net_paid_amount"] == 1000
    assert row["total_usd"] is None


def test_unknown_refund_amount_does_not_report_the_original_gross_as_net():
    from sales_conversion import record_order

    record_order(_order(total="1000"), domain="osakiusa.com")
    row = record_order({"id": "r-unknown", "order_id": "5001", "transactions": []}, domain="osakiusa.com", topic="refunds/create")
    assert row["financial_status"] == "partially_refunded"
    assert row["net_paid_amount"] is None
    assert row["total_usd"] is None


def test_authenticated_webhook_topics_drive_payment_and_refund_lifecycle(client):
    payload = {**_order(total="1000"), "financial_status": "pending"}
    assert _post_order(client, payload).status_code == 200
    assert _post_order(client, payload, topic="orders/paid").status_code == 200
    assert _post_order(client, _refund("r1", "100"), topic="refunds/create").status_code == 200
    with wm.warranty_db_session() as db:
        row = db.query(sm.SalesConversion).one()
        assert row.financial_status == "partially_refunded"
        assert row.net_paid_amount == 900


def test_additive_schema_upgrade_preserves_legacy_rows_as_unknown():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE sales_conversions (id INTEGER PRIMARY KEY, shopify_order_id VARCHAR UNIQUE, total_usd FLOAT, currency VARCHAR)")
        connection.exec_driver_sql("INSERT INTO sales_conversions (id, shopify_order_id, total_usd, currency) VALUES (1, 'legacy', 500, 'USD')")
    sm.ensure_sales_conversion_schema(engine)
    sm.ensure_sales_conversion_schema(engine)
    with engine.connect() as connection:
        row = connection.exec_driver_sql("SELECT total_usd,financial_status,net_paid_amount,refund_events FROM sales_conversions").one()
    assert tuple(row) == (500, "unknown", None, "{}")
    engine.dispose()
