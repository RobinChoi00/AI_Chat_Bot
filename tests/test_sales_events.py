"""Observed actions cannot manufacture purchases or cross-store attribution."""

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import warranty_models as wm
import sales_events as events
from sales_models import SalesSession


@pytest.fixture(autouse=True)
def database(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    wm.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setenv("SALES_ANALYTICS_SIGNING_SECRET", "analytics-tests-signing-key-at-least-32-characters")
    monkeypatch.setenv("ADMIN_API_KEY", "analytics-test-admin")
    with wm.warranty_db_session() as db:
        db.add(SalesSession(session_id="sales-session-1", domain="osakiusa.com"))
        db.add(SalesSession(session_id="titan-session", domain="titanchair.com"))
    yield
    engine.dispose()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(events.router)
    return TestClient(app)


def payload(**changes):
    body = {
        "event_id": str(uuid4()),
        "session_id": "sales-session-1",
        "domain": "osakiusa.com",
        "event_token": events.issue_event_token("sales-session-1", "osakiusa.com"),
        "event_type": "product_click",
        "product_url": "https://www.osakiusa.com/products/sample-chair?email=private%40example.test#private",
    }
    body.update(changes)
    return body


def post(client, body, origin="https://osakiusa.com"):
    return client.post("/api/v1/sales/events", json=body, headers={"Origin": origin})


def test_actions_are_idempotent_and_never_store_query_strings_or_customer_data(client):
    body = payload()
    first = post(client, body)
    duplicate = post(client, body)
    assert first.status_code == 200
    assert first.json()["recorded"] is True
    assert duplicate.status_code == 200
    assert duplicate.json()["recorded"] is False
    with wm.warranty_db_session() as db:
        rows = db.query(events.SalesEvent).all()
        assert len(rows) == 1
        assert rows[0].product_url == "https://osakiusa.com/products/sample-chair"
        assert rows[0].domain == "osakiusa.com"
    assert "private" not in first.text
    assert "event_token" not in first.text


@pytest.mark.parametrize("kind", ["purchase", "paid", "order_paid", "refund"])
def test_public_clients_cannot_claim_a_purchase(client, kind):
    assert post(client, payload(event_type=kind)).status_code == 422


def test_rejects_pii_metadata_unknown_fields_and_unbounded_event_ids(client):
    assert post(client, payload(customer_email="private@example.test")).status_code == 422
    assert post(client, payload(event_id="not-a-uuid")).status_code == 422
    assert post(client, payload(session_id="private@example.test")).status_code == 422


def test_requires_token_for_same_session_and_store(client):
    assert post(client, payload(event_token="forged")).status_code == 403
    assert post(client, payload(domain="titanchair.com"), origin="https://titanchair.com").status_code == 403
    assert post(client, payload(session_id="titan-session")).status_code == 403
    assert events.issue_event_token("sales-session-1", "titanchair.com") is None
    assert events.issue_event_token("does-not-exist", "osakiusa.com") is None


def test_rejects_cross_store_origin_and_product_url(client):
    assert post(client, payload(), origin="https://titanchair.com").status_code == 403
    assert post(client, payload(), origin="https://osakiusa.com:bad").status_code == 403
    assert post(client, payload(), origin="https://osakiusa.com:444").status_code == 403
    assert post(client, payload(product_url="https://titanchair.com/products/sample-chair")).status_code == 422
    assert post(client, payload(product_url="https://osakiusa.com.evil.test/products/sample-chair")).status_code == 422
    assert post(client, payload(product_url="https://osakiusa.com/customer/account")).status_code == 422
    assert post(client, payload(product_url="https://osakiusa.com:bad/products/sample-chair")).status_code == 422


def test_canonical_store_alias_can_track_a_real_product_click(client):
    body = payload(
        session_id="titan-session",
        domain="osakichair.com",
        event_token=events.issue_event_token("titan-session", "osakichair.com"),
        product_url="https://titanchair.com/products/sample-chair",
    )
    assert post(client, body, origin="https://www.osakichair.com").status_code == 200


def test_expired_and_deleted_session_tokens_are_rejected(client, monkeypatch):
    monkeypatch.setattr(events.time, "time", lambda: 1000000)
    body = payload()
    monkeypatch.setattr(events.time, "time", lambda: 1000000 + 86401)
    assert post(client, body).status_code == 403
    monkeypatch.setattr(events.time, "time", lambda: 1000000)
    with wm.warranty_db_session() as db:
        db.query(SalesSession).filter(SalesSession.session_id == "sales-session-1").delete()
    assert post(client, body).status_code == 404


def test_production_missing_signing_secret_does_not_issue_tokens(client, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SALES_ANALYTICS_SIGNING_SECRET")
    monkeypatch.delenv("ADMIN_SESSION_SECRET", raising=False)
    assert events.analytics_available() is False
    assert events.issue_event_token("sales-session-1", "osakiusa.com") is None
    assert post(client, payload(event_token="unavailable")).status_code == 503


def test_reused_uuid_cannot_change_an_existing_observation(client):
    body = payload()
    assert post(client, body).status_code == 200
    body["event_type"] = "add_to_cart"
    assert post(client, body).status_code == 409


def test_limits_unique_events_per_session_without_rejecting_a_retry(client):
    body = payload(product_url="https://osakiusa.com/products/sample-chair")
    assert post(client, body).status_code == 200
    with wm.warranty_db_session() as db:
        for _ in range(299):
            db.add(events.SalesEvent(
                event_id=str(uuid4()), session_id="sales-session-1", domain="osakiusa.com",
                event_type="add_to_cart",
            ))
    assert post(client, body).json()["recorded"] is False
    assert post(client, payload()).status_code == 429


def test_actual_funnel_preserves_order_unknown_steps_and_cohort_boundaries():
    now = wm._now_cst()
    with wm.warranty_db_session() as db:
        samples = [
            ("complete", "product_click", -30),
            ("complete", "add_to_cart", -20),
            ("complete", "checkout_started", -10),
            ("out-of-order", "add_to_cart", -30),
            ("out-of-order", "checkout_started", -20),
            ("out-of-order", "product_click", -10),
            ("click-only", "product_click", -10),
            ("future", "product_click", 10),
            ("outside-cohort", "product_click", -10),
        ]
        for sid, kind, minutes in samples:
            db.add(events.SalesEvent(
                event_id=str(uuid4()), session_id=sid, domain="osakiusa.com", event_type=kind,
                created_at=now + timedelta(minutes=minutes),
            ))
    result = events.aggregate_sales_events(
        ["complete", "out-of-order", "click-only", "future"], now - timedelta(hours=1), now,
        domain="osakiusa.com",
    )
    assert result["event_counts"] == {"tracking_started": 0, "product_click": 3, "add_to_cart": 2, "checkout_started": 2}
    assert result["observed_sessions"] == {"tracking_started": 0, "product_click": 3, "add_to_cart": 2, "checkout_started": 2}
    assert result["ordered_sessions"] == {"product_click": 3, "add_to_cart": 1, "checkout_started": 1}
    assert result["measured_sessions"] == 3
    assert result["measurement_coverage_pct"] == 75.0
    assert result["denominators"]["cohort_sessions"] == 4
    assert result["denominators"]["measured_sessions"] == 3
    assert result["rates_pct"] == {"consultation_to_click": 100.0, "click_to_cart": 33.3, "cart_to_checkout": 100.0}
    assert events.aggregate_sales_events(
        ["complete"], now - timedelta(hours=1), now, domain="osaki"
    )["ordered_sessions"]["checkout_started"] == 1
    assert events.aggregate_sales_events([], now - timedelta(hours=1), now)["rates_pct"] == {
        "consultation_to_click": None, "click_to_cart": None, "cart_to_checkout": None,
    }


def test_tracking_readiness_measures_zero_clicks_without_counting_uninstrumented_sessions(client):
    body = payload(event_type="tracking_started", product_url=None)
    assert post(client, body).status_code == 200
    assert post(client, body).json()["recorded"] is False
    now = wm._now_cst()
    result = events.aggregate_sales_events(
        ["sales-session-1", "not-instrumented"], now - timedelta(hours=1), now,
    )
    assert result["observed_sessions"]["tracking_started"] == 1
    assert result["observed_sessions"]["product_click"] == 0
    assert result["measured_sessions"] == 1
    assert result["measurement_coverage_pct"] == 50.0
    assert result["rates_pct"]["consultation_to_click"] == 0.0
    assert result["rates_pct"]["click_to_cart"] is None
    unmeasured = events.aggregate_sales_events(
        ["not-instrumented"], now - timedelta(hours=1), now,
    )
    assert unmeasured["measured_sessions"] == 0
    assert unmeasured["measurement_coverage_pct"] == 0.0
    assert unmeasured["rates_pct"]["consultation_to_click"] is None


def test_admin_event_inspection_requires_authentication(client):
    assert client.get("/admin/sales/events").status_code == 401
    assert post(client, payload()).status_code == 200
    response = client.get("/admin/sales/events", headers={"X-Admin-Key": "analytics-test-admin"})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["events"][0]["product_url"] == "https://osakiusa.com/products/sample-chair"
    assert "private" not in response.text


def test_real_sales_chat_actions_signed_order_lifecycle_and_metrics_join(monkeypatch):
    """Exercise real HTTP/router/database joins with only response/SMTP stubbed."""
    import sales_leads
    import sales_metrics
    import sales_router
    from sales_agent import SalesReply

    webhook_secret = "integration-webhook-test-secret"
    monkeypatch.setenv("SHOPIFY_WEBHOOK_SECRET", webhook_secret)
    monkeypatch.delenv("OSAKI_SHOPIFY_WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr(sales_metrics, "_ADMIN_API_KEY", "analytics-test-admin")
    monkeypatch.setattr(sales_router, "respond", lambda *args, **kwargs: SalesReply(
        reply="This chair matches your stated budget. View its product details.",
        intent="recommend",
        products=[{"product_url": "https://osakiusa.com/products/sample-chair"}],
        tools_used=["catalog.resolve_product", "cta.product_url", "stage:recommend"],
    ))
    # Keep lead persistence/delivery state real; replace only SMTP side effects.
    monkeypatch.setattr(sales_leads, "_resolve_transport", lambda: lambda *args, **kwargs: True)
    monkeypatch.setattr(sales_leads, "_send_shopper_receipt", lambda **kwargs: None)

    app = FastAPI()
    app.include_router(sales_router.router)
    app.include_router(events.router)
    app.include_router(sales_metrics.router)
    with TestClient(app) as client:
        reply = client.post("/api/v1/sales/chat", json={
            "session_id": "sales-session-1", "domain": "osakiusa.com",
            "message": "Recommend a chair under $3000", "channel": "web",
        })
        assert reply.status_code == 200, reply.text
        token = reply.json()["analytics_token"]
        assert isinstance(token, str) and token
        for kind in ("tracking_started", "product_click", "add_to_cart", "checkout_started"):
            response = post(client, {
                "event_id": str(uuid4()), "session_id": "sales-session-1",
                "domain": "osakiusa.com", "event_token": token, "event_type": kind,
                "product_url": "https://osakiusa.com/products/sample-chair" if kind in {"product_click", "add_to_cart"} else None,
            })
            assert response.status_code == 200, response.text
        lead = client.post("/api/v1/sales/lead", json={
            "session_id": "sales-session-1", "domain": "osakiusa.com",
            "email": "integration-buyer@example.test", "reason": "save_pick",
        })
        assert lead.status_code == 200, lead.text

        ordered_at = datetime.now(timezone.utc)
        order = {
            "id": "integration-order-1", "email": "integration-buyer@example.test",
            "total_price": "2500.00", "currency": "USD", "financial_status": "pending",
            "created_at": ordered_at.isoformat(), "updated_at": ordered_at.isoformat(),
        }

        def webhook(payload, topic):
            raw = json.dumps(payload).encode("utf-8")
            signature = base64.b64encode(hmac.new(
                webhook_secret.encode("utf-8"), raw, hashlib.sha256,
            ).digest()).decode("ascii")
            response = client.post("/api/v1/sales/shopify/webhook", content=raw, headers={
                "Content-Type": "application/json", "X-Shopify-Hmac-Sha256": signature,
                "X-Shopify-Shop-Domain": "osakiusa.com", "X-Shopify-Topic": topic,
            })
            assert response.status_code == 200, response.text
            return response.json()

        def metrics():
            response = client.get("/admin/sales/metrics", headers={"X-Admin-Key": "analytics-test-admin"})
            assert response.status_code == 200, response.text
            return response.json()

        assert webhook(order, "orders/create")["attributed"] is True
        pending = metrics()
        assert pending["totals"]["paid_orders_seen"] == 0
        assert pending["totals"]["attributed_revenue_usd"] == 0.0

        order["financial_status"] = "paid"
        order["updated_at"] = (ordered_at + timedelta(seconds=1)).isoformat()
        assert webhook(order, "orders/paid")["attributed"] is True
        assert webhook(order, "orders/paid")["duplicate"] is True
        # Two paid orders by one conversation still count as one paid session.
        second_order = {**order, "id": "integration-order-2", "total_price": "100.00"}
        assert webhook(second_order, "orders/paid")["attributed"] is True
        result = metrics()
        totals = result["totals"]
        assert totals["started"] == 2  # The other fixture session has no tracking.
        assert totals["paid_orders_seen"] == 2
        assert totals["paid_sessions"] == 1
        assert totals["session_to_order_rate_pct"] == 50.0
        assert totals["attributed_revenue_usd"] == 2600.0
        funnel = result["behavior_funnel"]
        assert funnel["measured_sessions"] == 1
        assert funnel["measurement_coverage_pct"] == 50.0
        assert funnel["ordered_sessions"] == {"product_click": 1, "add_to_cart": 1, "checkout_started": 1}
        assert funnel["rates_pct"] == {"consultation_to_click": 100.0, "click_to_cart": 100.0, "cart_to_checkout": 100.0}
