"""Paid conversion rates count people/sessions, never multiple orders as people."""
from datetime import timedelta
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import warranty_models as wm
import sales_models as sm
import sales_events  # register analytics table on the shared ORM
import sales_metrics


def test_paid_session_rates_are_bounded_and_revenue_keeps_currency(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(sales_metrics, "_ADMIN_API_KEY", "metrics-upgrade-key")
    now = wm._now_cst()
    with wm.warranty_db_session() as db:
        db.add_all([
            sm.SalesSession(session_id="paid", domain="osakiusa.com", created_at=now),
            sm.SalesSession(session_id="unknown", domain="osakiusa.com", created_at=now),
            sm.SalesSession(session_id="older", domain="osakiusa.com", created_at=now - timedelta(days=90)),
        ])
        for _ in range(2):
            db.add(sm.SalesLead(session_id="paid", domain="osakiusa.com", created_at=now))
        for order_id, session, state, amount, currency in [
            ("1", "paid", "paid", 10, "USD"), ("2", "paid", "paid", 20, "USD"),
            ("3", "older", "paid", 50, "EUR"), ("4", "unknown", "unknown", 999, "USD"),
            ("5", "paid", "refunded", 0, "USD"),
        ]:
            db.add(sm.SalesConversion(shopify_order_id=order_id, session_id=session,
                domain="osakiusa.com", financial_status=state, net_paid_amount=amount,
                total_usd=amount, currency=currency, created_at=now))
    app = FastAPI(); app.include_router(sales_metrics.router)
    with TestClient(app) as client:
        response = client.get("/admin/sales/metrics?days=7", headers={"X-Admin-Key": "metrics-upgrade-key"})
    assert response.status_code == 200
    body = response.json(); totals = body["totals"]
    assert totals["started"] == 2
    assert totals["paid_orders_seen"] == 3
    assert totals["paid_sessions"] == 1
    assert totals["session_to_order_rate_pct"] == 50.0
    assert totals["lead_to_order_rate_pct"] == 100.0
    assert totals["payment_unknown_orders"] == 1
    assert totals["attributed_revenue_usd"] == 30
    assert body["attributed_revenue_by_currency"] == {"USD": 30, "EUR": 50}
    assert body["behavior_funnel"]["observed_sessions"]["product_click"] == 0
    engine.dispose()
