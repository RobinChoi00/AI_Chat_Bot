"""Admin Sales funnel metrics."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import sales_metrics as metrics_mod  # noqa: E402
import sales_models as sm  # noqa: E402
import warranty_models as wm  # noqa: E402

_ADMIN_KEY = "sales-metrics-key"
_CST = pytz.timezone("America/Chicago")


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
    monkeypatch.setattr(metrics_mod, "_ADMIN_API_KEY", _ADMIN_KEY)
    yield


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(metrics_mod.router)
    return TestClient(app)


def _seed() -> None:
    now = datetime.now(_CST)
    with wm.warranty_db_session() as db:
        db.add_all(
            [
                sm.SalesSession(
                    session_id="recommended",
                    domain="osakiusa.com",
                    channel="tidio",
                    status="active",
                    created_at=now,
                ),
                sm.SalesSession(
                    session_id="handoff",
                    domain="titanchair.com",
                    channel="web",
                    status="handoff",
                    created_at=now - timedelta(days=1),
                ),
            ]
        )
        db.add_all(
            [
                sm.SalesMessage(
                    session_id="recommended",
                    role="user",
                    content="recommend",
                    intent="recommend",
                ),
                sm.SalesMessage(
                    session_id="recommended",
                    role="user",
                    content="[button]",
                    intent="recommend",
                ),
                sm.SalesMessage(
                    session_id="recommended",
                    role="assistant",
                    content="three picks",
                    intent="recommend",
                    tools_used=json.dumps(["cases.lookup"]),
                ),
                sm.SalesMessage(
                    session_id="handoff",
                    role="assistant",
                    content="human",
                    intent="discount",
                    handoff="discount",
                ),
            ]
        )
        db.add(
            sm.SalesLead(
                session_id="handoff",
                email="buyer@example.com",
                domain="titanchair.com",
                forwarded="failed",
                forwarded_error="smtp",
                created_at=now,
            )
        )


def test_metrics_requires_admin_key(client):
    assert client.get("/admin/sales/metrics").status_code == 401


def test_metrics_empty(client):
    response = client.get(
        "/admin/sales/metrics?days=7",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["started"] == 0
    assert len(body["daily"]) == 7


def test_metrics_funnel_and_delivery_failure(client):
    _seed()
    response = client.get(
        "/admin/sales/metrics?days=30",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert response.status_code == 200
    totals = response.json()["totals"]
    assert totals["started"] == 2
    assert totals["engaged"] == 1
    assert totals["engagement_rate_pct"] == 50.0
    assert totals["recommended"] == 1
    assert totals["recommend_rate_pct"] == 50.0
    assert totals["handoffs"] == 1
    assert totals["leads"] == 1
    assert totals["lead_forward_failed"] == 1
    assert totals["lead_forward_failure_rate_pct"] == 100.0


def test_metrics_domain_filter(client):
    _seed()
    response = client.get(
        "/admin/sales/metrics?domain=osakiusa",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert response.status_code == 200
    assert response.json()["totals"]["started"] == 1
