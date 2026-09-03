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


# ---------------------------------------------------------------------------
# Answer quality — CSAT and per-question drop-off
# ---------------------------------------------------------------------------


def _seed_funnel() -> None:
    """Three shoppers who quit at different points in the interview."""
    now = datetime.now(_CST)
    walks = {
        "finished": ["ask_height", "ask_weight", "ask_space", "ask_goal", "recommend"],
        "quit_at_weight": ["ask_height", "ask_weight"],
        "quit_at_space": ["ask_height", "ask_weight", "ask_space"],
    }
    with wm.warranty_db_session() as db:
        for sid, stages in walks.items():
            db.add(
                sm.SalesSession(
                    session_id=sid,
                    domain="osakiusa.com",
                    channel="tidio",
                    created_at=now,
                )
            )
            for stage in stages:
                db.add(
                    sm.SalesMessage(
                        session_id=sid,
                        role="assistant",
                        content=stage,
                        intent="recommend",
                        tools_used=json.dumps([f"stage:{stage}"]),
                    )
                )


def test_question_funnel_shows_where_shoppers_quit(client):
    _seed_funnel()
    response = client.get(
        "/admin/sales/metrics?days=7", headers={"X-Admin-Key": _ADMIN_KEY}
    )
    assert response.status_code == 200
    funnel = {row["stage"]: row for row in response.json()["question_funnel"]}

    assert funnel["ask_height"]["reached"] == 3
    assert funnel["ask_height"]["dropped"] == 0

    # Weight is where one of the three stops.
    assert funnel["ask_weight"]["reached"] == 3
    assert funnel["ask_weight"]["dropped"] == 1

    # Of the two who answered space, only one reaches the goal question.
    assert funnel["ask_space"]["reached"] == 2
    assert funnel["ask_space"]["dropped"] == 1


def test_unclear_rate_is_reported(client):
    """The 27% production figure needs to be visible to be driven down."""
    now = datetime.now(_CST)
    with wm.warranty_db_session() as db:
        db.add(sm.SalesSession(session_id="s1", domain="osakiusa.com", created_at=now))
        db.add_all(
            [
                sm.SalesMessage(session_id="s1", role="assistant", content="a", intent="unclear"),
                sm.SalesMessage(session_id="s1", role="assistant", content="b", intent="specs"),
                sm.SalesMessage(session_id="s1", role="assistant", content="c", intent="price"),
                sm.SalesMessage(session_id="s1", role="assistant", content="d", intent="price"),
            ]
        )
    response = client.get(
        "/admin/sales/metrics?days=7", headers={"X-Admin-Key": _ADMIN_KEY}
    )
    totals = response.json()["totals"]
    assert totals["unclear_turns"] == 1
    assert totals["unclear_rate_pct"] == 25.0


def test_csat_score_and_breakdown(client):
    now = datetime.now(_CST)
    with wm.warranty_db_session() as db:
        db.add(sm.SalesSession(session_id="s1", domain="osakiusa.com", created_at=now))
        db.add_all(
            [
                sm.SalesFeedback(
                    session_id="s1", rating="helpful", intent="specs",
                    domain="osakiusa.com", created_at=now,
                ),
                sm.SalesFeedback(
                    session_id="s1", rating="helpful", intent="specs",
                    domain="osakiusa.com", created_at=now,
                ),
                sm.SalesFeedback(
                    session_id="s1", rating="not_helpful", intent="price",
                    domain="osakiusa.com", created_at=now,
                ),
            ]
        )
    response = client.get(
        "/admin/sales/metrics?days=7", headers={"X-Admin-Key": _ADMIN_KEY}
    )
    body = response.json()
    assert body["totals"]["csat_rated"] == 3
    assert body["totals"]["csat_score_pct"] == 66.7

    by_intent = {row["intent"]: row for row in body["feedback_by_intent"]}
    assert by_intent["specs"]["score_pct"] == 100.0
    assert by_intent["price"]["score_pct"] == 0.0


def test_feedback_rejects_an_unknown_rating():
    from sales_models import record_feedback

    assert record_feedback("s1", rating="meh") is None
    assert record_feedback("s1", rating="helpful") is not None
