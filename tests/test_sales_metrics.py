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
import sales_events as events_mod  # noqa: E402
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
        "finished": ["ask_height", "ask_goal", "recommend"],
        "quit_at_goal": ["ask_height", "ask_goal"],
        "quit_at_height": ["ask_height"],
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


def test_question_funnel_does_not_label_active_sessions_as_abandoned(client):
    _seed_funnel()
    response = client.get(
        "/admin/sales/metrics?days=7", headers={"X-Admin-Key": _ADMIN_KEY}
    )
    assert response.status_code == 200
    funnel = {row["stage"]: row for row in response.json()["question_funnel"]}

    assert funnel["ask_height"]["reached"] == 3
    assert funnel["ask_height"]["dropped"] is None
    assert funnel["ask_height"]['active_waiting'] == 1
    assert funnel["ask_height"]['continued'] == 2

    assert funnel["ask_goal"]["reached"] == 2
    assert funnel["ask_goal"]["dropped"] is None
    assert funnel["ask_goal"]['active_waiting'] == 1

    assert funnel["recommend"]["reached"] == 1
    assert funnel["recommend"]["dropped"] is None
    assert funnel["recommend"]['answer_provided'] == 1
    assert "ask_weight" not in funnel
    assert "ask_space" not in funnel


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


def _metrics(client, query=""):
    response = client.get(f"/admin/sales/metrics{query}", headers={"X-Admin-Key": _ADMIN_KEY})
    assert response.status_code == 200, response.text
    return response.json()


def test_all_time_empty_and_default_range_contract(client, monkeypatch):
    now = _CST.localize(datetime(2026, 10, 23, 12))
    monkeypatch.setattr(metrics_mod, "_now_cst", lambda: now)
    lifetime = _metrics(client, "?period=all")
    assert lifetime["range"] == {
        "period": "all", "days": None, "start": None, "start_date": None,
        "end": now.isoformat(), "end_date": "2026-10-23",
        "daily_mode": "observed",
    }
    assert lifetime["daily"] == []
    assert lifetime["totals"]["started"] == 0
    assert lifetime["behavior_funnel"]["measured_sessions"] == 0
    default = _metrics(client)
    explicit = _metrics(client, "?period=range&days=30")
    assert default == explicit
    assert default["range"]["period"] == "range"
    assert default["range"]["days"] == 30
    assert default["range"]["start_date"] == "2026-09-24"
    assert len(default["daily"]) == 30


@pytest.mark.parametrize("query", ["period=invalid", "days=0", "days=181", "period=all&days=0", "period=all&days=181"])
def test_period_and_legacy_days_validation(client, query):
    assert client.get(f"/admin/sales/metrics?{query}", headers={"X-Admin-Key": _ADMIN_KEY}).status_code == 422


def test_all_time_includes_old_panels_and_excludes_future_records(client, monkeypatch):
    now = _CST.localize(datetime(2026, 10, 23, 12))
    monkeypatch.setattr(metrics_mod, "_now_cst", lambda: now)
    old = datetime(2020, 1, 1, 10)
    recent = datetime(2026, 10, 22, 10)
    future = datetime(2026, 10, 24, 10)
    with wm.warranty_db_session() as db:
        for label, when, domain in (
            ("old", old, "osakiusa.com"), ("recent", recent, "osakiusa.com"),
            ("future", future, "osakiusa.com"), ("other", datetime(2018, 1, 1), "titanchair.com"),
        ):
            db.add(sm.SalesSession(session_id=label, domain=domain, channel="web", created_at=when))
            db.add(sm.SalesMessage(session_id=label, role="assistant", content="Recommendation", intent="recommend",
                tools_used=json.dumps(["cases.lookup", "stage:recommend"]), created_at=when))
            db.add(sm.SalesLead(session_id=label, email="test@example.test", domain=domain, forwarded="sent", created_at=when))
            db.add(sm.SalesFeedback(session_id=label, rating="helpful", intent="recommend", domain=domain, created_at=when))
            db.add(sm.SalesConversion(shopify_order_id=label, session_id=label, domain=domain, financial_status="paid",
                total_usd=100, total_amount=100, net_paid_amount=100, currency="USD", created_at=when))
            for index, kind in enumerate(events_mod.ACTION_TYPES):
                db.add(events_mod.SalesEvent(event_id=f"{label}-{kind}", session_id=label, domain=domain,
                    event_type=kind, created_at=when + timedelta(minutes=index)))
        # Future messages/events on an otherwise eligible cohort cannot enter ALL TIME.
        db.add(sm.SalesMessage(session_id="old", role="assistant", content="Future", intent="unclear", created_at=future))
        db.add(events_mod.SalesEvent(event_id="future-old", session_id="old", domain="osakiusa.com",
            event_type="product_click", created_at=future))
        db.add(events_mod.SalesEvent(event_id="orphan", session_id="not-a-session", domain="osakiusa.com",
            event_type="product_click", created_at=datetime(2010, 1, 1)))

    lifetime = _metrics(client, "?period=all&days=7&domain=osakiusa.com")
    ranged = _metrics(client, "?period=range&days=7&domain=osakiusa.com")
    assert lifetime["range"]["start_date"] == "2020-01-01"
    assert lifetime["range"]["days"] is None
    for key in ("started", "recommended", "leads", "csat_rated", "orders_seen", "paid_orders_seen", "paid_sessions", "assistant_turns"):
        assert lifetime["totals"][key] == 2, key
        assert ranged["totals"][key] == 1, key
    assert lifetime["totals"]["attributed_revenue_usd"] == 200
    assert ranged["totals"]["attributed_revenue_usd"] == 100
    assert lifetime["totals"]["unclear_turns"] == 0
    assert lifetime["behavior_funnel"]["event_counts"]["product_click"] == 2
    assert ranged["behavior_funnel"]["event_counts"]["product_click"] == 1
    assert lifetime["behavior_funnel"]["ordered_sessions"]["checkout_started"] == 2
    assert lifetime["question_funnel"][-1]["reached"] == 2
    assert lifetime["by_domain"] == [{"domain": "osakiusa.com", "count": 2}]
    daily = {row["day"]: row for row in lifetime["daily"]}
    assert daily["2020-01-01"] == {"day": "2020-01-01", "started": 1, "recommended": 1, "leads": 1}
    assert [row["day"] for row in lifetime["daily"]] == ["2020-01-01", "2026-10-22"]
    assert _metrics(client, "?period=all")["range"]["start_date"] == "2018-01-01"


@pytest.mark.parametrize("source", ["lead", "feedback", "conversion", "message", "event"])
def test_all_time_start_uses_earliest_relevant_record(client, monkeypatch, source):
    now = _CST.localize(datetime(2026, 10, 23, 12))
    monkeypatch.setattr(metrics_mod, "_now_cst", lambda: now)
    earliest = datetime(2019, 1, 2, 10)
    with wm.warranty_db_session() as db:
        if source == "lead":
            db.add(sm.SalesLead(session_id="old", email="test@example.test", domain="osakiusa.com", created_at=earliest))
        elif source == "feedback":
            db.add(sm.SalesFeedback(session_id="old", rating="helpful", domain="osakiusa.com", created_at=earliest))
        elif source == "conversion":
            db.add(sm.SalesConversion(shopify_order_id="old", domain="osakiusa.com", created_at=earliest))
        else:
            db.add(sm.SalesSession(session_id="old", domain="osakiusa.com", created_at=datetime(2020, 1, 1)))
            if source == "message":
                db.add(sm.SalesMessage(session_id="old", role="user", content="Imported old question", created_at=earliest))
            else:
                db.add(events_mod.SalesEvent(event_id="old-event", session_id="old", domain="osakiusa.com",
                    event_type="product_click", created_at=earliest))
    body = _metrics(client, "?period=all&domain=osakiusa.com")
    assert body["range"]["start_date"] == "2019-01-02"
    assert body["range"]["start"].startswith("2019-01-02T10:00:00")
    if source == "lead":
        assert body["daily"][0]["day"] == "2019-01-02"
    assert _metrics(client, "?period=all&domain=titanchair.com")["range"]["start"] is None


def test_all_time_cohort_over_sqlite_parameter_limit(client, monkeypatch):
    import sqlite3

    now = _CST.localize(datetime(2026, 10, 23, 12))
    monkeypatch.setattr(metrics_mod, "_now_cst", lambda: now)
    old = datetime(2020, 1, 1)
    count = 1100
    with wm.warranty_db_session() as db:
        connection = db.connection().connection.driver_connection
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        db.add_all([sm.SalesSession(session_id=f"s-{i}", domain="osakiusa.com", created_at=old) for i in range(count)])
        db.add_all([sm.SalesMessage(session_id=f"s-{i}", role="assistant", content="Pick", intent="recommend",
            tools_used='["cases.lookup"]', created_at=old) for i in range(count)])
        db.add_all([events_mod.SalesEvent(event_id=f"event-{i}", session_id=f"s-{i}", domain="osakiusa.com",
            event_type="product_click", created_at=old) for i in range(count)])
    body = _metrics(client, "?period=all")
    assert body["totals"]["started"] == count
    assert body["totals"]["recommended"] == count
    assert body["totals"]["assistant_turns"] == count
    assert body["behavior_funnel"]["observed_sessions"]["product_click"] == count


def test_all_time_preserves_undated_legacy_records_without_fabricating_daily_dates(client):
    with wm.warranty_db_session() as db:
        db.add(sm.SalesSession(session_id="undated", domain="osakiusa.com"))
        db.add(sm.SalesMessage(session_id="undated", role="assistant", content="Pick", intent="recommend", tools_used='["cases.lookup"]'))
        db.add(sm.SalesLead(session_id="undated", email="test@example.test", domain="osakiusa.com"))
        db.add(sm.SalesFeedback(session_id="undated", rating="helpful", domain="osakiusa.com"))
        db.add(sm.SalesConversion(shopify_order_id="undated", session_id="undated", domain="osakiusa.com",
            financial_status="paid", currency="USD", total_usd=100, total_amount=100, net_paid_amount=100))
        db.flush()
        for model in (sm.SalesSession, sm.SalesMessage, sm.SalesLead, sm.SalesFeedback, sm.SalesConversion):
            db.query(model).update({model.created_at: None})
    body = _metrics(client, "?period=all")
    for key in ("started", "recommended", "assistant_turns", "leads", "csat_rated", "paid_orders_seen"):
        assert body["totals"][key] == 1, key
        assert _metrics(client)["totals"][key] == 0, key
    assert body["range"]["start"] is None
    assert body["range"]["start_date"] is None
    assert body["daily"] == []


def test_all_time_sparse_daily_does_not_expand_a_legacy_sentinel_date(client):
    with wm.warranty_db_session() as db:
        db.add(sm.SalesSession(session_id="legacy", domain="osakiusa.com", created_at=datetime(1, 1, 1)))
    body = _metrics(client, "?period=all")
    assert body["totals"]["started"] == 1
    assert body["range"]["start_date"] is None
    assert body["range"]["daily_mode"] == "observed"
    assert body["daily"] == []
