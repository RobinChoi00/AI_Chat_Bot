"""
tests/test_warranty_metrics.py
==============================
Admin completion-rate dashboard endpoint.

Uses an in-memory SQLite DB so the tests can seed tickets with arbitrary
statuses / timestamps and verify the aggregation math without hitting the
production DB or any LLM path.
"""

from __future__ import annotations

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

import warranty_models as wm  # noqa: E402

_ADMIN_KEY = "test-admin-key"
_CST = pytz.timezone("America/Chicago")


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    import warranty_metrics as metrics_mod
    import warranty_workflow as wf

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
    monkeypatch.setattr(wf, "_SessionFactory", mem_session_factory)
    monkeypatch.setattr(metrics_mod, "_ADMIN_API_KEY", _ADMIN_KEY, raising=False)

    yield


@pytest.fixture
def client():
    from warranty_metrics import router as metrics_router

    app = FastAPI()
    app.include_router(metrics_router)
    return TestClient(app)


def _seed_ticket(
    ticket_id: str,
    *,
    session_id: str = "s1",
    domain: str = "osakichair.com",
    status: str = "in_progress",
    issue_type: str = "defect",
    node_id: str = "root",
    created_days_ago: int = 0,
    updated_hours_ago: int = 0,
    admin_decision: str | None = None,
    customer_email: str | None = None,
    troubleshooting_outcome: str | None = None,
    turns_count: int = 0,
) -> None:
    now = datetime.now(_CST)
    created_at = now - timedelta(days=created_days_ago)
    updated_at = now - timedelta(hours=updated_hours_ago)
    collected = {}
    if customer_email:
        collected["customer_email"] = customer_email
    if troubleshooting_outcome:
        collected["troubleshooting_outcome"] = troubleshooting_outcome

    with wm.warranty_db_session() as db:
        ticket = wm.WarrantyTicket(
            ticket_id=ticket_id,
            session_id=session_id,
            domain=domain,
            current_node_id=node_id,
            status=status,
            issue_type=issue_type,
            model_name="OS-4000T",
            collected_data="{}",
            admin_decision=admin_decision,
            created_at=created_at,
            updated_at=updated_at,
        )
        for key, value in collected.items():
            ticket.set_collected(key, value)
        db.add(ticket)
        for i in range(turns_count):
            db.add(
                wm.WarrantyTurn(
                    ticket_id=ticket_id,
                    node_id=f"n{i}",
                    node_type="question",
                    node_prompt="",
                    customer_answer="",
                    answer_key="k",
                )
            )


def test_metrics_requires_admin_key(client):
    resp = client.get("/admin/warranty/metrics")
    assert resp.status_code == 401


def test_metrics_empty_db_returns_zero_totals(client):
    resp = client.get(
        "/admin/warranty/metrics", headers={"X-Admin-Key": _ADMIN_KEY}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["totals"]["started"] == 0
    assert body["totals"]["completion_rate_pct"] == 0.0
    assert body["by_status"] == []
    assert body["by_issue_type"] == []
    assert body["top_terminals"] == []
    # trend must still cover the full window with zero-days
    assert len(body["daily_started"]) == 30


def test_metrics_funnel_math(client):
    _seed_ticket(
        "t1",
        status="in_progress",
        issue_type="defect",
        node_id="root",
        updated_hours_ago=1,
    )
    _seed_ticket(
        "t2",
        status="in_progress",
        issue_type="defect",
        node_id="root",
        updated_hours_ago=48,
    )
    _seed_ticket(
        "t3",
        status="send_info",
        issue_type="installation",
        node_id="install_send_video",
        customer_email="a@b.com",
        turns_count=4,
    )
    _seed_ticket(
        "t4",
        status="awaiting_admin_review",
        issue_type="defect",
        node_id="defect_replace_terminal",
        customer_email="c@d.com",
        turns_count=6,
    )
    _seed_ticket(
        "t5",
        status="resolved",
        issue_type="defect",
        node_id="defect_replace_terminal",
        admin_decision="replacement",
        customer_email="e@f.com",
        turns_count=8,
    )

    resp = client.get(
        "/admin/warranty/metrics?days=30",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()

    totals = body["totals"]
    assert totals["started"] == 5
    # Three tickets reached a terminal (t3, t4, t5)
    assert totals["reached_terminal"] == 3
    assert totals["completion_rate_pct"] == 60.0
    # Three tickets have a customer_email
    assert totals["contact_captured"] == 3
    assert totals["contact_rate_pct"] == 60.0
    # Only t5 has an admin decision and is resolved
    assert totals["admin_decided"] == 1
    assert totals["resolved"] == 1
    assert totals["resolved_rate_pct"] == round(1 / 3 * 100.0, 1)
    assert totals["self_service_started"] == 0
    assert totals["self_service_resolved"] == 0
    assert totals["self_service_resolution_rate_pct"] == 0.0
    assert totals["escalated_after_self_service"] == 0
    # t2 is in_progress AND older than the abandon threshold (default 6h)
    assert totals["abandoned"] == 1
    # Median turns for t3/t4/t5 = 6
    assert totals["median_turns_to_terminal"] == 6

    statuses = {row["status"]: row["count"] for row in body["by_status"]}
    assert statuses["in_progress"] == 2
    assert statuses["send_info"] == 1
    assert statuses["awaiting_admin_review"] == 1
    assert statuses["resolved"] == 1

    issues = {row["issue_type"]: row for row in body["by_issue_type"]}
    assert issues["defect"]["count"] == 4
    assert issues["defect"]["completed"] == 2
    assert issues["installation"]["count"] == 1
    assert issues["installation"]["completion_rate_pct"] == 100.0

    top = {row["node_id"]: row["count"] for row in body["top_terminals"]}
    assert top["defect_replace_terminal"] == 2
    assert top["install_send_video"] == 1
    # Only completed tickets contribute to top_terminals
    assert "root" not in top


def test_metrics_tracks_self_service_resolution_and_escalation(client):
    _seed_ticket(
        "self-resolved",
        status="resolved",
        node_id="defect_power_terminal",
        admin_decision="self_resolved",
        troubleshooting_outcome="resolved",
    )
    _seed_ticket(
        "still-broken",
        status="awaiting_admin_review",
        node_id="defect_power_terminal",
        troubleshooting_outcome="unresolved",
    )
    _seed_ticket(
        "unsafe",
        status="awaiting_admin_review",
        node_id="defect_power_terminal",
        troubleshooting_outcome="unable_to_attempt",
    )

    resp = client.get(
        "/admin/warranty/metrics?days=30",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert resp.status_code == 200
    totals = resp.json()["totals"]
    assert totals["self_service_started"] == 3
    assert totals["self_service_resolved"] == 1
    assert totals["self_service_resolution_rate_pct"] == 33.3
    assert totals["escalated_after_self_service"] == 2
    assert totals["admin_decided"] == 0


def test_metrics_domain_filter(client):
    _seed_ticket("d1", domain="osakichair.com", status="resolved")
    _seed_ticket("d2", domain="titan.com", status="resolved")

    resp = client.get(
        "/admin/warranty/metrics?domain=osaki",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["totals"]["started"] == 1
    assert body["by_domain"][0]["domain"] == "osakichair.com"


def test_metrics_days_window_excludes_older(client):
    _seed_ticket("old", status="resolved", created_days_ago=45)
    _seed_ticket("new", status="resolved", created_days_ago=1)

    resp = client.get(
        "/admin/warranty/metrics?days=14",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["totals"]["started"] == 1


def test_metrics_rejects_out_of_range_days(client):
    resp = client.get(
        "/admin/warranty/metrics?days=999",
        headers={"X-Admin-Key": _ADMIN_KEY},
    )
    assert resp.status_code == 422
