"""Historical admin ticket pages expose all records with matching totals."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import warranty_models as wm
import warranty_router as wr
import warranty_workflow as wf


@pytest.fixture
def client(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    wm.Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(wm, "_SessionFactory", factory)
    monkeypatch.setattr(wf, "_SessionFactory", factory)
    monkeypatch.setattr(wr, "_ADMIN_API_KEY", "pagination-test-key")
    app = FastAPI()
    app.include_router(wr.router)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def seed(count, *, prefix="ticket", status="in_progress", domain="osakichair.com", channel="web"):
    with wm.warranty_db_session() as db:
        for index in range(count):
            db.add(wm.WarrantyTicket(
                ticket_id=f"{prefix}-{index:04d}",
                session_id=f"session-{prefix}-{index}",
                domain=domain,
                status=status,
                current_node_id="root",
                collected_data=json.dumps({"channel": channel}),
                # Include records older than every dashboard time window.
                created_at=datetime(2020, 1, 1) + timedelta(seconds=index),
            ))


def get_page(client, query=""):
    response = client.get(
        f"/admin/warranty/tickets?{query}", headers={"X-Admin-Key": "pagination-test-key"}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_all_historical_tickets_reachable_with_true_total(client):
    seed(205)
    first = get_page(client, "limit=100")
    second = get_page(client, "limit=100&offset=100")
    last = get_page(client, "limit=100&offset=200")
    assert [page["total"] for page in (first, second, last)] == [205, 205, 205]
    assert [len(page["tickets"]) for page in (first, second, last)] == [100, 100, 5]
    ids = [ticket["ticket_id"] for page in (first, second, last) for ticket in page["tickets"]]
    assert len(set(ids)) == 205
    assert ids[0] == "ticket-0204"
    assert ids[-1] == "ticket-0000"
    beyond = get_page(client, "limit=100&offset=300")
    assert beyond["total"] == 205
    assert beyond["tickets"] == []


def test_page_total_uses_same_filters_as_ticket_rows(client):
    seed(105, prefix="matching", status="awaiting_admin_review", channel="phone")
    seed(2, prefix="other-status", status="resolved", channel="phone")
    seed(3, prefix="other-channel", status="awaiting_admin_review", channel="web")
    seed(4, prefix="other-domain", status="awaiting_admin_review", channel="phone", domain="titan.com")
    page = get_page(client, "status=awaiting_admin_review&domain=osaki&channel=phone&limit=100&offset=100")
    assert page["total"] == 105
    assert len(page["tickets"]) == 5
    assert all(ticket["ticket_id"].startswith("matching-") for ticket in page["tickets"])
    assert get_page(client, "channel=web")["total"] == 3


def test_pagination_still_requires_admin_authentication(client):
    assert client.get("/admin/warranty/tickets?limit=100&offset=100").status_code == 401
