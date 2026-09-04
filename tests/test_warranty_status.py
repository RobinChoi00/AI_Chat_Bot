"""Public warranty case-status lookup."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402
from warranty_status import (  # noqa: E402
    LOOKUP_NOT_FOUND,
    emails_match,
    lookup_public_case,
    public_status_label,
)


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    import warranty_workflow as wf

    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")

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
    yield


@pytest.fixture
def client():
    from fastapi import FastAPI
    from warranty_router import router  # noqa: WPS433

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_emails_match_is_case_insensitive():
    assert emails_match("Buyer@Example.com", "buyer@example.com") is True
    assert emails_match("buyer@example.com", "other@example.com") is False


def test_public_status_label_uses_decision_when_resolved():
    assert public_status_label("resolved", "approved") == "Approved"
    assert public_status_label("awaiting_admin_review") == "Under review"


def test_lookup_requires_matching_email():
    from warranty_case_ref import case_reference_for_ticket  # noqa: WPS433
    from warranty_workflow import WarrantyEngine

    ticket_id, _ = WarrantyEngine.start_session("status-lookup", "osakiusa.com")
    with wm.warranty_db_session() as db:
        row = db.query(wm.WarrantyTicket).filter(wm.WarrantyTicket.ticket_id == ticket_id).first()
        assert row is not None
        row.status = "awaiting_admin_review"
        row.model_name = "OS-4000T"
        row.issue_type = "defect"
        row.created_at = datetime(2026, 9, 4, 12, 0)
        case_ref = case_reference_for_ticket(row)
        row.set_collected("customer_contact_email", "buyer@example.com")
        row.set_collected("case_reference", case_ref)

    found = lookup_public_case(case_reference=case_ref, email="buyer@example.com")
    assert found is not None
    assert found["case_reference"] == case_ref
    assert found["status_label"] == "Under review"
    assert found.get("ticket_id") is None

    assert lookup_public_case(case_reference=case_ref, email="other@example.com") is None


def test_status_endpoint_hides_mismatch(client):
    from warranty_workflow import WarrantyEngine

    ticket_id, _ = WarrantyEngine.start_session("status-http", "osakiusa.com")
    with wm.warranty_db_session() as db:
        row = db.query(wm.WarrantyTicket).filter(wm.WarrantyTicket.ticket_id == ticket_id).first()
        assert row is not None
        row.status = "awaiting_admin_review"
        row.created_at = datetime(2026, 9, 4, 15, 0)
        row.set_collected("customer_contact_email", "buyer@example.com")
        from warranty_case_ref import case_reference_for_ticket  # noqa: WPS433

        case_ref = case_reference_for_ticket(row)
        row.set_collected("case_reference", case_ref)

    ok = client.post(
        "/api/v1/warranty/status",
        json={"case_reference": case_ref, "email": "buyer@example.com"},
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["found"] is True
    assert body["case_reference"] == case_ref
    assert "ticket_id" not in body

    miss = client.post(
        "/api/v1/warranty/status",
        json={"case_reference": case_ref, "email": "stranger@example.com"},
    )
    assert miss.status_code == 404
    assert miss.json()["detail"] == LOOKUP_NOT_FOUND

    bad = client.post(
        "/api/v1/warranty/status",
        json={"case_reference": "nope", "email": "buyer@example.com"},
    )
    assert bad.status_code == 422
