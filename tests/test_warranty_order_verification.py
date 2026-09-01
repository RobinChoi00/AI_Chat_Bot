"""Shopify-backed warranty purchase verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402
from delivery_lookup import TrackingSnapshot  # noqa: E402
from warranty_order_verification import verify_ticket_purchase  # noqa: E402


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
    with wm.warranty_db_session() as db:
        db.add(
            wm.WarrantyTicket(
                ticket_id="t1",
                session_id="s1",
                domain="osakiusa.com",
                current_node_id="issue_type",
                status="in_progress",
                model_name="Osaki OS-Champ",
                collected_data="{}",
            )
        )


def _saved_verification() -> dict:
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id="t1").one()
        return json.loads(ticket.get_collected()["shopify_purchase_verification"])


def test_skips_without_checkout_email(monkeypatch):
    monkeypatch.setattr(
        "warranty_consent.get_chat_consent",
        lambda _sid: None,
    )
    result = verify_ticket_purchase(
        ticket_id="t1",
        session_id="s1",
        domain="osakiusa.com",
        expected_model="Osaki OS-Champ",
    )
    assert result["status"] == "skipped_no_email"
    assert _saved_verification()["status"] == "skipped_no_email"


def test_verified_matching_order_persists_eligibility(monkeypatch):
    monkeypatch.setattr(
        "warranty_consent.get_chat_consent",
        lambda _sid: SimpleNamespace(contact_email="buyer@example.com"),
    )
    snapshot = TrackingSnapshot(
        source="shopify",
        available=True,
        order_number="#1234",
        purchase_date="January 10, 2026",
        product_names=["Osaki OS-Champ Massage Chair"],
        looked_up_at="2026-09-01T12:00:00+00:00",
    )
    monkeypatch.setattr(
        "delivery_lookup.safe_lookup_by_order_or_email",
        lambda _email, _domain: snapshot,
    )

    result = verify_ticket_purchase(
        ticket_id="t1",
        session_id="s1",
        domain="osakiusa.com",
        expected_model="Osaki OS-Champ",
    )
    assert result["status"] == "verified_model_match"
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id="t1").one()
        collected = ticket.get_collected()
        assert collected["order_number"] == "#1234"
        assert collected["warranty_eligibility_status"] == "in_warranty"


def test_model_mismatch_is_review_signal_not_eligibility(monkeypatch):
    monkeypatch.setattr(
        "warranty_consent.get_chat_consent",
        lambda _sid: SimpleNamespace(contact_email="buyer@example.com"),
    )
    snapshot = TrackingSnapshot(
        source="shopify",
        available=True,
        order_number="#999",
        purchase_date="January 10, 2026",
        product_names=["Titan Grande XL"],
        looked_up_at="2026-09-01T12:00:00+00:00",
    )
    monkeypatch.setattr(
        "delivery_lookup.safe_lookup_by_order_or_email",
        lambda _email, _domain: snapshot,
    )

    result = verify_ticket_purchase(
        ticket_id="t1",
        session_id="s1",
        domain="osakiusa.com",
        expected_model="Osaki OS-Champ",
    )
    assert result["status"] == "model_mismatch"
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id="t1").one()
        assert "warranty_eligibility_status" not in ticket.get_collected()
