"""Attention queue prioritizes safety, then newest; ack clears without claim decision."""

import json
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import warranty_models as models
from warranty_handoff_review import router
from warranty_notification_jobs import WarrantyNotificationJob


def _seed_attention_db(monkeypatch, tmp_path):
    key = "synthetic-admin-key"
    monkeypatch.setenv("ADMIN_API_KEY", key)
    engine = create_engine(f"sqlite:///{tmp_path / 'review.db'}", connect_args={"check_same_thread": False})
    models.Base.metadata.create_all(engine)
    monkeypatch.setattr(models, "_SessionFactory", sessionmaker(bind=engine))
    with models.warranty_db_session() as db:
        def add(tid, *, status="awaiting_admin_review", state=None, safety=False, ack=None, day=1, source=True):
            source_key = tid + ":support"
            collected = {"human_support_requested": True, "customer_contact_email": "not-returned@example.invalid"}
            if source:
                collected["human_support_notification_source"] = source_key
            if safety:
                collected["human_support_reason"] = "safety_hazard"
            if ack:
                collected["human_support_staff_accepted_at"] = ack
            db.add(models.WarrantyTicket(ticket_id=tid, session_id=tid, current_node_id="root",
                                         status=status, created_at=datetime(2026, 11, day), collected_data=json.dumps(collected)))
            if state:
                db.add(WarrantyNotificationJob(kind="handoff", source_key=source_key, ticket_id=tid, status=state))

        add("AAAAAA-safety", state="dead_letter", safety=True, day=3)
        add("BBBBBB-missing", source=False)
        add("CCCCCC-retry", state="failed", day=2)
        add("DDDDDD-completed", state="completed")
        add("EEEEEE-queued", state="pending")
        add("FFFFFF-closed", status="resolved", state="dead_letter")
        add("AAAAAB-acknowledged", state="failed", ack="2026-11-19T17:00:00-06:00")
        db.add(WarrantyNotificationJob(kind="handoff", source_key="historical-wrong-source", ticket_id="BBBBBB-missing", status="completed"))
        db.add(models.WarrantyTicket(ticket_id="AAAAAC-broken-json", session_id="broken", current_node_id="root",
                                     status="awaiting_admin_review", collected_data="invalid"))
    return key, engine


def test_attention_lists_only_unconfirmed_open_handoffs(monkeypatch, tmp_path):
    key, engine = _seed_attention_db(monkeypatch, tmp_path)

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.get("/admin/warranty/handoff-attention").status_code in (401, 403)
        response = client.get("/admin/warranty/handoff-attention?limit=2", headers={"X-Admin-Key": key})
        assert response.status_code == 200
        payload = response.json()
        assert (payload["total"], payload["safety_reviews"]) == (3, 1)
        # Safety first, then newest non-safety (day 2 retry before day 1 missing).
        assert [row["ticket_id"] for row in payload["items"]] == ["AAAAAA-safety", "CCCCCC-retry"]
        assert [row["notification_state"] for row in payload["items"]] == ["dead_letter", "failed"]
        assert payload["items"][0].get("created_at")
        assert "example.invalid" not in response.text and "source_key" not in response.text

    with models.warranty_db_session() as db:
        assert db.query(WarrantyNotificationJob).filter_by(ticket_id="AAAAAA-safety").one().status == "dead_letter"
    engine.dispose()


def test_handoff_ack_clears_attention_without_changing_status(monkeypatch, tmp_path):
    key, engine = _seed_attention_db(monkeypatch, tmp_path)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.post(
            "/admin/warranty/CCCCCC-retry/handoff-ack",
            json={"note": "Freshdesk reconciled"},
        ).status_code in (401, 403)

        first = client.post(
            "/admin/warranty/CCCCCC-retry/handoff-ack",
            headers={"X-Admin-Key": key},
            json={"note": "Freshdesk reconciled", "accepted_by": "ops"},
        )
        assert first.status_code == 200
        body = first.json()
        assert body["staff_acceptance_recorded"] is True
        assert body["already_acknowledged"] is False
        assert body["accepted_at"]

        second = client.post(
            "/admin/warranty/CCCCCC-retry/handoff-ack",
            headers={"X-Admin-Key": key},
            json={},
        )
        assert second.status_code == 200
        assert second.json()["already_acknowledged"] is True

        listed = client.get("/admin/warranty/handoff-attention", headers={"X-Admin-Key": key})
        assert listed.status_code == 200
        payload = listed.json()
        assert payload["total"] == 2
        assert "CCCCCC-retry" not in [row["ticket_id"] for row in payload["items"]]

        missing = client.post(
            "/admin/warranty/no-such-ticket/handoff-ack",
            headers={"X-Admin-Key": key},
            json={},
        )
        assert missing.status_code == 404

        no_request = client.post(
            "/admin/warranty/AAAAAC-broken-json/handoff-ack",
            headers={"X-Admin-Key": key},
            json={},
        )
        assert no_request.status_code == 409

    with models.warranty_db_session() as db:
        row = db.query(models.WarrantyTicket).filter_by(ticket_id="CCCCCC-retry").one()
        collected = row.get_collected()
        assert collected["human_support_staff_accepted_at"]
        assert collected["human_support_staff_accepted_by"] == "ops"
        assert collected["human_support_staff_ack_note"] == "Freshdesk reconciled"
        assert row.status == "awaiting_admin_review"
    engine.dispose()
