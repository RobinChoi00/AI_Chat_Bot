"""Measured repair outcomes, delivery service and progressive consent remain distinct."""
import json
from datetime import timedelta
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import warranty_models as wm
import warranty_metrics as metrics
import warranty_quality as quality
import warranty_router as router
import warranty_workflow as workflow
import customer_journey as journey
import sales_metrics
import sales_models  # Register the tables before the isolated DB is created.

HEADERS = {"X-Admin-Key": "measured-tests"}
TERMINAL = "defect_power_remote_replace_terminal"


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(engine)
    monkeypatch.setattr(wm, "_engine", engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(metrics, "_ADMIN_API_KEY", "measured-tests")
    monkeypatch.setattr(sales_metrics, "_ADMIN_API_KEY", "measured-tests")
    monkeypatch.setenv("ADMIN_API_KEY", "measured-tests")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "measured-tests-signing-secret-at-least-32")
    monkeypatch.setenv("APP_RELEASE_ID", "measured-release")
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")
    monkeypatch.setattr("warranty_assistant_message.build_assistant_message_bundle", lambda **_kw: {})
    monkeypatch.setattr("warranty_email.send_warranty_transcript_email", lambda **_kw: pytest.fail("Metrics/outcome actions must not send email"))
    app = FastAPI()
    app.include_router(metrics.router)
    app.include_router(quality.router)
    app.include_router(router.router)
    app.include_router(sales_metrics.router)
    with TestClient(app) as http:
        yield http
    engine.dispose()


def seed(name, *, issue="defect", node=TERMINAL, status="send_info", outcome=None, decision=None, collected=None, age=0):
    with wm.warranty_db_session() as db:
        row = wm.WarrantyTicket(ticket_id=name, session_id=f"session-{name}", domain="osakiusa.com", model_name="OS-4000T",
            issue_type=issue, current_node_id=node, status=status, admin_decision=decision, created_at=wm._now_cst()-timedelta(days=age))
        for key, value in (collected or {}).items():
            row.set_collected(key, value)
        if outcome:
            row.set_collected("troubleshooting_outcome", outcome)
        db.add(row)


def measured(client, query="all_time=true"):
    response = client.get(f"/admin/warranty/metrics?{query}", headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize('path', [
    '/admin/warranty/metrics?all_time=true&cohort=current',
    '/admin/warranty/quality?all_time=true&cohort=current',
    '/admin/sales/metrics?period=all&cohort=current',
])
def test_filtered_response_exposes_metadata_contract_required_by_admin_pages(client, path):
    response = client.get(path, headers=HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['cohort_metadata']['selected'] == 'current'
    assert body['cohort_metadata']['current_release'] == 'measured-release'
    assert body['cohort_metadata'] == body['cohort']


def test_early_handoff_is_not_diagnosis_and_department_is_preserved(client):
    seed("repair-terminal")
    seed("early-human", issue=None, node="root", status="awaiting_admin_review", collected={"human_support_requested": True, "human_support_reason": "customer_request", "human_support_department": "warranty"})
    seed("early-safety", issue=None, node="issue_type", status="awaiting_admin_review", collected={"human_support_requested": True, "human_support_reason": "safety_hazard"})
    seed("delivery-human", issue="delivery", node="delivery_intent_q", status="awaiting_admin_review", collected={"human_support_requested": True, "human_support_reason": "callback_request", "human_support_department": "sales"})
    body = measured(client)
    assert body["totals"]["reached_terminal"] == 1
    assert body["repair"]["consultations"] == 2
    assert body["repair"]["diagnosis_completed"] == 1
    assert body["repair"]["diagnosis_completion_rate_pct"] == 50
    assert body["purpose_counts"] == {"repair": 2, "delivery": 1, "other": 0, "unknown": 1}
    assert {row["department"]: row["count"] for row in body["handoffs"]["by_department"]} == {"sales": 1, "warranty": 2}
    assert {row["node_id"] for row in body["top_terminals"]} == {TERMINAL}
    assert sum(row["completed"] for row in body["daily_started"]) == 1


def test_latest_result_progress_inability_deferral_and_unknown_have_distinct_denominators(client):
    seed("fixed", outcome="resolved", decision="self_resolved", status="resolved")
    seed("not-fixed", outcome="unresolved")
    seed("unable", outcome="unable_to_attempt")
    seed("deferred", outcome="deferred")
    seed("progress", outcome="steps_completed")
    seed("unknown", status="resolved", decision="closed")
    body = measured(client)
    assert body["repair"]["latest_outcomes"] == {"fixed": 1, "not_fixed": 1, "unable_to_attempt": 1, "deferred": 1, "unknown": 2}
    assert body["repair"]["result_reported"] == 2
    assert body["repair"]["confirmed_fix_rate_reported_pct"] == 50
    assert body["repair"]["confirmed_fix_rate_all_pct"] == 16.7
    assert body["totals"]["self_service_started"] == 2
    assert body["totals"]["self_service_resolution_rate_pct"] == 50
    quality.record_outcome_observation("fixed", "resolved")
    quality.record_outcome_observation("fixed", "still_broken", event_id="later-report", source="customer_followup")
    updated = measured(client)
    assert updated["repair"]["latest_outcomes"]["fixed"] == 0
    assert updated["repair"]["latest_outcomes"]["not_fixed"] == 2
    assert updated["repair"]["reopened_after_fix"] == 1
    assert updated["totals"]["self_service_resolved"] == 0
    other_endpoint = client.get("/admin/warranty/quality?all_time=true", headers=HEADERS).json()
    assert other_endpoint["repair"] == updated["repair"]


def test_old_inability_observation_is_not_mislabeled_as_a_confirmed_failed_repair(client):
    seed("old-unable", outcome="unable_to_attempt")
    with wm.warranty_db_session() as db:
        db.add(quality.WarrantyOutcomeObservation(ticket_id="old-unable", event_id="initial:old-unable:still_broken", outcome="still_broken", source="self_service"))
    body = measured(client)
    assert body["repair"]["latest_outcomes"]["unable_to_attempt"] == 1
    assert body["repair"]["latest_outcomes"]["not_fixed"] == 0
    assert body["repair"]["confirmed_fix_rate_reported_pct"] is None


def test_delivery_results_never_inflate_repair_fix_rate(client):
    seed("repair", outcome="resolved", decision="self_resolved", status="resolved")
    seed("old-delivery-fix", issue="delivery", node="delivery_status_terminal", status="resolved", decision="self_resolved", outcome="resolved")
    seed("delivery-later", issue="delivery", node="delivery_intent_q", outcome="deferred", collected={"delivery_service_outcome": "deferred", "tracking_snapshot": json.dumps({"available": True})})
    seed("delivery-review", issue="delivery", node="delivery_intent_q", collected={"delivery_service_outcome": "needs_review", "tracking_snapshot": json.dumps({"available": False})})
    body = measured(client)
    assert body["repair"]["consultations"] == 1 and body["repair"]["latest_outcomes"]["fixed"] == 1
    assert body["repair"]["confirmed_fix_rate_all_pct"] == 100
    assert body["delivery"]["consultations"] == 3
    assert body["delivery"]["deferred"] == 1 and body["delivery"]["needs_review"] == 1 and body["delivery"]["unknown"] == 1
    assert body["delivery"]["latest_lookup_available"] == 1 and body["delivery"]["latest_lookup_unavailable"] == 1
    assert body["totals"]["self_service_resolved"] == 1


def test_empty_samples_are_unmeasured_and_all_time_domain_cohorts_preserve_history(client):
    empty = measured(client)
    assert empty["repair"]["confirmed_fix_rate_all_pct"] is None
    assert empty["repair"]["confirmed_fix_rate_reported_pct"] is None
    assert empty["repair"]["diagnosis_completion_rate_pct"] is None
    seed("legacy", age=400)
    seed("new")
    with wm.warranty_db_session() as db:
        journey.register_conversation(db, surface="warranty", conversation_id="new", session_id="session-new", domain="osakiusa.com", is_new=True)
    assert measured(client)["totals"]["started"] == 2
    assert measured(client, "days=7")["totals"]["started"] == 1
    assert measured(client, "all_time=true&cohort=current")["totals"]["started"] == 1
    assert measured(client, "all_time=true&cohort=legacy")["totals"]["started"] == 1
    assert measured(client, "all_time=true&domain=other")["totals"]["started"] == 0
    assert client.get("/admin/warranty/metrics?cohort=bad%20value", headers=HEADERS).status_code == 422
    assert client.get("/admin/warranty/metrics").status_code == 401


def test_delivery_can_defer_then_request_review_without_fake_steps_or_resolution(client):
    seed("delivery", issue="delivery", node="delivery_status_terminal")
    endpoint = "/api/v1/warranty/delivery/troubleshooting-outcome"
    deferred = client.post(endpoint, json={"outcome": "deferred"})
    assert deferred.status_code == 200, deferred.text
    assert deferred.json()["deferred"] is True and deferred.json()["self_service_resolved"] is False
    with wm.warranty_db_session() as db:
        row = db.query(wm.WarrantyTicket).filter_by(ticket_id="delivery").one()
        assert row.status == "send_info" and row.admin_decision is None
    review = client.post(endpoint, json={"outcome": "unresolved"})
    assert review.status_code == 200, review.text
    assert review.json()["self_service_resolved"] is False
    assert measured(client)["delivery"]["needs_review"] == 1


def test_delivery_answer_requires_verified_lookup_and_is_service_not_repair(client):
    seed("delivery", issue="delivery", node="delivery_intent_q", status="in_progress")
    endpoint = "/api/v1/warranty/delivery/troubleshooting-outcome"
    assert client.post(endpoint, json={"outcome": "delivery_answered"}).status_code == 409
    with wm.warranty_db_session() as db:
        db.query(wm.WarrantyTicket).filter_by(ticket_id="delivery").one().set_collected("tracking_snapshot", json.dumps({"available": True}))
    response = client.post(endpoint, json={"outcome": "delivery_answered"})
    assert response.status_code == 200, response.text
    assert response.json()["service_result"] is True and response.json()["self_service_resolved"] is False
    assert client.post(endpoint, json={"outcome": "delivery_answered"}).status_code == 200
    body = measured(client)
    assert body["delivery"]["answered"] == 1 and body["repair"]["consultations"] == 0
    token = quality.issue_outcome_token("delivery", "session-delivery")
    assert client.post("/api/v1/warranty/delivery/follow-up", json={"token": token, "outcome": "fixed", "event_id": "wrong-purpose"}).status_code == 409


def test_returning_to_result_choice_withdraws_old_result_and_new_choice_is_latest(client):
    seed("repair")
    endpoint = "/api/v1/warranty/repair/troubleshooting-outcome"
    assert client.post(endpoint, json={"outcome": "unable_to_attempt"}).status_code == 200
    assert measured(client)["repair"]["latest_outcomes"]["unable_to_attempt"] == 1
    assert client.post("/api/v1/warranty/repair/troubleshooting-back").status_code == 200
    assert measured(client)["repair"]["latest_outcomes"]["unknown"] == 1
    assert client.post(endpoint, json={"outcome": "unable_to_attempt"}).status_code == 200
    assert measured(client)["repair"]["latest_outcomes"]["unable_to_attempt"] == 1
    with wm.warranty_db_session() as db:
        before = db.query(quality.WarrantyOutcomeObservation).count()
    assert client.post(endpoint, json={"outcome": "unable_to_attempt"}).status_code == 200
    with wm.warranty_db_session() as db:
        assert db.query(quality.WarrantyOutcomeObservation).count() == before


def test_contact_request_is_saved_without_claiming_a_failed_notification_was_sent(client, monkeypatch):
    seed("repair")
    monkeypatch.setattr("warranty_summary.summarize_warranty_case", lambda **_kw: {"summary": "Reported repair issue", "source": "rules"})
    monkeypatch.setattr("warranty_email.send_warranty_transcript_email", lambda **_kw: False)
    monkeypatch.setattr("warranty_email.notify_email_only_contact_async", lambda **_kw: None)
    monkeypatch.setattr("warranty_freshdesk_case.schedule_freshdesk_case_creation", lambda *_a, **_kw: {})
    monkeypatch.setattr(router, "_try_customer_receipt", lambda *_a, **_kw: {"sent": False})
    response = client.post("/api/v1/warranty/repair/contact", json={"customer_email": "customer@example.test"})
    assert response.status_code == 200, response.text
    assert response.json()["email_notified"] is False
    state = client.get("/api/v1/warranty/session/session-repair").json()["ticket"]
    assert state["review_requested"] is True and state["contact_submitted"] is True
    assert state["notification_sent"] is False
    assert "customer@example.test" not in json.dumps(state)


def test_uploaded_evidence_remains_submitted_after_session_reload(client, monkeypatch, tmp_path):
    seed("repair")
    monkeypatch.setattr(router, "_UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr("warranty_email.notify_evidence_upload_async", lambda **_kw: None)
    monkeypatch.setattr("warranty_freshdesk_case.schedule_freshdesk_case_creation", lambda *_a, **_kw: {})
    monkeypatch.setattr("warranty_freshdesk_case.note_evidence_upload", lambda *_a, **_kw: None)
    monkeypatch.setattr(router, "_try_customer_receipt", lambda *_a, **_kw: {"sent": False})
    response = client.post("/api/v1/warranty/repair/evidence", data={"evidence_type": "photo", "customer_email": "customer@example.test"},
                           files={"file": ("photo.png", b"\x89PNG\r\n\x1a\nmock-image", "image/png")})
    assert response.status_code == 200, response.text
    assert response.json()["contact_submitted"] is True and response.json()["email_notified"] is None
    state = client.get("/api/v1/warranty/session/session-repair").json()["ticket"]
    assert state["review_requested"] is True and state["contact_submitted"] is True
    assert state["notification_sent"] is None
    assert "customer@example.test" not in json.dumps(state)
