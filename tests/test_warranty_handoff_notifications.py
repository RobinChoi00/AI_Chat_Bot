"""A human request must survive transport failures without claiming a reply."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_warranty_conversation import chat
import warranty_models as wm
import warranty_notification_jobs as jobs


@pytest.fixture
def freshdesk_settings(monkeypatch):
    monkeypatch.setenv("WARRANTY_HANDOFF_CHANNEL", "freshdesk")
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "synthetic-helpdesk.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "synthetic-api-secret")
    monkeypatch.setenv("FRESHDESK_HANDOFF_REQUESTER_ID", "123456789012")
    monkeypatch.setenv("FRESHDESK_HANDOFF_GROUP_ID", "321098765432")
    monkeypatch.setenv("FRESHDESK_HANDOFF_ADMIN_ORIGIN", "https://help.example.test")
    monkeypatch.setattr("sales_email_transport._send", lambda *a: pytest.fail("the selected helpdesk route must not email"))


def _fake_success():
    return SimpleNamespace(status_code=201, json=lambda: {"id": 4567, "email": "never-log@example.test"})


def test_handoff_notifies_once_and_reports_only_confirmed_submission(chat, monkeypatch):
    client, ticket_id = chat
    submitted = []
    monkeypatch.setattr("sales_email_transport._send", lambda recipient, subject, body: submitted.append((recipient, subject, body)) or True)
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.admin_note = "PRIVATE STAFF NOTE"
    response = client.post("/api/v1/warranty/session/customer/handoff", json={"message": "I need a specialist"})
    assert response.status_code == 200, response.text
    assert response.json()["handoff"]["notification_sent"] is False
    assert response.json()["handoff"]["notification_status"] == "pending"
    assert jobs.process_pending_notifications() == 1
    assert len(submitted) == 1
    assert "I need a specialist" in submitted[0][2]
    assert "buyer@example.test" in submitted[0][2]
    assert "PRIVATE STAFF NOTE" not in submitted[0][2]
    state = client.get("/api/v1/warranty/session/customer").json()
    assert state["handoff"]["notification_sent"] is True
    assert "submitted to our support inbox" in state["handoff"]["message"]
    assert "callback has not been booked" in state["handoff"]["message"]
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "I need a specialist again"})
    assert jobs.process_pending_notifications() == 0
    assert len(submitted) == 1


def test_transport_failure_remains_retryable_and_does_not_claim_success(chat, monkeypatch):
    client, _ = chat
    monkeypatch.setattr("sales_email_transport._send", lambda *a: False)
    response = client.post("/api/v1/warranty/session/customer/handoff", json={"message": "Please call me back"})
    assert response.status_code == 200
    assert jobs.process_pending_notifications() == 0
    state = client.get("/api/v1/warranty/session/customer").json()
    assert state["handoff"]["notification_status"] == "failed"
    assert state["handoff"]["notification_sent"] is False
    assert "callback has not been booked" in state["assistant_message"]
    assert "notification has not been confirmed" in state["assistant_message"] or "queued" in state["assistant_message"]


def test_safety_escalation_after_a_completed_support_request_gets_a_new_notification(chat, monkeypatch):
    client, _ = chat
    calls = []
    monkeypatch.setattr("sales_email_transport._send", lambda *a: calls.append(a) or True)
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "A person please"})
    assert jobs.process_pending_notifications() == 1
    response = client.post("/api/v1/warranty/session/customer/handoff", json={"message": "The chair caught fire"})
    assert response.json()["handoff"]["reason"] == "safety_hazard"
    assert response.json()["handoff"]["notification_sent"] is False
    assert "Stop using" in response.json()["assistant_message"]
    assert jobs.process_pending_notifications() == 1
    assert len(calls) == 2
    assert "Safety concern" in calls[-1][1]


def test_admin_queue_is_authenticated_and_completed_jobs_cannot_be_replayed(chat, monkeypatch):
    client, _ = chat
    monkeypatch.setenv("ADMIN_API_KEY", "synthetic-operator-key")
    monkeypatch.setattr("sales_email_transport._send", lambda *a: False)
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "A person please"})
    jobs.process_pending_notifications()
    from notification_admin import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as admin:
        assert admin.get("/admin/warranty/notification-jobs").status_code == 401
        headers = {"X-Admin-Key": "synthetic-operator-key"}
        response = admin.get("/admin/warranty/notification-jobs?status=failed", headers=headers)
        assert response.status_code == 200
        assert "buyer@example.test" not in response.text
        job_id = response.json()["items"][0]["id"]
        assert admin.post(f"/admin/warranty/notification-jobs/{job_id}/retry").status_code == 401
        retry = admin.post(f"/admin/warranty/notification-jobs/{job_id}/retry", headers=headers)
        assert retry.json() == {"job_id": job_id, "queued": True, "sent": False}
        monkeypatch.setattr("sales_email_transport._send", lambda *a: True)
        assert jobs.process_pending_notifications() == 1
        assert admin.post(f"/admin/warranty/notification-jobs/{job_id}/retry", headers=headers).status_code == 409


def test_freshdesk_handoff_is_durable_staff_only_and_distinguishes_acceptance(chat, monkeypatch, freshdesk_settings):
    client, ticket_id = chat
    outbound = []
    def post(url, **kwargs):
        outbound.append((url, kwargs))
        return _fake_success()
    monkeypatch.setattr("warranty_handoff_delivery.requests.post", post)
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.model_name = "Private model entry"
        ticket.admin_note = "PRIVATE STAFF NOTE"
        ticket.set_collected("order_number", "PRIVATE ORDER NUMBER")
        ticket.set_collected("customer_contact_email", "buyer@example.test")
        ticket.set_collected("freshdesk_ticket_id", "7654")
    response = client.post("/api/v1/warranty/session/customer/handoff", json={"message": "A person please; my chair caught fire"})
    assert response.status_code == 200, response.text
    assert response.json()["handoff"]["notification_status"] == "pending"
    assert jobs.process_pending_notifications() == 1
    assert len(outbound) == 1
    url, kwargs = outbound[0]
    assert url == "https://synthetic-helpdesk.freshdesk.com/api/v2/tickets"
    assert kwargs["auth"] == ("synthetic-api-secret", "X")
    assert kwargs["allow_redirects"] is False and kwargs["timeout"] == (3.0, 8.0)
    payload = kwargs["json"]
    assert payload["requester_id"] == 123456789012
    assert payload["group_id"] == 321098765432
    assert payload["priority"] == 4
    assert payload["status"] == 2
    assert "Warranty" in payload["description"]
    assert f"https://help.example.test/admin/warranty/{ticket_id}" in payload["description"]
    assert "WR-" in payload["subject"] and "Safety review" in payload["subject"]
    for discarded in ("caught fire", "buyer@example.test", "Private model entry", "PRIVATE STAFF NOTE", "PRIVATE ORDER NUMBER", "synthetic-api-secret"):
        assert discarded not in str(payload)
    assert {"email", "phone", "cc_emails", "attachments"}.isdisjoint(payload)

    state = client.get("/api/v1/warranty/session/customer").json()
    assert state["handoff"]["notification_status"] == "freshdesk_submitted"
    assert state["handoff"]["submitted_via"] == "freshdesk"
    assert state["handoff"]["notification_sent"] is False
    assert "helpdesk accepted" in state["handoff"]["message"]
    assert "callback has not been confirmed" in state["handoff"]["message"]
    assert "4567" not in str(state["handoff"])
    with wm.warranty_db_session() as db:
        row = db.query(jobs.WarrantyNotificationJob).one()
        dispatch = db.get(jobs.WarrantyHandoffDispatch, row.id)
        assert dispatch.state == "submitted" and dispatch.provider_ticket_id == 4567
        assert row.status == "completed"
        assert db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one().get_collected()["freshdesk_ticket_id"] == "7654"
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "A person please"})
    assert jobs.process_pending_notifications() == 0
    assert len(outbound) == 1


@pytest.mark.parametrize("outcome,code,dispatch_state", [
    ("timeout", "freshdesk_submission_uncertain", "uncertain"),
    ("malformed", "freshdesk_submission_uncertain", "uncertain"),
    ("provider_503", "freshdesk_submission_uncertain", "uncertain"),
    ("provider_401", "freshdesk_authentication_failed", "rejected"),
])
def test_freshdesk_non_idempotent_failure_is_never_reposted(chat, monkeypatch, freshdesk_settings, outcome, code, dispatch_state, caplog):
    client, _ = chat
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        if outcome == "timeout":
            raise requests.Timeout("private-provider-information@example.test")
        if outcome == "malformed":
            return SimpleNamespace(status_code=201, json=lambda: {"private": "private-provider-information@example.test"})
        return SimpleNamespace(status_code=int(outcome.rsplit("_", 1)[-1]), text="private-provider-information@example.test")
    monkeypatch.setattr("warranty_handoff_delivery.requests.post", post)
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "Can I speak to a person?"})
    assert jobs.process_pending_notifications() == 0
    with wm.warranty_db_session() as db:
        row = db.query(jobs.WarrantyNotificationJob).one()
        job_id = int(row.id)
        assert row.status == "dead_letter" and row.last_error == code
        assert db.get(jobs.WarrantyHandoffDispatch, job_id).state == dispatch_state
    assert not jobs.retry_notification(job_id)
    assert jobs.process_pending_notifications() == 0
    assert len(calls) == 1
    assert "private-provider-information" not in caplog.text


@pytest.mark.parametrize("accepted", [False, True])
def test_freshdesk_worker_restart_after_external_attempt_never_reposts(chat, monkeypatch, freshdesk_settings, accepted):
    client, _ = chat
    class SimulatedCrash(BaseException):
        pass

    calls = []
    def post(url, **kwargs):
        calls.append(url)
        if not accepted:
            raise SimulatedCrash()
        return _fake_success()
    monkeypatch.setattr("warranty_handoff_delivery.requests.post", post)
    finish = jobs._finish_notification
    if accepted:
        def fail_finalize(*args, **kwargs):
            raise SimulatedCrash()
        monkeypatch.setattr(jobs, "_finish_notification", fail_finalize)
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "Can I speak to a person?"})
    with pytest.raises(SimulatedCrash):
        jobs.process_pending_notifications()
    with wm.warranty_db_session() as db:
        row = db.query(jobs.WarrantyNotificationJob).one()
        assert row.status == "processing"
        row.lease_expires_at = datetime(2000, 1, 1) - timedelta(seconds=1)
    monkeypatch.setattr(jobs, "_finish_notification", finish)
    assert jobs.process_pending_notifications() == int(accepted)
    with wm.warranty_db_session() as db:
        row = db.query(jobs.WarrantyNotificationJob).one()
        assert row.status == ("completed" if accepted else "dead_letter")
        if not accepted:
            assert row.last_error == "freshdesk_submission_uncertain"
    assert len(calls) == 1


def test_enabling_freshdesk_does_not_reroute_or_resend_existing_failed_job(chat, monkeypatch, freshdesk_settings):
    client, _ = chat
    monkeypatch.setenv("WARRANTY_HANDOFF_CHANNEL", "smtp")
    from sales_email_transport import SalesEmailError

    def reject(*_):
        raise SalesEmailError("smtp_authentication_failed")
    monkeypatch.setattr("sales_email_transport._send", reject)
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "Can I speak to a person?"})
    with wm.warranty_db_session() as db:
        row = db.query(jobs.WarrantyNotificationJob).one()
        row.attempts = jobs.MAX_NOTIFICATION_ATTEMPTS - 1
    assert jobs.process_pending_notifications() == 0
    monkeypatch.setenv("WARRANTY_HANDOFF_CHANNEL", "freshdesk")
    monkeypatch.setattr("warranty_handoff_delivery.requests.post", lambda *a, **k: pytest.fail("do not repost historic job"))
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "Still want a person"})
    assert jobs.process_pending_notifications() == 0
    with wm.warranty_db_session() as db:
        assert db.query(jobs.WarrantyHandoffDispatch).count() == 0
        row = db.query(jobs.WarrantyNotificationJob).one()
        assert row.status == "dead_letter" and row.last_error == "smtp_authentication_failed"


def test_missing_admin_origin_fails_closed_before_external_handoff(chat, monkeypatch, freshdesk_settings):
    client, _ = chat
    monkeypatch.setenv("FRESHDESK_HANDOFF_ADMIN_ORIGIN", "https://help.example.test@outside.test")
    monkeypatch.setattr("warranty_handoff_delivery.requests.post", lambda *a, **k: pytest.fail("invalid settings must not send"))
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "Can I speak to a person?"})
    assert jobs.process_pending_notifications() == 0
    with wm.warranty_db_session() as db:
        row = db.query(jobs.WarrantyNotificationJob).one()
        assert row.status == "dead_letter" and row.last_error == "freshdesk_handoff_not_configured"
        assert db.get(jobs.WarrantyHandoffDispatch, row.id).state == "queued"


def test_global_freshdesk_create_switch_stops_new_and_already_queued_submissions(chat, monkeypatch, freshdesk_settings):
    client, _ = chat
    client.post("/api/v1/warranty/session/customer/handoff", json={"message": "Can I speak to a person?"})
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "0")
    monkeypatch.setattr("warranty_handoff_delivery.requests.post", lambda *a, **k: pytest.fail("global create switch must stop provider calls"))
    assert jobs.process_pending_notifications() == 0
    with wm.warranty_db_session() as db:
        row = db.query(jobs.WarrantyNotificationJob).one()
        assert row.status == "dead_letter" and row.last_error == "freshdesk_handoff_not_configured"
