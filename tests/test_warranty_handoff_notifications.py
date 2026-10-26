"""A human request must survive transport failures without claiming a reply."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_warranty_conversation import chat
import warranty_models as wm
import warranty_notification_jobs as jobs


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
    assert "haven't sent a notification" in state["assistant_message"]


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
