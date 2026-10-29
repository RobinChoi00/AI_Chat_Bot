"""Incident instructions and the original report survive later acknowledgements."""

import pytest

from tests.test_warranty_conversation import chat
import warranty_notification_jobs as jobs


@pytest.mark.parametrize("language,report,required", [
    ("es", "Se ven llamas debajo de mi silla.", "Deje de usar la silla"),
    ("ko", "의자 아래에서 연기가 나고 있어요.", "사용을"),
])
def test_localized_safety_survives_refresh_acknowledgement_and_notification(chat, monkeypatch, language, report, required):
    client, ticket_id = chat
    submitted = []
    monkeypatch.setattr("sales_email_transport._send", lambda *args: submitted.append(args) or True)
    first = client.post("/api/v1/warranty/session/customer/handoff", json={"message": report})
    assert first.status_code == 200, first.text
    handoff = first.json()["handoff"]
    assert handoff["reason"] == "safety_hazard"
    assert handoff["language"] == language
    assert required in handoff["message"]
    assert handoff["notification_sent"] is False
    assert "Stop using the chair" not in handoff["message"]
    acknowledged = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "Thanks, I will stop using it."})
    assert acknowledged.status_code == 200, acknowledged.text
    assert acknowledged.json()["handoff"]["language"] == language
    assert required in acknowledged.json()["handoff"]["message"]
    assert jobs.process_pending_notifications() == 1
    assert len(submitted) == 1
    assert report in submitted[0][2]
    state = client.get("/api/v1/warranty/session/customer").json()
    assert state["handoff"]["language"] == language
    assert state["handoff"]["notification_sent"] is True
    assert required in state["handoff"]["message"]


def test_existing_safety_case_preserves_its_report_before_acknowledgement(chat, monkeypatch):
    client, ticket_id = chat
    import warranty_models as models

    original = "Se ven llamas debajo de mi silla."
    with models.warranty_db_session() as db:
        ticket = db.query(models.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.status = "awaiting_admin_review"
        ticket.set_collected("human_support_requested", True)
        ticket.set_collected("human_support_reason", "safety_hazard")
        ticket.set_collected("human_support_request", original)
    submitted = []
    monkeypatch.setattr("sales_email_transport._send", lambda *args: submitted.append(args) or True)
    response = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": "Thanks, I will stop using it."})
    assert response.status_code == 200, response.text
    assert response.json()["handoff"]["language"] == "es"
    with models.warranty_db_session() as db:
        saved = db.query(models.WarrantyTicket).filter_by(ticket_id=ticket_id).one().get_collected()
        assert saved["human_support_safety_report"] == original
    assert jobs.process_pending_notifications() == 1
    assert original in submitted[0][2]


@pytest.mark.parametrize("language_request,language,required", [
    ("영어로 답해주세요", "en", "Stop using the chair"),
    ("Please respond in español", "es", "Deje de usar la silla"),
    ("Responde en coreano, por favor", "ko", "사용을 즉시 중단"),
])
def test_safety_case_respects_an_explicit_language_switch_without_resuming_tests(chat, language_request, language, required):
    client, ticket_id = chat
    original = "The power cable is melting."
    first = client.post("/api/v1/warranty/session/customer/handoff", json={"message": original})
    assert first.status_code == 200, first.text
    assert first.json()["handoff"]["reason"] == "safety_hazard"
    response = client.post(f"/api/v1/warranty/{ticket_id}/answer", json={"answer": language_request})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["handoff"]["language"] == language
    assert payload["handoff"]["reason"] == "safety_hazard"
    assert required in payload["assistant_message"]
    assert payload["handoff"]["notification_sent"] is False
    import warranty_models as models

    with models.warranty_db_session() as db:
        saved = db.query(models.WarrantyTicket).filter_by(ticket_id=ticket_id).one().get_collected()
        assert saved["human_support_safety_report"] == original
