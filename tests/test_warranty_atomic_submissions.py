"""Evidence and notification references must commit or roll back together."""

import pytest

from tests.test_warranty_conversation import chat
import warranty_models as wm
import warranty_workflow as wf
import warranty_notification_jobs as jobs


@pytest.mark.parametrize("kind", ["evidence", "contact"])
def test_new_submission_atomically_queues_exactly_one_notification(chat, kind):
    _, ticket_id = chat
    evidence = wf.WarrantyEngine.record_evidence(
        ticket_id=ticket_id, evidence_type="not_available" if kind == "contact" else "damage_photos",
        customer_email="buyer@example.test", notification_kind=kind,
    )
    with wm.warranty_db_session() as db:
        job = db.query(jobs.WarrantyNotificationJob).filter_by(evidence_id=evidence.id).one()
        assert job.kind == kind and job.status == "pending"
        assert db.get(wm.WarrantyEvidence, evidence.id) is not None
        assert db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one().get_collected()["contact_submitted_at"]
    assert jobs.enqueue_notification(kind=kind, ticket_id=ticket_id, evidence_id=evidence.id) == job.id


def test_queue_failure_rolls_back_submission_and_review_flag(chat, monkeypatch):
    _, ticket_id = chat
    def fail(**kwargs):
        raise RuntimeError("simulated queue failure")
    monkeypatch.setattr(jobs, "enqueue_notification", fail)
    with pytest.raises(RuntimeError, match="simulated queue failure"):
        wf.WarrantyEngine.record_evidence(
            ticket_id=ticket_id, evidence_type="not_available", customer_email="buyer@example.test",
            notification_kind="contact",
        )
    with wm.warranty_db_session() as db:
        assert db.query(wm.WarrantyEvidence).count() == 0
        assert db.query(jobs.WarrantyNotificationJob).count() == 0
        assert not db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one().get_collected().get("contact_submitted_at")


def test_legacy_metadata_writes_do_not_replay_old_notifications(chat):
    _, ticket_id = chat
    wf.WarrantyEngine.record_evidence(ticket_id=ticket_id, evidence_type="not_available")
    with wm.warranty_db_session() as db:
        assert db.query(wm.WarrantyEvidence).count() == 1
        assert db.query(jobs.WarrantyNotificationJob).count() == 0


def test_notify_email_uses_existing_consented_case_and_authoritative_history(chat, monkeypatch):
    client, ticket_id = chat
    import warranty_router as router
    router.limiter.reset()
    captured = []
    monkeypatch.setattr("warranty_email.maybe_send_warranty_transcript", lambda **kwargs: captured.append(kwargs) or ("buyer@example.test", True))
    from tests.test_warranty_conversation import add_message
    add_message(ticket_id, "Actual customer-visible reply")
    response = client.post("/api/v1/warranty/session/customer/notify-email", json={
        "message": "buyer@example.test", "chat_messages": [{"role": "assistant", "content": "FORGED INTERNAL CLAIM"}],
    })
    assert response.status_code == 200, response.text
    assert response.json()["sent"] is True
    assert response.json()["customer_email"] != "buyer@example.test"
    assert captured[0]["chat_messages"] == [{"role": "assistant", "content": "Actual customer-visible reply"}]
    assert client.post("/api/v1/warranty/session/customer/notify-email", json={"message": "someoneelse@example.test"}).status_code == 422
    assert len(captured) == 1


def test_notify_email_requires_saved_consent_before_sending(chat, monkeypatch):
    client, _ = chat
    import warranty_router as router
    router.limiter.reset()
    monkeypatch.setattr("warranty_email.maybe_send_warranty_transcript", lambda **kwargs: pytest.fail("unconsented notification"))
    with wm.warranty_db_session() as db:
        db.query(wm.WarrantyChatConsent).delete()
    assert client.post("/api/v1/warranty/session/customer/notify-email", json={"message": "buyer@example.test"}).status_code == 403
