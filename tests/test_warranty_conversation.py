"""Customer history restoration must remain scoped, consented and paginated."""

from __future__ import annotations

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
import warranty_workflow as wf
from warranty_conversation import customer_conversation_page


@pytest.fixture
def chat(monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(wm, "_engine", engine)
    monkeypatch.setattr(wm, "_SessionFactory", factory)
    monkeypatch.setattr(wf, "_SessionFactory", factory)
    monkeypatch.setattr("warranty_step_enrichment.build_step_enrichment", lambda *a, **k: None)
    monkeypatch.setattr("warranty_terminal_enrichment.build_terminal_enrichment", lambda *a, **k: None)
    from warranty_router import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        consent = client.post("/api/v1/warranty/session/customer/consent", json={"domain": "osakiusa.com"})
        assert consent.status_code == 200
        email = client.post("/api/v1/warranty/session/customer/contact-email", json={"customer_email": "buyer@example.test"})
        assert email.status_code == 200
        ticket_id, _ = wf.WarrantyEngine.start_session("customer", "osakiusa.com")
        yield client, ticket_id
    engine.dispose()


def add_message(ticket_id, content, *, role="assistant", origin="api_response", session_id="customer"):
    with wm.warranty_db_session() as db:
        row = wm.WarrantyConversationMessage(
            ticket_id=ticket_id, session_id=session_id, exchange_id="synthetic",
            role=role, origin=origin, content=content,
        )
        db.add(row)
        db.flush()
        return int(row.id)


def test_refresh_restores_only_actual_customer_exchanges(chat):
    client, ticket_id = chat
    add_message(ticket_id, "My footrest stopped.", role="user", origin="customer_input")
    answer = "Please switch off before checking the external connection. " + "detail " * 1000
    add_message(ticket_id, answer)
    add_message(ticket_id, "STAFF NOTE", role="assistant", origin="admin_note")
    add_message(ticket_id, "INTERNAL WORKFLOW", role="assistant", origin="workflow_prompt")
    add_message(ticket_id, "OTHER SESSION", session_id="someone-else")
    add_message("other-ticket", "OTHER CASE")
    with wm.warranty_db_session() as db:
        row = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        row.admin_note = "INTERNAL DECISION"
    response = client.get("/api/v1/warranty/session/customer")
    assert response.status_code == 200
    messages = response.json()["conversation_messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "My footrest stopped."), ("assistant", answer),
    ]
    assert response.json()["conversation_has_more"] is False
    assert "STAFF NOTE" not in response.text
    assert "INTERNAL DECISION" not in response.text


def test_older_messages_are_bounded_stable_and_not_duplicated(chat):
    client, ticket_id = chat
    ids = [add_message(ticket_id, str(index)) for index in range(205)]
    state = client.get("/api/v1/warranty/session/customer").json()
    assert [m["id"] for m in state["conversation_messages"]] == ids[-100:]
    assert state["conversation_has_more"] is True
    cursor = state["conversation_before_id"]
    # A new reply cannot shift the boundary of an already requested older page.
    add_message(ticket_id, "new")
    older = client.get(f"/api/v1/warranty/session/customer/messages?before_id={cursor}").json()
    oldest = client.get(f"/api/v1/warranty/session/customer/messages?before_id={older['before_id']}").json()
    assert [m["id"] for m in oldest["messages"] + older["messages"] + state["conversation_messages"]] == ids
    assert older["has_more"] is True
    assert oldest["has_more"] is False
    assert oldest["before_id"] == ids[0]


def test_email_gate_does_not_expose_saved_history_until_completed(chat):
    client, ticket_id = chat
    add_message(ticket_id, "PRIVATE CHAT")
    with wm.warranty_db_session() as db:
        consent = db.query(wm.WarrantyChatConsent).filter_by(session_id="customer").one()
        consent.contact_email = None
        consent.email_gate_status = "pending"
    state = client.get("/api/v1/warranty/session/customer")
    assert "PRIVATE CHAT" not in state.text
    assert "conversation_messages" not in state.json()
    assert client.get("/api/v1/warranty/session/customer/messages").status_code == 403
    assert client.get("/api/v1/warranty/session/other/messages").status_code == 403
    client.post("/api/v1/warranty/session/customer/contact-email", json={"customer_email": "buyer@example.test"})
    assert client.get("/api/v1/warranty/session/customer/messages").json()["messages"][0]["content"] == "PRIVATE CHAT"


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "before_id=0", "before_id=-1"])
def test_invalid_pagination_is_rejected(chat, query):
    client, _ = chat
    assert client.get(f"/api/v1/warranty/session/customer/messages?{query}").status_code == 422


def test_repeated_replies_remain_distinct_and_no_legacy_history_is_invented(chat):
    _, ticket_id = chat
    assert customer_conversation_page(ticket_id=ticket_id, session_id="customer")["messages"] == []
    add_message(ticket_id, "Same answer")
    add_message(ticket_id, "Same answer")
    messages = customer_conversation_page(ticket_id=ticket_id, session_id="customer")["messages"]
    assert len(messages) == 2
    assert messages[0]["id"] != messages[1]["id"]


def test_completed_case_history_survives_refresh_without_becoming_an_open_case(chat):
    client, ticket_id = chat
    add_message(ticket_id, "Your case is closed.")
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.status = "resolved"
        ticket.admin_decision = "self_resolved"
    state = client.get("/api/v1/warranty/session/customer").json()
    assert state["ticket"]["status"] == "resolved"
    assert state["conversation_messages"][0]["content"] == "Your case is closed."
    assert client.get("/api/v1/warranty/session/customer/messages").json()["messages"][0]["content"] == "Your case is closed."
    assert wf.WarrantyEngine.get_active_session_ticket("customer") is None
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.admin_decision = "abandoned"
    assert client.get("/api/v1/warranty/session/customer").json()["ticket"] is None


def test_intake_email_does_not_make_the_contact_step_irreversible(chat):
    client, ticket_id = chat
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.current_node_id = "install_send_video"
        ticket.issue_type = "installation"
        ticket.status = "awaiting_evidence"
        ticket.set_collected("troubleshooting_outcome", "unresolved")
    # The mandatory intake email is present, but no review has been submitted.
    assert wf.WarrantyEngine.get_ticket(ticket_id).get_collected()["customer_contact_email"] == "buyer@example.test"
    response = client.post(f"/api/v1/warranty/{ticket_id}/troubleshooting-back")
    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "steps_completed"
    with wm.warranty_db_session() as db:
        ticket = db.query(wm.WarrantyTicket).filter_by(ticket_id=ticket_id).one()
        ticket.set_collected("troubleshooting_outcome", "unresolved")
        ticket.set_collected("contact_submitted_at", "2026-10-26T10:00:00")
    assert client.post(f"/api/v1/warranty/{ticket_id}/troubleshooting-back").status_code == 409
