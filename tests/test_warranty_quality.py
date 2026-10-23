"""Verified outcome history and explicitly reviewed, model-scoped knowledge."""
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
import warranty_quality as quality
import warranty_knowledge as knowledge
from warranty_resume import create_resume_token

HEADERS = {"X-Admin-Key": "quality-test-key"}


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setenv("ADMIN_API_KEY", "quality-test-key")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "quality-signing-secret-for-tests-only-32")
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "0")
    monkeypatch.setattr(quality.limiter, "enabled", False)
    for name in ("_load_fonz_error_entries", "_load_qa_entries", "_load_freshdesk_entries", "_load_freshdesk_kb_entries", "_load_autocheck_entries", "_load_fault_judgment_entries"):
        monkeypatch.setattr(knowledge, name, lambda: ())
    knowledge.clear_knowledge_cache()
    app = FastAPI()
    app.include_router(quality.router)
    with TestClient(app) as value:
        yield value
    knowledge.clear_knowledge_cache()
    engine.dispose()


def seed(ticket_id="case-1", session_id="session-1", status="resolved", outcome=None, age=0, domain="osakiusa.com", decision=None):
    with wm.warranty_db_session() as db:
        row = wm.WarrantyTicket(ticket_id=ticket_id, session_id=session_id, status=status, domain=domain, issue_type="defect", current_node_id="defect_power_remote_replace_terminal",
                                created_at=wm._now_cst() - timedelta(days=age), admin_decision=decision)
        if outcome:
            row.set_collected("troubleshooting_outcome", outcome)
        db.add(row)


def followup(client, outcome, *, ticket_id="case-1", session_id="session-1", event_id="event-one", token=None):
    return client.post(f"/api/v1/warranty/{ticket_id}/follow-up", json={
        "token": token or quality.issue_outcome_token(ticket_id, session_id),
        "event_id": event_id, "outcome": outcome,
    })


def draft():
    return {"title": "Horizon 4D remote response", "category": "remote", "model_names": ["Horizon 4D"],
            "diagnostic": "The remote does not respond after startup.",
            "customer_steps": ["Turn off the chair and unplug it for one minute."],
            "source_url": "https://support.example.com/manuals/horizon", "review_until": (wm._now_cst().date() + timedelta(days=30)).isoformat()}


def test_customer_followup_is_signed_bound_and_preserves_warranty_decision(client):
    seed(outcome="resolved", decision="approved")
    quality.record_outcome_observation("case-1", "resolved")
    response = followup(client, "still_broken")
    assert response.status_code == 200
    with wm.warranty_db_session() as db:
        row = db.query(wm.WarrantyTicket).filter_by(ticket_id="case-1").one()
        assert row.status == "resolved"
        assert row.admin_decision == "approved"
    totals = client.get("/admin/warranty/quality?all_time=true", headers=HEADERS).json()
    assert totals["latest_confirmed_fixed"] == 0
    assert totals["latest_still_broken"] == 1
    assert totals["reopened_after_fix"] == 1
    assert totals["followed_up"] == 1


def test_unknown_admin_closed_is_not_reported_as_fixed(client):
    seed(decision="closed")
    seed("second", "session-2", outcome="resolved")
    result = client.get("/admin/warranty/quality?all_time=true", headers=HEADERS).json()
    assert result["total_consultations"] == 2
    assert result["outcome_unknown"] == 1
    assert result["latest_confirmed_fixed"] == 1
    assert result["confirmed_fix_rate_all_pct"] == 50
    assert result["confirmed_fix_rate_reported_pct"] == 100


def test_followup_deduplicates_retries_but_keeps_later_changes(client):
    seed(outcome="resolved")
    first = followup(client, "fixed")
    retry = followup(client, "fixed")
    assert first.status_code == retry.status_code == 200
    assert first.json()["observation_id"] == retry.json()["observation_id"]
    assert followup(client, "still_broken").status_code == 409
    assert followup(client, "still_broken", event_id="event-two").status_code == 200
    with wm.warranty_db_session() as db:
        assert db.query(quality.WarrantyOutcomeObservation).count() == 2


def test_initial_progress_not_counted_as_resolution(client):
    seed(status="send_info")
    assert quality.record_outcome_observation("case-1", "steps_completed") is None
    with wm.warranty_db_session() as db:
        assert db.query(quality.WarrantyOutcomeObservation).count() == 0


def test_fix_after_initial_failure_is_not_a_reopening(client):
    seed(outcome="resolved")
    quality.record_outcome_observation("case-1", "unresolved")
    quality.record_outcome_observation("case-1", "resolved")
    result = quality.quality_metrics(all_time=True)
    assert result["latest_confirmed_fixed"] == 1
    assert result["reopened_after_fix"] == 0


def test_followup_rejects_unsigned_wrong_session_wrong_ticket_and_expired(client):
    seed()
    assert followup(client, "fixed", token="not-a-valid-signed-token").status_code == 401
    assert followup(client, "fixed", session_id="other-session").status_code == 401
    token = quality.issue_outcome_token("other-case", "session-1")
    assert followup(client, "fixed", token=token).status_code == 401
    token = quality.issue_outcome_token("case-1", "session-1", ttl_secs=-1)
    assert followup(client, "fixed", token=token).status_code == 401
    # A resume link is a different capability and cannot submit follow-up votes.
    assert followup(client, "fixed", token=create_resume_token("case-1", "session-1")).status_code == 401


def test_missing_signing_secret_fails_closed(client, monkeypatch):
    seed()
    token = quality.issue_outcome_token("case-1", "session-1")
    monkeypatch.delenv("ADMIN_SESSION_SECRET")
    assert quality.issue_outcome_token("case-1", "session-1") is None
    assert followup(client, "fixed", token=token).status_code == 401


def test_followup_only_after_diagnosis_and_valid_outcome(client):
    seed(status="in_progress")
    assert followup(client, "fixed").status_code == 409
    assert followup(client, "approved").status_code == 422


def test_quality_scope_respects_full_history_and_store(client):
    seed(age=400, outcome="resolved")
    seed("recent", "s2", outcome="unresolved")
    seed("other-store", "s3", domain="titanchair.com")
    recent = client.get("/admin/warranty/quality?days=30&domain=osakiusa.com", headers=HEADERS).json()
    assert recent["total_consultations"] == 1
    historical = client.get("/admin/warranty/quality?all_time=true&domain=osakiusa.com", headers=HEADERS).json()
    assert historical["total_consultations"] == 2
    assert historical["latest_confirmed_fixed"] == 1


def test_admin_quality_endpoints_require_auth(client):
    assert client.get("/admin/warranty/quality").status_code == 401
    assert client.get("/admin/warranty/knowledge-reviews").status_code == 401
    assert client.post("/admin/warranty/knowledge-reviews", json=draft()).status_code == 401


def test_knowledge_requires_explicit_approval_and_model_matching(client):
    response = client.post("/admin/warranty/knowledge-reviews", json=draft(), headers=HEADERS)
    assert response.status_code == 201
    review_id = response.json()["id"]
    assert quality.approved_knowledge_entries() == ()
    assert knowledge.load_knowledge_entries() == ()
    approved = client.post(f"/admin/warranty/knowledge-reviews/{review_id}/decision", json={"decision": "approved", "reviewed_by": "support-reviewer"}, headers=HEADERS)
    assert approved.status_code == 200
    assert approved.json()["reviewed_at"]
    entries = knowledge.load_knowledge_entries()
    assert len(entries) == 1
    assert entries[0].source_url == draft()["source_url"]
    assert knowledge._entry_matches_requested_model(entries[0], "Horizon 4D")
    assert not knowledge._entry_matches_requested_model(entries[0], "Different model")
    assert not knowledge._entry_matches_requested_model(entries[0], "")
    revoked = client.post(f"/admin/warranty/knowledge-reviews/{review_id}/decision", json={"decision": "rejected", "reviewed_by": "support-reviewer"}, headers=HEADERS)
    assert revoked.status_code == 200
    assert knowledge.load_knowledge_entries() == ()


def test_expired_knowledge_never_enters_retrieval(client):
    data = draft()
    review_id = client.post("/admin/warranty/knowledge-reviews", json=data, headers=HEADERS).json()["id"]
    with wm.warranty_db_session() as db:
        row = db.query(quality.WarrantyKnowledgeReview).filter_by(id=review_id).one()
        row.status = "approved"
        row.review_until = "2020-01-01"
    assert quality.approved_knowledge_entries() == ()
    response = client.post(f"/admin/warranty/knowledge-reviews/{review_id}/decision", json={"decision": "approved", "reviewed_by": "reviewer"}, headers=HEADERS)
    assert response.status_code == 409


@pytest.mark.parametrize("change", ["rejected", "expired"])
def test_remote_review_change_invalidates_cached_corpus_and_ready_vectors(client, monkeypatch, change):
    """Simulate another worker's committed decision without local invalidation."""
    from types import SimpleNamespace
    import warranty_embeddings as embeddings

    monkeypatch.setenv("WARRANTY_EMBEDDING_CACHE_DIR", "")
    embeddings.clear_embedding_cache()
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in kwargs["input"]])))
    monkeypatch.setattr(embeddings, "_openai_client", lambda: fake_client)

    review_id = client.post("/admin/warranty/knowledge-reviews", json=draft(), headers=HEADERS).json()["id"]
    response = client.post(f"/admin/warranty/knowledge-reviews/{review_id}/decision",
                          json={"decision": "approved", "reviewed_by": "reviewer"}, headers=HEADERS)
    assert response.status_code == 200
    cached = knowledge.load_knowledge_entries()
    assert len(cached) == 1
    assert knowledge.load_knowledge_entries() is cached
    assert embeddings.ensure_index()
    assert embeddings.is_available()

    # No clear_knowledge_cache / decision API call in this process.
    with wm.warranty_db_session() as db:
        row = db.query(quality.WarrantyKnowledgeReview).filter_by(id=review_id).one()
        if change == "rejected":
            row.status = "rejected"
        else:
            row.review_until = "2020-01-01"
    assert embeddings.is_available()  # The old index is still present initially.
    assert knowledge.load_knowledge_entries() == ()
    # Search reloads the current corpus before using ready in-memory vectors.
    assert embeddings.semantic_search("remote not responding") is None
    assert not embeddings.is_available()
    embeddings.clear_embedding_cache()


@pytest.mark.parametrize("change", [
    {"customer_steps": ["Replace the main PCB and dispatch a technician."]},
    {"customer_steps": ["Open the back panel and inspect the wiring."]},
    {"customer_steps": ["Contact customer@example.com for repairs."]},
    {"diagnostic": "Customer jane@example.com reported the issue."},
    {"source_url": "https://support.example.com/manual?token=private"},
    {"model_names": ["all"]},
    {"review_until": "2020-01-01"},
])
def test_drafts_reject_unsafe_unscoped_expired_and_contact_data(client, change):
    data = draft()
    data.update(change)
    assert client.post("/admin/warranty/knowledge-reviews", json=data, headers=HEADERS).status_code == 422
