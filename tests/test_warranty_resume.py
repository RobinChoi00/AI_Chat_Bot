"""
tests/test_warranty_resume.py
=============================
Unit + HTTP tests for the "Save & continue later" warranty resume flow.
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402
import warranty_resume as wr  # noqa: E402


TEST_SECRET = "a" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ADMIN_SESSION_SECRET", TEST_SECRET)
    yield


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    import warranty_workflow as wf  # noqa: WPS433

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
def client(monkeypatch):
    monkeypatch.setattr(
        wr,
        "_send_resume_email_async",
        lambda **kwargs: None,  # do not hit SMTP in tests
    )
    from warranty_router import router as warranty_router  # noqa: WPS433

    app = FastAPI()
    app.include_router(warranty_router)
    app.include_router(wr.router)
    return TestClient(app)


# --- Token round-trip ------------------------------------------------------


def test_token_roundtrip_ok():
    tok = wr.create_resume_token("t-abc", "s-1", ttl_secs=60)
    payload = wr.verify_resume_token(tok)
    assert payload is not None
    assert payload["tid"] == "t-abc"
    assert payload["sid"] == "s-1"


def test_token_rejects_tampered_signature():
    tok = wr.create_resume_token("t-abc", "s-1")
    body, _sig = tok.split(".")
    tampered = f"{body}.AAAA"
    assert wr.verify_resume_token(tampered) is None


def test_token_rejects_expired():
    tok = wr.create_resume_token("t-abc", "s-1", ttl_secs=-10)
    assert wr.verify_resume_token(tok) is None


def test_token_rejects_wrong_secret(monkeypatch):
    tok = wr.create_resume_token("t-abc", "s-1", ttl_secs=60)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "b" * 40)
    assert wr.verify_resume_token(tok) is None


def test_token_requires_min_secret_length(monkeypatch):
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "short")
    with pytest.raises(RuntimeError):
        wr.create_resume_token("t", "s")


# --- HTTP end-to-end -------------------------------------------------------


def _register_and_start(client, session_id: str) -> str:
    client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": "OS-4000T", "domain": "osaki.com"},
    )
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    return resp.json()["ticket"]["ticket_id"]


def test_resume_link_requires_active_ticket(client):
    resp = client.post(
        "/api/v1/warranty/session/no-ticket/resume-link",
        json={"customer_email": "buyer@example.com"},
    )
    assert resp.status_code == 404


def test_resume_link_rejects_bad_email(client):
    session_id = "sess-resume-bad-email"
    _register_and_start(client, session_id)
    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/resume-link",
        json={"customer_email": "not-an-email"},
    )
    assert resp.status_code == 422


def test_resume_link_and_verify_roundtrip(client):
    session_id = "sess-resume-happy"
    ticket_id = _register_and_start(client, session_id)

    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/resume-link",
        json={"customer_email": "buyer@example.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sent"] is True
    assert body["customer_email"] == "b***r@example.com"

    # Directly generate a token — the SMTP side-effect is mocked, so we
    # cannot pluck the URL out of the email. We verify the round-trip via
    # the public token helpers instead.
    token = wr.create_resume_token(ticket_id, session_id)
    verify = client.get(f"/api/v1/warranty/resume/{token}")
    assert verify.status_code == 200
    data = verify.json()
    assert data["ticket_id"] == ticket_id
    assert data["session_id"] == session_id
    assert data["status"] == "in_progress"
    assert data["expires_at"] > int(time.time())


def test_resume_endpoint_rejects_bad_token(client):
    resp = client.get("/api/v1/warranty/resume/definitely-not-a-token")
    assert resp.status_code == 400


def test_resume_endpoint_404_when_ticket_missing(client):
    token = wr.create_resume_token("does-not-exist", "s-x")
    resp = client.get(f"/api/v1/warranty/resume/{token}")
    assert resp.status_code == 404
