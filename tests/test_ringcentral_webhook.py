"""
tests/test_ringcentral_webhook.py
=================================
RingCentral webhook validation helpers.
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_router import router  # noqa: E402
from ringcentral_webhook import validation_token_response, verify_webhook_request  # noqa: E402


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_validation_token_response_echoes_header():
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/rc/on-call-enter",
        "headers": [(b"validation-token", b"abc123")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    response = validation_token_response(request)
    assert response is not None
    assert response.status_code == 200
    assert response.headers["Validation-Token"] == "abc123"


def test_verify_webhook_request_rejects_bad_token(monkeypatch):
    monkeypatch.setenv("RC_WEBHOOK_VERIFICATION_TOKEN", "secret-token")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/rc/on-call-enter",
        "headers": [(b"verification-token", b"wrong")],
    }

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    request = Request(scope, receive)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        verify_webhook_request(request)
    assert exc.value.status_code == 401


def test_verify_webhook_request_fails_closed_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("RC_WEBHOOK_VERIFICATION_TOKEN", raising=False)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/rc/on-call-enter",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    request = Request(scope, receive)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        verify_webhook_request(request)
    assert exc.value.status_code == 503


def test_on_call_enter_validation_token_returns_200(client):
    res = client.post(
        "/rc/on-call-enter",
        headers={"Validation-Token": "setup-token"},
    )
    assert res.status_code == 200
    assert res.headers["Validation-Token"] == "setup-token"


def test_on_call_enter_rejects_invalid_verification_token(client, monkeypatch):
    monkeypatch.setenv("RC_WEBHOOK_VERIFICATION_TOKEN", "expected-secret")
    res = client.post(
        "/rc/on-call-enter",
        json={"sessionId": "s-test"},
        headers={"Verification-Token": "bad"},
    )
    assert res.status_code == 401


def test_webhook_rejects_invalid_json(client):
    res = client.post(
        "/rc/on-call-enter",
        content=b"{not-json",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 400


def test_webhook_rejects_oversized_body(client, monkeypatch):
    monkeypatch.setenv("RC_WEBHOOK_MAX_BODY_BYTES", "1024")
    res = client.post(
        "/rc/on-call-enter",
        content=b"x" * 1025,
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 413


def test_webhook_rejects_missing_session_id(client):
    res = client.post("/rc/on-call-enter", json={"partyId": "p-1"})
    assert res.status_code == 422
