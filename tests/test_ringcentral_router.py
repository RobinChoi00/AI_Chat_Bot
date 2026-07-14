"""
tests/test_ringcentral_router.py
================================
HTTP tests for RingCentral webhook endpoints (handlers mocked).
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_router import router  # noqa: E402


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_rc_health(client):
    res = client.get("/rc/health")
    assert res.status_code == 200
    assert res.json()["service"] == "ringcentral-ivr"


def test_rc_health_production_detects_stopped_worker(client, monkeypatch):
    production_values = {
        "APP_ENV": "production",
        "RC_CLIENT_ID": "client",
        "RC_CLIENT_SECRET": "secret",
        "RC_USER_JWT": "jwt",
        "PUBLIC_BASE_URL": "https://api.example.com",
        "RC_WARRANTY_TRANSFER_EXTENSION": "3",
        "RC_SMS_FROM_NUMBER": "+12145550123",
        "RC_WEBHOOK_VERIFICATION_TOKEN": "verification",
        "RC_EVENT_WORKER_ENABLED": "true",
    }
    for key, value in production_values.items():
        monkeypatch.setenv(key, value)

    with patch("ringcentral_router._worker_thread", None):
        res = client.get("/rc/health")

    assert res.status_code == 503
    assert res.json()["checks"]["EVENT_WORKER"] is False


def test_on_call_enter_returns_204(client):
    session_id = f"s-{uuid.uuid4().hex}"
    payload = {
        "sessionId": session_id,
        "partyId": "p-app",
        "inParty": {
            "id": "p-caller",
            "from": {"phoneNumber": "+15551234567"},
        },
    }
    with patch("ringcentral_router.handle_call_enter") as mock_enter:
        res = client.post("/rc/on-call-enter", json=payload)
    assert res.status_code == 204
    matching = [call for call in mock_enter.call_args_list if call.args[0] == payload]
    assert len(matching) == 1


def test_on_command_update_returns_204(client):
    session_id = f"s-{uuid.uuid4().hex}"
    payload = {
        "sessionId": session_id,
        "command": "Play",
        "status": "Completed",
        "partyId": "p-caller",
    }
    with patch("ringcentral_router.handle_command_update") as mock_update:
        res = client.post("/rc/on-command-update", json=payload)
    assert res.status_code == 204
    matching = [call for call in mock_update.call_args_list if call.args[0] == payload]
    assert len(matching) == 1


def test_duplicate_webhook_is_acknowledged_but_processed_once(client):
    payload = {
        "sessionId": f"s-{uuid.uuid4().hex}",
        "partyId": "p-app",
        "inParty": {"id": "p-caller"},
    }
    with patch("ringcentral_router.handle_call_enter") as mock_enter:
        first = client.post("/rc/on-call-enter", json=payload)
        second = client.post("/rc/on-call-enter", json=payload)
    assert first.status_code == 204
    assert second.status_code == 204
    matching = [call for call in mock_enter.call_args_list if call.args[0] == payload]
    assert len(matching) == 1


def test_command_arriving_before_enter_is_retried_after_state_creation(client):
    session_id = f"s-{uuid.uuid4().hex}"
    command_payload = {
        "sessionId": session_id,
        "command": "Play",
        "status": "Completed",
        "partyId": "p-caller",
    }
    enter_payload = {
        "sessionId": session_id,
        "partyId": "p-app",
        "inParty": {"id": "p-caller"},
    }
    ready = {"value": False}

    def command_handler(_payload):
        if not ready["value"]:
            raise RuntimeError("call state not ready")

    def enter_handler(_payload):
        ready["value"] = True

    with (
        patch("ringcentral_router.handle_command_update", side_effect=command_handler) as mock_command,
        patch("ringcentral_router.handle_call_enter", side_effect=enter_handler),
    ):
        first = client.post("/rc/on-command-update", json=command_payload)
        second = client.post("/rc/on-call-enter", json=enter_payload)

    assert first.status_code == 204
    assert second.status_code == 204
    matching = [call for call in mock_command.call_args_list if call.args[0] == command_payload]
    assert len(matching) == 2
