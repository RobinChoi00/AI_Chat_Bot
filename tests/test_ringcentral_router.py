"""
tests/test_ringcentral_router.py
================================
HTTP tests for RingCentral webhook endpoints (handlers mocked).
"""

import sys
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


def test_on_call_enter_returns_204(client):
    payload = {
        "sessionId": "s-test",
        "partyId": "p-app",
        "inParty": {
            "id": "p-caller",
            "from": {"phoneNumber": "+15551234567"},
        },
    }
    with patch("ringcentral_router.handle_call_enter") as mock_enter:
        res = client.post("/rc/on-call-enter", json=payload)
    assert res.status_code == 204
    mock_enter.assert_called_once()


def test_on_command_update_returns_204(client):
    payload = {
        "sessionId": "s-test",
        "command": "Play",
        "status": "Completed",
        "partyId": "p-caller",
    }
    with patch("ringcentral_router.handle_command_update") as mock_update:
        res = client.post("/rc/on-command-update", json=payload)
    assert res.status_code == 204
    mock_update.assert_called_once()
