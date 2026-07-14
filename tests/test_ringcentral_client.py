"""
tests/test_ringcentral_client.py
================================
Unit tests for RingCentral client helpers (no live API).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import requests

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import ringcentral_client as rc  # noqa: E402


def test_forward_payload_prefers_extension_number(monkeypatch):
    monkeypatch.setattr(rc, "RC_WARRANTY_TRANSFER_EXTENSION", "103")
    monkeypatch.setattr(rc, "RC_WARRANTY_TRANSFER_TO", "+18888482630")
    assert rc._forward_payload("") == {"extensionNumber": "103"}


def test_forward_payload_uses_phone_when_no_extension(monkeypatch):
    monkeypatch.setattr(rc, "RC_WARRANTY_TRANSFER_EXTENSION", "")
    monkeypatch.setattr(rc, "RC_WARRANTY_TRANSFER_TO", "+18888482630")
    assert rc._forward_payload("") == {"phoneNumber": "+18888482630"}


def test_hangup_uses_delete(monkeypatch):
    class FakeResp:
        status_code = 204
        text = ""

    with patch.object(rc, "_request", return_value=FakeResp()) as mock_req:
        rc.hangup(session_id="s-1", party_id="p-1")
    mock_req.assert_called_once_with("DELETE", rc._party_url("s-1", "p-1"))


def test_request_retries_429_and_honors_retry_after(monkeypatch):
    class FakeResp:
        def __init__(self, status_code, retry_after=""):
            self.status_code = status_code
            self.headers = {"Retry-After": retry_after} if retry_after else {}

    monkeypatch.setenv("RC_API_MAX_RETRIES", "2")
    with (
        patch.object(rc, "_auth_headers", return_value={}),
        patch.object(
            rc.requests,
            "request",
            side_effect=[FakeResp(429, "1.5"), FakeResp(200)],
        ) as mock_request,
        patch.object(rc.time, "sleep") as mock_sleep,
    ):
        response = rc._request("GET", "https://example.invalid")

    assert response.status_code == 200
    assert mock_request.call_count == 2
    mock_sleep.assert_called_once_with(1.5)


def test_request_retries_transient_network_error(monkeypatch):
    class FakeResp:
        status_code = 204
        headers = {}

    monkeypatch.setenv("RC_API_MAX_RETRIES", "1")
    with (
        patch.object(rc, "_auth_headers", return_value={}),
        patch.object(
            rc.requests,
            "request",
            side_effect=[requests.ConnectionError("temporary"), FakeResp()],
        ),
        patch.object(rc.time, "sleep") as mock_sleep,
    ):
        response = rc._request("POST", "https://example.invalid", json_body={"x": 1})

    assert response.status_code == 204
    mock_sleep.assert_called_once()
