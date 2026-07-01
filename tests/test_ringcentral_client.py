"""
tests/test_ringcentral_client.py
================================
Unit tests for RingCentral client helpers (no live API).
"""

import sys
from pathlib import Path
from unittest.mock import patch

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
