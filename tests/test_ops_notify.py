"""Tests for ops_notify."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import ops_notify  # noqa: E402


def test_ops_alert_recipients_falls_back_to_config_default(monkeypatch):
    monkeypatch.delenv("OPS_ALERT_EMAIL", raising=False)
    monkeypatch.delenv("WARRANTY_TEAM_EMAIL", raising=False)
    monkeypatch.setattr("config.WARRANTY_TEAM_EMAIL", "service@osakititan.com", raising=False)
    assert ops_notify.ops_alert_recipients() == ["service@osakititan.com"]


def test_send_ops_alert_skips_without_smtp(monkeypatch):
    monkeypatch.setenv("WARRANTY_TEAM_EMAIL", "ops@example.com")
    monkeypatch.setattr("config.EMAIL_SENDER", "", raising=False)
    monkeypatch.setattr("config.EMAIL_PASSWORD", "", raising=False)
    assert ops_notify.send_ops_alert("subject", "body") is False
