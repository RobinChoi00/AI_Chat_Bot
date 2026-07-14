"""Tests for Freshdesk sync status tracking and dashboard."""

import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


@pytest.fixture
def status_path(tmp_path, monkeypatch):
    path = tmp_path / "freshdesk_sync_status.json"
    monkeypatch.setattr("freshdesk_status._STATUS_PATH", path)
    return path


def test_record_sync_result_persists_ticket_run(status_path):
    from freshdesk_status import _load_status, record_sync_result

    record_sync_result(
        "tickets",
        {
            "ok": True,
            "ticket_count": 12,
            "resolved_scanned": 40,
            "usable_qa_pairs": 12,
            "domain": "example.freshdesk.com",
            "message": "Saved 12 Freshdesk Q&A entries.",
        },
    )

    data = _load_status()
    assert data["tickets"]["ok"] is True
    assert data["tickets"]["ticket_count"] == 12
    assert data["tickets"]["domain"] == "example.freshdesk.com"
    assert data["tickets"]["last_sync_at"]


def test_record_sync_result_persists_kb_run(status_path):
    from freshdesk_status import _load_status, record_sync_result

    record_sync_result(
        "kb",
        {
            "ok": True,
            "article_count": 8,
            "domain": "example.freshdesk.com",
            "message": "Saved 8 Freshdesk KB articles.",
        },
    )

    data = _load_status()
    assert data["kb"]["article_count"] == 8
    assert data["kb"]["ok"] is True


def test_get_freshdesk_dashboard_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("FRESHDESK_DOMAIN", raising=False)
    monkeypatch.delenv("FRESHDESK_API_KEY", raising=False)

    tickets = tmp_path / "freshdesk_tickets.json"
    tickets.write_text(json.dumps([{"question": "q", "answer": "a"}]), encoding="utf-8")
    kb = tmp_path / "freshdesk_solutions.json"
    kb.write_text(json.dumps([{"title": "t", "description_text": "body"}]), encoding="utf-8")

    monkeypatch.setattr("freshdesk_status._TICKETS_PATH", tickets)
    monkeypatch.setattr("freshdesk_status._SOLUTIONS_PATH", kb)
    monkeypatch.setattr("freshdesk_status._STATUS_PATH", tmp_path / "missing_status.json")

    from freshdesk_status import get_freshdesk_dashboard

    dash = get_freshdesk_dashboard(probe_connection=False)
    assert dash["configured"] is False
    assert dash["files"]["tickets"]["count"] == 1
    assert dash["files"]["kb"]["count"] == 1
    assert dash["stale"]["tickets"] is True
    assert dash["stale"]["threshold_days"] == 7
