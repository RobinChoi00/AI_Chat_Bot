"""
tests/test_freshdesk_sync_search.py
===================================
Freshdesk ticket sync via Search API (Resolved/Closed only).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import freshdesk_sync as fs  # noqa: E402


class _FakeResp:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise fs.requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _install_env(monkeypatch):
    monkeypatch.setenv("FRESHDESK_DOMAIN", "acme.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")


def test_build_resolved_search_query():
    q = fs.build_resolved_search_query("2025-06-01", "2025-07-01")
    assert "status:4 OR status:5" in q
    assert "created_at:>'2025-06-01'" in q
    assert "created_at:<'2025-07-01'" in q


def test_iter_month_windows_newest_first():
    windows = list(fs.iter_month_windows(3))
    assert len(windows) == 3
    assert windows[0][0] < windows[0][1]
    assert windows[0][0] >= windows[1][0]


def test_fetch_resolved_tickets_uses_search_api(monkeypatch):
    _install_env(monkeypatch)
    monkeypatch.setattr(fs.time, "sleep", lambda _s: None)

    calls: list[str] = []

    def _fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith("/tickets") and "search" not in url:
            return _FakeResp(200, [{"id": 1}])
        if "/search/tickets" in url:
            return _FakeResp(
                200,
                {
                    "total": 1,
                    "results": [
                        {
                            "id": 101,
                            "status": 5,
                            "subject": "Power issue",
                            "description_text": "Chair will not turn on.",
                        }
                    ],
                },
            )
        if url.endswith("/tickets/101/conversations"):
            return _FakeResp(
                200,
                [
                    {
                        "incoming": False,
                        "private": False,
                        "body_text": "Toggle the back power switch OFF and ON.",
                    }
                ],
            )
        return _FakeResp(404, {})

    monkeypatch.setattr(fs.requests, "get", _fake_get)

    etl = fs.FreshdeskETL()
    tickets, stats = etl.fetch_resolved_tickets(months_back=1, max_pages=1)

    assert any("/search/tickets" in u for u in calls)
    assert stats["fetch_mode"] == "search"
    assert stats["resolved_scanned"] == 1
    assert stats["usable_qa_pairs"] == 1
    assert len(tickets) == 1
    assert tickets[0]["ticket_id"] == 101


def test_fetch_dedupes_ticket_ids(monkeypatch):
    _install_env(monkeypatch)
    monkeypatch.setattr(fs.time, "sleep", lambda _s: None)

    search_calls = {"n": 0}

    def _fake_get(url, **kwargs):
        if url.endswith("/tickets") and "search" not in url:
            return _FakeResp(200, [{"id": 1}])
        if "/search/tickets" in url:
            search_calls["n"] += 1
            return _FakeResp(
                200,
                {
                    "total": 2,
                    "results": [
                        {
                            "id": 55,
                            "status": 5,
                            "subject": "A",
                            "description_text": "Question one?",
                        },
                        {
                            "id": 55,
                            "status": 5,
                            "subject": "A dup",
                            "description_text": "Question one again?",
                        },
                    ],
                },
            )
        if url.endswith("/tickets/55/conversations"):
            return _FakeResp(
                200,
                [{"incoming": False, "private": False, "body_text": "Try outlet B."}],
            )
        return _FakeResp(404, {})

    monkeypatch.setattr(fs.requests, "get", _fake_get)

    etl = fs.FreshdeskETL()
    tickets, stats = etl.fetch_resolved_tickets(months_back=1, max_pages=1)

    assert stats["resolved_scanned"] == 1
    assert len(tickets) == 1


def test_fetch_respects_pages_budget(monkeypatch):
    _install_env(monkeypatch)
    monkeypatch.setattr(fs.time, "sleep", lambda _s: None)

    search_pages: list[int] = []

    def _fake_get(url, **kwargs):
        if url.endswith("/tickets") and "search" not in url:
            return _FakeResp(200, [{"id": 1}])
        if "/search/tickets" in url:
            page = kwargs.get("params", {}).get("page", 1)
            search_pages.append(page)
            tid = 1000 + page
            return _FakeResp(
                200,
                {
                    "total": 999,
                    "results": [
                        {
                            "id": tid,
                            "status": 5,
                            "subject": f"T{tid}",
                            "description_text": f"Question {tid}?",
                        }
                    ],
                },
            )
        if "/conversations" in url:
            return _FakeResp(
                200,
                [{"incoming": False, "private": False, "body_text": "Agent reply."}],
            )
        return _FakeResp(404, {})

    monkeypatch.setattr(fs.requests, "get", _fake_get)

    etl = fs.FreshdeskETL()
    _tickets, stats = etl.fetch_resolved_tickets(months_back=2, max_pages=2)

    assert stats["search_pages_fetched"] == 2
    assert len(search_pages) == 2


def test_sync_freshdesk_knowledge_includes_fetch_stats(monkeypatch, tmp_path):
    _install_env(monkeypatch)
    monkeypatch.setattr(fs, "_OUTPUT_PATH", tmp_path / "tickets.json")

    fake_stats = {
        "resolved_scanned": 10,
        "usable_qa_pairs": 2,
        "search_pages_fetched": 3,
        "month_windows_scanned": 1,
        "fetch_mode": "search",
    }

    with patch.object(
        fs.FreshdeskETL,
        "fetch_resolved_tickets",
        return_value=(
            [
                {
                    "ticket_id": 1,
                    "subject": "S",
                    "question": "Q?",
                    "answer": "A.",
                }
            ],
            fake_stats,
        ),
    ):
        result = fs.sync_freshdesk_knowledge(max_pages=3, months_back=6)

    assert result["ok"] is True
    assert result["fetch_mode"] == "search"
    assert result["resolved_scanned"] == 10
    assert result["search_pages_fetched"] == 3
