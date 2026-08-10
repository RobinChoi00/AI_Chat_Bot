"""
tests/test_freshdesk_solutions.py
=================================
Freshdesk KB (Solutions) probe + ingest + knowledge loader integration.

Network is fully mocked via ``requests.get``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import freshdesk_sync as fs  # noqa: E402
import warranty_knowledge as wk  # noqa: E402


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


def _wire_routes(monkeypatch, routes):
    """Route (url substring) → payload dict. Longest needle wins."""
    ordered = sorted(routes.items(), key=lambda kv: len(kv[0]), reverse=True)

    def _fake_get(url, **kwargs):
        for needle, resp in ordered:
            if needle in url:
                return resp
        return _FakeResp(404, {})

    monkeypatch.setattr(fs.requests, "get", _fake_get)


def test_probe_reports_category_and_article_counts(monkeypatch):
    _install_env(monkeypatch)
    _wire_routes(
        monkeypatch,
        {
            f"/api/v2/tickets": _FakeResp(200, [{"id": 1}]),
            f"/solutions/categories": _FakeResp(200, [{"id": 10, "name": "General"}]),
            f"/solutions/categories/10/folders": _FakeResp(
                200,
                [
                    {"id": 100, "name": "Setup", "articles_count": 3},
                    {"id": 101, "name": "Troubleshooting", "articles_count": 5},
                ],
            ),
        },
    )

    result = fs.probe_freshdesk_solutions()
    assert result["reachable"] is True
    assert result["categories"] == 1
    assert result["folders"] == 2
    assert result["articles"] == 8


def test_iter_solution_articles_yields_published_only(monkeypatch):
    _install_env(monkeypatch)
    _wire_routes(
        monkeypatch,
        {
            f"/api/v2/tickets": _FakeResp(200, [{"id": 1}]),
            f"/solutions/categories": _FakeResp(
                200, [{"id": 10, "name": "General"}]
            ),
            f"/solutions/categories/10/folders": _FakeResp(
                200, [{"id": 100, "name": "Setup"}]
            ),
            f"/solutions/folders/100/articles": _FakeResp(
                200,
                [
                    {
                        "id": 1001,
                        "title": "Draft article",
                        "status": 1,
                        "description_text": "Should be skipped",
                    },
                    {
                        "id": 1002,
                        "title": "How to reset power",
                        "status": 2,
                        "description_text": (
                            "1) Toggle the back power switch OFF for 10 seconds.\n"
                            "2) Try a different wall outlet.\n"
                            "3) Check the fuse behind the panel."
                        ),
                        "tags": ["power", "reset"],
                    },
                ],
            ),
        },
    )

    etl = fs.FreshdeskETL()
    articles = list(etl.iter_solution_articles(max_articles=10))
    assert len(articles) == 1
    assert articles[0]["title"] == "How to reset power"
    assert "reset" in articles[0]["tags"]


def test_sync_freshdesk_solutions_writes_file(monkeypatch, tmp_path):
    _install_env(monkeypatch)
    out = tmp_path / "solutions.json"
    monkeypatch.setattr(fs, "_SOLUTIONS_PATH", out)
    _wire_routes(
        monkeypatch,
        {
            f"/api/v2/tickets": _FakeResp(200, [{"id": 1}]),
            f"/solutions/categories": _FakeResp(
                200, [{"id": 10, "name": "General"}]
            ),
            f"/solutions/categories/10/folders": _FakeResp(
                200, [{"id": 100, "name": "Setup"}]
            ),
            f"/solutions/folders/100/articles": _FakeResp(
                200,
                [
                    {
                        "id": 1002,
                        "title": "How to reset power",
                        "status": 2,
                        "description_text": (
                            "Unplug the chair for 30 seconds, then plug back in. "
                            "Toggle the back power switch OFF and ON."
                        ),
                    }
                ],
            ),
        },
    )
    result = fs.sync_freshdesk_solutions(max_articles=100)
    assert result["ok"] is True
    assert result["article_count"] == 1
    assert out.is_file()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved[0]["title"] == "How to reset power"


def test_knowledge_loader_ingests_kb_articles(tmp_path, monkeypatch):
    kb_path = tmp_path / "solutions.json"
    kb_path.write_text(
        json.dumps(
            [
                {
                    "article_id": 1002,
                    "category": "Troubleshooting",
                    "folder": "Power",
                    "title": "How to reset power on the OS-4000T",
                    "description_text": (
                        "Unplug the chair for 30 seconds, then plug back in. "
                        "Toggle the back power switch OFF and ON. "
                        "Verify the wall outlet with another device."
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wk, "_FRESHDESK_KB_PATH", kb_path)
    wk.load_knowledge_entries.cache_clear()

    kb_entries = [e for e in wk.load_knowledge_entries() if e.source == "freshdesk_kb"]
    assert kb_entries, "KB article should surface at least one knowledge entry"
    assert kb_entries[0].category == "power"
    assert any("back power switch" in " ".join(e.customer_steps).lower() for e in kb_entries)
    assert all(wk._is_customer_safe_step(s) for s in kb_entries[0].customer_steps)


def test_knowledge_loader_skips_kb_without_diy_steps(tmp_path, monkeypatch):
    kb_path = tmp_path / "solutions.json"
    kb_path.write_text(
        json.dumps(
            [
                {
                    "article_id": 9,
                    "category": "Policy",
                    "folder": "Warranty",
                    "title": "Warranty coverage overview",
                    "description_text": (
                        "This policy document explains coverage terms for massage chairs "
                        "purchased through authorized dealers in the United States."
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wk, "_FRESHDESK_KB_PATH", kb_path)
    monkeypatch.setattr(wk, "_FRESHDESK_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(wk, "_QA_PATH", tmp_path / "qa.csv")
    monkeypatch.setattr(wk, "_AUTOCHECK_PATH", tmp_path / "ac.csv")
    wk.load_knowledge_entries.cache_clear()
    kb_entries = [e for e in wk.load_knowledge_entries() if e.source == "freshdesk_kb"]
    assert kb_entries == []


def test_kb_entries_are_returned_by_search(tmp_path, monkeypatch):
    kb_path = tmp_path / "solutions.json"
    kb_path.write_text(
        json.dumps(
            [
                {
                    "article_id": 1,
                    "category": "Troubleshooting",
                    "folder": "Power",
                    "title": "Chair will not power on troubleshooting",
                    "description_text": (
                        "Check the wall outlet is delivering power. "
                        "Unplug the chair for 30 seconds and plug it back in. "
                        "Toggle the back power switch OFF and ON."
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wk, "_FRESHDESK_KB_PATH", kb_path)
    monkeypatch.setattr(wk, "_FRESHDESK_PATH", tmp_path / "does-not-exist.json")
    monkeypatch.setattr(wk, "_QA_PATH", tmp_path / "qa.csv")
    monkeypatch.setattr(wk, "_AUTOCHECK_PATH", tmp_path / "ac.csv")
    wk.load_knowledge_entries.cache_clear()

    results = wk.search_knowledge(
        path_text="chair will not power on wall outlet",
        defect_category="power",
        model_name="OS-4000T",
    )
    assert any(r.source == "freshdesk_kb" for r in results)
