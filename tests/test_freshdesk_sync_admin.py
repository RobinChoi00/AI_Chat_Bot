"""Tests for Freshdesk sync admin endpoint and knowledge cache."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_knowledge as wk  # noqa: E402
import warranty_router as wr  # noqa: E402

ADMIN_KEY = "test-admin-key-fd"


@pytest.fixture()
def admin_client(monkeypatch):
    monkeypatch.setattr(wr, "_ADMIN_API_KEY", ADMIN_KEY)
    app = FastAPI()
    app.include_router(wr.router)
    return TestClient(app)


def test_clear_knowledge_cache(monkeypatch):
    calls = {"n": 0}

    def counting_load():
        calls["n"] += 1
        return []

    monkeypatch.setattr(wk, "_load_qa_entries", counting_load)
    monkeypatch.setattr(wk, "_load_freshdesk_entries", lambda: [])
    monkeypatch.setattr(wk, "_load_autocheck_entries", lambda: [])

    wk.clear_knowledge_cache()
    wk.load_knowledge_entries()
    wk.load_knowledge_entries()
    assert calls["n"] == 1

    wk.clear_knowledge_cache()
    wk.load_knowledge_entries()
    assert calls["n"] == 2


def test_admin_sync_freshdesk_requires_key(admin_client):
    res = admin_client.post("/admin/warranty/sync-freshdesk")
    assert res.status_code == 401


def test_admin_sync_freshdesk_success(admin_client):
    fake_result = {
        "ok": True,
        "ticket_count": 2,
        "output_path": "/tmp/freshdesk_tickets.json",
        "domain": "example.freshdesk.com",
        "message": "Saved 2 Freshdesk Q&A entries.",
    }

    with patch("freshdesk_sync.sync_freshdesk_knowledge", return_value=fake_result):
        res = admin_client.post(
            "/admin/warranty/sync-freshdesk",
            headers={"X-Admin-Key": ADMIN_KEY},
        )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["ticket_count"] == 2
    assert "knowledge_total_entries" in body


def test_admin_sync_freshdesk_missing_env(admin_client):
    with patch(
        "freshdesk_sync.sync_freshdesk_knowledge",
        side_effect=EnvironmentError("FRESHDESK_API_KEY is not set."),
    ):
        res = admin_client.post(
            "/admin/warranty/sync-freshdesk",
            headers={"X-Admin-Key": ADMIN_KEY},
        )

    assert res.status_code == 503
    assert "FRESHDESK_API_KEY" in res.json()["detail"]
