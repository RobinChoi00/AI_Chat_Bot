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
    monkeypatch.setattr(wk, "_load_freshdesk_kb_entries", lambda: [])
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
        with patch(
            "warranty_faiss_rebuilder.get_status", return_value={"running": False}
        ):
            with patch("warranty_faiss_rebuilder.rebuild_freshdesk_qa_index") as rebuild:
                res = admin_client.post(
                    "/admin/warranty/sync-freshdesk?rebuild_faiss=false",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["ticket_count"] == 2
    assert "knowledge_total_entries" in body
    assert "knowledge_yield" in body
    rebuild.assert_not_called()


def test_admin_sync_freshdesk_schedules_faiss_by_default(admin_client):
    fake_result = {
        "ok": True,
        "ticket_count": 2,
        "resolved_scanned": 10,
        "output_path": "/tmp/freshdesk_tickets.json",
        "domain": "example.freshdesk.com",
        "message": "Saved 2 Freshdesk Q&A entries.",
    }

    with patch("freshdesk_sync.sync_freshdesk_knowledge", return_value=fake_result):
        with patch(
            "warranty_faiss_rebuilder.get_status", return_value={"running": False}
        ):
            with patch("warranty_faiss_rebuilder.rebuild_freshdesk_qa_index") as rebuild:
                res = admin_client.post(
                    "/admin/warranty/sync-freshdesk",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )

    assert res.status_code == 200
    assert res.json().get("faiss_rebuild_scheduled") is True
    rebuild.assert_called_once()


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


def test_admin_probe_solutions_success(admin_client):
    payload = {"reachable": True, "categories": 2, "folders": 4, "articles": 30}
    with patch("freshdesk_sync.probe_freshdesk_solutions", return_value=payload):
        res = admin_client.get(
            "/admin/warranty/freshdesk-solutions/probe",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
    assert res.status_code == 200
    assert res.json()["articles"] == 30


def test_admin_probe_solutions_requires_key(admin_client):
    res = admin_client.get("/admin/warranty/freshdesk-solutions/probe")
    assert res.status_code == 401


def test_admin_sync_solutions_success(admin_client):
    payload = {
        "ok": True,
        "article_count": 5,
        "output_path": "/tmp/freshdesk_solutions.json",
        "message": "Saved 5 KB articles.",
    }
    with patch("freshdesk_sync.sync_freshdesk_solutions", return_value=payload):
        res = admin_client.post(
            "/admin/warranty/sync-freshdesk-solutions",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["article_count"] == 5
    assert "knowledge_freshdesk_kb_entries" in body


def test_admin_rebuild_faiss_schedules(admin_client, monkeypatch):
    import warranty_faiss_rebuilder as fr

    monkeypatch.setattr(fr, "get_status", lambda: {"running": False})

    called = {"n": 0}

    def _stub():
        called["n"] += 1
        return {"ok": True}

    monkeypatch.setattr(fr, "rebuild_freshdesk_qa_index", _stub)
    res = admin_client.post(
        "/admin/warranty/rebuild-faiss",
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["scheduled"] is True
    # BackgroundTasks run after the response is generated.
    assert called["n"] == 1


def test_admin_rebuild_faiss_skips_when_running(admin_client, monkeypatch):
    import warranty_faiss_rebuilder as fr

    monkeypatch.setattr(fr, "get_status", lambda: {"running": True, "ok": False})
    res = admin_client.post(
        "/admin/warranty/rebuild-faiss",
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert res.status_code == 200
    assert res.json()["scheduled"] is False


def test_admin_rebuild_faiss_wait_mode(admin_client, monkeypatch):
    import warranty_faiss_rebuilder as fr

    monkeypatch.setattr(fr, "get_status", lambda: {"running": False})
    monkeypatch.setattr(
        fr,
        "rebuild_freshdesk_qa_index",
        lambda: {"ok": True, "total_docs": 42},
    )
    res = admin_client.post(
        "/admin/warranty/rebuild-faiss?wait=true",
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["total_docs"] == 42
