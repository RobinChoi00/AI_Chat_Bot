"""
tests/test_chat_feedback.py
===========================
HTTP-level tests for the chat feedback (👍/👎) endpoints.
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    """Route the shared warranty engine/session at an in-memory SQLite instance."""
    import chat_feedback as cf  # noqa: WPS433

    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    mem_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=mem_engine,
        expire_on_commit=False,
    )
    wm.Base.metadata.create_all(bind=mem_engine)

    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", mem_session_factory)
    monkeypatch.setattr(cf, "_ADMIN_API_KEY", "test-admin-key")
    yield


@pytest.fixture
def client():
    from chat_feedback import router  # noqa: WPS433

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_submit_feedback_up_creates_row(client):
    resp = client.post(
        "/api/v1/feedback",
        json={
            "session_id": "s-1",
            "rating": "up",
            "message_content": "Here's how to reset your remote.",
            "context": "warranty",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["rating"] == "up"
    assert isinstance(body["feedback_id"], int)


def test_submit_feedback_dedupes_on_same_message(client):
    payload = {
        "session_id": "s-dedupe",
        "message_content": "Same message body.",
        "context": "warranty",
    }
    first = client.post("/api/v1/feedback", json={**payload, "rating": "up"})
    assert first.status_code == 200
    first_id = first.json()["feedback_id"]

    # Flip to 👎 with a comment — should update the same row.
    second = client.post(
        "/api/v1/feedback",
        json={**payload, "rating": "down", "comment": "Not helpful"},
    )
    assert second.status_code == 200
    assert second.json()["feedback_id"] == first_id
    assert second.json()["rating"] == "down"


def test_submit_feedback_rejects_bad_rating(client):
    resp = client.post(
        "/api/v1/feedback",
        json={
            "session_id": "s-2",
            "rating": "meh",
            "message_content": "content",
        },
    )
    assert resp.status_code == 422


def test_submit_feedback_rejects_empty_message(client):
    resp = client.post(
        "/api/v1/feedback",
        json={"session_id": "s-3", "rating": "up", "message_content": "   "},
    )
    assert resp.status_code == 422


def test_admin_list_and_summary_require_key(client):
    unauth = client.get("/admin/feedback")
    assert unauth.status_code == 401
    unauth2 = client.get("/admin/feedback/summary")
    assert unauth2.status_code == 401


def test_admin_summary_aggregates_ratings(client):
    for _ in range(3):
        client.post(
            "/api/v1/feedback",
            json={
                "session_id": f"s-up-{_}",
                "rating": "up",
                "message_content": f"msg-up-{_}",
            },
        )
    client.post(
        "/api/v1/feedback",
        json={
            "session_id": "s-down",
            "rating": "down",
            "message_content": "bad answer",
            "comment": "Wrong info",
        },
    )

    resp = client.get(
        "/admin/feedback/summary",
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert resp.status_code == 200
    bucket = resp.json()["summary"]["warranty"]
    assert bucket["up"] == 3
    assert bucket["down"] == 1
    assert bucket["total"] == 4
    assert bucket["up_ratio"] == 0.75


def test_admin_list_returns_rows_with_comment(client):
    client.post(
        "/api/v1/feedback",
        json={
            "session_id": "s-with-comment",
            "rating": "down",
            "message_content": "message with feedback comment",
            "comment": "This didn't answer my question.",
        },
    )
    resp = client.get(
        "/admin/feedback?rating=down",
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 1
    row = payload["rows"][0]
    assert row["comment"] == "This didn't answer my question."
    assert row["rating"] == "down"
