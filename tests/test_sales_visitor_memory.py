"""Returning Tidio visitors keep last-chair / height+goal across chats."""

from __future__ import annotations

import sys
from datetime import timedelta
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

PRIMARY = "Osaki OS-Pro Maestro 4D"
VISITOR = "11111111-2222-3333-4444-555555555555"
HEIGHT = 'Average (5\'4"–5\'11")'
GOAL = "Neck & Shoulders"


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
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
    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", mem_session_factory)

    import sales_models  # noqa: F401,WPS433

    wm.Base.metadata.create_all(bind=mem_engine)
    monkeypatch.setattr("sales_models._engine", mem_engine)
    yield


@pytest.fixture
def chat_client():
    from sales_router import router as sales_router  # noqa: WPS433
    from sales_tidio_router import router as tidio_router  # noqa: WPS433

    app = FastAPI()
    app.include_router(sales_router)
    app.include_router(tidio_router)
    return TestClient(app)


@pytest.fixture
def tidio_env(monkeypatch):
    monkeypatch.setenv("TIDIO_ENABLED", "1")
    monkeypatch.setenv("TIDIO_DOMAIN", "osakiusa.com")
    monkeypatch.setenv("TIDIO_PUBLIC_KEY", "pssxmoqpgzitub4925jec9c4nzfbvvam")
    monkeypatch.setenv("TIDIO_WEBHOOK_SECRET", "tidio-test-webhook-secret")
    monkeypatch.delenv("TIDIO_TURN_SECRET", raising=False)


def _memory_patch(**extra):
    data = {
        "pending_primary": PRIMARY,
        "pending_product_url": "https://osakiusa.com/products/osaki-os-pro-maestro-4d",
        "pending_pick_summary": f"{PRIMARY} — catalog",
        "recommend_prefs": {"height": HEIGHT, "goal": GOAL},
        "awaiting_email_for_pick": True,
        "human_offered": True,
    }
    data.update(extra)
    return data


def test_has_resume_memory_requires_chair_or_finished_fit():
    from sales_visitor_memory import has_resume_memory

    assert has_resume_memory({"pending_primary": PRIMARY})
    assert has_resume_memory({"recommend_prefs": {"height": HEIGHT, "goal": GOAL}})
    assert not has_resume_memory({"recommend_prefs": {"height": HEIGHT}})
    assert not has_resume_memory({})
    assert not has_resume_memory(None)


def test_hydrate_copies_last_chair_from_another_session():
    from sales_models import get_or_create_session, get_session_collected, merge_session_collected
    from sales_visitor_memory import hydrate_visitor_memory

    get_or_create_session("conv-1", domain="osakiusa.com", tidio_visitor_id=VISITOR)
    merge_session_collected("conv-1", _memory_patch())

    get_or_create_session("conv-2", domain="osakiusa.com", tidio_visitor_id=VISITOR)
    out = hydrate_visitor_memory("conv-2", VISITOR)

    assert out["pending_primary"] == PRIMARY
    assert out["recommend_prefs"]["height"] == HEIGHT
    assert out["recommend_prefs"]["goal"] == GOAL
    assert out["visitor_memory_hydrated"] is True
    assert out.get("awaiting_email_for_pick") is not True
    assert out.get("human_offered") is not True
    assert get_session_collected("conv-2")["pending_primary"] == PRIMARY


def test_hydrate_does_not_overwrite_in_progress_quiz():
    from sales_models import get_or_create_session, merge_session_collected
    from sales_visitor_memory import hydrate_visitor_memory

    get_or_create_session("conv-1", tidio_visitor_id=VISITOR)
    merge_session_collected("conv-1", _memory_patch())

    get_or_create_session("conv-2", tidio_visitor_id=VISITOR)
    merge_session_collected("conv-2", {"recommend_prefs": {"height": HEIGHT}})
    out = hydrate_visitor_memory("conv-2", VISITOR)

    assert out.get("pending_primary") in (None, "")
    assert out["recommend_prefs"] == {"height": HEIGHT}
    assert not out.get("visitor_memory_hydrated")


def test_hydrate_skips_missing_visitor_and_empty_prior():
    from sales_models import get_or_create_session, merge_session_collected
    from sales_visitor_memory import hydrate_visitor_memory

    get_or_create_session("conv-1", tidio_visitor_id=VISITOR)
    merge_session_collected("conv-1", {"last_quick_replies": [{"label": "Hi", "payload": "menu"}]})

    get_or_create_session("conv-2", tidio_visitor_id=VISITOR)
    empty = hydrate_visitor_memory("conv-2", VISITOR)
    assert not empty.get("pending_primary")
    assert not empty.get("visitor_memory_hydrated")

    get_or_create_session("conv-3")
    none = hydrate_visitor_memory("conv-3", None)
    assert none == {} or not none.get("visitor_memory_hydrated")


def test_hydrate_ignores_stale_sessions():
    from sales_models import SalesSession, get_or_create_session, merge_session_collected
    from sales_visitor_memory import hydrate_visitor_memory
    from warranty_models import _now_cst, warranty_db_session

    get_or_create_session("conv-old", tidio_visitor_id=VISITOR)
    merge_session_collected("conv-old", _memory_patch())
    with warranty_db_session() as db:
        row = db.query(SalesSession).filter(SalesSession.session_id == "conv-old").one()
        row.updated_at = _now_cst() - timedelta(days=40)

    get_or_create_session("conv-new", tidio_visitor_id=VISITOR)
    out = hydrate_visitor_memory("conv-new", VISITOR)
    assert not out.get("pending_primary")
    assert not out.get("visitor_memory_hydrated")


def test_hydrate_does_not_restore_after_start_over():
    from sales_models import get_or_create_session, merge_session_collected
    from sales_visitor_memory import hydrate_visitor_memory

    get_or_create_session("conv-1", tidio_visitor_id=VISITOR)
    merge_session_collected("conv-1", _memory_patch())

    get_or_create_session("conv-2", tidio_visitor_id=VISITOR)
    merge_session_collected(
        "conv-2",
        {
            "visitor_resume_declined": True,
            "recommend_prefs": None,
            "pending_primary": "",
        },
    )
    out = hydrate_visitor_memory("conv-2", VISITOR)
    assert out.get("visitor_resume_declined") is True
    assert not out.get("pending_primary")


def test_greeting_offers_resume_and_keeps_human_last():
    from sales_agent import respond

    result = respond(
        "hello",
        domain="osakiusa.com",
        prefs=_memory_patch(),
    )
    assert result.intent == "greeting"
    assert "still looking at" in result.reply.lower()
    assert "maestro" in result.reply.lower()
    assert "average (5" not in result.reply.lower()
    payloads = [q.payload for q in result.quick_replies]
    assert payloads[0] == "resume:continue"
    assert "resume:picks" in payloads
    assert "resume:reset" in payloads
    assert payloads[-1] == "human"
    assert payloads[0] != "human"
    assert "visitor.resume_offer" in (result.tools_used or [])


def test_fresh_hello_still_uses_generic_menu():
    from sales_agent import respond

    result = respond("hello", domain="osakiusa.com", prefs={})
    payloads = [q.payload for q in result.quick_replies]
    assert payloads[0] == "recommend"
    assert payloads[-1] == "human"
    assert "welcome back" not in result.reply.lower()


def test_resume_continue_returns_product_card():
    from sales_agent import respond

    result = respond(
        "",
        payload="resume:continue",
        domain="osakiusa.com",
        prefs={"pending_primary": PRIMARY},
    )
    assert result.intent == "specs"
    assert "still looking at" in result.reply.lower()
    assert "maestro" in result.reply.lower()
    assert "full quick specs" not in result.reply.lower()
    assert "heating:" not in result.reply.lower()
    payloads = [q.payload for q in result.quick_replies]
    assert any(p.startswith("open:") for p in payloads)
    assert "specs:osaki-os-pro-maestro-4d" in payloads
    assert payloads[-1] == "human"
    assert payloads[0] != "human"
    assert (result.prefs_patch or {}).get("visitor_resume_accepted") is True
    assert "visitor.resume_continue" in (result.tools_used or [])


def test_resume_picks_skips_height_question():
    from sales_agent import respond

    result = respond(
        "",
        payload="resume:picks",
        domain="osakiusa.com",
        prefs={"recommend_prefs": {"height": HEIGHT, "goal": GOAL}},
    )
    assert result.intent == "recommend"
    assert "user height" not in result.reply.lower()
    assert "fits neck" not in result.reply.lower()
    blurbs = [
        line
        for line in result.reply.splitlines()
        if " · " in line and not line.startswith("**") and "http" not in line.lower()
    ]
    assert len(blurbs) >= 2
    assert len(set(blurbs)) == len(blurbs)
    assert (result.prefs_patch or {}).get("visitor_resume_accepted") is True


def test_resume_reset_clears_memory_then_recommend_asks_height():
    from sales_agent import respond

    reset = respond(
        "",
        payload="resume:reset",
        domain="osakiusa.com",
        prefs=_memory_patch(),
    )
    assert reset.intent == "greeting"
    assert "recommend a chair" in reset.reply.lower() or "height" in reset.reply.lower()
    patch = reset.prefs_patch or {}
    assert patch.get("visitor_resume_declined") is True
    assert patch.get("recommend_prefs") is None
    assert patch.get("pending_primary") == ""

    declined = dict(_memory_patch())
    declined.update(patch)
    hello = respond("hello", domain="osakiusa.com", prefs=declined)
    assert "welcome back" not in hello.reply.lower()
    assert [q.payload for q in hello.quick_replies][0] == "recommend"

    ask = respond(
        "",
        payload="recommend",
        domain="osakiusa.com",
        prefs=declined,
    )
    assert "user height" in ask.reply.lower()


def test_chat_hydrates_across_sessions(chat_client):
    first = chat_client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "web-1",
            "tidio_visitor_id": VISITOR,
            "message": "tell me about the Maestro",
            "domain": "osakiusa.com",
        },
    )
    assert first.status_code == 200, first.text
    assert first.json()["intent"] == "specs"

    hello = chat_client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "web-2",
            "tidio_visitor_id": VISITOR,
            "message": "hello",
            "domain": "osakiusa.com",
        },
    )
    assert hello.status_code == 200, hello.text
    body = hello.json()
    assert body["intent"] == "greeting"
    assert "still looking at" in body["reply"].lower()
    payloads = [q["payload"] for q in body["quick_replies"]]
    assert payloads[0] == "resume:continue"
    assert payloads[-1] == "human"
    assert "visitor.resume_offer" in body["tools_used"]

    cont = chat_client.post(
        "/api/v1/sales/chat",
        json={
            "session_id": "web-2",
            "tidio_visitor_id": VISITOR,
            "message": "",
            "payload": "resume:continue",
            "domain": "osakiusa.com",
        },
    )
    assert cont.status_code == 200, cont.text
    assert cont.json()["intent"] == "specs"
    assert "maestro" in cont.json()["reply"].lower()


def test_tidio_new_conversation_offers_resume(chat_client, tidio_env):
    look = chat_client.post(
        "/api/v1/sales/tidio/turn",
        json={
            "contact_id": VISITOR,
            "session_id": "tidio-conv-1",
            "message": "tell me about the Maestro",
        },
    )
    assert look.status_code == 200, look.text
    assert look.json()["intent"] == "specs"

    hello = chat_client.post(
        "/api/v1/sales/tidio/turn",
        json={
            "contact_id": VISITOR,
            "session_id": "tidio-conv-2",
            "message": "hello",
        },
    )
    assert hello.status_code == 200, hello.text
    body = hello.json()
    assert body["intent"] == "greeting"
    assert "still looking at" in body["reply"].lower()
    assert body["button_1_payload"] == "resume:continue"
    assert body["button_1_label"] != "Talk to a human"
    labels = [q["label"] for q in body["quick_replies"]]
    assert labels[-1] == "Talk to a human"
    assert body["quick_replies"][-1]["payload"] == "human"

    reset = chat_client.post(
        "/api/v1/sales/tidio/turn",
        json={
            "contact_id": VISITOR,
            "session_id": "tidio-conv-2",
            "payload": "resume:reset",
            "message": "start over",
        },
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["button_1_payload"] == "recommend"

    again = chat_client.post(
        "/api/v1/sales/tidio/turn",
        json={
            "contact_id": VISITOR,
            "session_id": "tidio-conv-2",
            "payload": "recommend",
            "message": "recommend",
        },
    )
    assert again.status_code == 200, again.text
    assert "user height" in again.json()["reply"].lower()
