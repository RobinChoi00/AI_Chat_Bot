"""
tests/test_warranty_step_enrichment.py
======================================
Non-terminal Freshdesk-backed step enrichment for button-driven workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_step_enrichment as step_enrich  # noqa: E402
from warranty_knowledge import KnowledgeEntry  # noqa: E402


class _FakeEngine:
    def __init__(self, turns):
        self._turns = turns

    def get_turns(self, ticket_id: str):
        return self._turns


def _turn(answer_key: str, *, node_id: str = "defect_problem_type", prompt: str = ""):
    return SimpleNamespace(
        answer_key=answer_key,
        customer_answer=answer_key,
        node_id=node_id,
        node_prompt=prompt,
    )


def test_build_step_enrichment_skips_root():
    engine = _FakeEngine([_turn("warranty", node_id="root")])
    ticket = SimpleNamespace(ticket_id="t1", issue_type="defect", model_name="OS-4000T")
    node = {"node_id": "root", "type": "question", "prompt": "How can we help?"}
    assert step_enrich.build_step_enrichment(engine, ticket, node) is None


def test_build_step_enrichment_skips_without_turns():
    engine = _FakeEngine([])
    ticket = SimpleNamespace(ticket_id="t1", issue_type="defect", model_name="OS-4000T")
    node = {"node_id": "defect_power", "type": "question", "prompt": "What happens?"}
    assert step_enrich.build_step_enrichment(engine, ticket, node) is None


def test_build_step_enrichment_skips_before_issue_type_selected():
    engine = _FakeEngine([_turn("warranty", node_id="root")])
    ticket = SimpleNamespace(ticket_id="t1", issue_type="", model_name="Maestro")
    node = {
        "node_id": "issue_type",
        "type": "question",
        "prompt": "What type of problem are you experiencing with your chair?",
    }
    assert step_enrich.build_step_enrichment(engine, ticket, node) is None


def test_build_step_enrichment_uses_freshdesk_tips(monkeypatch):
    fake_matches = [
        KnowledgeEntry(
            source="freshdesk",
            category="power",
            title="Power not turning on",
            diagnostic="Chair does not power on",
            customer_steps=(
                "Toggle the back power switch OFF for 10 seconds, then ON.",
                "Try a different wall outlet.",
            ),
        )
    ]

    monkeypatch.setattr(step_enrich, "search_knowledge", lambda **kwargs: fake_matches)

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("defect", node_id="issue_type"),
            _turn("power", node_id="defect_problem_type"),
        ]
    )
    ticket = SimpleNamespace(ticket_id="t1", issue_type="defect", model_name="OS-4000T")
    node = {
        "node_id": "defect_power_back_switch",
        "type": "question",
        "prompt": "When you toggle the back switch, do you hear a click?",
    }

    result = step_enrich.build_step_enrichment(engine, ticket, node)
    assert result is not None
    assert result["phase"] == "workflow_step"
    assert result["sources"] == ["freshdesk"]
    assert "From similar support cases" in result["message"]
    assert "Toggle the back power switch" in result["message"]
    assert result["message"].rstrip().endswith("do you hear a click?")


def test_format_step_message_keeps_prompt_at_end():
    msg = step_enrich.format_step_message(
        base_prompt="Which part is affected?",
        summary="This looks like an **air inflation** issue.",
        tips=["Check hose connections."],
    )
    assert "Which part is affected?" in msg
    assert msg.index("air inflation") < msg.index("Which part is affected?")


def test_serialize_ticket_state_includes_step_enrichment(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import warranty_models as wm
    import warranty_workflow as wf
    from warranty_router import router

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
    monkeypatch.setattr(wf, "_SessionFactory", mem_session_factory)

    fake_step = {
        "message": "Freshdesk tip\n\nNext question?",
        "phase": "workflow_step",
        "sources": ["freshdesk"],
    }
    monkeypatch.setattr(
        "warranty_step_enrichment.build_step_enrichment",
        lambda engine, ticket, node: fake_step,
    )

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    session_id = "step-enrich-api"
    reg = client.post(
        f"/api/v1/warranty/session/{session_id}/register-model",
        json={"model": "OS-4000T", "domain": "osaki.com"},
    )
    assert reg.status_code == 200

    resp = client.post(
        f"/api/v1/warranty/session/{session_id}/quick-start",
        json={"issue_type": "defect", "domain": "osaki.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("assistant_message") == "Freshdesk tip\n\nNext question?"
    assert body.get("step_enrichment", {}).get("phase") == "workflow_step"
