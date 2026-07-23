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

    monkeypatch.setattr(step_enrich, "contextual_search_knowledge", lambda **kwargs: fake_matches)

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
    assert "What you can try" in result["message"] or "symptoms like yours" in result["message"].lower()
    assert "support cases" not in result["message"].lower()
    assert "Toggle the back power switch" in result["message"]
    assert result["message"].rstrip().endswith("do you hear a click?")


def test_build_step_enrichment_boosts_similar_freshdesk(monkeypatch):
    """High-overlap Freshdesk tips should lead, with soft similar-symptom framing."""
    qa = KnowledgeEntry(
        source="qa_csv",
        category="power",
        title="Generic power tip",
        diagnostic="General power note",
        customer_steps=("Confirm the power cord and outlet are working before follow-up.",),
    )
    freshdesk = KnowledgeEntry(
        source="freshdesk",
        category="power",
        title="Chair will not turn on after move",
        diagnostic="Customer said the chair will not turn on and the back switch clicks.",
        customer_steps=(
            "Toggle the back power switch OFF for 10 seconds, then ON.",
            "Try a different wall outlet and verify the cord is seated.",
        ),
    )
    monkeypatch.setattr(
        step_enrich,
        "contextual_search_knowledge",
        lambda **kwargs: [qa, freshdesk],
    )

    ticket = SimpleNamespace(ticket_id="t1", issue_type="defect", model_name="OS-4000T")
    ticket.get_collected = lambda: {
        "intake_summary": "Chair will not turn on and the back switch clicks."
    }
    ticket.set_collected = lambda key, value: None

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("defect", node_id="issue_type"),
            _turn("power", node_id="defect_problem_type"),
        ]
    )
    node = {
        "node_id": "defect_power_back_switch",
        "type": "question",
        "prompt": "When you toggle the back switch, do you hear a click?",
    }

    result = step_enrich.build_step_enrichment(engine, ticket, node)
    assert result is not None
    assert result.get("similar_symptom_match") is True
    assert result["sources"][0] == "freshdesk"
    assert "symptoms like yours" in result["message"].lower()
    assert "Toggle the back power switch" in result["message"]
    assert "support cases" not in result["message"].lower()
    assert "ticket #" not in result["message"].lower()
    assert result.get("top_match") is None


def test_pick_step_tips_prefers_freshdesk_when_flagged():
    qa = KnowledgeEntry(
        source="qa_csv",
        category="power",
        title="QA power",
        diagnostic="qa",
        customer_steps=("Confirm the power cord and outlet are working before follow-up.",),
    )
    freshdesk = KnowledgeEntry(
        source="freshdesk",
        category="power",
        title="FD power",
        diagnostic="fd",
        customer_steps=("Toggle the back power switch OFF for 10 seconds, then ON.",),
    )
    tips = step_enrich._pick_step_tips(
        [qa, freshdesk],
        (),
        prefer_freshdesk=True,
    )
    assert tips
    assert "Toggle the back power switch" in tips[0]


def test_build_step_enrichment_uses_intake_summary_in_search(monkeypatch):
    captured: dict = {}

    def _fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(step_enrich, "contextual_search_knowledge", _fake_search)

    ticket = SimpleNamespace(ticket_id="t1", issue_type="defect", model_name="OS-4000T")
    ticket.get_collected = lambda: {"intake_summary": "Footrest air not inflating."}
    ticket.set_collected = lambda key, value: None

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("defect", node_id="issue_type"),
            _turn("air", node_id="defect_problem_type"),
        ]
    )
    node = {
        "node_id": "defect_air_footrest",
        "type": "question",
        "prompt": "Does air blow through the hose?",
    }

    result = step_enrich.build_step_enrichment(engine, ticket, node)
    assert result is not None
    assert result["phase"] == "workflow_step"
    assert result.get("top_match") is None
    assert result.get("tips")
    assert "Footrest air not inflating" in captured.get("path_text", "")
    assert "You mentioned" in result["message"]


def test_build_step_enrichment_skips_delivery_path(monkeypatch):
    fake_matches = [
        KnowledgeEntry(
            source="freshdesk",
            category="voice",
            title="Voice crackling",
            diagnostic="Voice PCB",
            customer_steps=(
                "Replacing the Voice PCB often fixes crackling speaker issues.",
                "Checking the footrest mechanism can help with any related problems.",
            ),
        )
    ]
    monkeypatch.setattr(step_enrich, "contextual_search_knowledge", lambda **kwargs: fake_matches)

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("delivery", node_id="issue_type"),
            _turn("no_tracking", node_id="delivery_tracking_q"),
        ]
    )
    ticket = SimpleNamespace(ticket_id="t1", issue_type="delivery", model_name="OS-4000T")
    node = {
        "node_id": "delivery_get_name",
        "type": "question_text",
        "prompt": "Please provide your order number OR the email address used at checkout.",
    }

    assert step_enrich.build_step_enrichment(engine, ticket, node) is None


def test_build_step_enrichment_skips_defect_before_category(monkeypatch):
    """Before power/air/etc. is chosen, generic KB often matches wrong tickets."""

    def _should_not_search(**kwargs):
        raise AssertionError("search_knowledge should not run before defect category is known")

    monkeypatch.setattr("warranty_knowledge.search_knowledge", _should_not_search)

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("defect", node_id="issue_type"),
        ]
    )
    ticket = SimpleNamespace(ticket_id="t1", issue_type="defect", model_name="OS-4000T")
    node = {
        "node_id": "defect_problem_type",
        "type": "question",
        "prompt": "What type of problem are you experiencing with your chair?",
    }

    assert step_enrich.build_step_enrichment(engine, ticket, node) is None


def test_format_step_message_keeps_prompt_at_end():
    msg = step_enrich.format_step_message(
        base_prompt="Which part is affected?",
        summary="This looks like an **air inflation** issue.",
        tips=["Check hose connections."],
    )
    assert "Which part is affected?" in msg
    assert msg.index("air inflation") < msg.index("Which part is affected?")


def test_serialize_ticket_state_hides_internal_step_enrichment(monkeypatch):
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
    assert "step_enrichment" not in body

    get_resp = client.get(f"/api/v1/warranty/session/{session_id}")
    assert get_resp.status_code == 200
    get_body = get_resp.json()
    assert get_body.get("assistant_message") == "Freshdesk tip\n\nNext question?"
    assert "step_enrichment" not in get_body


def test_serialize_ticket_state_hides_terminal_source_metadata(monkeypatch):
    from warranty_router import _serialize_ticket_state

    ticket = SimpleNamespace(
        ticket_id="public-terminal",
        status="in_progress",
        issue_type="defect",
        model_name="OS-4000XT",
        customer_message="",
        admin_decision="",
        created_at=None,
    )
    ticket.get_collected = lambda: {}
    engine = SimpleNamespace(can_go_back=lambda ticket_id: False)
    node = {
        "node_id": "terminal-test",
        "type": "terminal",
        "prompt": "Try these checks.",
        "options": [],
    }
    monkeypatch.setattr(
        "warranty_assistant_message.build_assistant_message_bundle",
        lambda **kwargs: {
            "assistant_message": "Try these checks.",
            "step_enrichment": None,
            "terminal_enrichment": {
                "message": "Try these checks.",
                "diagnosis": {
                    "summary": "Remote check.",
                    "steps": ["Reconnect the cable."],
                    "sources": ["freshdesk"],
                    "top_match": "Past ticket subject",
                    "fonz_match": {"error_code": "01"},
                },
            },
        },
    )

    payload = _serialize_ticket_state("session-public", ticket, node, engine=engine)
    public_diagnosis = payload["terminal_enrichment"]["diagnosis"]
    assert public_diagnosis == {
        "summary": "Remote check.",
        "steps": ["Reconnect the cable."],
    }


def test_pick_step_tips_skips_logistics_and_uses_fallback():
    bad = KnowledgeEntry(
        source="freshdesk",
        category="mech",
        title="#2885577 You received a message from Angelica via Warranty Inquiry form",
        diagnostic="Warranty inquiry",
        customer_steps=(
            "Our technician will reach out to arrange a return visit.",
            "We have asked the technician to prioritize your repair.",
        ),
    )
    tips = step_enrich._pick_step_tips(
        [bad],
        (
            "Note which recline function fails and whether the stuck part moves when the chair powers off.",
            "Try the same function from the side panel buttons if your model has them.",
        ),
    )
    assert tips
    assert all("technician" not in t.lower() for t in tips)
    assert any("recline function" in t.lower() for t in tips)


def test_build_step_enrichment_hides_unhelpful_top_match(monkeypatch):
    fake_matches = [
        KnowledgeEntry(
            source="freshdesk",
            category="mech",
            title="#2885577 You received a message from Angelica via Warranty Inquiry form",
            diagnostic="Repair follow-up",
            customer_steps=(
                "Our technician will reach out to arrange a return visit.",
            ),
        )
    ]
    monkeypatch.setattr(step_enrich, "contextual_search_knowledge", lambda **kwargs: fake_matches)

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("defect", node_id="issue_type"),
            _turn("recline", node_id="defect_problem_type"),
        ]
    )
    ticket = SimpleNamespace(
        ticket_id="t1",
        issue_type="defect",
        model_name="Osaki 4D Achilles",
    )
    node = {
        "node_id": "defect_recline_which",
        "type": "question",
        "prompt": "Which recline function is not working?",
    }

    result = step_enrich.build_step_enrichment(engine, ticket, node)
    assert result is not None
    assert result.get("top_match") is None
    assert result.get("tips")
    assert all("technician" not in t.lower() for t in result["tips"])


def test_step_enrichment_hides_unconfirmed_error_code_rows(monkeypatch):
    error_row = KnowledgeEntry(
        source="fonz_error_code",
        category="remote",
        title="4000CS — error 1",
        diagnostic="Remote error code 1",
        customer_steps=("Press and hold a remote key for 40 seconds.",),
    )
    monkeypatch.setattr(
        step_enrich,
        "contextual_search_knowledge",
        lambda **kwargs: [error_row],
    )

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("defect", node_id="issue_type"),
            _turn("remote", node_id="defect_problem_type"),
        ]
    )
    ticket = SimpleNamespace(
        ticket_id="t1",
        issue_type="defect",
        model_name="Osaki OS-4000XT",
    )
    node = {
        "node_id": "defect_remote_power_q",
        "type": "question",
        "prompt": "Does the remote have power?",
    }

    result = step_enrich.build_step_enrichment(engine, ticket, node)

    assert result is not None
    assert result.get("top_match") is None
    assert "fonz_error_code" not in result.get("sources", [])
    assert "4000CS" not in result.get("message", "")


def test_build_step_enrichment_avoids_ungrounded_qa_title(monkeypatch):
    fake_matches = [
        KnowledgeEntry(
            source="qa_csv",
            category="power",
            title="Red blinking light.",
            diagnostic="Blinking red light on chair",
            customer_steps=(
                "Note what happens when you toggle the back power switch.",
                "Confirm the power cord and outlet are working.",
            ),
        )
    ]
    monkeypatch.setattr(step_enrich, "contextual_search_knowledge", lambda **kwargs: fake_matches)

    engine = _FakeEngine(
        [
            _turn("warranty", node_id="root"),
            _turn("defect", node_id="issue_type"),
            _turn("power", node_id="defect_problem_type"),
        ]
    )
    ticket = SimpleNamespace(ticket_id="t1", issue_type="defect", model_name="OS-4000T")
    node = {
        "node_id": "defect_power_remote_on_q",
        "type": "question",
        "prompt": "Does the chair's remote control turn ON when you press the power button?",
    }

    result = step_enrich.build_step_enrichment(engine, ticket, node)
    assert result is not None
    assert "Red blinking light" not in result["message"]
    assert "power" in result["message"].lower()
    assert "What you can try" in result["message"]


def test_title_grounded_requires_customer_overlap():
    assert step_enrich._title_grounded_in_customer_context(
        "Red blinking light.",
        "power issue selected",
    ) is False
    assert step_enrich._title_grounded_in_customer_context(
        "Footrest air not inflating",
        "Footrest air not inflating on my 3D LTX",
    ) is True
