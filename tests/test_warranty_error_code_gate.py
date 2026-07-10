"""Tests for pre-terminal error-code gate (engine intercept)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm
import warranty_error_code_gate as gate
from warranty_workflow import WarrantyEngine


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    import warranty_workflow as wf

    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
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
    yield


def _walk(ticket_id: str, answers: list[str]) -> dict:
    result: dict = {}
    for ans in answers:
        result = WarrantyEngine.submit_answer(ticket_id, ans)
    return result


def test_intercept_before_air_pump_terminal_for_supported_model():
    ticket_id, _ = WarrantyEngine.start_session("gate-air", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "3D LTX")

    result = _walk(ticket_id, [
        "warranty",
        "defect",
        "air",
        "feet_calves",
        "never_worked",
    ])

    assert result["next_node_id"] == gate.GATE_VISIBLE_ID
    assert result["is_terminal"] is False

    t = WarrantyEngine.get_ticket(ticket_id)
    assert t is not None
    assert t.get_collected().get(gate.COL_PENDING_TERMINAL) == "defect_air_pump_terminal"
    assert str(t.status) == "in_progress"


def test_gate_no_reaches_original_terminal():
    ticket_id, _ = WarrantyEngine.start_session("gate-no", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "3D LTX")

    _walk(ticket_id, [
        "warranty",
        "defect",
        "air",
        "feet_calves",
        "never_worked",
    ])
    result = WarrantyEngine.submit_answer(ticket_id, "error_code_no")

    assert result["next_node_id"] == "defect_air_pump_terminal"
    assert result["is_terminal"] is True
    t = WarrantyEngine.get_ticket(ticket_id)
    assert t is not None
    assert str(t.status) == "awaiting_admin_review"
    assert t.get_collected().get(gate.COL_GATE_COMPLETED) == "skipped"


def test_gate_yes_and_code_enriches_terminal():
    ticket_id, _ = WarrantyEngine.start_session("gate-yes", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "3D LTX")

    _walk(ticket_id, [
        "warranty",
        "defect",
        "air",
        "feet_calves",
        "never_worked",
    ])
    WarrantyEngine.submit_answer(ticket_id, "error_code_yes")
    result = WarrantyEngine.submit_answer(ticket_id, "pick_C6")

    assert result["next_node_id"] == "defect_air_pump_terminal"
    assert result["is_terminal"] is True

    t = WarrantyEngine.get_ticket(ticket_id)
    assert t is not None
    assert t.get_collected().get(gate.COL_ERROR_CODE) == "C6"

    node = WarrantyEngine.get_current_node(ticket_id)
    from warranty_terminal_enrichment import build_terminal_enrichment  # noqa: WPS433

    enrichment = build_terminal_enrichment(WarrantyEngine, t, node)
    assert enrichment is not None
    message = str(enrichment.get("message") or "")
    assert "C6" in message
    assert "MOS" in message or "air pump" in message.lower()


def test_gate_lookup_failure_message():
    ticket_id, _ = WarrantyEngine.start_session("gate-bad", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "3D LTX")

    _walk(ticket_id, [
        "warranty",
        "defect",
        "air",
        "feet_calves",
        "never_worked",
    ])
    WarrantyEngine.submit_answer(ticket_id, "error_code_yes")
    WarrantyEngine.submit_answer(ticket_id, "error_code_other")
    WarrantyEngine.submit_answer(ticket_id, "ZZZZ")

    t = WarrantyEngine.get_ticket(ticket_id)
    assert t is not None
    assert t.get_collected().get(gate.COL_FONZ_LOOKUP_FAILED) == "1"

    node = WarrantyEngine.get_current_node(ticket_id)
    from warranty_terminal_enrichment import build_terminal_enrichment  # noqa: WPS433

    enrichment = build_terminal_enrichment(WarrantyEngine, t, node)
    message = str(enrichment.get("message") or "")
    assert "not listed" in message.lower() or "verify" in message.lower()


def test_intake_error_code_skips_gate():
    ticket_id, _ = WarrantyEngine.start_session("gate-intake", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "3D LTX")

    from warranty_intake_context import persist_intake_summary  # noqa: WPS433

    with wm.warranty_db_session() as db:
        row = (
            db.query(wm.WarrantyTicket)
            .filter(wm.WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        assert row is not None
        persist_intake_summary(
            row,
            raw_message="My 3D LTX shows error code C6 on the remote",
        )

    result = _walk(ticket_id, [
        "warranty",
        "defect",
        "air",
        "feet_calves",
        "never_worked",
    ])

    assert result["next_node_id"] == "defect_air_pump_terminal"
    assert result["is_terminal"] is True
    t2 = WarrantyEngine.get_ticket(ticket_id)
    assert t2 is not None
    assert t2.get_collected().get(gate.COL_ERROR_CODE) == "C6"


def test_soft_hints_when_no_code():
    from types import SimpleNamespace

    ticket = SimpleNamespace(
        model_name="3D LTX",
        defect_type="air",
        issue_type="defect",
    )
    ticket.get_collected = lambda: {}
    ticket.set_collected = lambda k, v: None

    diagnosis = {"summary": "Air pump review needed.", "steps": [], "sources": []}
    merged = gate.merge_fonz_into_diagnosis(diagnosis, ticket)
    summary = str(merged.get("summary") or "")
    assert "error code" in summary.lower() or merged.get("fonz_suggestions")


def test_no_intercept_for_voice_defect():
    ticket_id, _ = WarrantyEngine.start_session("gate-voice", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "3D LTX")

    result = _walk(ticket_id, [
        "warranty",
        "defect",
        "voice",
        "voice_no_response",
    ])

    assert result["next_node_id"] == "defect_voice_not_working_terminal"
    assert result["is_terminal"] is True


def test_no_intercept_when_model_has_no_error_code_entry():
    ticket_id, _ = WarrantyEngine.start_session("gate-nomodel", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "OS-4000T")

    result = _walk(ticket_id, [
        "warranty",
        "defect",
        "air",
        "feet_calves",
        "never_worked",
    ])

    assert result["next_node_id"] == "defect_air_pump_terminal"
    assert result["is_terminal"] is True


def test_skip_gate_when_error_code_already_collected():
    ticket_id, _ = WarrantyEngine.start_session("gate-skip", "osakiusa.com")
    WarrantyEngine.set_model_name(ticket_id, "3D LTX")
    with wm.warranty_db_session() as db:
        row = (
            db.query(wm.WarrantyTicket)
            .filter(wm.WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        assert row is not None
        row.set_collected(gate.COL_ERROR_CODE, "C6")

    result = _walk(ticket_id, [
        "warranty",
        "defect",
        "air",
        "feet_calves",
        "never_worked",
    ])

    assert result["next_node_id"] == "defect_air_pump_terminal"
    assert result["is_terminal"] is True
