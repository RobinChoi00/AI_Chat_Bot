"""Tests for free-text warranty intake extraction and engine prefill."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import warranty_intake  # noqa: E402
import warranty_models as wm  # noqa: E402
from warranty_intake import apply_prefill_to_engine, extract_workflow_prefill  # noqa: E402
from warranty_workflow import WarrantyEngine, _NODES  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})()


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.chat = type("Chat", (), {"completions": self})()  # type: ignore[attr-defined]
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(json.dumps(self._payload))


# ---------------------------------------------------------------------------
# extract_workflow_prefill
# ---------------------------------------------------------------------------


def test_extract_returns_empty_when_text_blank():
    out = extract_workflow_prefill(free_text="   ", nodes=_NODES)
    assert out["answer_keys"] == []
    assert out["source"] == "empty"


def test_extract_high_confidence_keeps_only_valid_keys(monkeypatch):
    payload = {
        "answer_keys": ["warranty", "defect", "air", "footrest", "made_up_key"],
        "model_name": "OS-4000T",
        "confidence": "high",
        "summary": "Footrest air not inflating on OS-4000T.",
    }
    monkeypatch.setattr(warranty_intake, "_openai_client", lambda: _FakeClient(payload))
    out = extract_workflow_prefill(
        free_text="OS-4000T footrest air not inflating",
        nodes=_NODES,
    )
    assert out["source"] == "llm"
    assert out["answer_keys"][:4] == ["warranty", "defect", "air", "footrest"]
    assert "made_up_key" not in out["answer_keys"]
    assert out["model_name"] == "OS-4000T"
    assert "footrest" in out["summary"].lower()


def test_extract_low_confidence_drops_result(monkeypatch):
    payload = {
        "answer_keys": ["warranty", "defect"],
        "confidence": "medium",
        "summary": "Not sure",
    }
    monkeypatch.setattr(warranty_intake, "_openai_client", lambda: _FakeClient(payload))
    out = extract_workflow_prefill(free_text="I have a problem", nodes=_NODES)
    assert out["answer_keys"] == []
    assert out["source"] == "empty"


def test_extract_injects_warranty_root_when_missing(monkeypatch):
    payload = {
        "answer_keys": ["defect", "power"],
        "confidence": "high",
        "summary": "Power issue.",
    }
    monkeypatch.setattr(warranty_intake, "_openai_client", lambda: _FakeClient(payload))
    out = extract_workflow_prefill(free_text="chair will not turn on", nodes=_NODES)
    assert out["answer_keys"][0] == "warranty"
    assert "defect" in out["answer_keys"]
    assert "power" in out["answer_keys"]


def test_extract_returns_empty_when_no_client(monkeypatch):
    monkeypatch.setattr(warranty_intake, "_openai_client", lambda: None)
    out = extract_workflow_prefill(free_text="something wrong", nodes=_NODES)
    assert out["answer_keys"] == []
    assert out["source"] == "empty"


def test_extract_model_only_skips_llm_and_defect_branch(monkeypatch):
    """Typing only a model name must not pre-select a symptom path."""
    import product_catalog as pc

    monkeypatch.setattr(
        pc,
        "looks_like_model_only",
        lambda text: "Maestro" if text.strip().lower() == "maestro" else None,
    )

    def _boom():
        raise AssertionError("LLM should not run for model-only intake")

    monkeypatch.setattr(warranty_intake, "_openai_client", _boom)

    out = extract_workflow_prefill(free_text="Maestro", nodes=_NODES)
    assert out["source"] == "model_only"
    assert out["answer_keys"] == ["warranty"]
    assert out["model_name"] == "Maestro"
    assert out["confidence"] == "high"


# ---------------------------------------------------------------------------
# apply_prefill_to_engine
# ---------------------------------------------------------------------------


def test_apply_prefill_walks_workflow_until_terminal():
    ticket_id, _root = WarrantyEngine.start_session("s-1", "osakiusa.com")
    result = apply_prefill_to_engine(
        engine=WarrantyEngine,
        ticket_id=ticket_id,
        nodes=_NODES,
        answer_keys=["warranty", "defect", "heat", "too_hot"],
    )
    assert result["applied"] == ["warranty", "defect", "heat", "too_hot"]
    assert result["stopped_reason"] == "terminal"
    final = result["final_node"]
    assert final is not None
    assert final["node_id"] == "defect_heating_too_hot_terminal"


def test_apply_prefill_stops_at_question_text_node():
    ticket_id, _root = WarrantyEngine.start_session("s-2", "osakiusa.com")
    result = apply_prefill_to_engine(
        engine=WarrantyEngine,
        ticket_id=ticket_id,
        nodes=_NODES,
        answer_keys=["warranty", "installation", "OS-4000T"],
    )
    assert result["applied"] == ["warranty", "installation"]
    assert result["stopped_reason"] == "question_text"
    final = result["final_node"]
    assert final is not None
    assert final["node_id"] == "install_model"


def test_apply_prefill_skips_invalid_branch_key():
    ticket_id, _root = WarrantyEngine.start_session("s-3", "osakiusa.com")
    result = apply_prefill_to_engine(
        engine=WarrantyEngine,
        ticket_id=ticket_id,
        nodes=_NODES,
        answer_keys=["warranty", "defect", "footrest_or_no_air"],
    )
    assert result["applied"] == ["warranty", "defect"]
    assert "footrest_or_no_air" in result["skipped"]
    assert result["stopped_reason"] == "no_match"
