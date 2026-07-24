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
    monkeypatch.setattr(warranty_intake, "_keyword_workflow_prefill", lambda text: None)
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
    monkeypatch.setattr(warranty_intake, "_keyword_workflow_prefill", lambda text: None)
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
    monkeypatch.setattr(warranty_intake, "_keyword_workflow_prefill", lambda text: None)
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


def test_extract_keyword_footrest_air_without_llm(monkeypatch):
    monkeypatch.setattr(warranty_intake, "_openai_client", lambda: None)
    out = extract_workflow_prefill(
        free_text="OS-4000T footrest air not inflating",
        nodes=_NODES,
    )
    assert out["source"] == "keyword"
    assert out["answer_keys"] == ["warranty", "defect", "air", "footrest"]
    assert out["confidence"] == "high"


def test_normalize_air_footrest_keys_prefers_air_location_path():
    assert warranty_intake._normalize_air_footrest_keys(
        ["warranty", "defect", "footrest", "air"]
    ) == ["warranty", "defect", "air", "footrest"]
    assert warranty_intake._normalize_air_footrest_keys(
        ["warranty", "defect", "footrest", "air"]
    )[-1] == "footrest"


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


def test_extract_hiro_error_code_68_routes_to_footrest_via_fonz(monkeypatch):
    """Hiro code 68 is calf/footrest — never invent power from the code alone."""

    def _boom():
        raise AssertionError("LLM must not run when Fonz resolves the error code")

    monkeypatch.setattr(warranty_intake, "_openai_client", _boom)

    out = extract_workflow_prefill(free_text="hiro error code 68", nodes=_NODES)
    assert out["source"] == "fonz"
    assert out["answer_keys"] == ["warranty", "defect", "footrest"]
    assert "power" not in out["answer_keys"]
    assert "power" not in (out["summary"] or "").lower()
    assert "68" in out["summary"]
    assert out["confidence"] == "high"
    assert out["model_name"]


def test_extract_hiro_68_works_without_product_catalog(monkeypatch):
    """Prod may lack Shopify CSV — still match Hiro via Fonz model tokens."""
    import product_catalog as pc

    monkeypatch.setattr(pc, "resolve_model_name", lambda raw: None)
    monkeypatch.setattr(pc, "looks_like_model_only", lambda raw: None)
    monkeypatch.setattr(
        warranty_intake,
        "_openai_client",
        lambda: (_ for _ in ()).throw(AssertionError("LLM should not run")),
    )

    out = extract_workflow_prefill(free_text="hiro error code 68", nodes=_NODES)
    assert out["source"] == "fonz"
    assert out["answer_keys"] == ["warranty", "defect", "footrest"]
    assert "Hiro" in (out["model_name"] or "")


def test_extract_sanitizes_llm_power_guess_for_code_only(monkeypatch):
    """If LLM somehow runs, strip invented defect keys for model+code-only text."""
    # Force Fonz miss so LLM path runs, then sanitize.
    monkeypatch.setattr(
        warranty_intake,
        "_fonz_prefill_from_text",
        lambda text: None,
    )
    payload = {
        "answer_keys": ["warranty", "defect", "power"],
        "model_name": "Hiro",
        "confidence": "high",
        "summary": "Customer reports error code 99, indicating a power-related defect.",
    }
    monkeypatch.setattr(warranty_intake, "_openai_client", lambda: _FakeClient(payload))
    out = extract_workflow_prefill(free_text="hiro error code 99", nodes=_NODES)
    assert out["source"] == "llm"
    assert out["answer_keys"] == ["warranty"]
    assert "power" not in out["answer_keys"]
    assert "power-related" not in (out["summary"] or "").lower()
    assert "99" in out["summary"]


def test_hiro_68_category_is_footrest_not_power():
    from error_code_lookup import entry_workflow_category, knowledge_category_to_defect_key, lookup_error_code
    from warranty_knowledge import _infer_category

    hit = lookup_error_code("hiro", "68")
    assert hit is not None
    blob = f"{hit.get('meaning') or ''} {hit.get('troubleshooting') or ''}"
    assert _infer_category(blob) == "footrest"
    cat = entry_workflow_category(hit)
    assert cat == "footrest"
    assert knowledge_category_to_defect_key(cat, meaning=str(hit.get("meaning") or "")) == "footrest"


# ---------------------------------------------------------------------------
# apply_prefill_to_engine
# ---------------------------------------------------------------------------


def test_apply_prefill_stops_at_error_code_safety_gate():
    ticket_id, _root = WarrantyEngine.start_session("s-1", "osakiusa.com")
    result = apply_prefill_to_engine(
        engine=WarrantyEngine,
        ticket_id=ticket_id,
        nodes=_NODES,
        answer_keys=["warranty", "defect", "heat", "too_hot"],
    )
    assert result["applied"] == ["warranty", "defect", "heat", "too_hot"]
    assert result["stopped_reason"] == "done"
    final = result["final_node"]
    assert final is not None
    assert final["node_id"] == "defect_error_code_visible_q"

    completed = WarrantyEngine.submit_answer(ticket_id, "error_code_no")
    assert completed["is_terminal"] is True
    assert completed["next_node_id"] == "defect_heating_too_hot_terminal"


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
