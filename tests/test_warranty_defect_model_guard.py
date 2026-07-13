"""Defect intake requires chair model before entering defect branch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm
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


def test_defect_without_model_is_blocked_by_router_guard():
    from warranty_router import _guard_defect_requires_model  # noqa: WPS433

    ticket_id, _ = WarrantyEngine.start_session("defect-guard", "osakiusa.com")
    WarrantyEngine.submit_answer(ticket_id, "warranty")

    msg = _guard_defect_requires_model(WarrantyEngine, ticket_id, "defect")
    assert msg
    assert "model" in msg.lower()

    WarrantyEngine.set_model_name(ticket_id, "3D LTX")
    assert _guard_defect_requires_model(WarrantyEngine, ticket_id, "defect") is None
