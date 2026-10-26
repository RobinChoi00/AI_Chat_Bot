"""Replay the discovered customer failures through the real public APIs.

Models and external providers are unavailable. These checks protect the actual
degraded-mode behavior, not an estimate of live AI answer quality.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from script import evaluate_llm_quality as evaluation

CASES = [case for case in json.loads(evaluation.CONVERSATIONS_PATH.read_text())["cases"] if case["id"].startswith("simulation-")]


@pytest.fixture
def customer_api(monkeypatch):
    import sales_agent
    import sales_router
    import warranty_models as models
    import warranty_router
    import warranty_workflow
    import requests
    import smtplib

    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1")
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "0")
    monkeypatch.setenv("WARRANTY_NOTIFICATION_WORKER_ENABLED", "0")
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "0")
    monkeypatch.setenv("WARRANTY_STEP_PARAPHRASE", "0")
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for module in ("warranty_nlp", "warranty_intake", "warranty_summary", "warranty_step_paraphrase"):
        monkeypatch.setattr(__import__(module), "_openai_client", lambda: None)
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **kw: pytest.fail("Simulation must not make an external provider request"))
    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **kw: pytest.fail("Simulation must not send actual email"))
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *a, **kw: pytest.fail("Simulation must not send actual email"))
    monkeypatch.setattr(sales_agent, "fetch_live_stock", lambda *a, **kw: None)
    monkeypatch.setattr(sales_agent, "fetch_stock_many", lambda *a, **kw: {})
    monkeypatch.setattr("warranty_order_verification.verify_ticket_purchase", lambda **kw: {"status": "simulation"})
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(models, "_engine", engine)
    monkeypatch.setattr(models, "_SessionFactory", factory)
    monkeypatch.setattr(warranty_workflow, "_SessionFactory", factory)
    warranty_router.limiter.reset()
    app = FastAPI()
    app.include_router(warranty_router.router)
    app.include_router(sales_router.router)
    with TestClient(app) as client:
        yield client
    engine.dispose()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_previously_failed_customer_dialogue(customer_api, case):
    result = evaluation.run_case(case, customer_api, base_url="http://testserver", domain="osakiusa.com")
    assert result["passed"], json.dumps(result, ensure_ascii=False)
    assert result.get("unrun_turns", 0) == 0
