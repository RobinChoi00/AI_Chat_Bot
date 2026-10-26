"""Browser recording consent is enforced before any processing or persistence."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def consent_chat(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "app"))
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "1")
    module = importlib.import_module("app.main")
    wm = importlib.import_module("warranty_models")
    consent = importlib.import_module("warranty_consent")
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    wm.Base.metadata.create_all(engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    for name in ("vs_products", "vs_qa", "vs_web"):
        monkeypatch.setattr(module, name, object())
    monkeypatch.setattr(module.WarrantyEngine, "get_active_session_ticket", lambda _: None)
    scope = importlib.import_module("app.scope_classifier")
    scope_check = Mock(return_value=SimpleNamespace(is_blocked=False))
    monkeypatch.setattr(scope, "evaluate_scope", scope_check)
    model_call = Mock(side_effect=AssertionError("Greeting must not call the model"))
    monkeypatch.setattr(module, "openai_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=model_call)),
        responses=SimpleNamespace(create=model_call),
    ))
    persist = Mock()
    monkeypatch.setattr(module, "_persist_chat_log", persist)
    monkeypatch.setattr(module, "_persist_usage_log", lambda **_: None)
    traces = importlib.import_module("app.request_trace")
    trace_persist = Mock()
    monkeypatch.setattr(traces, "persist_trace", trace_persist)
    yield TestClient(module.app), consent, scope_check, model_call, persist, trace_persist
    engine.dispose()


@pytest.mark.parametrize("path, body", [
    ("/api/v1/chat", {"user_query": "hello"}),
    ("/api/v1/sales/chat", {"message": "hello"}),
    ("/api/v1/sales/lead", {"email": "buyer@example.invalid"}),
])
def test_general_chat_rejects_missing_consent_before_processors_and_storage(consent_chat, path, body):
    client, _, scope, model, logs, traces = consent_chat
    response = client.post(path, json={"session_id": "no-consent", **body})
    assert response.status_code == 403
    assert "privacy notice" in response.json()["detail"]
    for call in (scope, model, logs, traces):
        call.assert_not_called()


def test_general_chat_accepts_recorded_consent_without_contact_email(consent_chat):
    client, consent, _, model, _, _ = consent_chat
    consent.record_chat_consent("consented", domain="osakiusa.com", policy_store="osakiusa.com")
    response = client.post("/api/v1/chat", json={"session_id": "consented", "user_query": "hello"})
    assert response.status_code == 200, response.text
    assert response.text.strip()
    model.assert_not_called()


def test_consent_for_another_session_cannot_unlock_general_chat(consent_chat):
    client, consent, scope, model, logs, traces = consent_chat
    consent.record_chat_consent("session-a", domain="osakiusa.com", policy_store="osakiusa.com")
    response = client.post("/api/v1/chat", json={"session_id": "session-b", "user_query": "hello"})
    assert response.status_code == 403
    for call in (scope, model, logs, traces):
        call.assert_not_called()
