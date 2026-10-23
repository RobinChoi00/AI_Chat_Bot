"""Operational observations do not impersonate live quality measurements."""
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import ai_quality as quality
import warranty_models as wm


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(wm, "_SessionFactory", factory)
    monkeypatch.setenv("ADMIN_API_KEY", "quality-test-key")
    monkeypatch.setattr(quality, "_release_id", lambda: "release-test")
    app = FastAPI()
    app.include_router(quality.router)
    with TestClient(app) as client:
        yield client
    engine.dispose()


def test_quality_health_requires_admin_and_does_not_return_secret(client):
    assert client.get("/admin/ai/health").status_code == 401
    response = client.get("/admin/ai/health", headers={"X-Admin-Key": "quality-test-key"})
    assert response.status_code == 200
    assert "quality-test-key" not in response.text
    body = response.json()
    assert body["latency"]["sales"]["samples"] == 0
    assert body["latency"]["sales"]["p95_ms"] is None
    assert body["model_calls"]["roles"]["agent"]["latest_call"] is None
    assert body["model_calls"]["roles"]["router"]["latest_successful_call"] is None
    assert "not verified" in body["scope"]


def test_response_timing_percentiles_and_surface_isolation(client):
    for milliseconds in range(1, 21):
        quality.record_response_timing("sales", milliseconds)
    quality.record_response_timing("warranty", 100)
    quality.record_response_timing("private-email@example.invalid", 3)
    summary = quality.timing_summary()
    assert summary["sales"] == {"samples": 20, "p50_ms": 10, "p95_ms": 19, "mean_ms": 10.5}
    assert summary["warranty"]["samples"] == 1
    with wm.warranty_db_session() as db:
        assert {column.name for column in quality.AIResponseTiming.__table__.columns} == {"id", "surface", "elapsed_ms", "created_at"}
        assert db.query(quality.AIResponseTiming).count() == 21


def test_timing_retention_keeps_only_recent_samples(client):
    quality.record_response_timing("sales", 10)
    with wm.warranty_db_session() as db:
        db.add(quality.AIResponseTiming(surface="sales", elapsed_ms=5000, created_at=wm._now_cst() - timedelta(days=31)))
    quality.record_response_timing("sales", 20)
    assert quality.timing_summary()["sales"]["samples"] == 2
    with wm.warranty_db_session() as db:
        assert db.query(quality.AIResponseTiming).count() == 2


def test_model_call_failure_does_not_replace_last_success_or_invent_provider_identity(client):
    quality.record_model_call(role="agent", requested_model="candidate-alias",
        provider_model="candidate-2026-10-23", succeeded=True)
    quality.record_model_call(role="agent", requested_model="new-candidate",
        provider_model="unverified-model", succeeded=False, error_code="permission")
    quality.record_model_call(role="router", requested_model="router-alias", succeeded=True)
    body = client.get("/admin/ai/health", headers={"X-Admin-Key": "quality-test-key"}).json()
    agent = body["model_calls"]["roles"]["agent"]
    assert agent["calls"] == 2 and agent["successful_calls"] == agent["failed_calls"] == 1
    assert agent["latest_call"]["requested_model"] == "new-candidate"
    assert agent["latest_call"]["provider_model"] is None
    assert agent["latest_call"]["error_code"] == "permission"
    assert agent["latest_successful_call"]["requested_model"] == "candidate-alias"
    assert agent["latest_successful_call"]["provider_model"] == "candidate-2026-10-23"
    assert agent["latest_successful_call"]["release_id"] == "release-test"
    assert datetime.fromisoformat(agent["latest_call"]["observed_at"]).tzinfo is not None
    assert body["model_calls"]["roles"]["router"]["latest_successful_call"]["provider_model"] is None


def test_model_observations_and_configuration_never_copy_raw_metadata(client, monkeypatch):
    import config

    monkeypatch.setattr(config, "AGENT_MODEL", "configured-alias")
    monkeypatch.setattr(config, "MODEL_CONFIGURATION", {
        "agent": {"effective_model": "configured-alias", "default_model": "code-default",
                  "source": "env", "OPENAI_API_KEY": "private-token"},
        "router": {"source": "default", "default_model": "router-default"},
    }, raising=False)
    quality.record_model_call(role="agent", requested_model="configured-alias", succeeded=False,
        error_code="Provider said private@example.invalid with private-token")
    quality.record_model_call(role="agent", requested_model="configured-alias", succeeded=True,
        provider_model="https://provider.invalid/private-token")
    quality.record_model_call(role="agent", requested_model="sk-private-token", succeeded=True)
    quality.record_model_call(role="customer@example.invalid", requested_model="configured-alias", succeeded=True)
    quality.record_model_call(role="router", requested_model="router-alias", succeeded=False, error_code={"private": "private-token"})
    body = client.get("/admin/ai/health", headers={"X-Admin-Key": "quality-test-key"}).json()
    rendered = json.dumps(body)
    assert "private-token" not in rendered and "private@example.invalid" not in rendered
    assert body["models"]["agent"] == "configured-alias"
    assert body["model_configuration"]["agent"] == {
        "effective_model": "configured-alias", "default_model": "code-default", "source": "env"}
    assert body["app_release_id"] == "release-test"
    assert body["model_calls"]["roles"]["router"]["latest_call"]["error_code"] == "unexpected_error"
    with wm.warranty_db_session() as db:
        rows = db.query(quality.AIModelCallObservation).order_by(quality.AIModelCallObservation.id).all()
        assert len(rows) == 3
        assert rows[0].error_code == "unexpected_error"
        assert rows[1].provider_model is None
        assert {column.name for column in quality.AIModelCallObservation.__table__.columns} == {
            "id", "role", "requested_model", "provider_model", "succeeded", "error_code", "release_id", "created_at"}


def test_model_observation_retention_is_bounded_and_health_does_not_add_calls(client, monkeypatch):
    monkeypatch.setattr(quality, "_MODEL_MAX_OBSERVATIONS", 3)
    quality.record_model_call(role="agent", requested_model="expired-model", succeeded=True)
    with wm.warranty_db_session() as db:
        db.query(quality.AIModelCallObservation).update({"created_at": wm._now_cst() - timedelta(days=31)})
    for index in range(4):
        quality.record_model_call(role="agent", requested_model=f"model-{index}", succeeded=True)
    for _ in range(2):
        response = client.get("/admin/ai/health", headers={"X-Admin-Key": "quality-test-key"})
        assert response.status_code == 200
    with wm.warranty_db_session() as db:
        assert [row.requested_model for row in db.query(quality.AIModelCallObservation).order_by(quality.AIModelCallObservation.id)] == ["model-1", "model-2", "model-3"]


def test_observation_failure_never_breaks_a_customer_call(monkeypatch, caplog):
    def unavailable():
        raise RuntimeError("private-token private@example.invalid")

    monkeypatch.setattr(quality, "_release_id", unavailable)
    quality.record_model_call(role="agent", requested_model="candidate", succeeded=False, error_code="timeout")
    assert "could not be recorded" in caplog.text
    assert "private-token" not in caplog.text
    assert "private@example.invalid" not in caplog.text
