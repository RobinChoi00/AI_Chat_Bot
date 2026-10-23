"""Operational observations do not impersonate live quality measurements."""
from datetime import timedelta
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
