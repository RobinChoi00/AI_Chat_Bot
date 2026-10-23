"""Usage persistence must not turn unpriced model calls into free requests."""
import asyncio
import importlib
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def usage_app(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'app'))
    monkeypatch.setenv('APP_ENV', 'development')
    module = importlib.import_module('app.main')
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    module.OpenAIUsageLog.__table__.create(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(module, 'SessionLocal', factory)
    monkeypatch.setattr(module, 'require_admin_key', lambda *_: None)
    yield module, factory
    engine.dispose()


def save(module, model, cost, calls=1):
    module._persist_usage_log(session_id='synthetic-usage', domain='example.invalid',
        model=model, call_count=calls, prompt_tokens=100, cached_tokens=0,
        completion_tokens=20, estimated_cost_usd=cost, elapsed_ms=5, cache_hit=False)


def test_unknown_model_cost_persists_null_and_cannot_be_counted_as_free(usage_app):
    module, factory = usage_app
    save(module, 'gpt-6.1-sol', None)
    with factory() as db:
        row = db.query(module.OpenAIUsageLog).one()
        assert row.estimated_cost_usd is None
    summary = asyncio.run(module.cost_summary(days=7, x_admin_key='synthetic-key'))
    assert summary['requests'] == 1
    assert summary['total_cost_usd'] is None
    assert summary['avg_cost_per_request_usd'] is None
    assert summary['unknown_cost_requests'] == 1
    assert summary['pricing_available'] is False
    assert summary['by_model']['gpt-6.1-sol']['cost_usd'] is None


def test_mixed_pricing_shows_known_subtotal_without_claiming_complete_total(usage_app):
    module, _ = usage_app
    save(module, 'known-fixture', .25)
    save(module, 'unknown-fixture', None)
    summary = asyncio.run(module.cost_summary(days=7, x_admin_key='synthetic-key'))
    assert summary['known_cost_usd'] == .25
    assert summary['total_cost_usd'] is None
    assert summary['by_model']['known-fixture']['cost_usd'] == .25
    assert summary['by_model']['unknown-fixture']['cost_usd'] is None


def test_known_cost_and_zero_call_cache_remain_measured(usage_app):
    module, _ = usage_app
    save(module, 'known-fixture', .25)
    save(module, 'cache-fixture', 0, calls=0)
    summary = asyncio.run(module.cost_summary(days=7, x_admin_key='synthetic-key'))
    assert summary['total_cost_usd'] == .25
    assert summary['avg_cost_per_request_usd'] == .125
    assert summary['pricing_available'] is True
    assert summary['unknown_cost_requests'] == 0
