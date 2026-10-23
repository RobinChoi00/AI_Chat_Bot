"""Provider execution evidence survives trace persistence without inventing history."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine

from app import request_trace as rt


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "_table_ensured", False)
    value = create_engine(f"sqlite:///{tmp_path / 'traces.db'}")
    yield value
    value.dispose()


def trace():
    return rt.RequestTrace("request", "session", "osakiusa.com", "Synthetic question", model="gpt-6.1-sol")


def test_configured_model_does_not_imply_provider_execution():
    value = trace()
    value.cache_hit = True
    row = value.to_dict()
    assert row["model"] == "gpt-6.1-sol"
    assert row["requested_models"] == row["provider_models"] == []
    assert row["model_calls_successful"] == row["model_calls_failed"] == 0


def test_records_calls_independently_and_drops_untrusted_model_text():
    value = trace()
    value.record_model_call("gpt-6.1-sol", "gpt-6.1-sol-2026-10-23")
    value.record_model_call("gpt-6.1-sol", "gpt-6.1-sol-2026-10-23")
    value.record_model_call("gpt-6.1-sol", "private-provider-error@example.invalid", succeeded=False)
    value.record_model_call("gpt-6.1-sol", "untrusted raw response with customer details")
    row = value.to_dict()
    assert row["requested_models"] == ["gpt-6.1-sol"]
    assert row["provider_models"] == ["gpt-6.1-sol-2026-10-23"]
    assert row["model_calls_successful"] == 3
    assert row["model_calls_failed"] == 1
    assert row["model_calls_unverified"] == 1
    assert "private-provider" not in json.dumps(row)
    assert "untrusted raw" not in json.dumps(row)


def test_actual_provider_snapshot_round_trips(engine):
    value = trace()
    value.record_model_call("gpt-6.1-sol", "gpt-6.1-sol-2026-10-23")
    value.record_model_call("gpt-6.1-sol", succeeded=False)
    value.finish("Synthetic answer")
    rt.persist_trace(value, engine)
    row = rt.get_trace(engine, "request")
    assert row is not None
    assert row["requested_models"] == ["gpt-6.1-sol"]
    assert row["provider_models"] == ["gpt-6.1-sol-2026-10-23"]
    assert row["model_calls_successful"] == row["model_calls_failed"] == 1
    assert row["model_calls_unverified"] == 0
    assert row["final_response"] == "Synthetic answer"


def test_legacy_schema_and_rows_gain_unknown_execution_evidence(engine):
    with engine.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE request_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL, session_id TEXT NOT NULL,
                domain TEXT DEFAULT 'unknown', model TEXT DEFAULT '',
                user_query TEXT DEFAULT '', scope_blocked INTEGER DEFAULT 0,
                cache_hit INTEGER DEFAULT 0, short_circuit TEXT,
                forced_first_tool TEXT, warranty_mode INTEGER DEFAULT 0,
                guard_sanitized INTEGER DEFAULT 0, turn_count INTEGER DEFAULT 0,
                tools_called TEXT DEFAULT '[]', total_latency_ms INTEGER DEFAULT 0,
                final_response_len INTEGER DEFAULT 0, turns_json TEXT DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.exec_driver_sql("INSERT INTO request_traces (request_id, session_id, model) VALUES ('legacy', 'old-session', 'gpt-5.6')")
    row = rt.get_trace(engine, "legacy")
    assert row is not None
    assert row["model"] == "gpt-5.6"
    assert row["requested_models"] == row["provider_models"] == []
    assert row["model_calls_successful"] == row["model_calls_failed"] == row["model_calls_unverified"] == 0
    assert row["final_response"] == ""
    rt.persist_trace(trace(), engine)
    assert rt.get_trace(engine, "request") is not None


def test_invalid_stored_model_shapes_do_not_escape_as_evidence(engine):
    rt.persist_trace(trace(), engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("UPDATE request_traces SET provider_models = ?", ('["gpt-6.1-sol", "gpt-6.1-sol", "raw text with email", {"private": true}]',))
    row = rt.get_trace(engine, "request")
    assert row is not None
    assert row["provider_models"] == ["gpt-6.1-sol"]
