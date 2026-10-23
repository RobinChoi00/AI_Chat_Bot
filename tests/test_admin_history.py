"""All-source stored conversation history without importing the application."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import warranty_models as wm
import admin_history as history

HEADERS = {"X-Admin-Key": "history-test-key"}


def schema():
    meta = MetaData()
    tables = {}
    tables["warranty_tickets"] = Table("warranty_tickets", meta,
        Column("id", Integer, primary_key=True), Column("ticket_id", String), Column("session_id", String), Column("domain", String),
        Column("status", String), Column("created_at", DateTime), Column("updated_at", DateTime), Column("model_name", String),
        Column("issue_type", String), Column("collected_data", Text))
    tables["warranty_turns"] = Table("warranty_turns", meta,
        Column("id", Integer, primary_key=True), Column("ticket_id", String), Column("node_prompt", Text), Column("customer_answer", Text), Column("created_at", DateTime))
    tables["sales_sessions"] = Table("sales_sessions", meta,
        Column("id", Integer, primary_key=True), Column("session_id", String), Column("domain", String), Column("status", String),
        Column("created_at", DateTime), Column("updated_at", DateTime), Column("last_intent", String))
    tables["sales_messages"] = Table("sales_messages", meta,
        Column("id", Integer, primary_key=True), Column("session_id", String), Column("role", String), Column("content", Text), Column("created_at", DateTime))
    tables["chat_logs"] = Table("chat_logs", meta,
        Column("id", Integer, primary_key=True), Column("session_id", String), Column("domain", String), Column("user_query", Text), Column("bot_response", Text), Column("created_at", DateTime))
    return meta, tables


@pytest.fixture
def data(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    meta, tables = schema()
    meta.create_all(engine)
    monkeypatch.setattr(wm, "_SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setenv("ADMIN_API_KEY", "history-test-key")
    app = FastAPI()
    app.include_router(history.router)
    with TestClient(app) as client:
        yield client, engine, tables
    engine.dispose()


def insert(engine, table, rows):
    with engine.begin() as conn:
        conn.execute(table.insert(), rows if isinstance(rows, list) else [rows])


def get(client, path="/admin/history"):
    response = client.get(path, headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def test_all_historical_sources_and_pages_accessible(data):
    client, engine, tables = data
    old = datetime(2019, 1, 1)
    insert(engine, tables["warranty_tickets"], [dict(id=i + 1, ticket_id=f"w-{i}", session_id=f"w-session-{i}", domain="osakiusa.com",
        status="resolved", created_at=old + timedelta(days=i), updated_at=old + timedelta(days=i), model_name="Chair", issue_type="defect", collected_data="{}") for i in range(105)])
    insert(engine, tables["sales_sessions"], [dict(id=i + 1, session_id=f"s-session-{i}", domain="titanchair.com", status="closed",
        created_at=old + timedelta(days=i), updated_at=old + timedelta(days=i), last_intent="recommend") for i in range(102)])
    insert(engine, tables["chat_logs"], [dict(id=i + 1, session_id=f"g-session-{i}", domain="osakiusa.com", user_query=f"Question {i}", bot_response=f"Answer {i}", created_at=old + timedelta(days=i)) for i in range(6)])
    pages = [get(client, f"/admin/history?limit=100&offset={offset}") for offset in (0, 100, 200)]
    assert [page["total"] for page in pages] == [213, 213, 213]
    assert [len(page["items"]) for page in pages] == [100, 100, 13]
    records = [item for page in pages for item in page["items"]]
    assert len({(item["source"], item["record_id"]) for item in records}) == 213
    assert pages[0]["counts"] == {"warranty": 105, "sales": 102, "general": 6}
    assert {item["source"] for item in records} == {"warranty", "sales", "general"}
    assert records[0]["created_at"].startswith("2019-04-15")
    assert records[0]["ticket_id"] == "w-104"
    assert get(client, "/admin/history?offset=500")["items"] == []


def test_general_groups_session_and_domain_but_keeps_anonymous_records(data):
    client, engine, tables = data
    rows = [
        (1, "shared", "osakiusa.com"), (2, "shared", "osakiusa.com"), (3, "shared", "titanchair.com"),
        (4, None, "osakiusa.com"), (5, None, "osakiusa.com"), (6, "", "osakiusa.com"), (7, " ", "osakiusa.com"),
    ]
    insert(engine, tables["chat_logs"], [dict(id=key, session_id=session, domain=domain, user_query=f"q{key}", bot_response=f"a{key}", created_at=datetime(2020, 1, key)) for key, session, domain in rows])
    result = get(client)
    assert result["total"] == 6
    shared = next(item for item in result["items"] if item["record_id"] == "1")
    assert shared["message_count"] == 4
    assert shared["created_at"].startswith("2020-01-01")
    assert shared["updated_at"].startswith("2020-01-02")
    assert get(client, "/admin/history?source=general&domain=osaki")["total"] == 5
    assert client.get("/admin/history/general/2", headers=HEADERS).status_code == 404


def test_domain_counts_before_source_filter(data):
    client, engine, tables = data
    insert(engine, tables["warranty_tickets"], dict(id=1, ticket_id="w-1", session_id="w-1", domain="osakiusa.com", status="resolved", created_at=datetime(2020, 1, 1), collected_data="{}"))
    insert(engine, tables["sales_sessions"], [dict(id=1, session_id="s-1", domain="osakiusa.com", status="closed"), dict(id=2, session_id="s-2", domain="titanchair.com", status="closed")])
    result = get(client, "/admin/history?source=sales&domain=osaki")
    assert result["total"] == 1
    assert result["counts"] == {"warranty": 1, "sales": 1, "general": 0}
    assert result["items"][0]["source"] == "sales"
    assert get(client, "/admin/history?domain=%")["total"] == 0  # literal substring, not SQL wildcard


def test_warranty_detail_includes_full_pairs_and_timeline_without_raw_bag(data):
    client, engine, tables = data
    prompt = "Full saved diagnosis " + "x" * 4000
    answer = "Full customer report " + "y" * 3500
    insert(engine, tables["warranty_tickets"], dict(id=1, ticket_id="public-case-id", session_id="session", domain="osakiusa.com", status="resolved",
        created_at=datetime(2020, 1, 1), updated_at=datetime(2020, 1, 2), collected_data=json.dumps({"saved_path": "/private/evidence", "chat_timeline": [
            {"role": "assistant", "text": "Stored enrichment response", "created_at": "2020-01-01T16:01:00+00:00"},
            {"role": "system", "text": "private internal note", "created_at": "2020-01-01T16:02:00+00:00"},
        ]})))
    insert(engine, tables["warranty_turns"], dict(id=1, ticket_id="public-case-id", node_prompt=prompt, customer_answer=answer, created_at=datetime(2020, 1, 1, 10)))
    first = get(client, "/admin/history/warranty/1?limit=2")
    second = get(client, "/admin/history/warranty/1?limit=2&offset=2")
    assert first["total"] == second["total"] == 3
    assert first["conversation"]["message_count"] == 3
    assert first["conversation"]["ticket_id"] == "public-case-id"
    assert [(message["role"], message["content"]) for message in first["messages"]] == [("assistant", prompt), ("user", answer)]
    assert second["messages"][0]["content"] == "Stored enrichment response"
    serialized = json.dumps(first)
    assert "/private/evidence" not in serialized and "collected_data" not in serialized
    assert "private internal note" not in serialized


def test_sales_detail_complete_paginated_and_same_timestamp_order(data):
    client, engine, tables = data
    insert(engine, tables["sales_sessions"], dict(id=1, session_id="sales-session", domain="osakiusa.com", status="active"))
    long_content = "Full answer " + "z" * 5000
    insert(engine, tables["sales_messages"], [dict(id=1, session_id="sales-session", role="user", content="Budget 5000", created_at=datetime(2020, 1, 1)),
        dict(id=2, session_id="sales-session", role="assistant", content=long_content, created_at=datetime(2020, 1, 1)),
        dict(id=3, session_id="sales-session", role="system", content="internal", created_at=datetime(2020, 1, 1))])
    first = get(client, "/admin/history/sales/1?limit=1")
    second = get(client, "/admin/history/sales/1?limit=1&offset=1")
    assert first["total"] == second["total"] == 2
    assert first["messages"][0]["role"] == "user"
    assert second["messages"][0]["content"] == long_content


def test_general_detail_groups_and_retains_complete_responses(data):
    client, engine, tables = data
    long_response = "Saved answer " + "r" * 6000
    insert(engine, tables["chat_logs"], [dict(id=1, session_id="same", domain="osakiusa.com", user_query="first", bot_response=long_response, created_at=datetime(2020, 1, 1)),
        dict(id=2, session_id="same", domain="osakiusa.com", user_query="second", bot_response="next", created_at=datetime(2020, 1, 2)),
        dict(id=3, session_id="same", domain="titanchair.com", user_query="other store", bot_response="other", created_at=datetime(2020, 1, 2))])
    result = get(client, "/admin/history/general/1?limit=2&offset=1")
    assert result["total"] == 4
    assert [message["content"] for message in result["messages"]] == [long_response, "second"]
    assert result["conversation"]["domain"] == "osakiusa.com"


def test_missing_optional_general_table_is_zero_not_an_error(data):
    client, engine, tables = data
    tables["chat_logs"].drop(engine)
    result = get(client)
    assert result["counts"]["general"] == 0
    assert result["total"] == 0
    assert client.get("/admin/history/general/1", headers=HEADERS).status_code == 404


def test_empty_or_malformed_timeline_keeps_valid_turns(data):
    client, engine, tables = data
    insert(engine, tables["warranty_tickets"], dict(id=1, ticket_id="case", session_id="session", domain="osakiusa.com", status="resolved", collected_data="not json"))
    insert(engine, tables["warranty_turns"], dict(id=1, ticket_id="case", node_prompt="Prompt", customer_answer="", created_at=datetime(2020, 1, 1)))
    assert get(client, "/admin/history/warranty/1")["total"] == 1


def test_history_authentication_and_validation(data, monkeypatch):
    client, _, _ = data
    for path in ("/admin/history", "/admin/history/warranty/1"):
        assert client.get(path).status_code == 401
    for query in ("limit=0", "limit=201", "offset=-1", "source=unknown"):
        assert client.get(f"/admin/history?{query}", headers=HEADERS).status_code == 422
    assert client.get("/admin/history/warranty/0", headers=HEADERS).status_code == 422
    assert client.get("/admin/history/sales/999", headers=HEADERS).status_code == 404
    monkeypatch.delenv("ADMIN_API_KEY")
    assert client.get("/admin/history", headers=HEADERS).status_code == 503


def test_real_database_failures_propagate_instead_of_empty_history(data, monkeypatch):
    client, _, _ = data
    def failed_tables(_db):
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(history, "_tables", failed_tables)
    with pytest.raises(RuntimeError, match="database unavailable"):
        client.get("/admin/history", headers=HEADERS)
