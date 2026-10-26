"""Retention operates only on temporary DBs, with an explicit hold release."""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from privacy_retention import legal_hold_active, purge_chat_records, safe_evidence_path

NOW = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
OLD = "2020-01-01 12:00:00"
RECENT = "2026-10-25 12:00:00"


@pytest.fixture
def records(tmp_path, monkeypatch):
    for key in ("PRIVACY_RETENTION_LEGAL_HOLD", "CHAT_LOG_RETENTION_DAYS", "TRACE_RETENTION_DAYS", "CLOSED_CASE_RETENTION_DAYS"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "records.db"
    root = tmp_path / "uploads"
    root.mkdir()
    evidence = root / "private-name.jpg"
    evidence.write_bytes(b"evidence")
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE chat_logs (id INTEGER PRIMARY KEY, session_id TEXT, created_at TEXT, user_query TEXT);
            CREATE TABLE request_traces (id INTEGER PRIMARY KEY, session_id TEXT, created_at TEXT);
            CREATE TABLE warranty_tickets (id INTEGER PRIMARY KEY, ticket_id TEXT, session_id TEXT, status TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE warranty_chat_consents (id INTEGER PRIMARY KEY, session_id TEXT, updated_at TEXT, contact_email TEXT);
            CREATE TABLE warranty_turns (id INTEGER PRIMARY KEY, ticket_id TEXT, created_at TEXT);
            CREATE TABLE warranty_conversation_messages (id INTEGER PRIMARY KEY, ticket_id TEXT, session_id TEXT, created_at TEXT);
            CREATE TABLE warranty_evidences (id INTEGER PRIMARY KEY, ticket_id TEXT, file_path TEXT, created_at TEXT);
            CREATE TABLE warranty_outcome_observations (id INTEGER PRIMARY KEY, ticket_id TEXT, observed_at TEXT);
            CREATE TABLE conversation_releases (id INTEGER PRIMARY KEY, surface TEXT, conversation_id TEXT, session_id TEXT);
            CREATE TABLE customer_journey_events (id INTEGER PRIMARY KEY, surface TEXT, conversation_id TEXT, session_id TEXT, created_at TEXT);
            CREATE TABLE sales_sessions (id INTEGER PRIMARY KEY, session_id TEXT, status TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE sales_messages (id INTEGER PRIMARY KEY, session_id TEXT, created_at TEXT);
            CREATE TABLE sales_leads (id INTEGER PRIMARY KEY, session_id TEXT, updated_at TEXT);
            CREATE TABLE sales_feedback (id INTEGER PRIMARY KEY, session_id TEXT, created_at TEXT);
            CREATE TABLE sales_events (id INTEGER PRIMARY KEY, session_id TEXT, created_at TEXT);
            CREATE TABLE sales_conversions (id INTEGER PRIMARY KEY, session_id TEXT, email TEXT, matched_by TEXT, total_usd REAL, created_at TEXT, lifecycle_updated_at TEXT);
        """)
        db.executemany("INSERT INTO warranty_tickets VALUES (?, ?, ?, ?, ?, ?)", [
            (1, "expired-ticket", "expired-session", "closed", OLD, OLD),
            (2, "active-ticket", "active-session", "awaiting_admin", OLD, OLD),
            (3, "recent-ticket", "recent-session", "resolved", OLD, RECENT),
            (4, "new-child-ticket", "child-session", "closed", OLD, OLD),
        ])
        db.executemany("INSERT INTO chat_logs VALUES (?, ?, ?, ?)", [
            (1, "expired-session", OLD, "private customer message"),
            (2, "active-session", OLD, "private customer message"),
            (3, "general-old", OLD, "private customer message"),
            (4, "general-recent", OLD, "private customer message"),
            (5, "general-recent", RECENT, "private customer message"),
            (6, "recent-session", OLD, "private customer message"),
        ])
        db.execute("INSERT INTO request_traces VALUES (1, 'general-old', ?)", (OLD,))
        db.execute("INSERT INTO warranty_chat_consents VALUES (1, 'expired-session', ?, 'secret@example.com')", (OLD,))
        db.executemany("INSERT INTO warranty_turns VALUES (?, ?, ?)", [(1, "expired-ticket", OLD), (2, "new-child-ticket", RECENT)])
        db.execute("INSERT INTO warranty_conversation_messages VALUES (1, 'expired-ticket', 'expired-session', ?)", (OLD,))
        db.execute("INSERT INTO warranty_evidences VALUES (1, 'expired-ticket', ?, ?)", (str(evidence), OLD))
        db.execute("INSERT INTO warranty_outcome_observations VALUES (1, 'expired-ticket', ?)", (OLD,))
        db.execute("INSERT INTO conversation_releases VALUES (1, 'warranty', 'expired-ticket', 'expired-session')")
        db.execute("INSERT INTO customer_journey_events VALUES (1, 'warranty', 'expired-ticket', 'expired-session', ?)", (OLD,))
        db.executemany("INSERT INTO sales_sessions VALUES (?, ?, ?, ?, ?)", [(1, "old-sale", "closed", OLD, OLD), (2, "active-sale", "handoff", OLD, OLD)])
        for table in ("sales_messages", "sales_leads", "sales_feedback", "sales_events"):
            db.executemany(f"INSERT INTO {table} VALUES (?, ?, ?)", [(1, "old-sale", OLD), (2, "active-sale", OLD)])
        db.execute("INSERT INTO sales_conversions VALUES (1, 'old-sale', 'secret@example.com', 'lead_email', 2000, ?, ?)", (OLD, OLD))
    return path, root, evidence


def test_legal_hold_is_on_by_default_and_invalid_values_fail_closed(monkeypatch):
    monkeypatch.delenv("PRIVACY_RETENTION_LEGAL_HOLD", raising=False)
    assert legal_hold_active()
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "not-sure")
    assert legal_hold_active()
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    assert not legal_hold_active()


def test_dry_run_has_counts_without_pii_or_side_effects(records, caplog):
    path, root, evidence = records
    result = purge_chat_records(db_path=path, upload_root=root, now=NOW)
    assert result["ok"] and not result["applied"]
    assert result["candidate_rows"]["warranty_tickets"] == 1
    assert result["candidate_rows"]["chat_logs"] == 2
    assert result["candidate_rows"]["sales_sessions"] == 1
    assert result["candidate_files"] == 1
    assert evidence.exists()
    report = json.dumps(result) + caplog.text
    for private in ("secret@example.com", "private customer message", "expired-ticket", "private-name.jpg", str(root)):
        assert private not in report
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM warranty_tickets").fetchone()[0] == 4


def test_apply_blocked_by_default_hold(records):
    path, root, evidence = records
    result = purge_chat_records(db_path=path, upload_root=root, now=NOW, apply=True)
    assert not result["ok"] and result["blocked_by_legal_hold"]
    assert result["deleted_rows"] == {}
    assert evidence.exists()


def test_apply_deletes_expired_closed_conversations_preserves_active_and_recent(records, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "off")
    path, root, evidence = records
    result = purge_chat_records(db_path=path, upload_root=root, now=NOW, apply=True)
    assert result["ok"] and result["applied"]
    assert result["deleted_files"] == 1 and not evidence.exists()
    assert result["deidentified_rows"] == {"sales_conversions": 1}
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT ticket_id FROM warranty_tickets ORDER BY id").fetchall() == [("active-ticket",), ("recent-ticket",), ("new-child-ticket",)]
        assert db.execute("SELECT id FROM chat_logs ORDER BY id").fetchall() == [(2,), (4,), (5,), (6,)]
        for table in ("warranty_chat_consents", "warranty_conversation_messages", "warranty_evidences", "warranty_outcome_observations", "conversation_releases", "customer_journey_events"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert db.execute("SELECT session_id FROM sales_sessions").fetchall() == [("active-sale",)]
        assert db.execute("SELECT session_id, email, matched_by, total_usd FROM sales_conversions").fetchone() == (None, None, None, 2000)


@pytest.mark.parametrize("use_symlink", [False, True])
def test_evidence_outside_upload_root_retains_case(records, monkeypatch, tmp_path, use_symlink):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    path, root, _ = records
    outside = tmp_path / "private-outside.txt"
    outside.write_text("keep")
    value = outside
    if use_symlink:
        value = root / "link.txt"
        value.symlink_to(outside)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE warranty_evidences SET file_path = ?", (str(value),))
    result = purge_chat_records(db_path=path, upload_root=root, now=NOW, apply=True)
    assert result["unsafe_file_paths"] == 1 and outside.read_text() == "keep"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM warranty_tickets WHERE id=1").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM warranty_evidences").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM warranty_chat_consents").fetchone()[0] == 1


def test_invalid_dates_and_recent_linked_sales_activity_preserve_cases(records, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    path, root, evidence = records
    with sqlite3.connect(path) as db:
        db.execute("UPDATE warranty_tickets SET updated_at = NULL WHERE id=1")
        db.execute("UPDATE sales_leads SET updated_at = ? WHERE id=1", (RECENT,))
    result = purge_chat_records(db_path=path, upload_root=root, now=NOW, apply=True)
    assert result["ok"] and evidence.exists()
    assert "warranty_tickets" not in result["candidate_rows"]
    assert "sales_sessions" not in result["candidate_rows"]


def test_evidence_shared_with_active_case_is_preserved(records, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    path, root, evidence = records
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO warranty_evidences VALUES (2, 'active-ticket', ?, ?)", (str(evidence), OLD))
    result = purge_chat_records(db_path=path, upload_root=root, now=NOW, apply=True)
    assert result["shared_file_paths"] == 1 and evidence.exists()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM warranty_tickets").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM warranty_evidences").fetchone()[0] == 2


def test_missing_database_is_not_created_and_invalid_retention_is_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_LOG_RETENTION_DAYS", "0")
    monkeypatch.setenv("TRACE_RETENTION_DAYS", "broken")
    path = tmp_path / "missing.db"
    result = purge_chat_records(db_path=path)
    assert not result["ok"] and not path.exists()
    assert result["retention_days"]["chat_logs"] == 90
    assert result["retention_days"]["request_traces"] == 30


def test_incomplete_case_schema_prevents_any_deletion(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    path = tmp_path / "incomplete.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE warranty_tickets (id INTEGER PRIMARY KEY)")
        db.execute("CREATE TABLE chat_logs (id INTEGER PRIMARY KEY, session_id TEXT, created_at TEXT)")
        db.execute("INSERT INTO chat_logs VALUES (1, 'old', ?)", (OLD,))
    result = purge_chat_records(db_path=path, apply=True, now=NOW)
    assert not result["ok"] and not result["applied"]
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0] == 1


def test_legacy_evidence_purge_honors_global_legal_hold(tmp_path, monkeypatch):
    import warranty_models as wm
    from warranty_evidence_purge import purge_old_evidence

    monkeypatch.delenv("PRIVACY_RETENTION_LEGAL_HOLD", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    wm.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    @contextmanager
    def session():
        with factory() as db:
            yield db
            db.commit()

    monkeypatch.setattr(wm, "warranty_db_session", session)
    evidence = tmp_path / "evidence.jpg"
    evidence.write_bytes(b"keep")
    with session() as db:
        db.add(wm.WarrantyTicket(ticket_id="old", session_id="s", current_node_id="end", status="closed", created_at=datetime(2020, 1, 1), updated_at=datetime(2020, 1, 1)))
        db.add(wm.WarrantyEvidence(ticket_id="old", evidence_type="photo", file_path=str(evidence), created_at=datetime(2020, 1, 1)))
    result = purge_old_evidence(days=90, apply=True, upload_root=tmp_path)
    assert result["blocked_by_legal_hold"] and evidence.exists()
    with session() as db:
        assert db.query(wm.WarrantyEvidence).count() == 1
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    result = purge_old_evidence(days=90, apply=True, upload_root=tmp_path)
    assert result["ok"] and not evidence.exists()
    engine.dispose()


def test_upload_root_itself_is_not_a_deletion_target(tmp_path):
    assert safe_evidence_path(str(tmp_path), tmp_path) is None
