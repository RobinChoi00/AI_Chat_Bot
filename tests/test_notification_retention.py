"""Retention of queue references and private resume recipients in temporary DBs."""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from privacy_retention import purge_chat_records

NOW = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
OLD = "2020-01-01 12:00:00"
RECENT = "2026-10-25 12:00:00"
OLD_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
RECENT_EPOCH = datetime(2026, 10, 25, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def records(tmp_path, monkeypatch):
    for name in ("PRIVACY_RETENTION_LEGAL_HOLD", "CHAT_LOG_RETENTION_DAYS", "TRACE_RETENTION_DAYS", "CLOSED_CASE_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "notification-retention.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE warranty_tickets (id INTEGER PRIMARY KEY, ticket_id TEXT, session_id TEXT,
                status TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE warranty_notification_jobs (id INTEGER PRIMARY KEY, ticket_id TEXT, kind TEXT,
                source_key TEXT, status TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE warranty_resume_requests (attempt_id TEXT PRIMARY KEY, ticket_id TEXT,
                recipient_email TEXT, status TEXT, created_at REAL, updated_at REAL);
        """)
        db.execute("INSERT INTO warranty_tickets VALUES (1, 'private-ticket', 'private-session', 'closed', ?, ?)", (OLD, OLD))
        db.execute("INSERT INTO warranty_notification_jobs VALUES (1, 'private-ticket', 'resume', 'private-attempt', 'completed', ?, ?)", (OLD, OLD))
        db.execute("INSERT INTO warranty_resume_requests VALUES ('private-attempt', 'private-ticket', 'private@example.test', 'sent', ?, ?)", (OLD_EPOCH, OLD_EPOCH))
    return path


def counts(path):
    with sqlite3.connect(path) as db:
        return {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("warranty_tickets", "warranty_notification_jobs", "warranty_resume_requests")}


def test_expired_closed_case_deletes_queue_and_private_recipient_rows(records, monkeypatch, caplog):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and result["applied"]
    assert result["deleted_rows"] == {"warranty_tickets": 1, "warranty_notification_jobs": 1, "warranty_resume_requests": 1}
    assert set(counts(records).values()) == {0}
    report = json.dumps(result) + caplog.text
    assert all(private not in report for private in ("private@example.test", "private-ticket", "private-session", "private-attempt", str(records)))


def test_legal_hold_blocks_notification_cleanup_and_dry_run_never_deletes(records):
    before = counts(records)
    dry = purge_chat_records(db_path=records, now=NOW)
    assert dry["ok"] and not dry["applied"] and dry["legal_hold"]
    assert dry["candidate_rows"]["warranty_resume_requests"] == 1
    held = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert held["blocked_by_legal_hold"] and not held["applied"]
    assert counts(records) == before


@pytest.mark.parametrize("status", ["pending", "processing", "failed", "unknown-future-state"])
def test_outstanding_jobs_preserve_even_old_closed_cases_and_recipients(records, monkeypatch, status):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "off")
    with sqlite3.connect(records) as db:
        db.execute("UPDATE warranty_notification_jobs SET status=?", (status,))
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and result["applied"]
    assert result["deleted_rows"] == {}
    assert set(counts(records).values()) == {1}


@pytest.mark.parametrize("surface,value", [
    ("job", RECENT), ("job", None), ("job", "invalid-date"),
    ("resume", RECENT_EPOCH), ("resume", None), ("resume", "invalid-date"),
])
def test_recent_or_unknown_delivery_activity_preserves_case(records, monkeypatch, surface, value):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    table = "warranty_notification_jobs" if surface == "job" else "warranty_resume_requests"
    with sqlite3.connect(records) as db:
        db.execute(f"UPDATE {table} SET updated_at=?", (value,))
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and result["deleted_rows"] == {}
    assert set(counts(records).values()) == {1}


def test_active_case_and_same_session_sibling_remain_protected(records, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    with sqlite3.connect(records) as db:
        db.execute("INSERT INTO warranty_tickets VALUES (2, 'active-sibling', 'private-session', 'in_progress', ?, ?)", (OLD, OLD))
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and result["deleted_rows"] == {}
    assert counts(records) == {"warranty_tickets": 2, "warranty_notification_jobs": 1, "warranty_resume_requests": 1}


@pytest.mark.parametrize("job_status", ["completed", "dead_letter"])
def test_old_orphaned_jobs_and_recipient_rows_are_cleaned(records, monkeypatch, job_status):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    with sqlite3.connect(records) as db:
        db.execute("DELETE FROM warranty_tickets")
        db.execute("UPDATE warranty_notification_jobs SET status=?", (job_status,))
        db.execute("UPDATE warranty_resume_requests SET status='failed'")
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and result["applied"]
    assert result["deleted_rows"] == {"warranty_notification_jobs": 1, "warranty_resume_requests": 1}
    assert set(counts(records).values()) == {0}


@pytest.mark.parametrize("status,remaining", [
    ("pending", 1), ("delivering", 1), ("unknown-future-state", 1), ("sent", 0),
])
def test_orphan_recipient_rows_require_a_terminal_delivery_state(records, monkeypatch, status, remaining):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    with sqlite3.connect(records) as db:
        db.execute("DELETE FROM warranty_tickets")
        db.execute("DROP TABLE warranty_notification_jobs")
        db.execute("UPDATE warranty_resume_requests SET status=?", (status,))
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and result["applied"]
    with sqlite3.connect(records) as db:
        assert db.execute("SELECT COUNT(*) FROM warranty_resume_requests").fetchone()[0] == remaining


def test_utc_queue_datetime_and_epoch_cutoffs_are_not_read_as_local_time(records, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    cutoff_minus_two_hours = datetime(2024, 10, 26, 10, tzinfo=timezone.utc)
    with sqlite3.connect(records) as db:
        db.execute("UPDATE warranty_notification_jobs SET updated_at=?", (cutoff_minus_two_hours.replace(tzinfo=None).isoformat(),))
        db.execute("UPDATE warranty_resume_requests SET updated_at=?", (cutoff_minus_two_hours.timestamp(),))
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and set(counts(records).values()) == {0}


def test_missing_new_tables_still_allows_existing_retention(records, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    with sqlite3.connect(records) as db:
        db.execute("DROP TABLE warranty_resume_requests")
        db.execute("DROP TABLE warranty_notification_jobs")
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert result["ok"] and result["deleted_rows"] == {"warranty_tickets": 1}


def test_incomplete_existing_delivery_schema_fails_closed(records, monkeypatch):
    monkeypatch.setenv("PRIVACY_RETENTION_LEGAL_HOLD", "0")
    with sqlite3.connect(records) as db:
        db.execute("ALTER TABLE warranty_notification_jobs RENAME TO saved_jobs")
        db.execute("CREATE TABLE warranty_notification_jobs (id INTEGER PRIMARY KEY)")
    result = purge_chat_records(db_path=records, apply=True, now=NOW)
    assert not result["ok"] and not result["applied"] and result["error"] == "retention_failed"
    with sqlite3.connect(records) as db:
        assert db.execute("SELECT COUNT(*) FROM warranty_tickets").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM warranty_resume_requests").fetchone()[0] == 1
