from contextlib import closing
from datetime import datetime, timezone
import sqlite3

import pytest

from script.check_answer_quality_cohort import cohort_readiness


RELEASE = 'a' * 40
OTHER_RELEASE = 'b' * 40
NOW = datetime(2026, 11, 20, 18, tzinfo=timezone.utc)
PREVIOUS = '2026-11-20 08:00:00'
RECENT = '2026-11-20 11:00:00'


@pytest.fixture
def audit_db(tmp_path):
    path = tmp_path / 'audit?test.db'
    with closing(sqlite3.connect(path)) as db:
        db.executescript('''
            CREATE TABLE conversation_releases (surface, conversation_id, session_id, initial_release, latest_release, mixed);
            CREATE TABLE sales_sessions (id INTEGER PRIMARY KEY, session_id, channel);
            CREATE TABLE sales_messages (id INTEGER PRIMARY KEY, session_id, role, content, created_at);
            CREATE TABLE warranty_chat_consents (session_id TEXT UNIQUE, accepted_at);
            CREATE TABLE warranty_tickets (ticket_id, session_id);
            CREATE TABLE warranty_conversation_messages (session_id, ticket_id, exchange_id, role, origin, content, created_at);
        ''')
    return path


def _consent(db, sid, when):
    db.execute('INSERT OR IGNORE INTO warranty_chat_consents VALUES (?, ?)', (sid, when))


def _sales(db, sid, *, release=RELEASE, latest=RELEASE, mixed=0, channel='web',
           question='customer message', reply='assistant reply', consent=PREVIOUS, when=PREVIOUS):
    row = db.execute('INSERT INTO sales_sessions(session_id, channel) VALUES (?, ?)', (sid, channel)).lastrowid
    db.execute('INSERT INTO conversation_releases VALUES (?, ?, ?, ?, ?, ?)', ('sales', str(row), sid, release, latest, mixed))
    _consent(db, sid, consent)
    if question is not None:
        db.execute('INSERT INTO sales_messages(session_id, role, content, created_at) VALUES (?,?,?,?)',
                   (sid, 'user', question, when))
    if reply is not None:
        db.execute('INSERT INTO sales_messages(session_id, role, content, created_at) VALUES (?,?,?,?)',
                   (sid, 'assistant', reply, when))


def _warranty(db, ticket, sid, *, origin='api_response', exchange='a', actor='customer_input'):
    _consent(db, sid, PREVIOUS)
    db.execute('INSERT INTO warranty_tickets VALUES (?, ?)', (ticket, sid))
    db.execute('INSERT INTO conversation_releases VALUES (?, ?, ?, ?, ?, ?)', ('warranty', ticket, sid, RELEASE, RELEASE, 0))
    db.executemany('INSERT INTO warranty_conversation_messages VALUES (?,?,?,?,?,?,?)', [
        (sid, ticket, 'a', 'user', actor, 'question', PREVIOUS),
        (sid, ticket, exchange, 'assistant', origin, 'answer', PREVIOUS),
    ])


def test_readiness_counts_only_consent_based_reviewable_current_sessions(audit_db):
    with closing(sqlite3.connect(audit_db)) as db, db:
        _sales(db, 'shared')
        _sales(db, 'tidio', channel='tidio', question='[button:compare]')
        _sales(db, 'mixed', mixed=1)
        _sales(db, 'older', release=OTHER_RELEASE)
        _sales(db, 'newer', latest=OTHER_RELEASE)
        _sales(db, 'recent', when=RECENT)
        _sales(db, 'test', channel='test')
        _sales(db, 'no-consent', consent=None)
        _sales(db, 'late-consent', consent=RECENT)
        _sales(db, 'no-reply', reply=None)
        _sales(db, 'no-question', question=None)
        _sales(db, 'feedback-only', question='[button:feedback:up]')
        _warranty(db, 'w1', 'shared')
        _warranty(db, 'w2', 'shared')
        _warranty(db, 'w3', 'warranty', actor='user_action')
        _warranty(db, 'w4', 'legacy', origin='legacy_reply')
        _warranty(db, 'w5', 'unpaired', exchange='b')
    result = cohort_readiness(audit_db, RELEASE, now=NOW)
    assert result == {'release': RELEASE, 'candidate_sessions': 3, 'sales_candidate_sessions': 2,
                      'warranty_candidate_sessions': 2, 'sample_minimum': 30, 'sample_maximum': 50,
                      'ready_for_manual_screening': False, 'quiet_hours': 2}


def test_ready_for_review_at_thirty_without_emitting_ids(audit_db):
    with closing(sqlite3.connect(audit_db)) as db, db:
        for i in range(30):
            _sales(db, f'specific-session-{i}')
    report = cohort_readiness(audit_db, RELEASE, now=NOW)
    assert report['candidate_sessions'] == 30 and report['ready_for_manual_screening'] is True
    assert 'specific-session' not in str(report)


def test_exact_release_and_existing_database_required(tmp_path):
    path = tmp_path / 'not-there.db'
    with pytest.raises(ValueError):
        cohort_readiness(path, 'current', now=NOW)
    with pytest.raises(sqlite3.Error):
        cohort_readiness(path, RELEASE, now=NOW)
    assert not path.exists()
