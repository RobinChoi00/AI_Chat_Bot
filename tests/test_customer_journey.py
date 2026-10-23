"""Journey observations must preserve consent, provenance and unknown history."""
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import warranty_models as wm
import customer_journey as journey


@pytest.fixture
def store(monkeypatch):
    engine = create_engine('sqlite:///:memory:', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    wm.Base.metadata.create_all(engine)
    monkeypatch.setattr(wm, '_SessionFactory', sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setenv('APP_RELEASE_ID', 'release-a')
    app = FastAPI()
    app.include_router(journey.router)
    with TestClient(app) as client:
        yield client
    engine.dispose()


def consent(sid='visitor'):
    with wm.warranty_db_session() as db:
        db.add(wm.WarrantyChatConsent(session_id=sid, domain='osakiusa.com', accepted_at=datetime(2026, 10, 23)))


def body(**values):
    return {'event_id': str(uuid4()), 'session_id': 'visitor', 'domain': 'osakiusa.com', 'stage': 'email', 'event': 'shown', **values}


def add_case(db, cid, sid=None):
    row = wm.WarrantyTicket(ticket_id=cid, session_id=sid or cid, domain='osakiusa.com', current_node_id='root')
    db.add(row)
    db.flush()
    return row


def tagged(db, row, is_new=True):
    return journey.register_conversation(db, surface='warranty', conversation_id=row.ticket_id,
        session_id=row.session_id, domain=row.domain, is_new=is_new)


def case_ids(db, cohort):
    q = journey.apply_cohort(db, db.query(wm.WarrantyTicket), surface='warranty',
        id_column=wm.WarrantyTicket.ticket_id, cohort=cohort)
    return {r.ticket_id for r in q.all()}


def test_events_require_recording_consent_and_reject_customer_content(store):
    assert store.post('/api/v1/journey/events', json=body()).status_code == 403
    consent()
    assert store.post('/api/v1/journey/events', json=body(email='private@example.test')).status_code == 422
    assert store.post('/api/v1/journey/events', json=body(stage='private@example.test')).status_code == 422
    assert store.post('/api/v1/journey/events', json=body(session_id='private@example.test')).status_code == 422
    assert store.post('/api/v1/journey/events', json=body(domain='titanchair.com')).status_code == 422
    with wm.warranty_db_session() as db:
        assert db.query(journey.CustomerJourneyEvent).count() == 0


def test_client_retry_is_deduplicated_and_does_not_claim_server_outcome(store):
    consent()
    event = body()
    first = store.post('/api/v1/journey/events', json=event)
    retry = store.post('/api/v1/journey/events', json=event)
    assert first.json() == {'recorded': True, 'duplicate': False}
    assert retry.json() == {'recorded': True, 'duplicate': True}
    assert store.post('/api/v1/journey/events', json={**event, 'event': 'completed'}).status_code == 409
    with wm.warranty_db_session() as db:
        row = db.query(journey.CustomerJourneyEvent).one()
        assert row.source == 'client' and row.release_id == 'release-a'
        assert row.conversation_id is None
        assert db.query(wm.WarrantyTicket).count() == 0


def test_old_cases_are_never_relabelled_current_and_cross_release_is_mixed(store, monkeypatch):
    with wm.warranty_db_session() as db:
        add_case(db, 'unobserved-old')
        old = add_case(db, 'old-resumed')
        tagged(db, old, is_new=False)
        new = add_case(db, 'new-a')
        tagged(db, new)
        stable = add_case(db, 'stable-a')
        tagged(db, stable)
        assert case_ids(db, 'all') == {'unobserved-old', 'old-resumed', 'new-a', 'stable-a'}
        assert case_ids(db, 'legacy') == {'unobserved-old'}
        assert case_ids(db, 'current') == {'new-a', 'stable-a'}
        assert case_ids(db, 'mixed') == {'old-resumed'}
        monkeypatch.setenv('APP_RELEASE_ID', 'release-b')
        tagged(db, new, is_new=False)
        fresh = add_case(db, 'new-b')
        tagged(db, fresh)
        assert case_ids(db, 'current') == {'new-b'}
        assert case_ids(db, 'release-a') == {'stable-a'}
        assert case_ids(db, 'mixed') == {'old-resumed', 'new-a'}
        metadata = journey.cohort_metadata(db, surface='warranty', selected='current')
        assert metadata['available_releases'] == ['release-a', 'release-b']
        assert 'unknown' in metadata['legacy_note']


def test_empty_telemetry_is_unmeasured_not_zero_success(store):
    with wm.warranty_db_session() as db:
        result = journey.journey_summary(db, surface='warranty')
        assert result['measured'] is False
        assert result['measurement_started_at'] is None
        assert result['stages'] == []
        assert result['sessions_observed'] == 0


def test_observation_counts_use_distinct_sessions_and_missing_completion_is_not_exit(store):
    consent()
    for _ in range(2):
        assert store.post('/api/v1/journey/events', json=body()).status_code == 200
    assert store.post('/api/v1/journey/events', json=body(event='skipped')).status_code == 200
    with wm.warranty_db_session() as db:
        result = journey.journey_summary(db, surface='warranty')
        assert result['sessions_observed'] == 1
        assert result['client_events'] == 3
        assert result['server_events'] == 0
        stage = result['stages'][0]
        assert stage['shown'] == stage['skipped'] == 1
        assert stage['completed'] == stage['without_completion'] == 0
        assert 'not confirmed departure' in result['note']


def test_pre_case_observations_link_only_after_matching_real_case_exists(store):
    consent()
    assert store.post('/api/v1/journey/events', json=body()).status_code == 200
    with wm.warranty_db_session() as db:
        assert journey.journey_summary(db, surface='warranty')['unlinked_sessions'] == 1
        assert journey.journey_summary(db, surface='warranty', cohort='current')['measured'] is False
        row = add_case(db, 'created-case', 'visitor')
        tagged(db, row)
        assert journey.journey_summary(db, surface='warranty')['unlinked_sessions'] == 0
        assert journey.journey_summary(db, surface='warranty', cohort='current')['sessions_observed'] == 1


def test_client_observation_on_new_release_marks_existing_conversation_mixed(store, monkeypatch):
    consent()
    with wm.warranty_db_session() as db:
        tagged(db, add_case(db, 'case-a', 'visitor'))
    monkeypatch.setenv('APP_RELEASE_ID', 'release-b')
    assert store.post('/api/v1/journey/events', json=body()).status_code == 200
    with wm.warranty_db_session() as db:
        assert case_ids(db, 'release-a') == set()
        assert case_ids(db, 'mixed') == {'case-a'}


def test_pre_case_events_spanning_releases_do_not_become_strict_new_cohort(store, monkeypatch):
    consent()
    assert store.post('/api/v1/journey/events', json=body()).status_code == 200
    monkeypatch.setenv('APP_RELEASE_ID', 'release-b')
    with wm.warranty_db_session() as db:
        tagged(db, add_case(db, 'new-case-b', 'visitor'))
        assert case_ids(db, 'current') == set()
        assert case_ids(db, 'mixed') == {'new-case-b'}
        assert journey.journey_summary(db, surface='warranty', cohort='mixed')['sessions_observed'] == 1


def test_server_handoff_request_does_not_become_staff_completion(store):
    with wm.warranty_db_session() as db:
        journey.record_server_event(db, surface='sales', session_id='shopper', conversation_id='1',
            domain='osakiusa.com', stage='handoff', event='requested')
        db.flush()
        result = journey.journey_summary(db, surface='sales')
        assert result['server_events'] == 1 and result['client_events'] == 0
        assert result['stages'][0]['requested'] == 1
        assert result['stages'][0]['completed'] == 0


def test_invalid_cohort_cannot_change_query_and_invalid_release_is_not_exposed(store, monkeypatch):
    with wm.warranty_db_session() as db:
        with pytest.raises(__import__('fastapi').HTTPException):
            case_ids(db, 'release with spaces')
    monkeypatch.setenv('APP_RELEASE_ID', 'bad\nvalue')
    assert journey.current_release() == 'unconfigured'


def test_lifetime_observations_survive_legacy_sqlite_bind_limit(store):
    import sqlite3
    with wm.warranty_db_session() as db:
        connection = db.connection().connection.driver_connection
        previous = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        try:
            ids = [f'session-{i}' for i in range(1100)]
            db.add_all([journey.CustomerJourneyEvent(event_id=str(uuid4()), surface='sales',
                session_id=sid, conversation_id=str(i), domain='osakiusa.com',
                release_id='release-a', source='server', stage='response', event='completed')
                for i, sid in enumerate(ids)])
            db.flush()
            result = journey.journey_summary(db, surface='sales', session_ids=ids)
            assert result['sessions_observed'] == 1100
            assert result['server_events'] == 1100
        finally:
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)
