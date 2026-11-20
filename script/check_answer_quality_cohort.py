#!/usr/bin/env python3
"""Read only aggregate readiness for a release's real-conversation review."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo


_QUERY = """
WITH release_sessions AS (
    SELECT surface, conversation_id, session_id FROM conversation_releases
    WHERE initial_release = :release AND latest_release = :release AND mixed = 0
), eligible AS (
    SELECT r.surface, r.session_id
    FROM release_sessions r JOIN sales_sessions s
      ON r.surface = 'sales' AND r.conversation_id = CAST(s.id AS TEXT)
      AND r.session_id = s.session_id
    JOIN warranty_chat_consents c ON c.session_id = s.session_id AND c.accepted_at IS NOT NULL
    WHERE s.channel IN ('web', 'tidio')
      AND EXISTS (
        SELECT 1 FROM sales_messages u JOIN sales_messages a
          ON a.session_id = u.session_id AND a.id > u.id AND a.role = 'assistant'
          AND length(trim(a.content)) > 0
        WHERE u.session_id = s.session_id AND u.role = 'user'
          AND length(trim(u.content)) > 0 AND u.content NOT GLOB '[[]button:feedback:*'
          AND c.accepted_at <= u.created_at
      )
      AND NOT EXISTS (
        SELECT 1 FROM sales_messages recent
        WHERE recent.session_id = s.session_id AND recent.created_at > :quiet_cutoff
      )
    UNION ALL
    SELECT r.surface, r.session_id
    FROM release_sessions r JOIN warranty_tickets w
      ON r.surface = 'warranty' AND r.conversation_id = w.ticket_id
      AND r.session_id = w.session_id
    JOIN warranty_chat_consents c ON c.session_id = w.session_id AND c.accepted_at IS NOT NULL
    WHERE EXISTS (
        SELECT 1 FROM warranty_conversation_messages u JOIN warranty_conversation_messages a
          ON a.session_id = u.session_id AND a.ticket_id = u.ticket_id
          AND a.exchange_id = u.exchange_id AND a.role = 'assistant' AND a.origin = 'api_response'
          AND length(trim(a.content)) > 0
        WHERE u.ticket_id = w.ticket_id AND u.session_id = w.session_id
          AND u.role = 'user' AND u.origin IN ('customer_input', 'user_action')
          AND length(trim(u.content)) > 0 AND c.accepted_at <= u.created_at
      )
      AND NOT EXISTS (
        SELECT 1 FROM warranty_conversation_messages recent
        WHERE recent.ticket_id = w.ticket_id AND recent.session_id = w.session_id
          AND recent.origin IN ('customer_input', 'user_action', 'api_response')
          AND recent.created_at > :quiet_cutoff
      )
)
SELECT COUNT(DISTINCT session_id),
       COUNT(DISTINCT CASE WHEN surface = 'sales' THEN session_id END),
       COUNT(DISTINCT CASE WHEN surface = 'warranty' THEN session_id END)
FROM eligible
"""


def cohort_readiness(db_path: Path, release: str, *, now: datetime, quiet_hours: int = 2) -> dict:
    if not re.fullmatch(r'[a-f0-9]{40}', release):
        raise ValueError('Use the exact 40-character deployment commit.')
    if now.utcoffset() is None or quiet_hours < 0:
        raise ValueError('Use an aware clock and a non-negative quiet window.')
    # Application timestamps are stored as naive America/Chicago wall time.
    cutoff = (now - timedelta(hours=quiet_hours)).astimezone(ZoneInfo('America/Chicago'))
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True, timeout=8)) as db:
        db.execute('PRAGMA query_only=ON')
        total, sales, warranty = db.execute(_QUERY, {
            'release': release,
            'quiet_cutoff': cutoff.replace(tzinfo=None).isoformat(sep=' ', timespec='microseconds'),
        }).fetchone()
    return {'release': release, 'candidate_sessions': total, 'sales_candidate_sessions': sales,
            'warranty_candidate_sessions': warranty, 'sample_minimum': 30, 'sample_maximum': 50,
            'ready_for_manual_screening': total >= 30, 'quiet_hours': quiet_hours}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('/app/db_data/chat_history.db'))
    parser.add_argument('--release', default=os.getenv('APP_RELEASE_ID', ''))
    args = parser.parse_args()
    try:
        print(json.dumps(cohort_readiness(args.db, args.release, now=datetime.now(ZoneInfo('UTC')))))
    except (ValueError, sqlite3.Error) as exc:
        parser.exit(2, f'Cohort counter unavailable ({type(exc).__name__}).\n')


if __name__ == '__main__':
    main()
