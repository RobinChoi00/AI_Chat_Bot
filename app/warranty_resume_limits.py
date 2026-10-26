"""Atomic, restart-safe admission and delivery state for resume email requests.

SQLite's write reservation serializes competing workers on the single-host
deployment. A process lock also protects tests using one shared in-memory
connection. The notification queue stores only ``attempt_id``; recipients remain
in this private application table, never in queue payloads or rate-limit keys.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import text

_LOCK = threading.RLock()
_COOLDOWN_SECONDS = 60
_DELIVERY_LEASE_SECONDS = 300
_LIMITS = (("ip_hash", 60, 6), ("ip_hash", 3600, 10),
           ("session_hash", 3600, 3), ("recipient_hash", 3600, 3))


@contextmanager
def _transaction():
    import warranty_models as models

    with _LOCK, models._engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            connection.exec_driver_sql("""CREATE TABLE IF NOT EXISTS warranty_resume_requests (
                attempt_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
                ip_hash TEXT NOT NULL, session_hash TEXT NOT NULL, recipient_hash TEXT NOT NULL,
                recipient_email TEXT NOT NULL, created_at REAL NOT NULL,
                status TEXT NOT NULL, updated_at REAL NOT NULL, lease_token TEXT
            )""")
            columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(warranty_resume_requests)")}
            if "lease_token" not in columns:
                connection.exec_driver_sql("ALTER TABLE warranty_resume_requests ADD COLUMN lease_token TEXT")
            connection.exec_driver_sql("""CREATE INDEX IF NOT EXISTS ix_resume_request_time
                ON warranty_resume_requests(created_at)""")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _hash(kind: str, value: str) -> str:
    secret = os.environ.get("ADMIN_SESSION_SECRET", "")
    if len(secret.strip()) < 32:
        raise RuntimeError("Resume signing is unavailable")
    return hmac.new(secret.encode(), f"resume-limit:{kind}:{value}".encode(), hashlib.sha256).hexdigest()


def reserve_resume_attempt(*, attempt_id: str, ticket_id: str, session_id: str,
                           recipient: str, client_ip: str) -> None:
    """Commit admission, saved contact, and queue reference as one transaction."""
    from warranty_notification_jobs import WarrantyNotificationJob, initialize_notification_storage

    initialize_notification_storage()
    now = time.time()
    keys = {"ip_hash": _hash("ip", client_ip), "session_hash": _hash("session", session_id),
            "recipient_hash": _hash("recipient", recipient.lower())}
    with _transaction() as connection:
        ticket = connection.execute(text("SELECT collected_data FROM warranty_tickets WHERE ticket_id=:ticket_id AND session_id=:session_id"),
                                    {"ticket_id": ticket_id, "session_id": session_id}).first()
        if ticket is None:
            raise HTTPException(404, "Warranty case not found.")
        recent = connection.execute(text("""SELECT MAX(created_at) FROM warranty_resume_requests
            WHERE (session_hash=:session_hash OR recipient_hash=:recipient_hash)
            AND status IN ('pending','delivering','sent') AND created_at>:cutoff"""),
            {**keys, "cutoff": now - _COOLDOWN_SECONDS}).scalar()
        if recent is not None:
            wait = max(1, math.ceil(float(recent) + _COOLDOWN_SECONDS - now))
            raise HTTPException(429, f"A resume email was requested recently. Please try again in {wait} seconds.",
                                headers={"Retry-After": str(wait)})
        for column, window, maximum in _LIMITS:
            rows = connection.execute(text(f"SELECT created_at FROM warranty_resume_requests WHERE {column}=:key AND created_at>:cutoff ORDER BY created_at"),
                                      {"key": keys[column], "cutoff": now - window}).scalars().all()
            if len(rows) >= maximum:
                wait = max(1, math.ceil(float(rows[-maximum]) + window - now))
                raise HTTPException(429, "Too many resume email requests. Please try again later.",
                                    headers={"Retry-After": str(wait)})
        connection.execute(text("""INSERT INTO warranty_resume_requests
            (attempt_id,ticket_id,ip_hash,session_hash,recipient_hash,recipient_email,created_at,status,updated_at)
            VALUES (:attempt_id,:ticket_id,:ip_hash,:session_hash,:recipient_hash,:recipient,:now,'pending',:now)"""),
            {**keys, "attempt_id": attempt_id, "ticket_id": ticket_id, "recipient": recipient, "now": now})
        try:
            data = json.loads(ticket[0] or "{}")
        except (ValueError, TypeError):
            data = {}
        data.update(resume_email=recipient, resume_delivery_attempt_id=attempt_id,
                    resume_delivery_status="pending", resume_requested_at=datetime.fromtimestamp(now, timezone.utc).isoformat())
        connection.execute(text("UPDATE warranty_tickets SET collected_data=:data WHERE ticket_id=:ticket_id"),
                           {"data": json.dumps(data), "ticket_id": ticket_id})
        queued_at = datetime.fromtimestamp(now, timezone.utc).replace(tzinfo=None)
        connection.execute(WarrantyNotificationJob.__table__.insert().values(
            kind="resume", source_key=attempt_id, ticket_id=ticket_id, status="pending",
            attempts=0, created_at=queued_at, updated_at=queued_at,
        ))


def claim_resume_attempt(attempt_id: str) -> dict | None:
    """Lease a queued reference; duplicate workers never send the same request."""
    now = time.time()
    with _transaction() as connection:
        row = connection.execute(text("SELECT * FROM warranty_resume_requests WHERE attempt_id=:attempt_id"),
                                 {"attempt_id": attempt_id}).mappings().first()
        if row is None or row["status"] in {"sent", "superseded"}:
            return None
        if row["status"] == "delivering" and float(row["updated_at"]) > now - _DELIVERY_LEASE_SECONDS:
            raise RuntimeError("Resume delivery is already in progress")
        lease_token = uuid4().hex
        connection.execute(text("UPDATE warranty_resume_requests SET status='delivering',updated_at=:now,lease_token=:lease_token WHERE attempt_id=:attempt_id"),
                           {"attempt_id": attempt_id, "now": now, "lease_token": lease_token})
        return {**row, "lease_token": lease_token}


def finish_resume_attempt(attempt_id: str, *, sent: bool, lease_token: str) -> None:
    """Record transport acceptance without overwriting a later save request."""
    now = time.time()
    status = "sent" if sent else "failed"
    with _transaction() as connection:
        row = connection.execute(text("SELECT ticket_id FROM warranty_resume_requests WHERE attempt_id=:attempt_id AND lease_token=:lease_token"),
                                 {"attempt_id": attempt_id, "lease_token": lease_token}).first()
        if row is None:
            return
        connection.execute(text("UPDATE warranty_resume_requests SET status=:status,updated_at=:now,lease_token=NULL WHERE attempt_id=:attempt_id AND lease_token=:lease_token"),
                           {"status": status, "now": now, "attempt_id": attempt_id, "lease_token": lease_token})
        ticket = connection.execute(text("SELECT collected_data FROM warranty_tickets WHERE ticket_id=:ticket_id"),
                                    {"ticket_id": row[0]}).first()
        if ticket is None:
            return
        data = json.loads(ticket[0] or "{}")
        if data.get("resume_delivery_attempt_id") == attempt_id:
            data.update(resume_delivery_status=status, resume_delivery_checked_at=datetime.fromtimestamp(now, timezone.utc).isoformat())
            connection.execute(text("UPDATE warranty_tickets SET collected_data=:data WHERE ticket_id=:ticket_id"),
                               {"data": json.dumps(data), "ticket_id": row[0]})


def supersede_resume_attempt(attempt_id: str, *, lease_token: str) -> None:
    """A later explicit save replaces an unsent older recipient request."""
    with _transaction() as connection:
        connection.execute(text("""UPDATE warranty_resume_requests SET status='superseded',updated_at=:now,lease_token=NULL
            WHERE attempt_id=:attempt_id AND lease_token=:lease_token"""),
            {"attempt_id": attempt_id, "lease_token": lease_token, "now": time.time()})
