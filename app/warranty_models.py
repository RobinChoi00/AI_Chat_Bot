"""
warranty_models.py
==================
SQLAlchemy ORM for the Warranty AI Workflow system.

Uses the same SQLite file as the chat/usage logs (db_data/chat_history.db)
but keeps its own declarative Base so there is zero coupling to main.py's
ORM setup — avoiding circular imports when warranty_workflow.py imports here.

Tables
------
  warranty_tickets    – one row per warranty intake session
  warranty_turns      – one row per Q&A step within a session
  warranty_evidences  – uploaded evidence files (photos, videos, receipts)
  warranty_decisions  – admin decisions recorded against a ticket
  ringcentral_webhook_events – durable, idempotent webhook inbox
  ringcentral_call_states     – restart-safe IVR call state
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, cast

import pytz
from sqlalchemy import (
    Column, DateTime, Integer, String, Text,
    create_engine, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# DB setup — same file as main.py, independent engine/session factory
# ---------------------------------------------------------------------------

_DB_DIR = Path(__file__).resolve().parent.parent / "db_data"
_DB_DIR.mkdir(exist_ok=True)
_DATABASE_URL = f"sqlite:///{_DB_DIR}/chat_history.db"

_engine = create_engine(
    _DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)


@event.listens_for(_engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

_SessionFactory = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=_engine,
    expire_on_commit=False,  # keep attributes accessible after session.close()
)

Base = declarative_base()


@contextmanager
def warranty_db_session() -> Generator:
    """Context manager: opens a session, commits on exit, rolls back on error."""
    db = _SessionFactory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _now_cst() -> datetime:
    return datetime.now(pytz.timezone("America/Chicago"))


def _now_utc() -> datetime:
    """Return naive UTC for internal retry/lease timestamps."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WarrantyTicket(Base):
    """Top-level warranty case, one per customer intake conversation."""

    __tablename__ = "warranty_tickets"

    id             = Column(Integer, primary_key=True, index=True)
    ticket_id      = Column(String,  unique=True, index=True, nullable=False)
    session_id     = Column(String,  index=True,  nullable=False)
    domain         = Column(String,  index=True,  default="unknown")

    # Flowchart traversal state
    current_node_id = Column(String, nullable=False)
    status = Column(
        String, index=True, default="in_progress",
        # in_progress | awaiting_admin | awaiting_evidence |
        # send_info   | sales_handoff  | resolved | closed
    )

    # High-level fields extracted during traversal
    issue_type     = Column(String, nullable=True)   # installation / delivery / defect
    defect_type    = Column(String, nullable=True)   # air / cosmetic / remote / …
    model_name     = Column(String, nullable=True)   # chair model entered by customer

    # Flexible bag for any key→value collected from question_text nodes
    # (e.g. order_name, tracking_number).  Stored as JSON string.
    collected_data = Column(Text, default="{}")

    # Admin review fields (set by admin via /admin/warranty/{id}/decision)
    admin_decision = Column(String, nullable=True)   # replacement / tech_dispatch / …
    admin_note     = Column(Text,   nullable=True)
    decided_by     = Column(String, nullable=True)
    # Message sent to customer after admin decision (filled by admin, surfaced by chat)
    customer_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=_now_cst)
    updated_at = Column(DateTime, default=_now_cst, onupdate=_now_cst)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_collected(self) -> dict:
        try:
            raw = cast(str, self.collected_data) or "{}"
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}

    def set_collected(self, key: str, value: str) -> None:
        data = self.get_collected()
        data[key] = value
        self.collected_data = json.dumps(data)

    def to_dict(self) -> dict:
        return {
            "ticket_id":       self.ticket_id,
            "session_id":      self.session_id,
            "domain":          self.domain,
            "current_node_id": self.current_node_id,
            "status":          self.status,
            "issue_type":      self.issue_type,
            "defect_type":     self.defect_type,
            "model_name":      self.model_name,
            "collected_data":  self.get_collected(),
            "admin_decision":  self.admin_decision,
            "admin_note":      self.admin_note,
            "decided_by":      self.decided_by,
            "customer_message": self.customer_message,
            "created_at":      cast(datetime, self.created_at).isoformat() if self.created_at is not None else None,
            "updated_at":      cast(datetime, self.updated_at).isoformat() if self.updated_at is not None else None,
        }


class WarrantyChatConsent(Base):
    """Customer acceptance of live-chat privacy / recording notice (per browser session)."""

    __tablename__ = "warranty_chat_consents"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    domain = Column(String, index=True, default="unknown")
    policy_store = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    email_gate_status = Column(String, nullable=True)  # provided | skipped
    accepted_at = Column(DateTime, default=_now_cst)
    updated_at = Column(DateTime, default=_now_cst, onupdate=_now_cst)


class WarrantyTurn(Base):
    """One Q&A exchange within a warranty ticket (node visited + answer given)."""

    __tablename__ = "warranty_turns"

    id              = Column(Integer, primary_key=True, index=True)
    ticket_id       = Column(String,  index=True, nullable=False)
    node_id         = Column(String,  nullable=False)
    node_type       = Column(String,  nullable=True)   # question / instruction / question_text
    node_prompt     = Column(Text,    nullable=True)   # prompt shown to customer
    customer_answer = Column(Text,    nullable=True)   # raw customer input
    answer_key      = Column(String,  nullable=True)   # normalized answer_key matched
    created_at      = Column(DateTime, default=_now_cst)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "ticket_id":       self.ticket_id,
            "node_id":         self.node_id,
            "node_type":       self.node_type,
            "node_prompt":     self.node_prompt,
            "customer_answer": self.customer_answer,
            "answer_key":      self.answer_key,
            "created_at":      cast(datetime, self.created_at).isoformat() if self.created_at is not None else None,
        }


class WarrantyEvidence(Base):
    """An uploaded evidence file (photo, video, receipt) attached to a ticket."""

    __tablename__ = "warranty_evidences"

    id                = Column(Integer, primary_key=True, index=True)
    ticket_id         = Column(String,  index=True, nullable=False)
    evidence_type     = Column(String,  nullable=False)  # damage_photos / video_of_issue / …
    file_path         = Column(String,  nullable=True)   # local disk path
    original_filename = Column(String,  nullable=True)
    mime_type         = Column(String,  nullable=True)
    file_size_bytes   = Column(Integer, default=0)
    emailed           = Column(Integer, default=0)       # 1 = warranty team notified
    customer_email    = Column(String, nullable=True)    # email entered on upload form
    created_at        = Column(DateTime, default=_now_cst)

    def to_dict(self) -> dict:
        return {
            "id":                self.id,
            "ticket_id":         self.ticket_id,
            "evidence_type":     self.evidence_type,
            "file_path":         self.file_path,
            "original_filename": self.original_filename,
            "mime_type":         self.mime_type,
            "file_size_bytes":   self.file_size_bytes,
            "customer_email":    self.customer_email,
            "emailed":           bool(self.emailed),
            "created_at":        cast(datetime, self.created_at).isoformat() if self.created_at is not None else None,
        }

    def to_dict_public(self) -> dict:
        """Browser-safe evidence metadata (no server filesystem paths)."""
        return {
            "id":                self.id,
            "ticket_id":         self.ticket_id,
            "evidence_type":     self.evidence_type,
            "original_filename": self.original_filename,
            "mime_type":         self.mime_type,
            "file_size_bytes":   self.file_size_bytes,
            "emailed":           bool(self.emailed),
            "created_at":        cast(datetime, self.created_at).isoformat() if self.created_at is not None else None,
        }


class RingCentralWebhookEvent(Base):
    """Durable inbox row for an inbound RingCentral callback."""

    __tablename__ = "ringcentral_webhook_events"

    id = Column(Integer, primary_key=True, index=True)
    event_key = Column(String, unique=True, index=True, nullable=False)
    route = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=True)
    payload_json = Column(Text, nullable=False)
    status = Column(String, index=True, default="pending", nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    last_error = Column(String, nullable=True)
    next_attempt_at = Column(DateTime, index=True, nullable=True)
    created_at = Column(DateTime, default=_now_utc)
    updated_at = Column(DateTime, default=_now_utc, onupdate=_now_utc)
    completed_at = Column(DateTime, nullable=True)


class RingCentralCallState(Base):
    """Persistent mirror of the active in-memory IVR call context."""

    __tablename__ = "ringcentral_call_states"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    party_id = Column(String, nullable=False)
    ticket_id = Column(String, index=True, nullable=False)
    caller_phone = Column(String, nullable=True)
    phase = Column(String, nullable=False)
    awaiting_command = Column(String, nullable=True)
    last_audio_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now_utc)
    updated_at = Column(DateTime, default=_now_utc, onupdate=_now_utc)


# ---------------------------------------------------------------------------
# Create tables (idempotent — safe to call multiple times)
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=_engine)


def _migrate_warranty_schema() -> None:
    """Add columns introduced after initial deploy (SQLite ALTER TABLE)."""
    with _engine.connect() as conn:
        raw = conn.connection.driver_connection
        cursor = raw.cursor()
        cursor.execute("PRAGMA table_info(warranty_evidences)")
        cols = {row[1] for row in cursor.fetchall()}
        if "customer_email" not in cols:
            cursor.execute(
                "ALTER TABLE warranty_evidences ADD COLUMN customer_email TEXT"
            )
            raw.commit()

        cursor.execute("PRAGMA table_info(warranty_chat_consents)")
        consent_cols = {row[1] for row in cursor.fetchall()}
        if "contact_email" not in consent_cols:
            cursor.execute(
                "ALTER TABLE warranty_chat_consents ADD COLUMN contact_email TEXT"
            )
            raw.commit()
        if "email_gate_status" not in consent_cols:
            cursor.execute(
                "ALTER TABLE warranty_chat_consents ADD COLUMN email_gate_status TEXT"
            )
            raw.commit()


_migrate_warranty_schema()
