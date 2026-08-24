"""
sales_models.py
===============
SQLAlchemy tables backing the Sales AI (Tidio) chat.

Reuses the same SQLite file (``chat_history.db``) and the shared ``Base`` +
engine exported by ``warranty_models``, so migrations happen in one place and
we don't fork DB state between chat surfaces.

Tables
------
  sales_sessions  – one row per customer conversation (Tidio visitor)
  sales_messages  – append-only log of user + assistant turns
  sales_leads     – lead capture for human handoff / follow-up

Design notes
------------
- ``session_id`` is the primary identifier — it stays stable across page
  reloads and is the token Tidio can persist in the visitor's cookie/session.
- ``last_intent`` on the session table is a cache used by the admin dashboard;
  the authoritative per-message intent lives on ``sales_messages.intent``.
- Lead capture is *decoupled* from messages so we can list leads without
  scanning conversation history.
"""

from __future__ import annotations

import json
from typing import Optional, cast

from sqlalchemy import Column, DateTime, Integer, String, Text

from warranty_models import Base, _engine, _now_cst, warranty_db_session


class SalesSession(Base):
    """One row per Tidio (or other channel) sales conversation."""

    __tablename__ = "sales_sessions"

    id              = Column(Integer, primary_key=True, index=True)
    session_id      = Column(String,  unique=True, index=True, nullable=False)
    tidio_visitor_id = Column(String, index=True, nullable=True)
    channel         = Column(String, index=True, default="tidio")  # tidio | web | test
    domain          = Column(String, index=True, default="unknown")

    # Convenience mirrors of the latest classifier / handoff state.
    last_intent     = Column(String, index=True, nullable=True)
    last_message    = Column(Text,   nullable=True)
    status          = Column(String, index=True, default="active")  # active | handoff | closed

    contact_email   = Column(String, nullable=True)
    contact_phone   = Column(String, nullable=True)

    # Anything we want to persist without a schema migration (buttons chosen,
    # last recommendation set, remembered model, etc.).
    collected_data  = Column(Text, default="{}")

    created_at      = Column(DateTime, default=_now_cst, index=True)
    updated_at      = Column(DateTime, default=_now_cst, onupdate=_now_cst)

    def get_collected(self) -> dict:
        try:
            raw = cast(str, self.collected_data) or "{}"
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}

    def set_collected(self, key: str, value) -> None:
        data = self.get_collected()
        data[key] = value
        self.collected_data = json.dumps(data)

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "tidio_visitor_id": self.tidio_visitor_id,
            "channel":          self.channel,
            "domain":           self.domain,
            "last_intent":      self.last_intent,
            "last_message":     self.last_message,
            "status":           self.status,
            "contact_email":    self.contact_email,
            "contact_phone":    self.contact_phone,
            "collected_data":   self.get_collected(),
            "created_at":       self.created_at.isoformat() if self.created_at else None,
            "updated_at":       self.updated_at.isoformat() if self.updated_at else None,
        }


class SalesMessage(Base):
    """Append-only chat log — one row per user or assistant turn."""

    __tablename__ = "sales_messages"

    id           = Column(Integer, primary_key=True, index=True)
    session_id   = Column(String,  index=True, nullable=False)
    role         = Column(String,  nullable=False)  # "user" | "assistant"
    content      = Column(Text,    nullable=False)
    intent       = Column(String,  index=True, nullable=True)
    handoff      = Column(String,  index=True, nullable=True)   # None | intent label
    tools_used   = Column(Text,    nullable=True)               # JSON list
    created_at   = Column(DateTime, default=_now_cst, index=True)

    def to_dict(self) -> dict:
        try:
            tools = json.loads(cast(str, self.tools_used)) if self.tools_used else []
        except (TypeError, ValueError):
            tools = []
        return {
            "id":         self.id,
            "session_id": self.session_id,
            "role":       self.role,
            "content":    self.content,
            "intent":     self.intent,
            "handoff":    self.handoff,
            "tools_used": tools,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SalesLead(Base):
    """Contact info captured for a human handoff or follow-up."""

    __tablename__ = "sales_leads"

    id                = Column(Integer, primary_key=True, index=True)
    session_id        = Column(String, index=True, nullable=False)
    email             = Column(String, index=True, nullable=True)
    phone             = Column(String, nullable=True)
    domain            = Column(String, index=True, default="unknown")
    interest_summary  = Column(Text,   nullable=True)
    reason            = Column(String, index=True, nullable=True)  # discount | eta | human | ...
    forwarded         = Column(String, index=True, default="pending")  # pending | sent | failed
    forwarded_error   = Column(Text,   nullable=True)
    created_at        = Column(DateTime, default=_now_cst, index=True)
    updated_at        = Column(DateTime, default=_now_cst, onupdate=_now_cst)

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "session_id":       self.session_id,
            "email":            self.email,
            "phone":            self.phone,
            "domain":           self.domain,
            "interest_summary": self.interest_summary,
            "reason":           self.reason,
            "forwarded":        self.forwarded,
            "forwarded_error":  self.forwarded_error,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
            "updated_at":       self.updated_at.isoformat() if self.updated_at else None,
        }


Base.metadata.create_all(bind=_engine)


# ---------------------------------------------------------------------------
# Small helpers used by the router / agent
# ---------------------------------------------------------------------------


def get_or_create_session(
    session_id: str,
    *,
    domain: str = "unknown",
    channel: str = "tidio",
    tidio_visitor_id: Optional[str] = None,
) -> SalesSession:
    """Idempotent session upsert."""
    with warranty_db_session() as db:
        session = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == session_id)
            .one_or_none()
        )
        if session is None:
            session = SalesSession(
                session_id=session_id,
                domain=domain,
                channel=channel,
                tidio_visitor_id=tidio_visitor_id,
            )
            db.add(session)
            db.flush()
        else:
            # Update mutable fields if the caller provided them.
            if domain and domain != "unknown":
                session.domain = domain
            if channel and channel != "tidio":
                session.channel = channel
            if tidio_visitor_id and session.tidio_visitor_id != tidio_visitor_id:
                session.tidio_visitor_id = tidio_visitor_id
        db.expunge(session)
        return session


def record_message(
    session_id: str,
    *,
    role: str,
    content: str,
    intent: Optional[str] = None,
    handoff: Optional[str] = None,
    tools_used: Optional[list[str]] = None,
) -> None:
    """Append a message to the sales_messages log."""
    with warranty_db_session() as db:
        row = SalesMessage(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            handoff=handoff,
            tools_used=json.dumps(tools_used or []),
        )
        db.add(row)


def update_session_last_intent(
    session_id: str,
    *,
    intent: str,
    last_message: str,
    status: Optional[str] = None,
) -> None:
    with warranty_db_session() as db:
        row = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == session_id)
            .one_or_none()
        )
        if row is None:
            return
        row.last_intent = intent
        row.last_message = last_message[:2000]
        if status:
            row.status = status


def get_session_collected(session_id: str) -> dict:
    with warranty_db_session() as db:
        row = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == session_id)
            .one_or_none()
        )
        if row is None:
            return {}
        return row.get_collected()


def merge_session_collected(session_id: str, patch: dict) -> dict:
    """Shallow-merge ``patch`` into ``collected_data`` and return the new dict."""
    with warranty_db_session() as db:
        row = (
            db.query(SalesSession)
            .filter(SalesSession.session_id == session_id)
            .one_or_none()
        )
        if row is None:
            return dict(patch or {})
        data = row.get_collected()
        for key, value in (patch or {}).items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                merged = dict(data[key])
                merged.update(value)
                data[key] = merged
            else:
                data[key] = value
        row.collected_data = json.dumps(data)
        return data
