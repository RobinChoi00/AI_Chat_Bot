"""
chat_feedback.py
================
Persist customer thumbs-up / thumbs-down feedback on assistant messages.

Design
------
- Uses the same SQLite file as ``warranty_models`` (``chat_history.db``) via
  the shared ``Base`` and engine so migrations stay in one place.
- Content is deduplicated per (session_id, content_hash, rating) so repeated
  clicks by the same user overwrite the previous vote / comment instead of
  filling the table.
- No LLM calls here — this is a plain HTTP + SQL endpoint.

Public API
----------
  POST /api/v1/feedback         → { ok: true, feedback_id, rating }
  GET  /admin/feedback          → list rows (admin key required)
  GET  /admin/feedback/summary  → aggregate up/down counts (admin key required)
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from typing import List, Optional

import pytz
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, Text

from warranty_models import Base, _engine, warranty_db_session
from admin_auth import require_admin_key

logger = logging.getLogger(__name__)

_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

_MAX_MESSAGE_LEN = 4000
_MAX_COMMENT_LEN = 1000
_ALLOWED_RATINGS = frozenset({"up", "down"})
_ALLOWED_CONTEXTS = frozenset({"warranty", "chat"})


def _now_cst() -> datetime:
    return datetime.now(pytz.timezone("America/Chicago"))


class ChatFeedback(Base):
    """One row per (session, assistant message, rating) triple."""

    __tablename__ = "chat_feedback"

    id              = Column(Integer, primary_key=True, index=True)
    session_id      = Column(String,  index=True, nullable=False)
    rating          = Column(String,  index=True, nullable=False)  # "up" | "down"
    comment         = Column(Text,    nullable=True)
    message_content = Column(Text,    nullable=False)
    content_hash    = Column(String,  index=True, nullable=False)
    context         = Column(String,  index=True, default="warranty")
    domain          = Column(String,  nullable=True)
    ticket_id       = Column(String,  index=True, nullable=True)
    created_at      = Column(DateTime, default=_now_cst, index=True)
    updated_at      = Column(DateTime, default=_now_cst, onupdate=_now_cst)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "session_id":      self.session_id,
            "rating":          self.rating,
            "comment":         self.comment,
            "message_content": self.message_content,
            "content_hash":    self.content_hash,
            "context":         self.context,
            "domain":          self.domain,
            "ticket_id":       self.ticket_id,
            "created_at":      self.created_at.isoformat() if self.created_at else None,
            "updated_at":      self.updated_at.isoformat() if self.updated_at else None,
        }


Base.metadata.create_all(bind=_engine)


def _hash_content(text: str) -> str:
    """Short stable hash used to dedupe repeated votes on the same message."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _require_admin(x_admin_key: Optional[str]) -> None:
    require_admin_key(x_admin_key, _ADMIN_API_KEY)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=200)
    rating: str
    message_content: str
    comment: Optional[str] = None
    context: str = "warranty"
    domain: Optional[str] = None
    ticket_id: Optional[str] = None


@router.post("/api/v1/feedback")
async def submit_feedback(body: FeedbackRequest):
    """Persist a thumbs-up / thumbs-down vote on an assistant message."""
    rating = (body.rating or "").strip().lower()
    if rating not in _ALLOWED_RATINGS:
        raise HTTPException(status_code=422, detail=f"rating must be one of {sorted(_ALLOWED_RATINGS)}")

    context = (body.context or "warranty").strip().lower()
    if context not in _ALLOWED_CONTEXTS:
        raise HTTPException(status_code=422, detail=f"context must be one of {sorted(_ALLOWED_CONTEXTS)}")

    message_content = (body.message_content or "").strip()
    if not message_content:
        raise HTTPException(status_code=422, detail="message_content must not be empty.")
    if len(message_content) > _MAX_MESSAGE_LEN:
        message_content = message_content[:_MAX_MESSAGE_LEN]

    comment = (body.comment or "").strip() or None
    if comment and len(comment) > _MAX_COMMENT_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"comment is too long (max {_MAX_COMMENT_LEN} characters).",
        )

    content_hash = _hash_content(message_content)

    with warranty_db_session() as db:
        existing = (
            db.query(ChatFeedback)
            .filter(
                ChatFeedback.session_id == body.session_id,
                ChatFeedback.content_hash == content_hash,
            )
            .first()
        )
        if existing is not None:
            existing.rating = rating
            if comment is not None:
                existing.comment = comment
            existing.context = context
            existing.domain = body.domain
            existing.ticket_id = body.ticket_id
            db.flush()
            feedback_id = existing.id
        else:
            row = ChatFeedback(
                session_id=body.session_id,
                rating=rating,
                comment=comment,
                message_content=message_content,
                content_hash=content_hash,
                context=context,
                domain=body.domain,
                ticket_id=body.ticket_id,
            )
            db.add(row)
            db.flush()
            feedback_id = row.id

    logger.info(
        "chat_feedback stored session=%s rating=%s context=%s hash=%s",
        body.session_id,
        rating,
        context,
        content_hash,
    )

    return {
        "ok": True,
        "feedback_id": feedback_id,
        "rating": rating,
    }


@router.get("/admin/feedback")
async def list_feedback(
    rating: Optional[str] = None,
    context: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    x_admin_key: Optional[str] = Header(default=None),
):
    """List feedback rows for the admin dashboard."""
    _require_admin(x_admin_key)
    limit = max(1, min(limit, 500))
    with warranty_db_session() as db:
        q = db.query(ChatFeedback)
        if rating:
            q = q.filter(ChatFeedback.rating == rating.lower())
        if context:
            q = q.filter(ChatFeedback.context == context.lower())
        rows: List[ChatFeedback] = (
            q.order_by(ChatFeedback.created_at.desc()).limit(limit).offset(offset).all()
        )
    return {
        "total": len(rows),
        "offset": offset,
        "rows": [r.to_dict() for r in rows],
    }


@router.get("/admin/feedback/summary")
async def feedback_summary(x_admin_key: Optional[str] = Header(default=None)):
    """Aggregate counts: up / down / total per context."""
    _require_admin(x_admin_key)
    with warranty_db_session() as db:
        rows = db.query(ChatFeedback.context, ChatFeedback.rating).all()

    summary: dict[str, dict[str, int]] = {}
    for context, rating in rows:
        bucket = summary.setdefault(str(context or "unknown"), {"up": 0, "down": 0})
        if rating in bucket:
            bucket[rating] += 1

    for bucket in summary.values():
        bucket["total"] = bucket.get("up", 0) + bucket.get("down", 0)
        if bucket["total"]:
            bucket["up_ratio"] = round(bucket["up"] / bucket["total"], 3)
        else:
            bucket["up_ratio"] = 0.0

    return {"summary": summary}
