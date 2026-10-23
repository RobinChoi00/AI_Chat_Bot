"""Customer-confirmed outcome history and human-reviewed troubleshooting knowledge.

Outcome observations are append-only and never approve, reopen, or change a
warranty claim. Raw conversations are not automatically turned into knowledge.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.exc import IntegrityError

from admin_auth import require_admin_key
from cost_guard import limiter
from warranty_models import Base, WarrantyTicket, _engine, _now_cst, warranty_db_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["warranty-quality"])


class WarrantyOutcomeObservation(Base):
    __tablename__ = "warranty_outcome_observations"
    id = Column(Integer, primary_key=True)
    event_id = Column(String(100), unique=True, index=True, nullable=False)
    ticket_id = Column(String, index=True, nullable=False)
    outcome = Column(String(30), nullable=False)
    source = Column(String(30), nullable=False)
    observed_at = Column(DateTime, default=_now_cst, nullable=False, index=True)


class WarrantyKnowledgeReview(Base):
    __tablename__ = "warranty_knowledge_reviews"
    id = Column(Integer, primary_key=True)
    status = Column(String(20), default="pending", nullable=False, index=True)
    title = Column(String(160), nullable=False)
    category = Column(String(30), nullable=False)
    model_names_json = Column(Text, nullable=False)
    diagnostic = Column(Text, nullable=False)
    steps_json = Column(Text, nullable=False)
    source_url = Column(String(1000), nullable=False)
    review_until = Column(String(10), nullable=False)
    submitted_at = Column(DateTime, default=_now_cst, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String(100), nullable=True)


Base.metadata.create_all(bind=_engine)


def _ensure_quality_tables(db) -> None:
    # Standalone router tests and maintenance sessions may bind a fresh DB after
    # this module was imported. Create only these additive tables on that bind.
    for table in (WarrantyOutcomeObservation.__table__, WarrantyKnowledgeReview.__table__):
        table.create(bind=db.get_bind(), checkfirst=True)


def issue_outcome_token(ticket_id: str, session_id: str, ttl_secs: int = 30 * 86400) -> Optional[str]:
    """Mint a purpose-bound bearer token; missing signing configuration is optional."""
    from warranty_resume import _b64u_encode, _get_secret
    try:
        secret = _get_secret()
    except RuntimeError:
        return None
    payload = {"tid": ticket_id, "sid": session_id, "purpose": "warranty_followup", "exp": int(time.time()) + ttl_secs}
    body = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode())
    signature = _b64u_encode(hmac.new(secret, body.encode(), hashlib.sha256).digest())
    return f"{body}.{signature}"


# Compatibility alias for callers describing the customer-facing operation.
issue_followup_token = issue_outcome_token


def _verify_outcome_token(token: str) -> Optional[dict]:
    from warranty_resume import _b64u_decode, _b64u_encode, _get_secret
    if len(token) > 4096:
        return None
    try:
        body, signature = token.split(".", 1)
        secret = _get_secret()
        expected = _b64u_encode(hmac.new(secret, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_b64u_decode(body))
        if not isinstance(payload, dict) or payload.get("purpose") != "warranty_followup":
            return None
        if not isinstance(payload.get("tid"), str) or not isinstance(payload.get("sid"), str):
            return None
        if int(payload.get("exp", 0)) <= int(time.time()):
            return None
        return payload
    except (RuntimeError, ValueError, TypeError, UnicodeError):
        return None


def record_outcome_observation(ticket_id: str, outcome: str, event_id: Optional[str] = None, source: str = "self_service") -> Optional[int]:
    """Append an observation, deduplicating retries; ignores progress-only events."""
    normalized = {"resolved": "fixed", "unresolved": "still_broken", "unable_to_attempt": "still_broken"}.get(outcome, outcome)
    if normalized not in {"fixed", "still_broken"}:
        return None
    if source not in {"self_service", "customer_followup"}:
        raise ValueError("Unsupported outcome observation source")
    # Existing endpoint retries do not create a new observation for an unchanged
    # initial result. Follow-ups use a client event ID, allowing real later changes.
    event_id = event_id or f"initial:{ticket_id}:{normalized}"
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        previous = db.query(WarrantyOutcomeObservation).filter_by(event_id=event_id).first()
        if previous:
            if previous.ticket_id != ticket_id or previous.outcome != normalized:
                raise HTTPException(status_code=409, detail="Event ID already used for another observation.")
            return int(previous.id)
        if db.query(WarrantyTicket).filter_by(ticket_id=ticket_id).first() is None:
            raise HTTPException(status_code=404, detail="Warranty ticket not found.")
        row = WarrantyOutcomeObservation(event_id=event_id, ticket_id=ticket_id, outcome=normalized, source=source)
        db.add(row)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            previous = db.query(WarrantyOutcomeObservation).filter_by(event_id=event_id).first()
            if previous and previous.ticket_id == ticket_id and previous.outcome == normalized:
                return int(previous.id)
            raise HTTPException(status_code=409, detail="Event ID already used for another observation.")
        return int(row.id)


class FollowUpRequest(BaseModel):
    token: str = Field(min_length=20, max_length=4096)
    outcome: Literal["fixed", "still_broken"]
    event_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")


@router.post("/api/v1/warranty/{ticket_id}/follow-up")
@limiter.limit("20/minute")
async def record_customer_follow_up(request: Request, ticket_id: str, body: FollowUpRequest):
    payload = _verify_outcome_token(body.token)
    if not payload or payload["tid"] != ticket_id:
        raise HTTPException(status_code=401, detail="Invalid or expired follow-up access.")
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        ticket = db.query(WarrantyTicket).filter_by(ticket_id=ticket_id, session_id=payload["sid"]).first()
        if ticket is None:
            raise HTTPException(status_code=401, detail="Invalid or expired follow-up access.")
        if str(ticket.status or "in_progress") == "in_progress":
            raise HTTPException(status_code=409, detail="Follow-up is available after diagnosis or case closure.")
    observation_id = record_outcome_observation(ticket_id, body.outcome, body.event_id, source="customer_followup")
    return {"ok": True, "outcome": body.outcome, "observation_id": observation_id}


def quality_metrics(*, days: Optional[int] = 30, domain: Optional[str] = None, all_time: bool = False) -> dict:
    """Use latest customer report, keeping unknown results and reopenings distinct."""
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        query = db.query(WarrantyTicket)
        if days and not all_time:
            query = query.filter(WarrantyTicket.created_at >= (_now_cst() - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0))
        if domain:
            query = query.filter(WarrantyTicket.domain.contains(domain))
        tickets = query.all()
        observations = (db.query(WarrantyOutcomeObservation)
                        .join(WarrantyTicket, WarrantyTicket.ticket_id == WarrantyOutcomeObservation.ticket_id)
                        .filter(WarrantyTicket.id.in_(query.with_entities(WarrantyTicket.id)))
                        .order_by(WarrantyOutcomeObservation.observed_at, WarrantyOutcomeObservation.id).all())
    by_ticket: dict[str, list] = {}
    for row in observations:
        by_ticket.setdefault(str(row.ticket_id), []).append(row)
    fixed = still_broken = reopened = followed_up = 0
    for ticket in tickets:
        collected = ticket.get_collected()
        initial = str(collected.get("troubleshooting_outcome") or "")
        baseline = "fixed" if initial == "resolved" or str(ticket.admin_decision or "") == "self_resolved" else (
            "still_broken" if initial in {"unresolved", "unable_to_attempt"} else None
        )
        history = by_ticket.get(str(ticket.ticket_id), [])
        latest = str(history[-1].outcome) if history else baseline
        if latest == "fixed":
            fixed += 1
        elif latest == "still_broken":
            still_broken += 1
        if any(row.source == "customer_followup" for row in history):
            followed_up += 1
        ever_fixed = baseline == "fixed" and not any(row.source == "self_service" for row in history)
        for row in history:
            if ever_fixed and row.outcome == "still_broken":
                reopened += 1
                break
            ever_fixed = ever_fixed or row.outcome == "fixed"
    total = len(tickets)
    known = fixed + still_broken
    return {
        "total_consultations": total,
        "latest_confirmed_fixed": fixed,
        "latest_still_broken": still_broken,
        "outcome_unknown": total - known,
        "followed_up": followed_up,
        "reopened_after_fix": reopened,
        "confirmed_fix_rate_all_pct": round(fixed / total * 100, 1) if total else 0.0,
        "confirmed_fix_rate_reported_pct": round(fixed / known * 100, 1) if known else 0.0,
        "scope": {"days": None if all_time else days, "all_time": all_time, "domain": domain, "basis": "ticket_created_at"},
    }


@router.get("/admin/warranty/quality")
async def get_quality_metrics(days: int = Query(30, ge=1, le=180), all_time: bool = Query(False), domain: Optional[str] = None,
                              x_admin_key: Optional[str] = Header(default=None)):
    require_admin_key(x_admin_key)
    return quality_metrics(days=days, domain=domain, all_time=all_time)


class KnowledgeDraftRequest(BaseModel):
    title: str = Field(min_length=5, max_length=160)
    category: Literal["power", "remote", "air", "mech", "footrest", "heat", "misc", "voice", "general"]
    model_names: list[str] = Field(min_length=1, max_length=20)
    diagnostic: str = Field(min_length=10, max_length=1000)
    customer_steps: list[str] = Field(min_length=1, max_length=8)
    source_url: str = Field(max_length=1000)
    review_until: date

    @field_validator("title", "diagnostic")
    @classmethod
    def text_has_no_customer_contact(cls, value):
        from warranty_knowledge import _EMAIL_RE, _PHONE_RE
        if _EMAIL_RE.search(value) or _PHONE_RE.search(value):
            raise ValueError("Remove customer contact details from reviewed knowledge.")
        return value.strip()

    @field_validator("source_url")
    @classmethod
    def source_is_nonsecret_https(cls, value):
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an HTTPS source URL without credentials, query, or fragment.")
        return value

    @field_validator("model_names")
    @classmethod
    def model_names_are_explicit(cls, values):
        values = [value.strip() for value in values]
        if any(len(value) < 3 or len(value) > 100 or value.lower() in {"all", "any", "unknown", "*"} for value in values):
            raise ValueError("Explicit applicable model names are required.")
        return list(dict.fromkeys(values))

    @field_validator("customer_steps")
    @classmethod
    def steps_are_customer_safe(cls, values):
        from warranty_knowledge import _is_customer_safe_step
        values = [value.strip() for value in values]
        internal_access = re.compile(r"\b(?:pcb|circuit board|live wires?|wiring|power supply|solder\w*|disassembl\w*|back panel|internal panel|inside the chair|technician)\b", re.I)
        if any(not _is_customer_safe_step(value) or internal_access.search(value) for value in values):
            raise ValueError("Each step must pass the existing customer-safe troubleshooting filter.")
        return values

    @field_validator("review_until")
    @classmethod
    def expiry_is_current(cls, value):
        if value < _now_cst().date():
            raise ValueError("The review expiry cannot be in the past.")
        return value


class KnowledgeReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewed_by: str = Field(min_length=2, max_length=100)


def _review_dict(row):
    return {"id": row.id, "status": row.status, "title": row.title, "category": row.category,
            "model_names": json.loads(row.model_names_json), "diagnostic": row.diagnostic,
            "customer_steps": json.loads(row.steps_json), "source_url": row.source_url,
            "review_until": row.review_until, "reviewed_by": row.reviewed_by,
            "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None}


@router.post("/admin/warranty/knowledge-reviews", status_code=201)
async def submit_knowledge_draft(body: KnowledgeDraftRequest, x_admin_key: Optional[str] = Header(default=None)):
    require_admin_key(x_admin_key)
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        row = WarrantyKnowledgeReview(title=body.title, category=body.category, model_names_json=json.dumps(body.model_names),
                                     diagnostic=body.diagnostic, steps_json=json.dumps(body.customer_steps),
                                     source_url=body.source_url, review_until=body.review_until.isoformat(), status="pending")
        db.add(row)
        db.flush()
        return _review_dict(row)


@router.get("/admin/warranty/knowledge-reviews")
async def list_knowledge_reviews(status: Optional[Literal["pending", "approved", "rejected"]] = None,
                                limit: int = Query(100, ge=1, le=200), x_admin_key: Optional[str] = Header(default=None)):
    require_admin_key(x_admin_key)
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        query = db.query(WarrantyKnowledgeReview)
        if status:
            query = query.filter_by(status=status)
        return {"reviews": [_review_dict(row) for row in query.order_by(WarrantyKnowledgeReview.id.desc()).limit(limit).all()]}


@router.post("/admin/warranty/knowledge-reviews/{review_id}/decision")
async def decide_knowledge_review(review_id: int, body: KnowledgeReviewDecision, x_admin_key: Optional[str] = Header(default=None)):
    require_admin_key(x_admin_key)
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        row = db.query(WarrantyKnowledgeReview).filter_by(id=review_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Knowledge draft not found.")
        if body.decision == "approved" and row.review_until < _now_cst().date().isoformat():
            raise HTTPException(status_code=409, detail="Knowledge review expired; submit an updated draft.")
        row.status = body.decision
        row.reviewed_by = body.reviewed_by.strip()
        row.reviewed_at = _now_cst()
        result = _review_dict(row)
    from warranty_knowledge import clear_knowledge_cache
    clear_knowledge_cache()
    try:
        from warranty_embeddings import clear_embedding_cache
        clear_embedding_cache()
    except ImportError:
        pass
    return result


def approved_knowledge_entries():
    """Only current explicitly approved records may enter the search corpus."""
    from warranty_knowledge import KnowledgeEntry
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        rows = db.query(WarrantyKnowledgeReview).filter(
            WarrantyKnowledgeReview.status == "approved",
            WarrantyKnowledgeReview.review_until >= _now_cst().date().isoformat(),
        ).order_by(WarrantyKnowledgeReview.id).all()
    return tuple(KnowledgeEntry(
        source="reviewed_knowledge", category=row.category, title=row.title, diagnostic=row.diagnostic,
        customer_steps=tuple(json.loads(row.steps_json)), applicable_models=tuple(json.loads(row.model_names_json)),
        source_url=row.source_url, reviewed_at=row.reviewed_at.isoformat() if row.reviewed_at else None,
        review_until=row.review_until,
    ) for row in rows)
