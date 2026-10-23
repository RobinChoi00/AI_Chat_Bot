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
from collections import Counter
from sqlalchemy import Column, DateTime, Integer, String, Text, or_
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
    normalized = {"resolved": "fixed", "unresolved": "still_broken"}.get(outcome, outcome)
    if normalized not in {"fixed", "still_broken", "unable_to_attempt", "deferred", "delivery_answered", "unknown"}:
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
        if ticket_purpose(ticket) == "delivery":
            raise HTTPException(status_code=409, detail="Delivery feedback is separate from a chair repair outcome.")
        if not is_workflow_terminal(ticket) and not ticket.get_collected().get("troubleshooting_outcome") and str(ticket.admin_decision or "") != "self_resolved":
            raise HTTPException(status_code=409, detail="A repair outcome is available after diagnosis, not after an early handoff.")
    observation_id = record_outcome_observation(ticket_id, body.outcome, body.event_id, source="customer_followup")
    return {"ok": True, "outcome": body.outcome, "observation_id": observation_id}


def ticket_purpose(ticket) -> str:
    issue = str(getattr(ticket, "issue_type", "") or "").lower()
    node = str(getattr(ticket, "current_node_id", "") or "")
    collected = ticket.get_collected()
    if issue == "delivery" or node.startswith("delivery_") or collected.get("human_support_department") == "sales":
        return "delivery"
    if issue in {"defect", "installation"} or node.startswith(("defect_", "install_")) or collected.get("human_support_reason") == "safety_hazard":
        return "repair"
    if issue in {"sales", "order_cancel"} or node == "sales_routing":
        return "other"
    return "unknown"


def is_workflow_terminal(ticket) -> bool:
    """A queued handoff at a question is not a completed diagnostic workflow."""
    from warranty_workflow import _NODES

    node = _NODES.get(str(getattr(ticket, "current_node_id", "") or "")) or {}
    return node.get("type") == "terminal"


def _rate(part: int, total: int):
    return round(part / total * 100, 1) if total else None


def summarize_ticket_outcomes(tickets, observations) -> dict:
    """One current result per repair case; no result is inferred from closure."""
    by_ticket: dict[str, list] = {}
    for row in observations:
        by_ticket.setdefault(str(row.ticket_id), []).append(row)
    purposes = Counter({name: 0 for name in ("repair", "delivery", "other", "unknown")})
    outcomes = Counter({name: 0 for name in ("fixed", "not_fixed", "unable_to_attempt", "deferred", "unknown")})
    delivery = Counter({name: 0 for name in ("consultations", "latest_lookup_available", "latest_lookup_unavailable", "lookup_not_attempted", "answered", "needs_review", "unable_to_attempt", "deferred", "unknown")})
    reasons, departments = Counter(), Counter()
    followed_up = reopened = diagnosed = 0
    by_id = {}
    for ticket in tickets:
        collected = ticket.get_collected()
        purpose = ticket_purpose(ticket)
        purposes[purpose] += 1
        history = by_ticket.get(str(ticket.ticket_id), [])
        initial = str(collected.get("troubleshooting_outcome") or "")
        baseline = {"resolved": "fixed", "unresolved": "not_fixed", "unable_to_attempt": "unable_to_attempt", "deferred": "deferred"}.get(initial)
        if baseline is None and str(ticket.admin_decision or "") == "self_resolved":
            baseline = "fixed"
        normalized_history = []
        for row in history:
            value = {"still_broken": "not_fixed"}.get(str(row.outcome), str(row.outcome))
            # Older code stored inability as still_broken; its matching original
            # self-service observation can be disambiguated from the saved input.
            if initial == "unable_to_attempt" and row.source == "self_service" and str(row.event_id) == f"initial:{ticket.ticket_id}:still_broken":
                value = "unable_to_attempt"
            normalized_history.append(value)
        latest = normalized_history[-1] if normalized_history else baseline or "unknown"
        if latest not in outcomes:
            latest = "unknown"
        by_id[str(ticket.ticket_id)] = {"purpose": purpose, "outcome": latest, "diagnosis_completed": purpose == "repair" and is_workflow_terminal(ticket)}
        if collected.get("human_support_requested"):
            reasons[str(collected.get("human_support_reason") or "customer_request")] += 1
            department = str(collected.get("human_support_department") or ("sales" if purpose == "delivery" else "warranty"))
            departments[department] += 1
        if purpose == "delivery":
            delivery["consultations"] += 1
            snapshot = collected.get("tracking_snapshot")
            try:
                snapshot = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
            except (TypeError, ValueError):
                snapshot = None
            if isinstance(snapshot, dict) and isinstance(snapshot.get("available"), bool):
                delivery["latest_lookup_available" if snapshot["available"] else "latest_lookup_unavailable"] += 1
            else:
                delivery["lookup_not_attempted"] += 1
            service = str(collected.get("delivery_service_outcome") or "")
            service_key = "answered" if service == "delivery_answered" else service if service in {"needs_review", "unable_to_attempt", "deferred"} else "deferred" if initial == "deferred" else "unknown"
            delivery[service_key] += 1
            continue
        if purpose != "repair":
            continue
        outcomes[latest] += 1
        diagnosed += is_workflow_terminal(ticket)
        followed_up += any(row.source == "customer_followup" for row in history)
        ever_fixed = baseline == "fixed" and not any(row.source == "self_service" for row in history)
        for value in normalized_history:
            if ever_fixed and value == "not_fixed":
                reopened += 1
                break
            ever_fixed = ever_fixed or value == "fixed"
    reported = outcomes["fixed"] + outcomes["not_fixed"]
    return {
        "purpose_counts": dict(purposes),
        "repair": {"consultations": purposes["repair"], "diagnosis_completed": diagnosed,
            "diagnosis_completion_rate_pct": _rate(diagnosed, purposes["repair"]), "latest_outcomes": dict(outcomes),
            "result_reported": reported, "confirmed_fix_rate_all_pct": _rate(outcomes["fixed"], purposes["repair"]),
            "confirmed_fix_rate_reported_pct": _rate(outcomes["fixed"], reported),
            "followed_up": followed_up, "reopened_after_fix": reopened},
        "delivery": dict(delivery),
        "handoffs": {"requested": sum(reasons.values()), "by_reason": [{"reason": k, "count": v} for k, v in sorted(reasons.items())],
            "by_department": [{"department": k, "count": v} for k, v in sorted(departments.items())]},
        "_by_ticket": by_id,
    }


def quality_metrics(*, days: Optional[int] = 30, domain: Optional[str] = None, all_time: bool = False, cohort: str = "all") -> dict:
    """Latest repair results are separate from delivery service and unclassified cases."""
    with warranty_db_session() as db:
        _ensure_quality_tables(db)
        query = db.query(WarrantyTicket)
        now = _now_cst()
        query = query.filter(or_(WarrantyTicket.created_at <= now, WarrantyTicket.created_at.is_(None)))
        if days and not all_time:
            query = query.filter(WarrantyTicket.created_at >= (_now_cst() - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0))
        if domain:
            query = query.filter(WarrantyTicket.domain.contains(domain))
        from customer_journey import apply_cohort, cohort_metadata
        query = apply_cohort(db, query, surface="warranty", id_column=WarrantyTicket.ticket_id, cohort=cohort)
        tickets = query.all()
        observations = (db.query(WarrantyOutcomeObservation)
                        .join(WarrantyTicket, WarrantyTicket.ticket_id == WarrantyOutcomeObservation.ticket_id)
                        .filter(WarrantyTicket.id.in_(query.with_entities(WarrantyTicket.id)))
                        .filter(WarrantyOutcomeObservation.observed_at <= now)
                        .order_by(WarrantyOutcomeObservation.observed_at, WarrantyOutcomeObservation.id).all())
        cohort_info = cohort_metadata(db, surface="warranty", selected=cohort)
    summary = summarize_ticket_outcomes(tickets, observations)
    summary.pop("_by_ticket")
    repair = summary["repair"]
    fixed, still_broken = repair["latest_outcomes"]["fixed"], repair["latest_outcomes"]["not_fixed"]
    total = len(tickets)
    known = fixed + still_broken
    return {
        "total_consultations": total,
        "latest_confirmed_fixed": fixed,
        "latest_still_broken": still_broken,
        "outcome_unknown": total - known,
        "followed_up": repair["followed_up"],
        "reopened_after_fix": repair["reopened_after_fix"],
        "confirmed_fix_rate_all_pct": round(fixed / total * 100, 1) if total else 0.0,
        "confirmed_fix_rate_reported_pct": round(fixed / known * 100, 1) if known else 0.0,
        **summary,
        "cohort": cohort_info,
        "cohort_metadata": cohort_info,
        "scope": {"days": None if all_time else days, "all_time": all_time, "domain": domain, "basis": "ticket_created_at"},
    }


@router.get("/admin/warranty/quality")
async def get_quality_metrics(days: int = Query(30, ge=1, le=180), all_time: bool = Query(False), domain: Optional[str] = None, cohort: str = Query("all", max_length=100),
                              x_admin_key: Optional[str] = Header(default=None)):
    require_admin_key(x_admin_key)
    return quality_metrics(days=days, domain=domain, all_time=all_time, cohort=cohort)


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
