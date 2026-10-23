"""Consented journey observations and explicit conversation release provenance.

An observed event is not proof of departure, reading, resolution, or staff action.
Historical conversations are never assigned an invented release. Metadata contains
allowlisted stages only; customer messages, emails, order IDs and URLs are absent.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

import pytz
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, UniqueConstraint, cast, func

from cost_guard import limiter
from warranty_models import Base, WarrantyChatConsent, WarrantyTicket, _engine, _now_cst, warranty_db_session

router = APIRouter(tags=["customer-journey"])
STAGES = ("consent", "email", "model", "issue", "response", "help", "outcome", "evidence", "handoff", "delivery", "resume", "recommendation", "lead", "lead_delivery")
EVENTS = ("shown", "completed", "failed", "resumed", "requested", "skipped")
Stage = Literal["consent", "email", "model", "issue", "response", "help", "outcome", "evidence", "handoff", "delivery", "resume", "recommendation", "lead", "lead_delivery"]
Event = Literal["shown", "completed", "failed", "resumed", "requested", "skipped"]
_RELEASE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")


class ConversationRelease(Base):
    __tablename__ = "conversation_releases"
    __table_args__ = (UniqueConstraint("surface", "conversation_id", name="uq_conversation_release"),)
    id = Column(Integer, primary_key=True)
    surface = Column(String(20), nullable=False, index=True)
    conversation_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    domain = Column(String(100), nullable=False, index=True)
    initial_release = Column(String(80), nullable=False, index=True)
    latest_release = Column(String(80), nullable=False, index=True)
    observed_releases = Column(Text, nullable=False, default="[]")
    mixed = Column(Boolean, nullable=False, default=False, index=True)
    first_seen_at = Column(DateTime, nullable=False, default=_now_cst)
    last_seen_at = Column(DateTime, nullable=False, default=_now_cst)


class CustomerJourneyEvent(Base):
    __tablename__ = "customer_journey_events"
    id = Column(Integer, primary_key=True)
    event_id = Column(String(100), unique=True, nullable=False, index=True)
    surface = Column(String(20), nullable=False, index=True)
    conversation_id = Column(String(128), nullable=True, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    domain = Column(String(100), nullable=False, index=True)
    release_id = Column(String(80), nullable=False, index=True)
    source = Column(String(20), nullable=False)
    stage = Column(String(30), nullable=False, index=True)
    event = Column(String(20), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_now_cst, index=True)


def _ensure_tables(db):
    for table in (ConversationRelease.__table__, CustomerJourneyEvent.__table__):
        table.create(bind=db.connection(), checkfirst=True)


def current_release() -> str:
    value = os.getenv("APP_RELEASE_ID", "development").strip()
    return value if _RELEASE.fullmatch(value) else "unconfigured"


def _iso(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = pytz.timezone("America/Chicago").localize(value)
    return value.isoformat()


def register_conversation(db, *, surface, conversation_id, session_id, domain, is_new=False):
    """Call is_new=True only inside the transaction that creates the real case.

    A pre-existing case first observed here is mixed/legacy, never a new-current
    conversation. A subsequent release also moves a case into the mixed cohort.
    """
    _ensure_tables(db)
    if surface not in {"warranty", "sales"}:
        raise ValueError("Unsupported conversation surface")
    cid, sid, release = str(conversation_id), str(session_id), current_release()
    row = db.query(ConversationRelease).filter_by(surface=surface, conversation_id=cid).first()
    prior_releases = {str(value) for (value,) in db.query(CustomerJourneyEvent.release_id).filter_by(
        surface=surface, session_id=sid, domain=str(domain), conversation_id=None).distinct().all()}
    now = _now_cst()
    if row is None:
        row = ConversationRelease(surface=surface, conversation_id=cid, session_id=sid,
            domain=str(domain), initial_release=release if is_new else "legacy",
            latest_release=release, observed_releases=json.dumps(sorted(prior_releases | {release})),
            mixed=not is_new or bool(prior_releases - {release}), first_seen_at=now, last_seen_at=now)
        db.add(row)
    else:
        seen = json.loads(row.observed_releases or "[]")
        if release not in seen:
            seen.append(release)
        row.observed_releases = json.dumps(sorted(set(seen) | prior_releases))
        row.latest_release = release
        row.last_seen_at = now
        row.mixed = bool(row.mixed or row.initial_release != release or prior_releases - {release})
    db.flush()
    # Consented observations before a case was created can be associated only
    # once a real case on the same session/store exists. Unfinished pre-case
    # observations remain explicitly unlinked, not fabricated consultations.
    db.query(CustomerJourneyEvent).filter_by(surface=surface, session_id=sid,
        domain=str(domain), conversation_id=None).update({"conversation_id": cid}, synchronize_session=False)
    return row


def record_server_event(db, *, surface, session_id, conversation_id=None, domain,
                        stage, event, event_id=None):
    _ensure_tables(db)
    if surface not in {"warranty", "sales"} or stage not in STAGES or event not in EVENTS:
        raise ValueError("Unsupported journey observation")
    eid = str(event_id or uuid4())
    previous = db.query(CustomerJourneyEvent).filter_by(event_id=eid).first()
    if previous is not None:
        return previous
    row = CustomerJourneyEvent(event_id=eid, surface=surface,
        session_id=str(session_id), conversation_id=str(conversation_id) if conversation_id is not None else None,
        domain=str(domain), release_id=current_release(), source="server", stage=stage, event=event)
    db.add(row)
    return row


def _release_query(db, surface, cohort):
    if cohort not in {"all", "current", "legacy", "mixed"} and not _RELEASE.fullmatch(cohort):
        raise HTTPException(422, "Invalid release cohort.")
    q = db.query(ConversationRelease).filter(ConversationRelease.surface == surface)
    if cohort == "mixed":
        return q.filter(ConversationRelease.mixed.is_(True))
    if cohort in {"all", "legacy"}:
        return q
    release = current_release() if cohort == "current" else cohort
    return q.filter(ConversationRelease.initial_release == release, ConversationRelease.mixed.is_(False))


def apply_cohort(db, query, *, surface, id_column, cohort="all"):
    _ensure_tables(db)
    tags = _release_query(db, surface, cohort)
    if cohort == "all":
        return query
    ids = tags.with_entities(ConversationRelease.conversation_id)
    comparison = cast(id_column, String).in_(ids)
    return query.filter(~comparison if cohort == "legacy" else comparison)


def cohort_metadata(db, *, surface, selected="all"):
    _ensure_tables(db)
    _release_query(db, surface, selected)
    rows = db.query(ConversationRelease).filter_by(surface=surface).all()
    releases = {current_release()}
    for row in rows:
        releases.update(json.loads(row.observed_releases or "[]"))
    first = db.query(func.min(CustomerJourneyEvent.created_at)).filter(CustomerJourneyEvent.surface == surface).scalar()
    return {"selected": selected, "current_release": current_release(),
        "available_releases": sorted(releases), "measurement_started_at": _iso(first),
        "legacy_note": "Historical unversioned cases remain unknown. Current means newly created conversations observed only on this release; conversations spanning releases are mixed."}


def journey_summary(db, *, surface, session_ids=None, start=None, end=None, domain=None, cohort="all"):
    _ensure_tables(db)
    tags = _release_query(db, surface, cohort)
    q = db.query(CustomerJourneyEvent).filter(CustomerJourneyEvent.surface == surface)
    if start is not None:
        q = q.filter(CustomerJourneyEvent.created_at >= start)
    if end is not None:
        q = q.filter(CustomerJourneyEvent.created_at <= end)
    if domain:
        q = q.filter(CustomerJourneyEvent.domain.contains(domain))
    if cohort != "all":
        ids = tags.with_entities(ConversationRelease.conversation_id)
        tagged = CustomerJourneyEvent.conversation_id.in_(ids)
        # Unknown pre-case observations cannot be called historical/current.
        q = q.filter(CustomerJourneyEvent.conversation_id.is_not(None))
        q = q.filter(~tagged if cohort == "legacy" else tagged)
    ordered = q.order_by(CustomerJourneyEvent.created_at, CustomerJourneyEvent.id)
    if session_ids is None:
        rows = ordered.all()
    else:
        # SQLite deployments may retain the 999-bind limit. A lifetime cohort
        # must not be truncated or turned into a single unbounded IN clause.
        selected = sorted({str(s) for s in session_ids})
        rows = []
        for offset in range(0, len(selected), 400):
            rows.extend(ordered.filter(CustomerJourneyEvent.session_id.in_(selected[offset:offset + 400])).all())
        rows.sort(key=lambda row: (row.created_at, row.id))
    groups = defaultdict(set)
    for row in rows:
        groups[(str(row.stage), str(row.event))].add(str(row.session_id))
    stages = []
    for stage in STAGES:
        values = {event: len(groups[(stage, event)]) for event in EVENTS}
        if any(values.values()):
            stages.append({"stage": stage, **values, "without_completion": len(
                groups[(stage, "shown")] - groups[(stage, "completed")] - groups[(stage, "skipped")])})
    return {"measured": bool(rows), "measurement_start": "after_recording_consent" if surface == "warranty" else "server_conversation_requests",
        "measurement_started_at": _iso(rows[0].created_at) if rows else None,
        "sessions_observed": len({str(r.session_id) for r in rows}),
        "client_events": sum(r.source == "client" for r in rows),
        "server_events": sum(r.source == "server" for r in rows),
        "unlinked_sessions": len({str(r.session_id) for r in rows if r.conversation_id is None}),
        "stages": stages,
        "note": "Distinct sessions by observed stage; stages are conditional, not a mandatory sequence. Missing completion is not confirmed departure. Client reports are not server-verified outcomes. Pre-consent visits are unmeasured; unlinked pre-case sessions appear only in All cohorts. Event-time window may differ from conversation-creation cohorts."}


class JourneyEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: UUID
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    domain: str = Field(default="osakiusa.com", max_length=100)
    stage: Stage
    event: Event


@router.post("/api/v1/journey/events")
@limiter.limit("120/minute")
async def record_client_event(request: Request, body: JourneyEventRequest):
    from warranty_defaults import normalize_warranty_domain
    with warranty_db_session() as db:
        _ensure_tables(db)
        consent = db.query(WarrantyChatConsent).filter_by(session_id=body.session_id).first()
        if consent is None or consent.accepted_at is None:
            raise HTTPException(403, "Recording consent is required.")
        ticket = db.query(WarrantyTicket).filter_by(session_id=body.session_id).order_by(WarrantyTicket.id.desc()).first()
        domain = normalize_warranty_domain(str(ticket.domain if ticket is not None else consent.domain))
        if normalize_warranty_domain(body.domain) != domain:
            raise HTTPException(422, "Journey store must match the consented session.")
        eid = str(body.event_id)
        previous = db.query(CustomerJourneyEvent).filter_by(event_id=eid).first()
        if previous is not None:
            if previous.surface != "warranty" or previous.session_id != body.session_id or previous.stage != body.stage or previous.event != body.event:
                raise HTTPException(409, "Event identifier already used.")
            return {"recorded": True, "duplicate": True}
        if ticket is not None:
            register_conversation(db, surface='warranty', conversation_id=str(ticket.ticket_id),
                session_id=body.session_id, domain=domain, is_new=False)
        db.add(CustomerJourneyEvent(event_id=eid, surface="warranty", session_id=body.session_id,
            conversation_id=str(ticket.ticket_id) if ticket is not None else None,
            domain=domain, release_id=current_release(), source="client", stage=body.stage, event=body.event))
    return {"recorded": True, "duplicate": False}


for _table in (ConversationRelease.__table__, CustomerJourneyEvent.__table__):
    _table.create(bind=_engine, checkfirst=True)
