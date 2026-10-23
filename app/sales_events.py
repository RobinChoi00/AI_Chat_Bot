"""Observed storefront actions, separate from recommendations and paid orders.

Client events are useful funnel observations, not proof of a purchase. Only
the authenticated Shopify order integration may establish payment outcomes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable, Literal, Optional
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.exc import IntegrityError

from admin_auth import require_admin_key
from cost_guard import limiter
from sales_models import SalesSession
from store_config import (
    get_storefront_base_url,
    is_supported_storefront_domain,
    normalize_storefront_domain,
)
from warranty_models import Base, _engine, _now_cst, warranty_db_session

router = APIRouter(tags=["sales-events"])
ACTION_TYPES = ("product_click", "add_to_cart", "checkout_started")
EVENT_TYPES = ("tracking_started", *ACTION_TYPES)
_SESSION_ID = re.compile(r"^[A-Za-z0-9:_.-]{1,200}$")
_PRODUCT_PATH = re.compile(r"^/products/([a-z0-9][a-z0-9-]{0,199})/?$")
_DEV_SIGNING_KEY = secrets.token_bytes(32)


class SalesEvent(Base):
    """No names, email, phone, cart contents, IP, or free-text metadata."""

    __tablename__ = "sales_events"

    id = Column(Integer, primary_key=True)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    session_id = Column(String(200), nullable=False, index=True)
    domain = Column(String(200), nullable=False, index=True)
    event_type = Column(String(32), nullable=False, index=True)
    product_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=_now_cst, nullable=False, index=True)


Base.metadata.create_all(bind=_engine)


def _store(domain: str) -> str:
    if not is_supported_storefront_domain(domain):
        return ""
    return normalize_storefront_domain(get_storefront_base_url(domain))


def _signing_key() -> Optional[bytes]:
    for name in ("SALES_ANALYTICS_SIGNING_SECRET", "ADMIN_SESSION_SECRET"):
        value = os.getenv(name, "").strip()
        if len(value) >= 32:
            return value.encode("utf-8")
    if os.getenv("APP_ENV", "development").lower() == "production":
        return None
    # Local development only. Tokens expire when this process restarts.
    return _DEV_SIGNING_KEY


def analytics_available() -> bool:
    return _signing_key() is not None


def issue_event_token(session_id: str, domain: str) -> Optional[str]:
    """Issue a short-lived token for a saved conversation in this store.

    Optional analytics configuration must never prevent a Sales reply. The
    caller may omit the token when this returns None.
    """
    key = _signing_key()
    store = _store(domain)
    if key is None or not store or not _SESSION_ID.fullmatch(session_id):
        return None
    with warranty_db_session() as db:
        row = db.query(SalesSession).filter(SalesSession.session_id == session_id).first()
        if row is None or _store(str(row.domain or "")) != store:
            return None
    payload = {"sid": session_id, "store": store, "exp": int(time.time()) + 86400}
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(key, body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def _valid_token(token: str, session_id: str, store: str) -> bool:
    key = _signing_key()
    if key is None:
        return False
    try:
        body, signature = token.split(".", 1)
        expected = hmac.new(key, body.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        return (
            payload.get("sid") == session_id
            and payload.get("store") == store
            and isinstance(payload.get("exp"), int)
            and time.time() < payload["exp"] <= time.time() + 86400
        )
    except (ValueError, TypeError, UnicodeError, AttributeError):
        return False


class SalesEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    session_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9:_.-]+$")
    domain: str = Field(min_length=1, max_length=200)
    event_token: str = Field(min_length=1, max_length=1200)
    event_type: Literal["tracking_started", "product_click", "add_to_cart", "checkout_started"]
    product_url: Optional[str] = Field(default=None, max_length=1500)


def _product_url(raw: Optional[str], store: str) -> Optional[str]:
    if raw is None:
        return None
    parsed = urlparse(raw)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or _store(parsed.hostname or "") != store
        or not _PRODUCT_PATH.fullmatch(parsed.path)
    ):
        raise HTTPException(status_code=422, detail="Expected a same-store product URL.")
    # No query strings, fragments, campaign values, cart IDs, or customer data.
    return f"https://{store}{parsed.path.rstrip('/')}"


@router.post("/api/v1/sales/events")
@limiter.limit("120/minute")
async def record_sales_event(request: Request, body: SalesEventRequest):
    """Observe a browser action; never accept purchase/payment claims here."""
    store = _store(body.domain)
    if not store:
        raise HTTPException(status_code=422, detail="Unsupported storefront domain.")
    if not analytics_available():
        raise HTTPException(status_code=503, detail="Sales analytics signing is not configured.")
    if not _valid_token(body.event_token, body.session_id, store):
        raise HTTPException(status_code=403, detail="Invalid or expired analytics token.")
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed_origin = urlparse(origin)
            valid_origin = (
                parsed_origin.scheme == "https"
                and not parsed_origin.username
                and not parsed_origin.password
                and parsed_origin.port in (None, 443)
                and parsed_origin.path in ("", "/")
                and _store(parsed_origin.hostname or "") == store
            )
        except ValueError:
            valid_origin = False
        if not valid_origin:
            raise HTTPException(status_code=403, detail="Storefront origin does not match this session.")
    try:
        product_url = _product_url(body.product_url, store)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid product URL.") from None
    if body.event_type == "product_click" and product_url is None:
        raise HTTPException(status_code=422, detail="A product click requires its product URL.")
    if body.event_type == "tracking_started" and product_url is not None:
        raise HTTPException(status_code=422, detail="Tracking readiness does not require a product URL.")

    event_id = str(body.event_id)
    with warranty_db_session() as db:
        session = db.query(SalesSession).filter(SalesSession.session_id == body.session_id).first()
        if session is None:
            raise HTTPException(status_code=404, detail="Sales session not found.")
        if _store(str(session.domain or "")) != store:
            raise HTTPException(status_code=403, detail="Storefront does not match this session.")
        existing = db.query(SalesEvent).filter(SalesEvent.event_id == event_id).first()
        if existing is not None:
            if (
                existing.session_id != body.session_id
                or existing.domain != store
                or existing.event_type != body.event_type
                or existing.product_url != product_url
            ):
                raise HTTPException(status_code=409, detail="Event ID has already been used.")
            return {"ok": True, "recorded": False, "event_id": event_id}
        recent_count = db.query(SalesEvent).filter(
            SalesEvent.session_id == body.session_id,
            SalesEvent.created_at >= _now_cst() - timedelta(hours=1),
        ).count()
        if recent_count >= 300:
            raise HTTPException(status_code=429, detail="Session event limit reached.")
        db.add(SalesEvent(
            event_id=event_id,
            session_id=body.session_id,
            domain=store,
            event_type=body.event_type,
            product_url=product_url,
        ))
        try:
            db.flush()
        except IntegrityError:
            # Concurrent retries cannot create duplicate observations.
            db.rollback()
            raise HTTPException(status_code=409, detail="Event ID has already been used.") from None
    return {"ok": True, "recorded": True, "event_id": event_id}


def aggregate_sales_events(
    session_ids: Iterable[str],
    start: Optional[datetime],
    end: datetime,
    domain: Optional[str] = None,
) -> dict:
    """Observed and ordered funnel stages for the selected consultation cohort.

    A null start includes all stored history up to end. Ordered cart/checkout
    counts require each preceding observed action. We do
    not infer unobserved clicks or cart operations from a later event.
    """
    eligible = set(session_ids)
    counts = Counter({kind: 0 for kind in EVENT_TYPES})
    observed: dict[str, set[str]] = {kind: set() for kind in EVENT_TYPES}
    ordered: dict[str, set[str]] = {kind: set() for kind in ACTION_TYPES}
    if eligible:
        with warranty_db_session() as db:
            query = db.query(SalesEvent).filter(SalesEvent.created_at <= end)
            if start is not None:
                query = query.filter(SalesEvent.created_at >= start)
            if domain:
                store = _store(domain)
                query = query.filter(
                    SalesEvent.domain == store if store else SalesEvent.domain.contains(domain.strip().lower())
                )
            rows = query.order_by(SalesEvent.created_at.asc(), SalesEvent.id.asc()).all()
        for row in rows:
            sid = str(row.session_id)
            kind = str(row.event_type)
            if sid not in eligible or kind not in observed:
                continue
            counts[kind] += 1
            observed[kind].add(sid)
            if kind == "product_click":
                ordered[kind].add(sid)
            elif kind == "add_to_cart" and sid in ordered["product_click"]:
                ordered[kind].add(sid)
            elif kind == "checkout_started" and sid in ordered["add_to_cart"]:
                ordered[kind].add(sid)

    measured = set().union(*observed.values())

    def percent(part: int, whole: int) -> Optional[float]:
        return round(part / whole * 100, 1) if whole else None

    return {
        "event_counts": dict(counts),
        "observed_sessions": {kind: len(values) for kind, values in observed.items()},
        "ordered_sessions": {kind: len(values) for kind, values in ordered.items()},
        "measured_sessions": len(measured),
        "measurement_coverage_pct": percent(len(measured), len(eligible)),
        "rates_pct": {
            "consultation_to_click": percent(len(ordered["product_click"]), len(measured)),
            "click_to_cart": percent(len(ordered["add_to_cart"]), len(ordered["product_click"])),
            "cart_to_checkout": percent(len(ordered["checkout_started"]), len(ordered["add_to_cart"])),
        },
        "denominators": {
            "eligible_sessions": len(eligible),
            "cohort_sessions": len(eligible),
            "measured_sessions": len(measured),
            "clicked_sessions": len(ordered["product_click"]),
            "cart_sessions": len(ordered["add_to_cart"]),
        },
    }


@router.get("/admin/sales/events")
async def list_sales_events(
    session_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    x_admin_key: Optional[str] = Header(default=None),
):
    require_admin_key(x_admin_key)
    with warranty_db_session() as db:
        query = db.query(SalesEvent)
        if session_id:
            query = query.filter(SalesEvent.session_id == session_id)
        total = query.count()
        rows = query.order_by(SalesEvent.created_at.desc(), SalesEvent.id.desc()).limit(limit).offset(offset).all()
    return {
        "total": total,
        "offset": offset,
        "events": [
            {
                "event_id": row.event_id,
                "session_id": row.session_id,
                "domain": row.domain,
                "event_type": row.event_type,
                "product_url": row.product_url,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ],
    }
