"""Admin health observations for the AI workflow; never makes external calls."""
from __future__ import annotations

import logging
import math
import os
import re
from datetime import timedelta
from typing import Optional

import pytz
from fastapi import APIRouter, Header, Query
from sqlalchemy import Boolean, Column, DateTime, Integer, String

from admin_auth import require_admin_key
import warranty_models as wm

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin-ai-quality"])
_MODEL_RETENTION_DAYS = 30
_MODEL_MAX_OBSERVATIONS = 10000
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")
_MODEL_ERROR_CODES = frozenset({
    "provider_unavailable", "authentication", "permission", "rate_limit",
    "timeout", "connection", "invalid_request", "provider_error",
    "empty_response", "unexpected_error",
})


class AIResponseTiming(wm.Base):
    __tablename__ = "ai_response_timings"
    id = Column(Integer, primary_key=True)
    surface = Column(String, nullable=False, index=True)
    elapsed_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=wm._now_cst, index=True)


class AIModelCallObservation(wm.Base):
    """Model identities and outcomes only, never requests or provider errors."""

    __tablename__ = "ai_model_call_observations"
    id = Column(Integer, primary_key=True)
    role = Column(String(12), nullable=False, index=True)
    requested_model = Column(String(120), nullable=False, index=True)
    provider_model = Column(String(120), nullable=True)
    succeeded = Column(Boolean, nullable=False)
    error_code = Column(String(32), nullable=True)
    release_id = Column(String(80), nullable=False, index=True)
    created_at = Column(DateTime, default=wm._now_cst, nullable=False, index=True)


def _model_id(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.lower().startswith(("sk-", "sess-", "bearer:")):
        return None
    return value if _MODEL_ID.fullmatch(value) else None


def _release_id() -> str:
    from customer_journey import current_release

    return current_release()


def record_model_call(*, role, requested_model, provider_model=None, succeeded, error_code=None) -> None:
    """Best-effort, bounded observations of an actual provider call.

    The completion wrapper calls this after a response or failure. Configuration,
    cached replies and deterministic replies do not create model-call evidence.
    Error categories are allowlisted; exception messages cannot enter this table.
    """
    model = _model_id(requested_model)
    if not isinstance(role, str) or role not in {"agent", "router"} or model is None or not isinstance(succeeded, bool):
        return
    provider = _model_id(provider_model) if succeeded else None
    code = None if succeeded else error_code if isinstance(error_code, str) and error_code in _MODEL_ERROR_CODES else "unexpected_error"
    try:
        release = _release_id()
        with wm.warranty_db_session() as db:
            AIModelCallObservation.__table__.create(bind=db.connection(), checkfirst=True)
            db.add(AIModelCallObservation(role=role, requested_model=model,
                provider_model=provider, succeeded=succeeded, error_code=code,
                release_id=release))
            db.flush()
            db.query(AIModelCallObservation).filter(
                AIModelCallObservation.created_at < wm._now_cst() - timedelta(days=_MODEL_RETENTION_DAYS)
            ).delete(synchronize_session=False)
            oldest_kept = (db.query(AIModelCallObservation.id)
                .order_by(AIModelCallObservation.id.desc()).offset(_MODEL_MAX_OBSERVATIONS - 1).limit(1).scalar())
            if oldest_kept is not None:
                db.query(AIModelCallObservation).filter(
                    AIModelCallObservation.id < oldest_kept
                ).delete(synchronize_session=False)
    except Exception:
        logger.warning("Model-call observation could not be recorded")


def _model_call_dict(row) -> Optional[dict]:
    if row is None:
        return None
    at = row.created_at
    if at is not None and at.tzinfo is None:
        at = pytz.timezone("America/Chicago").localize(at)
    return {"requested_model": row.requested_model, "provider_model": row.provider_model,
        "succeeded": row.succeeded, "error_code": row.error_code,
        "release_id": row.release_id,
        "observed_at": at.isoformat() if at else None}


def _model_configuration(config) -> dict:
    """Expose the resolver's allowlisted provenance, never environment contents."""
    configured = getattr(config, "MODEL_CONFIGURATION", {})
    result = {}
    for role, value in (("agent", config.AGENT_MODEL), ("router", config.ROUTER_MODEL)):
        metadata = configured.get(role, {}) if isinstance(configured, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        source = metadata.get("source")
        result[role] = {"effective_model": _model_id(value),
            "default_model": _model_id(metadata.get("default_model")),
            "source": source if isinstance(source, str) and source in {"env", "default"} else "unknown"}
    return result


def model_call_summary(days: int = 7) -> dict:
    with wm.warranty_db_session() as db:
        AIModelCallObservation.__table__.create(bind=db.connection(), checkfirst=True)
        rows = (db.query(AIModelCallObservation)
            .filter(AIModelCallObservation.created_at >= wm._now_cst() - timedelta(days=days))
            .order_by(AIModelCallObservation.created_at, AIModelCallObservation.id).all())
    roles = {}
    for role in ("agent", "router"):
        calls = [row for row in rows if row.role == role]
        successful = [row for row in calls if row.succeeded]
        roles[role] = {"calls": len(calls), "successful_calls": len(successful),
            "failed_calls": len(calls) - len(successful),
            "latest_call": _model_call_dict(calls[-1]) if calls else None,
            "latest_successful_call": _model_call_dict(successful[-1]) if successful else None}
    return {"days": days, "retention_days": _MODEL_RETENTION_DAYS,
        "max_retained_calls": _MODEL_MAX_OBSERVATIONS, "roles": roles,
        "scope": "Retained model-call attempts, including deployment compatibility probes, separate from current configuration and customer-answer quality. Requested aliases and provider-returned IDs can differ. A past successful call does not verify current access; missing provider identity remains unknown. Cached and deterministic replies are not calls."}


def record_response_timing(surface: str, elapsed_ms: int) -> None:
    """Record no customer content, identifiers, or credentials. Fail open for chat."""
    if surface not in {"sales", "warranty"}:
        return
    try:
        with wm.warranty_db_session() as db:
            AIResponseTiming.__table__.create(bind=db.connection(), checkfirst=True)
            db.add(AIResponseTiming(surface=surface, elapsed_ms=max(0, int(elapsed_ms))))
            # Timing samples are operational data, with bounded retention.
            db.query(AIResponseTiming).filter(
                AIResponseTiming.created_at < wm._now_cst() - timedelta(days=30)
            ).delete(synchronize_session=False)
    except Exception:
        logger.warning("AI response timing could not be recorded")


def timing_summary(days: int = 7) -> dict:
    with wm.warranty_db_session() as db:
        AIResponseTiming.__table__.create(bind=db.connection(), checkfirst=True)
        rows = db.query(AIResponseTiming).filter(
            AIResponseTiming.created_at >= wm._now_cst() - timedelta(days=days)
        ).all()
    result = {}
    for surface in ("warranty", "sales"):
        values = sorted(int(row.elapsed_ms) for row in rows if row.surface == surface)
        result[surface] = {
            "samples": len(values),
            "p50_ms": values[math.ceil(len(values) * .50) - 1] if values else None,
            "p95_ms": values[math.ceil(len(values) * .95) - 1] if values else None,
            "mean_ms": round(sum(values) / len(values), 1) if values else None,
        }
    return result


@router.get("/admin/ai/health")
async def ai_health(
    days: int = Query(7, ge=1, le=30),
    x_admin_key: Optional[str] = Header(default=None),
) -> dict:
    require_admin_key(x_admin_key, os.getenv("ADMIN_API_KEY", ""))
    import config
    from sales_shopify_stock import stock_cache_status
    from warranty_embeddings import embedding_cache_status

    return {
        "models": {"agent": config.AGENT_MODEL, "router": config.ROUTER_MODEL, "embedding": config.EMBEDDING_MODEL},
        "model_configuration": _model_configuration(config),
        "app_release_id": _release_id(),
        "model_calls": model_call_summary(days),
        "latency": timing_summary(days),
        "latency_days": days,
        "latency_scope": "Successful JSON request duration; streamed general-chat completion timing remains in request traces.",
        "stock_cache": stock_cache_status(),
        "warranty_embeddings": embedding_cache_status(),
        "scope": "Local process observations; external integration access is not verified by this endpoint.",
    }
