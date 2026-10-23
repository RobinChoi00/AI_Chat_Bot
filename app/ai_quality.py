"""Admin health observations for the AI workflow; never makes external calls."""
from __future__ import annotations

import logging
import math
import os
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Header, Query
from sqlalchemy import Column, DateTime, Integer, String

from admin_auth import require_admin_key
import warranty_models as wm

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin-ai-quality"])


class AIResponseTiming(wm.Base):
    __tablename__ = "ai_response_timings"
    id = Column(Integer, primary_key=True)
    surface = Column(String, nullable=False, index=True)
    elapsed_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=wm._now_cst, index=True)


def record_response_timing(surface: str, elapsed_ms: int) -> None:
    """Record no customer content, identifiers, or credentials. Fail open for chat."""
    if surface not in {"sales", "warranty"}:
        return
    try:
        with wm.warranty_db_session() as db:
            AIResponseTiming.__table__.create(bind=db.get_bind(), checkfirst=True)
            db.add(AIResponseTiming(surface=surface, elapsed_ms=max(0, int(elapsed_ms))))
            # Timing samples are operational data, with bounded retention.
            db.query(AIResponseTiming).filter(
                AIResponseTiming.created_at < wm._now_cst() - timedelta(days=30)
            ).delete(synchronize_session=False)
    except Exception:
        logger.warning("AI response timing could not be recorded")


def timing_summary(days: int = 7) -> dict:
    with wm.warranty_db_session() as db:
        AIResponseTiming.__table__.create(bind=db.get_bind(), checkfirst=True)
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
    from config import AGENT_MODEL, EMBEDDING_MODEL, ROUTER_MODEL
    from sales_shopify_stock import stock_cache_status
    from warranty_embeddings import embedding_cache_status

    return {
        "models": {"agent": AGENT_MODEL, "router": ROUTER_MODEL, "embedding": EMBEDDING_MODEL},
        "latency": timing_summary(days),
        "latency_days": days,
        "latency_scope": "Successful JSON request duration; streamed general-chat completion timing remains in request traces.",
        "stock_cache": stock_cache_status(),
        "warranty_embeddings": embedding_cache_status(),
        "scope": "Local process observations; external integration access is not verified by this endpoint.",
    }
