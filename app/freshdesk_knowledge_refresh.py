"""
Post-sync cache invalidation and knowledge yield metrics.

After Freshdesk ticket/KB JSON is refreshed, warranty search must reload
keyword caches, semantic embedding indexes, and (for general chat) the
``freshdesk_qa`` FAISS slice.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def invalidate_warranty_knowledge_caches() -> None:
    """Clear in-memory keyword + semantic caches so the next query reloads JSON."""
    from warranty_knowledge import clear_knowledge_cache  # noqa: WPS433

    clear_knowledge_cache()
    try:
        from warranty_embeddings import clear_embedding_cache  # noqa: WPS433

        clear_embedding_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clear warranty embedding cache: %s", exc)


def _pct(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 1)


def build_knowledge_yield_stats(
    *,
    synced_ticket_rows: int = 0,
    synced_kb_articles: int = 0,
    resolved_scanned: int = 0,
) -> dict[str, Any]:
    """Count loaded knowledge entries and compute simple extraction yields."""
    from warranty_knowledge import load_knowledge_entries  # noqa: WPS433

    entries = load_knowledge_entries()
    freshdesk_entries = sum(1 for entry in entries if entry.source == "freshdesk")
    freshdesk_kb_entries = sum(
        1 for entry in entries if entry.source == "freshdesk_kb"
    )

    yield_stats = {
        "ticket_rows_to_knowledge_pct": _pct(
            freshdesk_entries, int(synced_ticket_rows or 0)
        ),
        "resolved_to_knowledge_pct": _pct(
            freshdesk_entries, int(resolved_scanned or 0)
        )
        if resolved_scanned
        else None,
        "kb_articles_to_knowledge_pct": _pct(
            freshdesk_kb_entries, int(synced_kb_articles or 0)
        ),
    }

    return {
        "knowledge_freshdesk_entries": freshdesk_entries,
        "knowledge_freshdesk_kb_entries": freshdesk_kb_entries,
        "knowledge_total_entries": len(entries),
        "knowledge_yield": yield_stats,
    }


def schedule_faiss_rebuild(
    background_tasks,
    *,
    enabled: bool,
    sync_ok: bool,
) -> dict[str, Any]:
    """Schedule ``freshdesk_qa`` FAISS rebuild when requested and sync succeeded."""
    if not enabled or not sync_ok:
        return {}

    try:
        from warranty_faiss_rebuilder import (  # noqa: WPS433
            get_status as faiss_status,
            rebuild_freshdesk_qa_index,
        )
    except ImportError:
        from app.warranty_faiss_rebuilder import (  # type: ignore  # noqa: WPS433
            get_status as faiss_status,
            rebuild_freshdesk_qa_index,
        )

    if faiss_status().get("running"):
        return {
            "faiss_rebuild_scheduled": False,
            "faiss_rebuild_reason": "already_running",
        }

    background_tasks.add_task(rebuild_freshdesk_qa_index)
    return {"faiss_rebuild_scheduled": True}


def log_ticket_sync_yield(
    *,
    ok: bool,
    ticket_count: int,
    resolved_scanned: int,
    stats: dict[str, Any],
) -> None:
    y = stats.get("knowledge_yield") or {}
    logger.info(
        "Freshdesk sync — ok=%s ticket_rows=%s resolved_scanned=%s "
        "knowledge_freshdesk=%s yield_ticket_rows=%s%% yield_resolved=%s",
        ok,
        ticket_count,
        resolved_scanned,
        stats.get("knowledge_freshdesk_entries"),
        y.get("ticket_rows_to_knowledge_pct"),
        y.get("resolved_to_knowledge_pct"),
    )


def log_kb_sync_yield(*, ok: bool, article_count: int, stats: dict[str, Any]) -> None:
    y = stats.get("knowledge_yield") or {}
    logger.info(
        "Freshdesk KB sync — ok=%s articles=%s knowledge_kb=%s yield_kb=%s%%",
        ok,
        article_count,
        stats.get("knowledge_freshdesk_kb_entries"),
        y.get("kb_articles_to_knowledge_pct"),
    )
