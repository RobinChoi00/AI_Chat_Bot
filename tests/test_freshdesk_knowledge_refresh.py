"""Tests for post-sync knowledge cache invalidation and yield stats."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_embeddings as we  # noqa: E402
import warranty_knowledge as wk  # noqa: E402
from freshdesk_knowledge_refresh import (  # noqa: E402
    build_knowledge_yield_stats,
    invalidate_warranty_knowledge_caches,
)


def test_invalidate_warranty_knowledge_caches_clears_embedding_state(monkeypatch):
    monkeypatch.setattr(wk, "clear_knowledge_cache", lambda: None)
    we._state["ready"] = True
    we._state["build_attempted"] = True

    invalidate_warranty_knowledge_caches()

    assert we._state["ready"] is False
    assert we._state["build_attempted"] is False


def test_build_knowledge_yield_stats(monkeypatch):
    class Entry:
        def __init__(self, source: str):
            self.source = source

    monkeypatch.setattr(
        wk,
        "load_knowledge_entries",
        lambda: [Entry("freshdesk"), Entry("freshdesk"), Entry("freshdesk_kb")],
    )

    stats = build_knowledge_yield_stats(
        synced_ticket_rows=10,
        synced_kb_articles=4,
        resolved_scanned=20,
    )
    assert stats["knowledge_freshdesk_entries"] == 2
    assert stats["knowledge_freshdesk_kb_entries"] == 1
    assert stats["knowledge_yield"]["ticket_rows_to_knowledge_pct"] == 20.0
    assert stats["knowledge_yield"]["resolved_to_knowledge_pct"] == 10.0
    assert stats["knowledge_yield"]["kb_articles_to_knowledge_pct"] == 25.0
