"""Tests for post-sync knowledge cache invalidation and yield stats."""

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import freshdesk_knowledge_refresh as refresh  # noqa: E402
import warranty_embeddings as we  # noqa: E402
import warranty_knowledge as wk  # noqa: E402
from freshdesk_knowledge_refresh import (  # noqa: E402
    build_knowledge_yield_stats,
    invalidate_warranty_knowledge_caches,
    rebuild_faiss_sync,
    run_llm_ticket_rescue,
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


def test_run_llm_ticket_rescue_skips_when_disabled(monkeypatch):
    import freshdesk_ticket_summarizer as summarizer

    monkeypatch.setattr(summarizer, "is_enabled", lambda: False)
    result = run_llm_ticket_rescue()
    assert result["enabled"] is False
    assert result.get("skipped") is True


def test_run_llm_ticket_rescue_summarizes(tmp_path, monkeypatch):
    import freshdesk_ticket_summarizer as summarizer

    tickets = tmp_path / "tickets.json"
    tickets.write_text(
        json.dumps(
            [
                {
                    "subject": "Power",
                    "question": "No power",
                    "answer": "We will follow up shortly.",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(summarizer, "is_enabled", lambda: True)
    monkeypatch.setattr(
        summarizer,
        "summarize_missing_tickets",
        lambda raw, **_kwargs: {
            "processed": 1,
            "rescued": 1,
            "skipped": 0,
            "cached": 0,
            "errors": 0,
        },
    )
    monkeypatch.setattr(refresh, "invalidate_warranty_knowledge_caches", lambda: None)
    result = run_llm_ticket_rescue(tickets_path=tickets)
    assert result["ok"] is True
    assert result["rescued"] == 1


def test_rebuild_faiss_sync_skips_when_running(monkeypatch):
    import warranty_faiss_rebuilder as faiss_rebuilder

    monkeypatch.setattr(faiss_rebuilder, "get_status", lambda: {"running": True})
    monkeypatch.setattr(
        faiss_rebuilder,
        "rebuild_freshdesk_qa_index",
        lambda: (_ for _ in ()).throw(AssertionError("should not rebuild")),
    )
    result = rebuild_faiss_sync()
    assert result["ok"] is True
    assert result["reason"] == "already_running"
