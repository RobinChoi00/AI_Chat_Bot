"""Tests for the in-memory semantic warranty index + hybrid search."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_embeddings  # noqa: E402
import warranty_knowledge  # noqa: E402
from warranty_knowledge import KnowledgeEntry, search_knowledge  # noqa: E402


def test_semantic_disabled_when_no_openai_key(monkeypatch):
    monkeypatch.setattr(warranty_embeddings, "_openai_client", lambda: None)
    warranty_embeddings.clear_embedding_cache()
    assert warranty_embeddings.ensure_index() is False
    assert warranty_embeddings.is_available() is False
    assert warranty_embeddings.semantic_search("anything") is None


def test_keyword_only_path_still_works_when_semantic_off(monkeypatch):
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "0")
    warranty_embeddings.clear_embedding_cache()
    results = search_knowledge(
        path_text="power outlet fuse chair will not turn on",
        defect_category="power",
        model_name="OS-4000T",
        limit=3,
    )
    assert isinstance(results, list)
    if results:
        assert all(isinstance(r, KnowledgeEntry) for r in results)


def test_search_knowledge_uses_semantic_when_available(monkeypatch):
    """Stub semantic_search to verify hybrid scoring promotes a low-keyword
    entry when its cosine similarity is high."""
    entries = warranty_knowledge.load_knowledge_entries()
    if not entries:
        return  # nothing to test if the data files are not present

    # Pick a target entry whose keyword score for our nonsense query is 0.
    target = entries[0]
    monkeypatch.setattr(
        warranty_embeddings,
        "semantic_search",
        lambda q, top_k=12, category=None: [(0.95, target)],
    )
    monkeypatch.setattr(warranty_embeddings, "ensure_index", lambda: True)
    monkeypatch.setattr(warranty_embeddings, "is_available", lambda: True)
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "1")

    results = search_knowledge(
        path_text="zzz_no_keyword_overlap_xxx",
        defect_category=None,
        limit=3,
    )
    assert results, "semantic match should surface even with zero keyword overlap"
    assert results[0].title == target.title


def test_search_knowledge_falls_back_when_semantic_raises(monkeypatch):
    """If the semantic layer raises, hybrid search must still degrade to the
    keyword-only path without bubbling the error."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated embedding outage")

    monkeypatch.setattr(warranty_embeddings, "semantic_search", _raise)
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "1")

    # No exception should escape — purely keyword scoring should handle it.
    results = search_knowledge(
        path_text="back switch chair power on fuse remote",
        defect_category="power",
        limit=3,
    )
    assert isinstance(results, list)
