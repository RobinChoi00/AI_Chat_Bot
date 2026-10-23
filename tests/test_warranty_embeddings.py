"""Tests for the in-memory semantic warranty index + hybrid search."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_embeddings  # noqa: E402
import warranty_knowledge  # noqa: E402
from warranty_knowledge import KnowledgeEntry, search_knowledge  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("WARRANTY_EMBEDDING_CACHE_DIR", str(tmp_path / "vectors"))
    warranty_embeddings.clear_embedding_cache()
    yield
    warranty_embeddings.clear_embedding_cache()


@pytest.fixture
def fake_embeddings(monkeypatch):
    entries = (
        KnowledgeEntry("reviewed", "power", "Chair will not start", "Check external power", ("Check the outlet",)),
        KnowledgeEntry("reviewed", "air", "Air pressure", "Check hose", ("Inspect the visible hose",)),
    )
    calls = []

    def create(*, model, input):
        calls.append((model, input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(index + 1), 2.0, 3.0]) for index, _ in enumerate(input)])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr(warranty_embeddings, "_openai_client", lambda: client)
    monkeypatch.setattr(warranty_embeddings, "_embed_model", lambda: "test-embedding-model")
    monkeypatch.setattr(warranty_embeddings, "load_knowledge_entries", lambda: entries)
    return entries, client, calls


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


def test_disk_cache_reuses_vectors_without_another_embedding_request(fake_embeddings, tmp_path):
    entries, _, calls = fake_embeddings
    assert warranty_embeddings.ensure_index()
    assert len(calls) == 1
    paths = list((tmp_path / "vectors").glob("*.json"))
    assert len(paths) == 1
    assert paths[0].stat().st_mode & 0o777 == 0o600
    raw = paths[0].read_text()
    assert "Chair will not start" not in raw
    assert "Check the outlet" not in raw

    warranty_embeddings.clear_embedding_cache()
    assert warranty_embeddings.ensure_index()
    assert len(calls) == 1
    assert warranty_embeddings.embedding_cache_status()["source"] == "disk"
    assert warranty_embeddings.embedding_cache_status()["entries"] == len(entries)


def test_changed_model_or_knowledge_invalidates_memory_and_disk(fake_embeddings, monkeypatch):
    entries, _, calls = fake_embeddings
    assert warranty_embeddings.ensure_index()
    monkeypatch.setattr(warranty_embeddings, "_embed_model", lambda: "new-test-model")
    assert warranty_embeddings.ensure_index()
    assert len(calls) == 2

    # Metadata affects applicability, even if the text sent for embedding is identical.
    changed = (KnowledgeEntry("new-reviewed-source", entries[0].category, entries[0].title,
                              entries[0].diagnostic, entries[0].customer_steps), entries[1])
    monkeypatch.setattr(warranty_embeddings, "load_knowledge_entries", lambda: changed)
    assert warranty_embeddings.ensure_index()
    assert len(calls) == 3


@pytest.mark.parametrize("damage", ["invalid_json", "nan", "zero", "dimension", "count", "checksum", "identity"])
def test_corrupt_cached_vectors_rebuild_instead_of_entering_search(fake_embeddings, tmp_path, damage):
    _, _, calls = fake_embeddings
    assert warranty_embeddings.ensure_index()
    path = next((tmp_path / "vectors").glob("*.json"))
    body = json.loads(path.read_text())
    if damage == "invalid_json":
        path.write_text("not valid JSON")
    else:
        if damage == "nan":
            body["vectors"][0][0] = float("nan")
        elif damage == "zero":
            body["vectors"][0] = [0, 0, 0]
        elif damage == "dimension":
            body["vectors"][1] = [1, 2]
        elif damage == "count":
            body["vectors"].pop()
        elif damage == "checksum":
            body["vectors"][0][0] += 1
        else:
            body["fingerprint"] = "wrong identity"
        path.write_text(json.dumps(body))

    warranty_embeddings.clear_embedding_cache()
    assert warranty_embeddings.ensure_index()
    assert len(calls) == 2
    assert warranty_embeddings.embedding_cache_status()["source"] == "api"


def test_cached_vectors_do_not_enable_search_without_query_credentials(fake_embeddings, monkeypatch):
    assert warranty_embeddings.ensure_index()
    warranty_embeddings.clear_embedding_cache()
    monkeypatch.setattr(warranty_embeddings, "_openai_client", lambda: None)
    assert warranty_embeddings.ensure_index() is False
    assert warranty_embeddings.semantic_search("power") is None


@pytest.mark.parametrize("bad_vector", [[float("inf"), 1, 2], [0, 0, 0], [True, 1, 2], [1, 2]])
def test_malformed_query_embeddings_fall_back(fake_embeddings, bad_vector):
    _, client, _ = fake_embeddings
    assert warranty_embeddings.ensure_index()
    client.embeddings.create = lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=bad_vector)])
    assert warranty_embeddings.semantic_search("power") is None


def test_persistence_can_be_disabled(fake_embeddings, monkeypatch, tmp_path):
    _, _, calls = fake_embeddings
    monkeypatch.setenv("WARRANTY_EMBEDDING_CACHE_DIR", "")
    assert warranty_embeddings.ensure_index()
    warranty_embeddings.clear_embedding_cache()
    assert warranty_embeddings.ensure_index()
    assert len(calls) == 2
    assert not list(tmp_path.rglob("*.json"))


def test_warmup_is_opt_in_and_runs_off_request_thread(fake_embeddings, monkeypatch):
    _, _, calls = fake_embeddings
    monkeypatch.delenv("WARRANTY_EMBEDDING_WARMUP", raising=False)
    assert warranty_embeddings.warmup_semantic_index() is None
    assert calls == []
    monkeypatch.setenv("WARRANTY_EMBEDDING_WARMUP", "1")
    monkeypatch.setenv("WARRANTY_SEMANTIC_SEARCH", "1")
    thread = warranty_embeddings.warmup_semantic_index()
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert warranty_embeddings.is_available()
    assert len(calls) == 1
