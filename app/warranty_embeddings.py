"""
Semantic index for warranty knowledge entries with an optional durable cache.

Design contract
---------------
- JSON vectors are cached under ignored data/embedding_cache by default.
  WARRANTY_EMBEDDING_CACHE_DIR selects another directory; an empty value disables
  persistence. The cache stores no customer queries or source document text.
- Cache identity includes the model and complete ordered knowledge content.
- Startup warming is opt-in with WARRANTY_EMBEDDING_WARMUP=1; it runs in a daemon
  thread. Loading this module never starts an API call.
- Embedding model: text-embedding-3-small (1536d, cheap, OpenAI).
- If the OpenAI client / API key is unavailable, every call degrades to
  return None, letting `warranty_knowledge.search_knowledge()` fall back to
  the legacy keyword-only path with zero behaviour change.
- Batched embedding requests (one API call per ~256 entries).
- Public surface is small on purpose:
    * `semantic_search(query, *, top_k)` → list[(score, KnowledgeEntry)]
    * `clear_embedding_cache()` for tests / freshdesk sync invalidation
    * `is_available()` so callers can short-circuit work
"""

from __future__ import annotations

import logging
import hashlib
import json
import math
import os
import threading
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from warranty_knowledge import KnowledgeEntry, load_knowledge_entries

logger = logging.getLogger(__name__)

_EMBED_MODEL_DEFAULT = "text-embedding-3-small"
_BATCH_SIZE = 256
_CACHE_VERSION = 1
_MAX_CACHE_BYTES = 128 * 1024 * 1024
_MAX_DIMENSIONS = 8192

_lock = threading.Lock()
_state: dict[str, object] = {
    "ready": False,
    "entries": (),
    "vectors": (),
    "model": "",
    "build_attempted": False,
    "fingerprint": "",
    "source": "unavailable",
    "built_at": None,
    "last_error": None,
}
_warmup_thread: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        from config import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT
    except ImportError:
        return None

    return OpenAI(
        api_key=api_key,
        timeout=float(OPENAI_REQUEST_TIMEOUT),
        max_retries=int(OPENAI_MAX_RETRIES),
    )


def _embed_model() -> str:
    try:
        from config import EMBEDDING_MODEL  # type: ignore
        if EMBEDDING_MODEL:
            return EMBEDDING_MODEL
    except Exception:
        pass
    return os.environ.get("OPENAI_EMBEDDING_MODEL", _EMBED_MODEL_DEFAULT)


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------


def _normalize(vec: list[float]) -> list[float]:
    if not isinstance(vec, (list, tuple)) or not 1 <= len(vec) <= _MAX_DIMENSIONS:
        raise ValueError("invalid embedding dimensions")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vec):
        raise ValueError("embedding contains invalid numeric values")
    norm = math.hypot(*vec)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("embedding has invalid magnitude")
    return [v / norm for v in vec]


def _cosine_pre_normalized(a: list[float], b: list[float]) -> float:
    """Dot product, assuming both vectors were L2-normalized at build time."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def _entry_text(entry: KnowledgeEntry) -> str:
    parts = [entry.title or "", entry.diagnostic or "", " ".join(entry.customer_steps or ())]
    return " ".join(p for p in parts if p).strip()


def _fingerprint(entries: tuple[KnowledgeEntry, ...], model: str) -> str:
    payload = {"version": _CACHE_VERSION, "model": model, "entries": [asdict(e) for e in entries]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _cache_directory() -> Optional[Path]:
    raw = os.getenv("WARRANTY_EMBEDDING_CACHE_DIR")
    if raw is not None:
        return Path(raw) if raw.strip() else None
    return Path(__file__).resolve().parent.parent / "data" / "embedding_cache"


def _cache_path(fingerprint: str) -> Optional[Path]:
    directory = _cache_directory()
    return directory / f"warranty-v{_CACHE_VERSION}-{fingerprint}.json" if directory is not None else None


def _vectors_digest(vectors: list) -> str:
    return hashlib.sha256(json.dumps(vectors, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _validated_vectors(raw: object, count: int) -> tuple[list[float], ...]:
    if not isinstance(raw, list) or len(raw) != count or not raw:
        raise ValueError("embedding count mismatch")
    vectors = tuple(_normalize(v) for v in raw)
    dimensions = len(vectors[0])
    if any(len(v) != dimensions for v in vectors):
        raise ValueError("inconsistent embedding dimensions")
    return vectors


def _read_cache(fingerprint: str, model: str, count: int) -> Optional[tuple[tuple[list[float], ...], str]]:
    path = _cache_path(fingerprint)
    if path is None:
        return None
    try:
        if path.stat().st_size > _MAX_CACHE_BYTES:
            raise ValueError("embedding cache too large")
        with path.open("rb") as stream:
            content = stream.read(_MAX_CACHE_BYTES + 1)
        if len(content) > _MAX_CACHE_BYTES:
            raise ValueError("embedding cache too large")
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
            raise ValueError("unsupported embedding cache version")
        if payload.get("fingerprint") != fingerprint or payload.get("model") != model:
            raise ValueError("embedding cache identity mismatch")
        raw = payload.get("vectors")
        vectors = _validated_vectors(raw, count)
        if payload.get("vectors_sha256") != _vectors_digest(raw):
            raise ValueError("embedding cache checksum mismatch")
        if payload.get("dimensions") != len(vectors[0]):
            raise ValueError("embedding cache dimension mismatch")
        created_at = payload.get("created_at")
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            raise ValueError("embedding cache timestamp is not timezone-aware")
        return vectors, created_at
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, OverflowError):
        logger.warning("warranty_embeddings: invalid or unreadable vector cache; rebuilding")
        return None


def _write_cache(fingerprint: str, model: str, vectors: tuple[list[float], ...]) -> None:
    path = _cache_path(fingerprint)
    if path is None:
        return
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = list(vectors)
        payload = {
            "version": _CACHE_VERSION, "fingerprint": fingerprint, "model": model,
            "dimensions": len(vectors[0]), "vectors": raw,
            "created_at": _state["built_at"],
            "vectors_sha256": _vectors_digest(raw),
        }
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # NamedTemporaryFile is private (0600); replacement is atomic.
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError):
        logger.warning("warranty_embeddings: vector cache could not be saved; using memory")
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _build_index_locked(entries: tuple[KnowledgeEntry, ...], model: str, fingerprint: str) -> bool:
    """Caller MUST hold _lock. Returns True if index is usable after this call."""
    if _state["ready"]:
        return True
    if _state["build_attempted"]:
        return bool(_state["ready"])
    _state["build_attempted"] = True

    client = _openai_client()
    if client is None:
        logger.info("warranty_embeddings: no OPENAI_API_KEY — semantic search disabled.")
        _state["last_error"] = "missing_credentials"
        return False

    if not entries:
        logger.info("warranty_embeddings: knowledge base is empty.")
        _state["last_error"] = "empty_knowledge"
        return False

    cached = _read_cache(fingerprint, model, len(entries))
    if cached is not None:
        vectors, created_at = cached
        _state.update(entries=entries, vectors=vectors, model=model, ready=True,
                      source="disk", built_at=created_at, last_error=None)
        return True

    texts = [_entry_text(e) for e in entries]
    vectors: list[list[float]] = []
    try:
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = client.embeddings.create(model=model, input=batch)
            for item in response.data:
                vectors.append(_normalize(list(item.embedding)))
        checked = _validated_vectors(vectors, len(entries))
    except Exception:
        # Exception text from an SDK may contain request details; status is enough.
        logger.warning("warranty_embeddings: embedding build failed; using keyword search")
        _state["last_error"] = "build_failed"
        return False

    _state["entries"] = entries
    _state["vectors"] = checked
    _state["model"] = model
    _state["ready"] = True
    _state["source"] = "api"
    _state["built_at"] = datetime.now(timezone.utc).isoformat()
    _state["last_error"] = None
    _write_cache(fingerprint, model, checked)
    logger.info(
        "warranty_embeddings: built semantic index — %d entries, model=%s.",
        len(entries),
        model,
    )
    return True


def ensure_index() -> bool:
    """Load/build matching knowledge vectors, reusing in-memory content when unchanged."""
    entries = tuple(load_knowledge_entries())
    model = _embed_model()
    if _state["ready"] and _state["entries"] is entries and _state["model"] == model:
        return True
    fingerprint = _fingerprint(entries, model)
    with _lock:
        if fingerprint != _state["fingerprint"]:
            _reset_state()
            _state["fingerprint"] = fingerprint
        return _build_index_locked(entries, model, fingerprint)


def is_available() -> bool:
    """True if a semantic index is built and ready to query."""
    return bool(_state["ready"])


def clear_embedding_cache() -> None:
    """Invalidate memory; unchanged versioned disk vectors remain reusable."""
    with _lock:
        _reset_state()


def _reset_state() -> None:
    _state.update(ready=False, build_attempted=False, entries=(), vectors=(), model="",
                  fingerprint="", source="unavailable", built_at=None, last_error=None)


def embedding_cache_status() -> dict:
    """Non-secret health metadata for an authenticated operations dashboard."""
    with _lock:
        vectors = _state["vectors"]
        return {
            "ready": bool(_state["ready"]), "source": _state["source"],
            "model": _state["model"], "entries": len(_state["entries"]),
            "dimensions": len(vectors[0]) if vectors else 0,
            "built_at": _state["built_at"], "last_error": _state["last_error"],
            "persistence_enabled": _cache_directory() is not None,
        }


def warmup_semantic_index() -> Optional[threading.Thread]:
    """Start one background warmup only when explicitly enabled by configuration."""
    global _warmup_thread
    if os.getenv("WARRANTY_EMBEDDING_WARMUP", "0") != "1":
        return None
    if os.getenv("WARRANTY_SEMANTIC_SEARCH", "1") != "1":
        return None
    with _lock:
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return _warmup_thread
        _warmup_thread = threading.Thread(target=_warm_index, name="warranty-index-warmup", daemon=True)
        _warmup_thread.start()
        return _warmup_thread


def _warm_index() -> None:
    try:
        ensure_index()
    except Exception:
        logger.warning("warranty_embeddings: startup warmup failed; keyword search remains available")
        with _lock:
            _state["last_error"] = "warmup_failed"


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def embed_query(text: str) -> Optional[list[float]]:
    """Embed a single user query into the same vector space."""
    if not text or not text.strip():
        return None
    client = _openai_client()
    if client is None:
        return None
    try:
        response = client.embeddings.create(model=_embed_model(), input=[text])
        vec = list(response.data[0].embedding)
        normalized = _normalize(vec)
        vectors = _state["vectors"]
        if vectors and len(normalized) != len(vectors[0]):
            raise ValueError("query embedding dimensions mismatch")
        return normalized
    except Exception:
        logger.warning("warranty_embeddings: query embed failed; using keyword search")
        return None


def semantic_search(
    query: str,
    *,
    top_k: int = 5,
    category: Optional[str] = None,
) -> Optional[list[tuple[float, KnowledgeEntry]]]:
    """
    Return [(cosine_similarity, entry), ...] sorted desc.

    Returns None if the index isn't usable — callers should then fall back to
    keyword scoring. An empty list means the index is ready but the query
    produced no embedding (e.g. blank query).
    """
    if not ensure_index():
        return None

    qvec = embed_query(query)
    if qvec is None:
        return [] if not query.strip() else None

    entries = _state["entries"]
    vectors = _state["vectors"]
    assert isinstance(entries, tuple) and isinstance(vectors, tuple)

    cat_lower = (category or "").lower() or None
    scored: list[tuple[float, KnowledgeEntry]] = []
    for entry, vec in zip(entries, vectors):
        if cat_lower and entry.category != cat_lower:
            continue
        sim = _cosine_pre_normalized(qvec, vec)
        scored.append((sim, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(top_k, 0)]
