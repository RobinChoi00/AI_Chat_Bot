"""
In-memory semantic index for warranty knowledge entries.

Design contract
---------------
- No disk persistence — built once at first call, kept in module-level cache,
  rebuilt on next process start.
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
import math
import os
import threading
from typing import Optional

from warranty_knowledge import KnowledgeEntry, load_knowledge_entries

logger = logging.getLogger(__name__)

_EMBED_MODEL_DEFAULT = "text-embedding-3-small"
_BATCH_SIZE = 256

_lock = threading.Lock()
_state: dict[str, object] = {
    "ready": False,
    "entries": (),
    "vectors": (),
    "model": "",
    "build_attempted": False,
}


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
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
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


def _build_index_locked() -> bool:
    """Caller MUST hold _lock. Returns True if index is usable after this call."""
    if _state["ready"]:
        return True
    if _state["build_attempted"]:
        return bool(_state["ready"])
    _state["build_attempted"] = True

    client = _openai_client()
    if client is None:
        logger.info("warranty_embeddings: no OPENAI_API_KEY — semantic search disabled.")
        return False

    entries = load_knowledge_entries()
    if not entries:
        logger.info("warranty_embeddings: knowledge base is empty.")
        return False

    texts = [_entry_text(e) for e in entries]
    model = _embed_model()
    vectors: list[list[float]] = []
    try:
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = client.embeddings.create(model=model, input=batch)
            for item in response.data:
                vectors.append(_normalize(list(item.embedding)))
    except Exception as exc:
        logger.warning("warranty_embeddings: embedding build failed: %s", exc)
        return False

    if len(vectors) != len(entries):
        logger.warning(
            "warranty_embeddings: vector count %d != entry count %d, aborting.",
            len(vectors),
            len(entries),
        )
        return False

    _state["entries"] = entries
    _state["vectors"] = tuple(vectors)
    _state["model"] = model
    _state["ready"] = True
    logger.info(
        "warranty_embeddings: built semantic index — %d entries, model=%s.",
        len(entries),
        model,
    )
    return True


def ensure_index() -> bool:
    """Build the index on first call; cheap no-op afterwards."""
    if _state["ready"]:
        return True
    if _state["build_attempted"] and not _state["ready"]:
        return False
    with _lock:
        return _build_index_locked()


def is_available() -> bool:
    """True if a semantic index is built and ready to query."""
    return bool(_state["ready"])


def clear_embedding_cache() -> None:
    """Wipe cached vectors so the next call rebuilds (used after freshdesk sync)."""
    with _lock:
        _state["ready"] = False
        _state["build_attempted"] = False
        _state["entries"] = ()
        _state["vectors"] = ()
        _state["model"] = ""


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
        return _normalize(vec)
    except Exception as exc:
        logger.warning("warranty_embeddings: query embed failed: %s", exc)
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
        return []

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
