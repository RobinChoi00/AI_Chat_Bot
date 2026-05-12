"""
Cost & reliability guardrails for the AI chat endpoint.

This module centralizes the cross-cutting concerns that protect both the
*wallet* and the *uptime* of the service:

1. **OpenAI usage logging** — every chat completion's token usage is
   captured (input, output, cached input) along with an estimated USD
   cost. Persisted to the `openai_usage_logs` SQLite table so you can
   roll up daily/weekly cost from real data instead of guesses.

2. **Response cache** — short-TTL in-memory LRU cache that lets identical
   FAQ-style questions skip the full agent loop. Cache key is a
   normalised tuple of (query, domain, history-tail). Queries containing
   PII signals (email, order id) are NEVER cached, since their answers
   are per-customer.

3. **Rate limiting** — slowapi-based per-IP throttling. Applied as a
   decorator on the chat endpoint.

The module is intentionally framework-agnostic where possible so it can
be unit-tested without spinning up FastAPI.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from cachetools import TTLCache
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

try:
    from config import (
        CHAT_CACHE_ENABLED,
        CHAT_CACHE_MAX_ENTRIES,
        CHAT_CACHE_TTL_SECONDS,
        MODEL_PRICING_USD_PER_1M,
        RATE_LIMIT_PER_HOUR,
        RATE_LIMIT_PER_MINUTE,
    )
except ImportError:  # pragma: no cover — config import path inside Docker
    from ..config import (  # type: ignore
        CHAT_CACHE_ENABLED,
        CHAT_CACHE_MAX_ENTRIES,
        CHAT_CACHE_TTL_SECONDS,
        MODEL_PRICING_USD_PER_1M,
        RATE_LIMIT_PER_HOUR,
        RATE_LIMIT_PER_MINUTE,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Rate limiting
# =============================================================================
#
# A single shared Limiter instance — slowapi tracks state per-IP in memory.
# In a multi-replica deployment you'd want to back this with Redis via
# `storage_uri="redis://..."` but for a single-pod setup in-memory is fine.

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_HOUR],
)


# =============================================================================
# 2. OpenAI usage logging
# =============================================================================

def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate USD cost from token counts using the pricing table.

    `cached_tokens` is the subset of `prompt_tokens` that hit OpenAI's
    automatic prompt cache (billed at half price). We bill the non-cached
    portion at the full input rate.
    """
    pricing = MODEL_PRICING_USD_PER_1M.get(model)
    if not pricing:
        return 0.0
    fresh_input = max(prompt_tokens - cached_tokens, 0)
    return (
        fresh_input * pricing["input"]
        + cached_tokens * pricing.get("cached_input", pricing["input"])
        + completion_tokens * pricing["output"]
    ) / 1_000_000


def extract_usage(response: Any) -> Tuple[int, int, int]:
    """Pull (prompt, cached, completion) token counts out of an OpenAI response.

    Returns zeros for any field that's missing — defensive because the SDK
    has shipped breaking shape changes a few times.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = 0
    # The cached-token count lives in `prompt_tokens_details.cached_tokens`
    # on newer SDK versions. Tolerate the old shape too.
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    else:
        cached = int(getattr(usage, "cached_tokens", 0) or 0)
    return prompt, cached, completion


class UsageRecorder:
    """Accumulates usage across the tool-call loop for one request.

    Each `record(response)` call adds the response's usage to the running
    totals. At the end we hand the totals to `flush()` to persist a single
    aggregated row.
    """

    __slots__ = (
        "model", "prompt_tokens", "cached_tokens", "completion_tokens",
        "call_count", "_started_at",
    )

    def __init__(self, model: str):
        self.model = model
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        self._started_at = time.time()

    def record(self, response: Any) -> None:
        p, c, o = extract_usage(response)
        self.prompt_tokens += p
        self.cached_tokens += c
        self.completion_tokens += o
        self.call_count += 1

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_cost_usd(
            self.model, self.prompt_tokens, self.cached_tokens, self.completion_tokens
        )

    @property
    def elapsed_ms(self) -> int:
        return int((time.time() - self._started_at) * 1000)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "elapsed_ms": self.elapsed_ms,
        }


# =============================================================================
# 3. Response cache
# =============================================================================

# In-memory LRU + TTL. cachetools is already in requirements.txt so no new dep.
_response_cache: "TTLCache[str, str]" = TTLCache(
    maxsize=CHAT_CACHE_MAX_ENTRIES, ttl=CHAT_CACHE_TTL_SECONDS
)
_cache_lock = threading.Lock()

_EMAIL_RE = re.compile(r"[\w\.\-+]+@[\w\.\-]+\.\w+")
_ORDER_ID_RE = re.compile(r"\b(OSKMC|OSKUS|TIDM|OSK|TI)\d{3,7}\b", re.IGNORECASE)
_HASH_ORDER_RE = re.compile(r"#\d{4,}")


def _query_has_pii(query: str) -> bool:
    """A query that mentions an email or order id is per-customer — never cache."""
    if not query:
        return False
    if _EMAIL_RE.search(query):
        return True
    if _ORDER_ID_RE.search(query):
        return True
    if _HASH_ORDER_RE.search(query):
        return True
    return False


def _normalise_query(query: str) -> str:
    """Cheap normalisation: lowercase, collapse whitespace, strip punctuation noise."""
    q = (query or "").lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[?!.,]+$", "", q)
    return q


def make_cache_key(
    query: str, target_domain: str, history: Optional[Iterable[Any]] = None
) -> Optional[str]:
    """Build a cache key, or return None if this query must NOT be cached.

    History is collapsed to the previous user message's first ~120 chars
    so multi-turn follow-ups don't collide with brand-new questions but
    also don't fragment the cache per session.
    """
    if not CHAT_CACHE_ENABLED:
        return None
    if not query or _query_has_pii(query):
        return None
    norm = _normalise_query(query)
    if len(norm) < 3:
        return None  # too short → probably "hi" etc., let LLM handle freshly

    prev_user = ""
    if history:
        for m in reversed(list(history)):
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
            if role == "user":
                prev_user = _normalise_query(content)[:120]
                break

    return f"v1|{target_domain}|{prev_user}|{norm}"


def cache_get(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    with _cache_lock:
        return _response_cache.get(key)


def cache_set(key: Optional[str], value: str) -> None:
    if not key or not value:
        return
    with _cache_lock:
        _response_cache[key] = value


def cache_stats() -> Dict[str, Any]:
    """Lightweight introspection for an admin/debug endpoint."""
    with _cache_lock:
        return {
            "enabled": CHAT_CACHE_ENABLED,
            "size": len(_response_cache),
            "maxsize": _response_cache.maxsize,
            "ttl_seconds": _response_cache.ttl,
        }


# =============================================================================
# 4. FAISS Reader/Writer lock
# =============================================================================
#
# Reads can run concurrently; writes (webhook-driven index updates) must run
# exclusively. Python doesn't ship a built-in RWLock so we implement a
# minimal one — fine for our low write rate.

class RWLock:
    """Writer-preference reader/writer lock.

    - `read()`: many readers at once, blocks if a writer is waiting/active.
    - `write()`: exclusive; waits for all current readers to drain.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readers_ok = threading.Condition(self._lock)
        self._writers_ok = threading.Condition(self._lock)
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    def _acquire_read(self) -> None:
        with self._lock:
            while self._writer_active or self._writers_waiting > 0:
                self._readers_ok.wait()
            self._readers += 1

    def _release_read(self) -> None:
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._writers_ok.notify()

    def _acquire_write(self) -> None:
        with self._lock:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._writers_ok.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1

    def _release_write(self) -> None:
        with self._lock:
            self._writer_active = False
            # Prefer waking another writer if any are queued; else wake readers.
            if self._writers_waiting > 0:
                self._writers_ok.notify()
            else:
                self._readers_ok.notify_all()

    # Context-manager helpers ------------------------------------------------
    def read(self) -> "_RWContext":
        return _RWContext(self._acquire_read, self._release_read)

    def write(self) -> "_RWContext":
        return _RWContext(self._acquire_write, self._release_write)


class _RWContext:
    def __init__(self, acquire: Callable[[], None], release: Callable[[], None]) -> None:
        self._acquire = acquire
        self._release = release

    def __enter__(self) -> "_RWContext":
        self._acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release()


# Global FAISS lock — main.py imports this.
faiss_rwlock = RWLock()


# =============================================================================
# 5. FastAPI helper — typed wrapper around the limiter for the chat endpoint
# =============================================================================

def get_request_ip(request: Request) -> str:
    """Convenience for logging / debugging — slowapi uses this internally."""
    return get_remote_address(request)
