"""
request_trace.py
================
Turn-level tracing for the chat agent loop.

Every ``/api/v1/chat`` request gets one ``RequestTrace`` (keyed by
``request_id``). Each agent loop iteration appends a ``TurnEvent``.
After the response is finalised the trace is persisted to SQLite so
admin/ops can reconstruct *exactly* what happened for any session.

Design contract
---------------
- ZERO coupling to OpenAI SDK types — callers pass plain values.
- All timestamps are America/Chicago (matches the rest of the codebase).
- The trace table lives in the same ``db_data/chat_history.db`` alongside
  chat_logs / openai_usage_logs.  Uses the same engine as main.py's ORM.
- ``persist()`` is safe to call from a BackgroundTask.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory structures (built during the request, persisted at the end)
# ---------------------------------------------------------------------------

@dataclass
class TurnEvent:
    """One iteration of the agent tool-call loop."""

    turn: int
    tool_name: Optional[str] = None
    tool_args_summary: Optional[str] = None
    tool_success: bool = True
    tool_result_len: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cached_tokens: int = 0
    llm_latency_ms: int = 0
    guard_triggered: bool = False
    guard_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "tool_name": self.tool_name,
            "tool_args_summary": self.tool_args_summary,
            "tool_success": self.tool_success,
            "tool_result_len": self.tool_result_len,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_cached_tokens": self.llm_cached_tokens,
            "llm_latency_ms": self.llm_latency_ms,
            "guard_triggered": self.guard_triggered,
            "guard_action": self.guard_action,
        }


@dataclass
class RequestTrace:
    """Accumulates trace data for one ``/api/v1/chat`` request."""

    request_id: str
    session_id: str
    domain: str
    user_query: str
    model: str = ""

    # Pre-agent outcomes
    scope_blocked: bool = False
    cache_hit: bool = False
    short_circuit: Optional[str] = None  # "welcome", "showroom", etc.
    forced_first_tool: Optional[str] = None
    warranty_mode: bool = False

    # Turn-level detail
    turns: List[TurnEvent] = field(default_factory=list)

    # Post-agent
    guard_sanitized: bool = False
    final_response_len: int = 0
    total_latency_ms: int = 0

    _started_at: float = field(default_factory=time.time, repr=False)

    def start_clock(self) -> None:
        self._started_at = time.time()

    def finish(self, response_text: str, guard_changed: bool = False) -> None:
        self.final_response_len = len(response_text)
        self.guard_sanitized = guard_changed
        self.total_latency_ms = int((time.time() - self._started_at) * 1000)

    def add_turn(self, event: TurnEvent) -> None:
        self.turns.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "domain": self.domain,
            "user_query": self.user_query[:500],
            "model": self.model,
            "scope_blocked": self.scope_blocked,
            "cache_hit": self.cache_hit,
            "short_circuit": self.short_circuit,
            "forced_first_tool": self.forced_first_tool,
            "warranty_mode": self.warranty_mode,
            "guard_sanitized": self.guard_sanitized,
            "final_response_len": self.final_response_len,
            "total_latency_ms": self.total_latency_ms,
            "turn_count": len(self.turns),
            "tools_called": [t.tool_name for t in self.turns if t.tool_name],
            "turns": [t.to_dict() for t in self.turns],
        }


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS request_traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    domain      TEXT    DEFAULT 'unknown',
    model       TEXT    DEFAULT '',
    user_query  TEXT    DEFAULT '',
    scope_blocked    INTEGER DEFAULT 0,
    cache_hit        INTEGER DEFAULT 0,
    short_circuit    TEXT,
    forced_first_tool TEXT,
    warranty_mode    INTEGER DEFAULT 0,
    guard_sanitized  INTEGER DEFAULT 0,
    turn_count       INTEGER DEFAULT 0,
    tools_called     TEXT    DEFAULT '[]',
    total_latency_ms INTEGER DEFAULT 0,
    final_response_len INTEGER DEFAULT 0,
    turns_json       TEXT    DEFAULT '[]',
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT_SQL = """
INSERT INTO request_traces (
    request_id, session_id, domain, model, user_query,
    scope_blocked, cache_hit, short_circuit, forced_first_tool,
    warranty_mode, guard_sanitized, turn_count, tools_called,
    total_latency_ms, final_response_len, turns_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_table_ensured = False


def persist_trace(trace: RequestTrace, engine: Any) -> None:
    """Write a completed trace to SQLite.  BackgroundTask-safe."""
    global _table_ensured  # noqa: PLW0603
    try:
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            if not _table_ensured:
                cur.execute(_TABLE_SQL)
                raw.commit()
                _table_ensured = True

            d = trace.to_dict()
            cur.execute(_INSERT_SQL, (
                d["request_id"],
                d["session_id"],
                d["domain"],
                d["model"],
                d["user_query"],
                int(d["scope_blocked"]),
                int(d["cache_hit"]),
                d["short_circuit"],
                d["forced_first_tool"],
                int(d["warranty_mode"]),
                int(d["guard_sanitized"]),
                d["turn_count"],
                json.dumps(d["tools_called"]),
                d["total_latency_ms"],
                d["final_response_len"],
                json.dumps(d["turns"]),
            ))
            raw.commit()
        finally:
            raw.close()
    except Exception:
        logger.exception("Failed to persist request trace")
