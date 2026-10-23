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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _model_id(value: Any) -> Optional[str]:
    """Accept bounded identifiers, never arbitrary provider/customer text."""
    return value if isinstance(value, str) and _MODEL_ID.fullmatch(value) else None


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
    # Controlled source identities only; never raw retrieval or customer IDs.
    source_evidence: List[Dict[str, Any]] = field(default_factory=list)
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
            "source_evidence": self.source_evidence,
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
    # Configuration is not proof that a provider executed a model. These fields
    # remain empty for cache hits, deterministic replies, and historical traces.
    requested_models: List[str] = field(default_factory=list)
    provider_models: List[str] = field(default_factory=list)
    model_calls_successful: int = 0
    model_calls_failed: int = 0
    model_calls_unverified: int = 0

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
    final_response: str = ""
    final_response_len: int = 0
    total_latency_ms: int = 0

    _started_at: float = field(default_factory=time.time, repr=False)

    def start_clock(self) -> None:
        self._started_at = time.time()

    def finish(self, response_text: str, guard_changed: bool = False) -> None:
        text = response_text or ""
        self.final_response = text[:2000]
        self.final_response_len = len(text)
        self.guard_sanitized = guard_changed
        self.total_latency_ms = int((time.time() - self._started_at) * 1000)

    def add_turn(self, event: TurnEvent) -> None:
        self.turns.append(event)

    def record_model_call(
        self, requested_model: str, provider_model: Optional[str] = None,
        succeeded: bool = True,
    ) -> None:
        """Record actual call evidence independently from the configured model."""
        requested = _model_id(requested_model)
        if requested and requested not in self.requested_models and len(self.requested_models) < 16:
            self.requested_models.append(requested)
        if not succeeded:
            self.model_calls_failed += 1
            return
        self.model_calls_successful += 1
        provider = _model_id(provider_model)
        if not requested or not provider:
            self.model_calls_unverified += 1
        if provider and provider not in self.provider_models and len(self.provider_models) < 16:
            self.provider_models.append(provider)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "domain": self.domain,
            "user_query": self.user_query[:500],
            "model": self.model,
            "requested_models": list(self.requested_models),
            "provider_models": list(self.provider_models),
            "model_calls_successful": self.model_calls_successful,
            "model_calls_failed": self.model_calls_failed,
            "model_calls_unverified": self.model_calls_unverified,
            "scope_blocked": self.scope_blocked,
            "cache_hit": self.cache_hit,
            "short_circuit": self.short_circuit,
            "forced_first_tool": self.forced_first_tool,
            "warranty_mode": self.warranty_mode,
            "guard_sanitized": self.guard_sanitized,
            "final_response": self.final_response,
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
    requested_models TEXT DEFAULT '[]',
    provider_models TEXT DEFAULT '[]',
    model_calls_successful INTEGER DEFAULT 0,
    model_calls_failed INTEGER DEFAULT 0,
    model_calls_unverified INTEGER DEFAULT 0,
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
    final_response   TEXT    DEFAULT '',
    turns_json       TEXT    DEFAULT '[]',
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT_SQL = """
INSERT INTO request_traces (
    request_id, session_id, domain, model, user_query,
    scope_blocked, cache_hit, short_circuit, forced_first_tool,
    warranty_mode, guard_sanitized, turn_count, tools_called,
    total_latency_ms, final_response_len, final_response, turns_json,
    requested_models, provider_models, model_calls_successful,
    model_calls_failed, model_calls_unverified
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_table_ensured = False


def _ensure_trace_schema(cur: Any) -> None:
    """Create the trace table and add columns introduced after the first deploy."""
    global _table_ensured  # noqa: PLW0603
    if _table_ensured:
        return
    cur.execute(_TABLE_SQL)
    existing = {row[1] for row in cur.execute("PRAGMA table_info(request_traces)")}
    if "final_response" not in existing:
        cur.execute("ALTER TABLE request_traces ADD COLUMN final_response TEXT DEFAULT ''")
    for name, definition in (
        ("requested_models", "TEXT DEFAULT '[]'"),
        ("provider_models", "TEXT DEFAULT '[]'"),
        ("model_calls_successful", "INTEGER DEFAULT 0"),
        ("model_calls_failed", "INTEGER DEFAULT 0"),
        ("model_calls_unverified", "INTEGER DEFAULT 0"),
    ):
        if name not in existing:
            cur.execute(f"ALTER TABLE request_traces ADD COLUMN {name} {definition}")
    _table_ensured = True


def persist_trace(trace: RequestTrace, engine: Any) -> None:
    """Write a completed trace to SQLite.  BackgroundTask-safe."""
    global _table_ensured  # noqa: PLW0603
    try:
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            _ensure_trace_schema(cur)
            raw.commit()

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
                d["final_response"],
                json.dumps(d["turns"]),
                json.dumps(d["requested_models"]),
                json.dumps(d["provider_models"]),
                d["model_calls_successful"],
                d["model_calls_failed"],
                d["model_calls_unverified"],
            ))
            raw.commit()
        finally:
            raw.close()
    except Exception:
        logger.exception("Failed to persist request trace")


def _rows(engine: Any, sql: str, params: tuple) -> List[tuple]:
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        _ensure_trace_schema(cur)
        raw.commit()
        cur.execute(sql, params)
        return list(cur.fetchall())
    finally:
        raw.close()


def list_traces(engine: Any, *, limit: int = 50, issues_only: bool = False) -> List[Dict[str, Any]]:
    """Recent chat traces for the admin page. Empty when the table is new."""
    limit = max(1, min(int(limit), 200))
    where = "WHERE guard_sanitized = 1 OR scope_blocked = 1" if issues_only else ""
    sql = f"""
        SELECT request_id, session_id, domain, user_query, short_circuit,
               scope_blocked, cache_hit, guard_sanitized, turn_count,
               tools_called, total_latency_ms, final_response_len, created_at
        FROM request_traces
        {where}
        ORDER BY id DESC
        LIMIT ?
    """
    try:
        found = _rows(engine, sql, (limit,))
    except Exception:
        logger.exception("Failed to list request traces")
        return []
    rows = []
    for row in found:
        tools = row[9] or "[]"
        try:
            parsed_tools = json.loads(tools)
        except json.JSONDecodeError:
            parsed_tools = []
        rows.append({
            "request_id": row[0],
            "session_id": row[1],
            "domain": row[2],
            "user_query": row[3] or "",
            "short_circuit": row[4],
            "scope_blocked": bool(row[5]),
            "cache_hit": bool(row[6]),
            "guard_sanitized": bool(row[7]),
            "turn_count": row[8] or 0,
            "tools_called": parsed_tools,
            "total_latency_ms": row[10] or 0,
            "final_response_len": row[11] or 0,
            "created_at": row[12],
        })
    return rows


def get_trace(engine: Any, request_id: str) -> Optional[Dict[str, Any]]:
    """One trace, including the answer text and each tool turn."""
    sql = """
        SELECT request_id, session_id, domain, model, user_query,
               scope_blocked, cache_hit, short_circuit, forced_first_tool,
               warranty_mode, guard_sanitized, turn_count, tools_called,
               total_latency_ms, final_response_len, final_response,
               turns_json, created_at, requested_models, provider_models,
               model_calls_successful, model_calls_failed, model_calls_unverified
        FROM request_traces
        WHERE request_id = ?
        ORDER BY id DESC
        LIMIT 1
    """
    try:
        found = _rows(engine, sql, (request_id,))
    except Exception:
        logger.exception("Failed to load request trace")
        return None
    if not found:
        return None
    row = found[0]
    try:
        tools = json.loads(row[12] or "[]")
    except json.JSONDecodeError:
        tools = []
    try:
        turns = json.loads(row[16] or "[]")
    except json.JSONDecodeError:
        turns = []
    return {
        "request_id": row[0],
        "session_id": row[1],
        "domain": row[2],
        "model": row[3],
        "user_query": row[4] or "",
        "scope_blocked": bool(row[5]),
        "cache_hit": bool(row[6]),
        "short_circuit": row[7],
        "forced_first_tool": row[8],
        "warranty_mode": bool(row[9]),
        "guard_sanitized": bool(row[10]),
        "turn_count": row[11] or 0,
        "tools_called": tools,
        "total_latency_ms": row[13] or 0,
        "final_response_len": row[14] or 0,
        "final_response": row[15] or "",
        "turns": turns,
        "created_at": row[17],
        "requested_models": _stored_model_ids(row[18]),
        "provider_models": _stored_model_ids(row[19]),
        "model_calls_successful": int(row[20] or 0),
        "model_calls_failed": int(row[21] or 0),
        "model_calls_unverified": int(row[22] or 0),
    }


def _stored_model_ids(value: Any) -> List[str]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return list(dict.fromkeys(item for item in parsed if _model_id(item)))[:16]
