"""Tests for the agent loop integration: tracing, guard gating, tool routing.

These tests validate the *structure* of the agent loop without calling
any external APIs. They exercise:
  1. RequestTrace / TurnEvent lifecycle
  2. Guard change detection
  3. LLM eval golden set structural validity
  4. Scope classifier → trace linkage
  5. Cache-hit short-circuit path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))


# -------------------------------------------------------------------
# 1. RequestTrace lifecycle
# -------------------------------------------------------------------

from request_trace import RequestTrace, TurnEvent


class TestRequestTraceLifecycle:
    def test_init_defaults(self):
        t = RequestTrace(
            request_id="r1", session_id="s1", domain="test.com",
            user_query="hello", model="gpt-4o",
        )
        assert t.request_id == "r1"
        assert t.scope_blocked is False
        assert t.cache_hit is False
        assert t.guard_sanitized is False
        assert t.total_latency_ms == 0
        assert len(t.turns) == 0

    def test_start_clock_and_finish(self):
        t = RequestTrace(
            request_id="r2", session_id="s1", domain="test.com",
            user_query="price", model="gpt-4o",
        )
        t.start_clock()
        t.finish("The price is $2,999.", guard_changed=False)
        assert t.final_response_len == len("The price is $2,999.")
        assert t.total_latency_ms >= 0
        assert t.guard_sanitized is False

    def test_finish_with_guard(self):
        t = RequestTrace(
            request_id="r3", session_id="s1", domain="test.com",
            user_query="price", model="gpt-4o",
        )
        t.start_clock()
        t.finish("Sanitized answer", guard_changed=True)
        assert t.guard_sanitized is True

    def test_add_turns(self):
        t = RequestTrace(
            request_id="r4", session_id="s1", domain="test.com",
            user_query="fix", model="gpt-4o",
        )
        t.add_turn(TurnEvent(turn=0, tool_name="get_repair_help", tool_success=True, tool_result_len=500))
        t.add_turn(TurnEvent(turn=1, tool_name=None))
        assert len(t.turns) == 2
        assert t.turns[0].tool_name == "get_repair_help"
        assert t.turns[1].tool_name is None

    def test_to_dict_structure(self):
        t = RequestTrace(
            request_id="r5", session_id="s1", domain="test.com",
            user_query="recommend", model="gpt-4o",
        )
        t.add_turn(TurnEvent(turn=0, tool_name="recommend_chairs", tool_success=True))
        t.finish("Here are...", guard_changed=False)
        d = t.to_dict()
        assert d["request_id"] == "r5"
        assert d["turn_count"] == 1
        assert d["tools_called"] == ["recommend_chairs"]
        assert len(d["turns"]) == 1
        assert d["turns"][0]["tool_name"] == "recommend_chairs"

    def test_scope_blocked_trace(self):
        t = RequestTrace(
            request_id="r6", session_id="s1", domain="test.com",
            user_query="write python", model="gpt-4o",
        )
        t.scope_blocked = True
        t.finish("", guard_changed=False)
        d = t.to_dict()
        assert d["scope_blocked"] is True
        assert d["turn_count"] == 0

    def test_cache_hit_trace(self):
        t = RequestTrace(
            request_id="r7", session_id="s1", domain="test.com",
            user_query="price", model="gpt-4o",
        )
        t.cache_hit = True
        t.finish("Cached answer", guard_changed=False)
        d = t.to_dict()
        assert d["cache_hit"] is True

    def test_short_circuit_welcome(self):
        t = RequestTrace(
            request_id="r8", session_id="s1", domain="test.com",
            user_query="hi", model="gpt-4o",
        )
        t.short_circuit = "welcome"
        t.finish("Welcome!", guard_changed=False)
        assert t.to_dict()["short_circuit"] == "welcome"


# -------------------------------------------------------------------
# 2. TurnEvent edge cases
# -------------------------------------------------------------------

class TestTurnEvent:
    def test_defaults(self):
        e = TurnEvent(turn=0)
        assert e.tool_name is None
        assert e.tool_success is True
        assert e.llm_prompt_tokens == 0

    def test_failed_tool(self):
        e = TurnEvent(turn=1, tool_name="lookup_order_status", tool_success=False, tool_result_len=0)
        d = e.to_dict()
        assert d["tool_success"] is False

    def test_guard_action(self):
        e = TurnEvent(turn=0, guard_triggered=True, guard_action="strip_price")
        d = e.to_dict()
        assert d["guard_triggered"] is True
        assert d["guard_action"] == "strip_price"


# -------------------------------------------------------------------
# 3. Persist trace (mock engine)
# -------------------------------------------------------------------

from request_trace import persist_trace, _TABLE_SQL, _INSERT_SQL


class TestPersistTrace:
    def test_persist_calls_sql(self):
        import request_trace
        request_trace._table_ensured = False

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_engine = MagicMock()
        mock_engine.raw_connection.return_value = mock_conn

        t = RequestTrace(
            request_id="p1", session_id="s1", domain="test.com",
            user_query="test", model="gpt-4o",
        )
        t.finish("done", guard_changed=False)

        persist_trace(t, mock_engine)

        mock_cursor.execute.assert_any_call(_TABLE_SQL)
        assert mock_cursor.execute.call_count >= 2
        mock_conn.commit.assert_called()
        mock_conn.close.assert_called_once()

    def test_persist_skips_table_creation_when_ensured(self):
        import request_trace
        request_trace._table_ensured = True

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_engine = MagicMock()
        mock_engine.raw_connection.return_value = mock_conn

        t = RequestTrace(
            request_id="p2", session_id="s1", domain="test.com",
            user_query="test", model="gpt-4o",
        )
        t.finish("ok", guard_changed=False)

        persist_trace(t, mock_engine)

        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert not any("CREATE TABLE" in c for c in calls)

        request_trace._table_ensured = False


# -------------------------------------------------------------------
# 4. LLM eval golden set integrity
# -------------------------------------------------------------------

class TestEvalGoldenSet:
    @pytest.fixture
    def cases(self):
        return json.loads((ROOT / "data" / "llm_eval_cases.json").read_text())

    def test_required_sections_exist(self, cases):
        for section in ("tool_selection", "faithfulness", "refusal", "language_consistency"):
            assert section in cases, f"Missing section: {section}"

    def test_all_ids_unique(self, cases):
        ids = []
        for section in ("tool_selection", "faithfulness", "refusal", "language_consistency"):
            for case in cases.get(section, []):
                assert case["id"] not in ids, f"Duplicate id: {case['id']}"
                ids.append(case["id"])

    def test_tool_selection_cases_have_expected_tool(self, cases):
        for case in cases["tool_selection"]:
            assert "expected_tool" in case, f"{case['id']} missing expected_tool"

    def test_refusal_cases_have_expect_refusal(self, cases):
        for case in cases["refusal"]:
            assert "expect_refusal" in case, f"{case['id']} missing expect_refusal"

    def test_language_cases_have_response_language(self, cases):
        for case in cases["language_consistency"]:
            assert "response_language" in case, f"{case['id']} missing response_language"

    def test_minimum_case_counts(self, cases):
        assert len(cases["tool_selection"]) >= 10
        assert len(cases["faithfulness"]) >= 3
        assert len(cases["refusal"]) >= 3
        assert len(cases["language_consistency"]) >= 2


# -------------------------------------------------------------------
# 5. Scope classifier → trace linkage
# -------------------------------------------------------------------

class TestScopeTraceIntegration:
    def test_scope_block_sets_trace_flag(self):
        from scope_classifier import evaluate_scope
        decision = evaluate_scope("Write me a Python script")
        trace = RequestTrace(
            request_id="st1", session_id="s1", domain="test.com",
            user_query="Write me a Python script", model="gpt-4o",
        )
        if decision.is_blocked:
            trace.scope_blocked = True
        trace.finish("", guard_changed=False)
        assert trace.scope_blocked is True
        assert trace.to_dict()["turn_count"] == 0

    def test_scope_allow_keeps_trace_open(self):
        from scope_classifier import evaluate_scope
        decision = evaluate_scope("How much is the Osaki Solo Flex?")
        trace = RequestTrace(
            request_id="st2", session_id="s1", domain="test.com",
            user_query="How much is the Osaki Solo Flex?", model="gpt-4o",
        )
        if decision.is_blocked:
            trace.scope_blocked = True
        assert trace.scope_blocked is False


# -------------------------------------------------------------------
# 6. Guard detection pattern
# -------------------------------------------------------------------

class TestGuardChangeDetection:
    def test_guard_change_detected(self):
        original = "The price is $999.99 and delivery is tomorrow"
        guarded = "Please check our website for current pricing."
        guard_changed = original != guarded
        trace = RequestTrace(
            request_id="gc1", session_id="s1", domain="test.com",
            user_query="price?", model="gpt-4o",
        )
        trace.finish(guarded, guard_changed=guard_changed)
        assert trace.guard_sanitized is True

    def test_no_guard_change(self):
        original = "The Solo Flex has 3D massage technology."
        trace = RequestTrace(
            request_id="gc2", session_id="s1", domain="test.com",
            user_query="specs?", model="gpt-4o",
        )
        trace.finish(original, guard_changed=False)
        assert trace.guard_sanitized is False
