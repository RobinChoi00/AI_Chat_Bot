"""Tests for answer_guard — hallucination blocking."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from answer_guard import sanitize_agent_response


def test_blocks_price_without_catalog_tool():
    out = sanitize_agent_response(
        "The Osaki Solo Flex is $4,999 today.",
        tools_called=[],
        user_query="how much is solo flex",
    )
    assert "$4,999" not in out
    assert "catalog" in out.lower() or "model" in out.lower()


def test_allows_price_from_tool_result():
    out = sanitize_agent_response(
        "The Osaki Solo Flex is $4,999.",
        tools_called=["search_chair_specs"],
        user_query="price for solo flex",
        tool_results=["BASE PRICE (USD): $4,999.00\nOther specs..."],
    )
    assert "$4,999" in out


def test_blocks_numbered_repair_steps_without_tool():
    out = sanitize_agent_response(
        "1. Turn off the chair.\n2. Remove the back panel.\n3. Check the fuse.",
        tools_called=[],
        user_query="how do I fix my chair",
    )
    assert "Remove the back panel" not in out
    assert "support" in out.lower()


def test_strips_discount_narrative():
    out = sanitize_agent_response(
        "Originally $7,999, now only $4,999!",
        tools_called=["search_chair_specs"],
        user_query="price",
        tool_results=["BASE PRICE (USD): $4,999.00"],
    )
    assert "Originally" not in out
