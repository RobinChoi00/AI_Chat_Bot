"""Tests for intent_router — forced first-tool selection."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from intent_router import infer_forced_tool


GOLDEN_INTENTS = [
    ("Where is your showroom?", "get_showroom_info"),
    ("My chair won't turn on", "get_repair_help"),
    ("How do I assemble the Osaki Duo?", "get_repair_help"),
    ("Track order OSKMC1234 john@example.com", "lookup_order_status"),
    ("I need to file a warranty claim", "start_warranty_workflow"),
    ("Recommend a chair under $5000", "recommend_chairs"),
    ("What's the price of Solo Flex?", "search_chair_specs"),
    ("How much is the Hypnos 4D?", "search_chair_specs"),
    ("Tell me about the Maestro LE dimensions", "search_chair_specs"),
    ("Hello there", None),
]


def test_golden_intent_routing():
    for query, expected in GOLDEN_INTENTS:
        assert infer_forced_tool(query) == expected, query


def test_recommend_before_price_when_both_match():
    assert infer_forced_tool("Recommend chairs around $3000") == "recommend_chairs"
