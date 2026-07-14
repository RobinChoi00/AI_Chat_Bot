"""Tool schemas stay strict so malformed model arguments cannot reach tools."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from agent_tools import TOOL_SCHEMAS  # noqa: E402


def test_all_tool_schemas_are_strict_and_closed():
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        parameters = function["parameters"]
        assert function["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_originally_optional_fields_accept_null_in_strict_mode():
    by_name = {item["function"]["name"]: item["function"] for item in TOOL_SCHEMAS}
    recommend = by_name["recommend_chairs"]["parameters"]["properties"]
    assert "null" in recommend["budget_min"]["type"]
    assert "null" in recommend["budget_max"]["type"]
    evidence = by_name["attach_warranty_evidence"]["parameters"]["properties"]
    assert "null" in evidence["original_filename"]["type"]
