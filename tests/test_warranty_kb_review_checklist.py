"""Ensure the repair KB review checklist stays usable for Admin quality work."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "data" / "warranty_kb_review_checklist.json"
FLOWCHART = ROOT / "data" / "warranty_flowchart.json"


def test_priority_terminals_exist_in_flowchart_and_have_checks():
    checklist = json.loads(CHECKLIST.read_text(encoding="utf-8"))
    flowchart = json.loads(FLOWCHART.read_text(encoding="utf-8"))
    nodes = flowchart.get("nodes") or flowchart
    assert isinstance(nodes, dict)
    rows = checklist["priority_terminals"]
    assert len(rows) >= 5
    for row in rows:
        node_id = row["node_id"]
        assert node_id in nodes, node_id
        assert row["issue_type"] in {"installation", "defect", "delivery"}
        assert row["check"] and all(isinstance(item, str) and item.strip() for item in row["check"])
    assert "B_first_answer_match" in checklist["success_metrics"]
    assert "A_confirmed_fix" in checklist["success_metrics"]
    assert "C_handoff_followthrough" in checklist["success_metrics"]
