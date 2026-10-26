"""Reviewed customer goals exercise public Sales APIs and saved session facts.

No live model, stock provider, or email delivery is used. Passing these cases
does not establish a general customer answer-quality score.
"""

import json
from pathlib import Path

import pytest

from script import evaluate_llm_quality as evaluation
from tests.test_evaluation_sales_dialogues import isolated_sales_api


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "sales_customer_quality_cases.json"
CASES = json.loads(CASES_PATH.read_text())["cases"]


class CustomerReplay:
    def __init__(self, client):
        self.client = client
        self.turns = []

    def post(self, path, **kwargs):
        response = self.client.post(path, **kwargs)
        if path.endswith("/api/v1/sales/chat"):
            from sales_models import get_session_collected

            self.turns.append({
                "response": response.json(),
                "saved": get_session_collected(kwargs["json"]["session_id"]),
            })
        return response


@pytest.mark.parametrize("case", CASES, ids=lambda row: row["id"])
def test_customer_goal_through_persisted_public_api(isolated_sales_api, monkeypatch, case):
    import smtplib
    import warranty_router

    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *args, **kwargs: pytest.fail("No real email in customer quality regression"))
    warranty_router.limiter.reset()
    replay = CustomerReplay(isolated_sales_api)
    result = evaluation.run_case(case, replay, base_url="http://testserver", domain="osakiusa.com")
    detail = json.dumps({"goal": case["goal"], "result": result, "turns": replay.turns}, ensure_ascii=False)
    assert result["passed"], detail
    assert result["unrun_turns"] == 0
    assert len(replay.turns) == len(case["turns"])
    for expected_turn, actual in zip(case["turns"], replay.turns):
        for path, expected in expected_turn.get("persisted_expect", {}).items():
            actual_value = evaluation._lookup(actual["saved"], path, object())
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                assert float(actual_value) == pytest.approx(expected, abs=0.02), (path, detail)
            else:
                assert actual_value == expected, (path, detail)


def test_customer_quality_fixtures_have_goals_and_saved_fact_checks():
    assert evaluation.validate_conversations({"cases": CASES}) == []
    assert all(case.get("goal") and case.get("manual_acceptance") for case in CASES)
    assert all(case.get("split") == "regression" for case in CASES)
    assert any(turn.get("persisted_expect") for case in CASES for turn in case["turns"])
