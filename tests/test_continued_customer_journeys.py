"""Continued conversation accounting must retain mistakes after recovery."""
from copy import deepcopy

import pytest

from script.simulate_continued_customer_journeys import summarize, validate_result


def result():
    return {"id": "customer-1", "area": "warranty", "customer_turns": 8,
            "transcript": [{"customer": "It still does not work", "response": {"actual": "reply"}}],
            "checks": [{"dimension": "understanding", "name": "Identify symptom", "status": "fail", "turn": 1},
                       {"dimension": "recovery", "name": "Explicit correction", "status": "pass", "turn": 3},
                       {"dimension": "accuracy", "name": "Live price", "status": "unmeasured", "turn": 4}],
            "final_outcome": "goal_met", "transport_request_count": 12,
            "recovery_attempts": [{"turn": 3, "action": "Rephrase", "recovered": True}]}


def test_successful_final_step_does_not_erase_earlier_misunderstanding():
    report = summarize([result()])
    assert report["outcomes"] == {"goal_met": 1}
    assert report["goal_met_after_earlier_failures"] == 1
    assert report["checkpoint_counts"] == {"pass": 1, "fail": 1, "unmeasured": 1}
    assert report["recovery_attempts"]["observed_success"] == 1
    assert report["live_model_score"] is None and report["universal_score_out_of_100"] is None


def test_failed_recovery_and_partial_goal_remain_visible():
    row = result()
    row["final_outcome"] = "partially_met"
    row["recovery_attempts"][0]["recovered"] = False
    report = summarize([row])
    assert report["outcomes"] == {"partially_met": 1}
    assert report["goal_met_after_earlier_failures"] == 0
    assert report["recovery_attempts"]["observed_failure"] == 1


@pytest.mark.parametrize("mutation", ["outcome", "status", "turns", "no-checks", "recovery"])
def test_incomplete_or_unknown_evidence_cannot_look_like_a_completed_goal(mutation):
    row = deepcopy(result())
    if mutation == "outcome":
        row["final_outcome"] = "purchase_completed"
    elif mutation == "status":
        row["checks"][0]["status"] = "skipped"
    elif mutation == "turns":
        row["customer_turns"] = 1
    elif mutation == "no-checks":
        row["checks"] = []
    else:
        row["recovery_attempts"][0]["recovered"] = "probably"
    with pytest.raises(ValueError):
        validate_result(row)


def test_runtime_error_is_reported_instead_of_becoming_a_goal_met():
    row = {"id": "runtime", "area": "sales_service", "execution_error": "RuntimeError"}
    report = summarize([row])
    assert report["execution_errors"] == 1
    assert report["outcomes"] == {}
