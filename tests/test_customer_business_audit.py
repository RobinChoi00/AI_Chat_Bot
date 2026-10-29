"""Business audit accounting must preserve failed journeys without invented weights."""
import pytest

from script.simulate_customer_business_quality import summarize


def rows():
    return [{"id": f"customer-{index}", "category": "ordinary", "family": f"goal-{index % 6}",
             "status": "pass", "level": "http_api"} for index in range(24)]


def test_counts_report_actual_customer_outcomes_without_universal_score():
    cases = rows()
    cases[0]["status"] = "fail"
    cases[1]["status"] = "error"
    report = summarize(cases, ["ordinary"])
    assert report["executed"] == 24
    assert report["passed"] == 22
    assert report["failed"] == report["errors"] == 1
    assert report["business_score_out_of_100"] is None
    assert report["categories"]["ordinary"]["behavior_families"] == 6


@pytest.mark.parametrize("mutation", ["duplicate", "missing-category", "unknown-status", "no-breadth"])
def test_incomplete_evidence_cannot_hide_customer_failures(mutation):
    cases = rows()
    if mutation == "duplicate":
        cases.append(cases[0])
    elif mutation == "missing-category":
        cases[0]["category"] = "unknown"
    elif mutation == "unknown-status":
        cases[0]["status"] = "skipped"
    else:
        for row in cases:
            row["family"] = "one-goal"
    with pytest.raises(ValueError):
        summarize(cases, ["ordinary"])
