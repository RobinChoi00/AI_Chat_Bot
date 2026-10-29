"""Audit accounting must preserve failures rather than reward duplicated variants."""
from copy import deepcopy

import pytest

from script.quality_v2_support import CheckFailure, Scenario
from script.simulate_chatbot_broad_quality import execute_scenario, score_results


def rows():
    return [{"id": f"case-{family}-{variant}", "category": "domain", "family": f"family-{family}",
             "status": "pass", "level": "function", "severity": "high"}
            for family in range(5) for variant in range(4)]


def test_family_weight_does_not_grow_with_repeated_successful_paraphrases():
    evidence = rows()
    for row in evidence:
        if row["family"] == "family-0":
            row["status"] = "fail"
    score = score_results(evidence, {"domain": 100})
    repeated = deepcopy(evidence)
    for index in range(100):
        repeated.append({**evidence[-1], "id": f"extra-{index}"})
    assert score["score_out_of_100"] == 80
    assert score_results(repeated, {"domain": 100})["score_out_of_100"] == 80


def test_unexpected_errors_reduce_score_and_remain_separate_from_failures():
    evidence = rows()
    evidence[0]["status"] = "error"
    evidence[1]["status"] = "fail"
    report = score_results(evidence, {"domain": 100})
    assert report["score_out_of_100"] == 90
    assert report["failed"] == report["errors"] == 1
    assert report["failed_by_severity"] == {"high": 2}


@pytest.mark.parametrize("mutation", ["duplicate", "missing-family", "unknown-category", "unknown-status"])
def test_missing_or_inconsistent_results_cannot_be_scored(mutation):
    evidence = rows()
    if mutation == "duplicate":
        evidence.append(evidence[-1])
    elif mutation == "missing-family":
        evidence = [row for row in evidence if row["family"] != "family-0"]
    elif mutation == "unknown-category":
        evidence[0]["category"] = "unmeasured"
    else:
        evidence[0]["status"] = "skipped"
    with pytest.raises(ValueError):
        score_results(evidence, {"domain": 100})


def test_failed_customer_contract_preserves_expected_and_observed(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import script.simulate_chatbot_broad_quality as audit

    @contextmanager
    def fake_context():
        yield SimpleNamespace(attempted_external_delivery=[])

    monkeypatch.setattr(audit, "offline_api", fake_context)

    def fail(_):
        raise CheckFailure("Budget was lost", expected=300, observed=3000)

    report = execute_scenario(Scenario("failure", "domain", "budget", "Keep stated budget", "high", fail))
    assert report["status"] == "fail"
    assert report["expected"] == 300 and report["observed"] == 3000


def test_swallowed_outbound_attempt_cannot_be_reported_as_a_passing_case(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import script.simulate_chatbot_broad_quality as audit

    @contextmanager
    def fake_context():
        yield SimpleNamespace(attempted_external_delivery=["blocked"])

    monkeypatch.setattr(audit, "offline_api", fake_context)
    case = Scenario("outbound", "domain", "isolation", "Never send during audit", "high", lambda _: {})
    report = execute_scenario(case)
    assert report["status"] == "fail" and report["observed"] == 1
