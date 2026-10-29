"""The offline quality score fails closed and executes its public-API contracts."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from script.simulate_chatbot_quality import WEIGHTS, offline_api, score_results


def results(passed=True):
    return [{"id": f"{category}-{index}", "category": category, "passed": passed}
            for category in WEIGHTS for index in range(8)]


def test_runtime_failures_reduce_the_fixed_weighted_score():
    rows = results()
    rows[0].update(passed=False, error_type="RuntimeError", error="offline delivery attempted")
    scored = score_results(rows)
    assert scored["score_out_of_100"] == 96.875
    assert scored["passed"] == 39 and scored["failed"] == 1
    assert scored["categories"]["safety"]["score"] == 21.875


def test_all_failures_cannot_be_reported_as_success():
    assert score_results(results(False))["score_out_of_100"] == 0


def test_swallowed_outbound_failures_remain_observable(monkeypatch):
    import requests

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "app"))
    with offline_api() as client:
        # An application may legitimately catch transport errors. The harness
        # must still know that this scenario attempted forbidden delivery.
        try:
            requests.get("https://blocked.example.test/")
        except RuntimeError:
            pass
        assert len(client.attempted_external_delivery) == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong-category"])
def test_incomplete_or_duplicated_evidence_cannot_produce_a_score(mutation):
    rows = results()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1]["id"] = rows[0]["id"]
    else:
        rows[-1]["category"] = "unmeasured"
    with pytest.raises(ValueError):
        score_results(rows)


def test_fixed_synthetic_behavior_contracts_through_real_offline_apis(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "quality.json"
    completed = subprocess.run([sys.executable, str(root / "script/simulate_chatbot_quality.py"), "--out", str(output)],
                               cwd=root, capture_output=True, text=True, timeout=90)
    report = json.loads(output.read_text())
    failures = [{"id": row["id"], "error": row.get("error")} for row in report["cases"] if not row["passed"]]
    assert completed.returncode == 0, (failures, completed.stderr)
    assert report["executed"] == report["passed"] == 40
    assert report["failed"] == 0
    assert report["score_out_of_100"] == 100
    assert report["live_model_score"] is None
    assert report["mode"] == "offline_synthetic_regression_simulation"
