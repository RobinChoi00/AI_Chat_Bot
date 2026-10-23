from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from script import verify_model_upgrade as probe


def response(model, *, content=None, tools=None):
    return NS(model=model, usage=NS(prompt_tokens=10, completion_tokens=8), choices=[NS(message=NS(content=content, tool_calls=tools))])


@pytest.fixture
def provider(monkeypatch):
    requests = []
    observer_calls = []
    call = NS(id="call_fixture_1", type="function", function=NS(name="fixture_lookup", arguments='{"id":"fixture-chair"}'))
    replies = [
        response("gpt-6.1-sol", tools=[call]),
        response("gpt-6.1-sol", content="The fixture chair costs $1,234."),
        response("gpt-6-luna", content='{"label":"human","confidence":"high"}'),
        response("gpt-6-luna", content='{"label":"warranty_redirect","confidence":"high"}'),
    ]

    class Session:
        def __init__(self, client, observer=None):
            self.observer = observer

        def create(self, **kwargs):
            requests.append(kwargs)
            if self.observer:
                self.observer(requested_model=kwargs["model"])
                observer_calls.append(True)
            return replies.pop(0)

    def create_completion(client, **kwargs):
        requests.append(kwargs)
        kwargs["observer"](requested_model=kwargs["model"])
        observer_calls.append(True)
        return replies.pop(0)

    adapter = ModuleType("app.responses_completion")
    adapter.ResponsesCompletionSession = Session
    runtime = ModuleType("model_runtime")
    runtime.create_completion = create_completion
    monkeypatch.setitem(sys.modules, "app.responses_completion", adapter)
    monkeypatch.setitem(sys.modules, "model_runtime", runtime)
    return NS(requests=requests, replies=replies, observer_calls=observer_calls)


def test_probe_uses_adapter_continuation_and_reports_actual_models_without_text(provider):
    result = probe.run_probe(object(), agent="gpt-6.1-sol", router="gpt-6-luna")
    assert result["passed"] is True
    assert result["provider_requests"] == len(provider.requests) == 4
    assert [row["provider_model"] for row in result["observations"]] == ["gpt-6.1-sol"] * 2 + ["gpt-6-luna"] * 2
    # A provider tool-call ID must survive continuation intact.
    continuation = provider.requests[1]["messages"]
    assert continuation[-1]["tool_call_id"] == continuation[-2]["tool_calls"][0]["id"] == "call_fixture_1"
    assert all(request["max_completion_tokens"] == 2048 for request in provider.requests[:2])
    assert all(request["max_output_tokens"] == 2048 for request in provider.requests[2:])
    assert len(provider.observer_calls) == 4
    report = json.dumps(result)
    assert "fixture-chair" not in report
    assert "$1,234" not in report
    assert result["customer_quality_evaluated"] is False
    assert result["customer_records_created"] is False


def test_wrong_provider_model_fails_before_using_wrong_response(provider):
    provider.replies[0].model = "gpt-5.6"
    with pytest.raises(probe.ProbeFailure, match="^provider_model_mismatch$"):
        probe.run_probe(object(), agent="gpt-6.1-sol", router="gpt-6-luna")
    assert len(provider.requests) == 1


def test_wrong_tool_arguments_fail_before_followup(provider):
    provider.replies[0].choices[0].message.tool_calls[0].function.arguments = '{"id":"untrusted-customer-id"}'
    with pytest.raises(probe.ProbeFailure, match="^agent_tool_arguments_invalid$"):
        probe.run_probe(object(), agent="gpt-6.1-sol", router="gpt-6-luna")
    assert len(provider.requests) == 1


def test_invalid_router_json_fails_without_false_success(provider):
    provider.replies[2].choices[0].message.content = "I will connect you."
    with pytest.raises(probe.ProbeFailure, match="^router_json_invalid$"):
        probe.run_probe(object(), agent="gpt-6.1-sol", router="gpt-6-luna")
    assert len(provider.requests) == 3


@pytest.mark.parametrize("actual,accepted", [("gpt-6.1-sol", True), ("gpt-6.1-sol-2026-10-23", True), ("gpt-6.1-sol-other", False), ("gpt-6-sol", False), (None, False)])
def test_provider_snapshot_aliases_are_narrow(actual, accepted):
    if accepted:
        assert probe._provider_model(NS(model=actual), "gpt-6.1-sol") == actual
    else:
        with pytest.raises(probe.ProbeFailure):
            probe._provider_model(NS(model=actual), "gpt-6.1-sol")


@pytest.mark.parametrize("extra", ["OPENAI_API_KEY=do-not-read", "OPENAI_AGENT_MODEL=duplicate", "export OPENAI_ROUTER_MODEL=bad", "OPENAI_AGENT_MODEL=$(untrusted)"])
def test_model_profile_rejects_extra_secret_keys_duplicates_and_shell_syntax(tmp_path, extra):
    path = tmp_path / "models.env"
    path.write_text("OPENAI_AGENT_MODEL=gpt-6.1-sol\nOPENAI_ROUTER_MODEL=gpt-6-luna\n" + extra + "\n")
    with pytest.raises(probe.ProbeFailure):
        probe.load_profile(path)


def test_default_cli_validates_profile_without_provider_access(tmp_path, capsys):
    path = tmp_path / "models.env"
    path.write_text("OPENAI_AGENT_MODEL=gpt-6.1-sol\nOPENAI_ROUTER_MODEL=gpt-6-luna\n")
    assert probe.main(["--profile", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["provider_access_tested"] is False


def test_live_cli_rejects_stale_runtime_configuration_without_network(tmp_path, monkeypatch, capsys):
    path = tmp_path / "models.env"
    path.write_text("OPENAI_AGENT_MODEL=gpt-6.1-sol\nOPENAI_ROUTER_MODEL=gpt-6-luna\n")
    monkeypatch.setenv("OPENAI_AGENT_MODEL", "gpt-5.6")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "gpt-4.1-mini")
    assert probe.main(["--live", "--profile", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["failure_code"] == "effective_model_profile_mismatch"


def test_live_cli_never_prints_provider_exception_body(tmp_path, monkeypatch, capsys):
    import openai

    path = tmp_path / "models.env"
    path.write_text("OPENAI_AGENT_MODEL=gpt-6.1-sol\nOPENAI_ROUTER_MODEL=gpt-6-luna\n")
    monkeypatch.setenv("OPENAI_AGENT_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-secret")

    def fail(**kwargs):
        assert kwargs["timeout"] == 20.0 and kwargs["max_retries"] == 0
        raise ValueError("synthetic-test-secret must never appear in report")

    monkeypatch.setattr(openai, "OpenAI", fail)
    assert probe.main(["--live", "--profile", str(path)]) == 1
    output = capsys.readouterr().out
    assert "synthetic-test-secret" not in output
    assert json.loads(output)["failure_code"] == "provider_probe_failed"


def test_observation_submission_contains_only_operational_metadata(provider):
    result = probe.run_probe(object(), agent="gpt-6.1-sol", router="gpt-6-luna")
    submitted = []
    assert probe._record_probe_observations(result, lambda **row: submitted.append(row)) == 4
    assert [row["role"] for row in submitted] == ["agent", "agent", "router", "router"]
    assert all(set(row) == {"role", "requested_model", "provider_model", "succeeded"} for row in submitted)
    assert all(row["succeeded"] is True for row in submitted)
    with pytest.raises(probe.ProbeFailure, match="cannot_record_failed_probe"):
        probe._record_probe_observations({"passed": False}, lambda **row: submitted.append(row))
    assert len(submitted) == 4


def test_recording_cannot_run_without_opted_in_live_probe(capsys):
    assert probe.main(["--record-observations"]) == 1
    assert json.loads(capsys.readouterr().out)["failure_code"] == "recording_requires_live_probe"


def test_probe_integrates_real_responses_adapter_and_router_without_database(monkeypatch):
    """Exercise actual production adapters against SDK-shaped synthetic replies."""
    calls = []
    accidental_observations = []
    observation_module = ModuleType("ai_quality")
    observation_module.record_model_call = lambda **row: accidental_observations.append(row)
    monkeypatch.setitem(sys.modules, "ai_quality", observation_module)
    monkeypatch.setitem(sys.modules, "app.ai_quality", observation_module)
    usage = NS(input_tokens=20, output_tokens=12, total_tokens=32, input_tokens_details=NS(cached_tokens=0))
    responses = [
        NS(id="resp_fixture_tool", model="gpt-6.1-sol", status="completed", error=None, usage=usage, output=[
            NS(id="reasoning_fixture", type="reasoning", summary=[], encrypted_content="synthetic-encrypted-state"),
            NS(id="fc_fixture", type="function_call", call_id="call_fixture_real", name="fixture_lookup", arguments='{"id":"fixture-chair"}', status="completed"),
        ]),
        NS(id="resp_fixture_answer", model="gpt-6.1-sol", status="completed", error=None, usage=usage, output=[
            NS(id="msg_fixture", type="message", status="completed", role="assistant", content=[NS(type="output_text", text="The fixture-chair costs $1,234.", annotations=[])]),
        ]),
    ]
    classifications = [response("gpt-6-luna", content='{"label":"human","confidence":"high"}'), response("gpt-6-luna", content='{"label":"warranty_redirect","confidence":"high"}')]

    def responses_create(**kwargs):
        calls.append(("responses", kwargs))
        return responses.pop(0)

    def chat_create(**kwargs):
        calls.append(("chat", kwargs))
        return classifications.pop(0)

    client = NS(responses=NS(create=responses_create), chat=NS(completions=NS(create=chat_create)))
    result = probe.run_probe(client, agent="gpt-6.1-sol", router="gpt-6-luna")
    assert result["passed"] is True and result["provider_requests"] == 4
    assert [kind for kind, _ in calls] == ["responses", "responses", "chat", "chat"]
    assert calls[0][1]["tool_choice"] == {"type": "function", "name": "fixture_lookup"}
    assert calls[0][1]["tools"][0]["strict"] is False
    assert "topic" in calls[0][1]["tools"][0]["parameters"]["properties"]
    assert "topic" not in calls[0][1]["tools"][0]["parameters"]["required"]
    continuation = calls[1][1]["input"]
    assert continuation[1]["encrypted_content"] == "synthetic-encrypted-state"
    assert continuation[-1]["call_id"] == continuation[2]["call_id"] == "call_fixture_real"
    assert all(kwargs["store"] is False for _, kwargs in calls)
    assert all("max_tokens" not in kwargs and "temperature" not in kwargs for _, kwargs in calls)
    assert accidental_observations == []
    assert "synthetic-encrypted-state" not in json.dumps(result)
