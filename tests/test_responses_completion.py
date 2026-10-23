"""Exercise provider contracts without network calls or customer records."""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace as NS

import httpx
import pytest
from openai import OpenAI

from app.responses_completion import ResponsesCompletionSession, ResponsesContractError


MODEL = "gpt-6.1-sol"
PROVIDER_MODEL = "gpt-6.1-sol-2026-10-01"
TOOLS = [{"type": "function", "function": {
    "name": "lookup_fixture", "description": "Read verified synthetic facts",
    "strict": True, "parameters": {"type": "object", "properties": {
        "model": {"type": "string"}}, "required": ["model"], "additionalProperties": False},
}}]
MESSAGES = [{"role": "system", "content": "Use only verified facts."},
            {"role": "user", "content": "What is the capacity of Fixture?"}]


def namespace(value):
    if isinstance(value, dict):
        return NS(**{key: namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [namespace(item) for item in value]
    return value


def reasoning(label="a"):
    return {"type": "reasoning", "id": "reasoning_" + label,
            "summary": [], "encrypted_content": "opaque-encrypted-" + label}


def function(label="a", name="lookup_fixture", arguments='{"model":"Fixture"}'):
    return {"type": "function_call", "id": "item_" + label, "call_id": "call_" + label,
            "name": name, "arguments": arguments, "status": "completed"}


def message(text="Verified capacity is 250 lb."):
    return {"type": "message", "id": "message_a", "role": "assistant", "status": "completed",
            "phase": "final_answer", "content": [{"type": "output_text", "text": text, "annotations": []}]}


def response(items, *, status="completed", error=None):
    return namespace({"id": "resp_a", "model": PROVIDER_MODEL, "status": status, "error": error,
        "output": items, "output_text": "".join(part.get("text", "") for item in items
            if item.get("type") == "message" for part in item.get("content", [])),
        "usage": {"input_tokens": 123, "input_tokens_details": {"cached_tokens": 45},
                  "output_tokens": 67, "output_tokens_details": {"reasoning_tokens": 56}, "total_tokens": 190}})


def client_with(*responses):
    calls = []
    iterator = iter(responses)

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return value

    return NS(responses=NS(create=create)), calls


def request(**kwargs):
    return {"model": MODEL, "messages": deepcopy(MESSAGES), "tools": deepcopy(TOOLS),
            "parallel_tool_calls": False, "max_completion_tokens": 2048,
            "reasoning_effort": "medium", "verbosity": "low", "store": False, **kwargs}


def tool_history(reply, *, base=None, output="Verified synthetic capacity: 250 lb"):
    result = deepcopy(base or MESSAGES)
    assistant = reply.choices[0].message
    result.append({"role": "assistant", "content": assistant.content, "tool_calls": [
        {"id": call.id, "type": "function", "function": {
            "name": call.function.name, "arguments": call.function.arguments}}
        for call in assistant.tool_calls]})
    result.extend({"role": "tool", "tool_call_id": call.id,
                   "name": call.function.name, "content": output}
                  for call in assistant.tool_calls)
    return result


def test_exact_call_ids_encrypted_reasoning_and_strict_schema_survive_continuation():
    client, calls = client_with(response([reasoning(), function()]), response([message()]))
    observations = []
    session = ResponsesCompletionSession(client, observer=lambda **values: observations.append(values))
    first = session.create(**request(tool_choice={"type": "function", "function": {"name": "lookup_fixture"}},
                                     safety_identifier="synthetic-hash"))
    assert first.choices[0].message.tool_calls[0].id == "call_a"
    assert first.model == PROVIDER_MODEL
    final = session.create(**request(messages=tool_history(first), tools=None))
    assert final.choices[0].message.content == "Verified capacity is 250 lb."
    assert calls[0]["tools"][0] == {"type": "function", **TOOLS[0]["function"]}
    assert calls[0]["tool_choice"] == {"type": "function", "name": "lookup_fixture"}
    assert calls[0]["parallel_tool_calls"] is False
    assert calls[0]["safety_identifier"] == "synthetic-hash"
    assert calls[0]["store"] is False
    assert calls[0]["include"] == ["reasoning.encrypted_content"]
    assert calls[0]["max_output_tokens"] == 2048
    assert "max_completion_tokens" not in calls[0]
    assert "messages" not in calls[0]
    assert "previous_response_id" not in calls[1]
    assert "tools" not in calls[1]
    assert calls[1]["input"][-3:] == [reasoning(), function(), {
        "type": "function_call_output", "call_id": "call_a", "output": "Verified synthetic capacity: 250 lb"}]
    assert observations == [{"role": "agent", "requested_model": MODEL,
        "provider_model": PROVIDER_MODEL, "succeeded": True, "error_code": None}] * 2


def test_multiple_rounds_replay_each_provider_output_once_and_in_order():
    client, calls = client_with(response([reasoning("a"), function("a")]),
                                response([reasoning("b"), function("b")]), response([message()]))
    session = ResponsesCompletionSession(client, observer=lambda **_: None)
    first = session.create(**request())
    history = tool_history(first)
    second = session.create(**request(messages=history))
    session.create(**request(messages=tool_history(second, base=history), tools=None))
    replay = calls[-1]["input"]
    assert [item["call_id"] for item in replay if item.get("type") == "function_call"] == ["call_a", "call_b"]
    assert [item["encrypted_content"] for item in replay if item.get("type") == "reasoning"] == [
        "opaque-encrypted-a", "opaque-encrypted-b"]
    assert len([item for item in replay if item.get("type") == "function_call_output"]) == 2


def test_stall_retry_does_not_replay_an_unaccepted_assistant_response():
    client, calls = client_with(response([reasoning(), message("One moment please.")]),
                                response([reasoning("b"), function("b")]))
    session = ResponsesCompletionSession(client, observer=lambda **_: None)
    session.create(**request())
    retry = session.create(**request(tool_choice={"type": "function", "function": {"name": "lookup_fixture"}}))
    assert calls[1]["input"] == MESSAGES
    assert retry.choices[0].message.tool_calls[0].id == "call_b"


def test_usage_including_cached_and_reasoning_tokens_is_mapped_without_inventing_cost():
    client, _ = client_with(response([message()]))
    result = ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(tools=None))
    assert result.usage.prompt_tokens == 123
    assert result.usage.completion_tokens == 67
    assert result.usage.prompt_tokens_details.cached_tokens == 45
    assert result.usage.completion_tokens_details.reasoning_tokens == 56
    assert result.usage.total_tokens == 190


@pytest.mark.parametrize("options", [
    {"store": True}, {"previous_response_id": "other-session"}, {"stream": True},
    {"temperature": 0.1}, {"max_tokens": 128}, {"parallel_tool_calls": True},
    {"reasoning_effort": "none"}, {"reasoning_effort": "minimal"},
    {"max_completion_tokens": True}, {"max_completion_tokens": 0},
    {"verbosity": "unknown"}, {"tool_choice": {"type": "function", "function": {"name": "unknown"}}},
])
def test_incompatible_or_unsafe_request_parameters_fail_before_api_call(options):
    client, calls = client_with()
    with pytest.raises(ResponsesContractError):
        ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(**options))
    assert calls == []


@pytest.mark.parametrize("items,status,error", [
    ([message()], "incomplete", None), ([message()], "failed", {"message": "private-provider-details"}),
    ([], "completed", None), ([reasoning()], "completed", None),
    ([function(name="unknown")], "completed", None),
    ([{"type": "web_search_call", "id": "unexpected-hosted-tool"}], "completed", None),
])
def test_failed_empty_or_unavailable_outputs_are_not_delivered_as_success(items, status, error):
    client, _ = client_with(response(items, status=status, error=error))
    observations = []
    with pytest.raises(ResponsesContractError) as failure:
        ResponsesCompletionSession(client, observer=lambda **values: observations.append(values)).create(**request())
    assert "private-provider-details" not in str(failure.value)
    assert observations[-1]["succeeded"] is False
    assert observations[-1]["provider_model"] == PROVIDER_MODEL
    assert "private-provider-details" not in json.dumps(observations)


@pytest.mark.parametrize("tools,choice", [(None, None), (TOOLS, "none")])
def test_synthesis_never_accepts_new_tool_calls(tools, choice):
    client, _ = client_with(response([function()]))
    with pytest.raises(ResponsesContractError, match="unavailable function"):
        ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(tools=tools, tool_choice=choice))


def test_provider_exception_is_reraised_without_copying_exception_text_to_observation():
    error = TimeoutError("private-customer-input-or-provider-body")
    client, _ = client_with(error)
    observations = []
    with pytest.raises(TimeoutError) as failure:
        ResponsesCompletionSession(client, observer=lambda **values: observations.append(values)).create(**request())
    assert failure.value is error
    assert observations == [{"role": "agent", "requested_model": MODEL,
        "provider_model": None, "succeeded": False, "error_code": "timeout"}]


def test_observer_failure_cannot_prevent_a_valid_answer():
    client, _ = client_with(response([message()]))

    def unavailable(**_):
        raise RuntimeError("synthetic-observer-failure")

    result = ResponsesCompletionSession(client, observer=unavailable).create(**request(tools=None))
    assert result.choices[0].message.content == "Verified capacity is 250 lb."


def test_request_local_state_rejects_another_sessions_call_identifiers():
    client, calls = client_with(response([reasoning(), function()]))
    first = ResponsesCompletionSession(client, observer=lambda **_: None)
    reply = first.create(**request())
    second = ResponsesCompletionSession(client, observer=lambda **_: None)
    with pytest.raises(ResponsesContractError, match="does not belong"):
        second.create(**request(messages=tool_history(reply)))
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["arguments", "duplicate_output", "missing_output", "duplicate_assistant"])
def test_replay_requires_exact_provider_call_and_one_matching_output(mutation):
    client, calls = client_with(response([reasoning(), function()]))
    session = ResponsesCompletionSession(client, observer=lambda **_: None)
    history = tool_history(session.create(**request()))
    if mutation == "arguments":
        history[-2]["tool_calls"][0]["function"]["arguments"] = '{"model":"Altered"}'
    elif mutation == "duplicate_output":
        history.append(deepcopy(history[-1]))
    elif mutation == "missing_output":
        history.pop()
    else:
        history.extend(deepcopy(history[-2:]))
    with pytest.raises(ResponsesContractError):
        session.create(**request(messages=history))
    assert len(calls) == 1


def test_refusal_is_exposed_as_text_without_fabricating_a_tool_or_fact():
    refusal = message()
    refusal["content"] = [{"type": "refusal", "refusal": "I cannot provide that information."}]
    client, _ = client_with(response([refusal]))
    result = ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(tools=None))
    assert result.choices[0].message.content == "I cannot provide that information."
    assert result.choices[0].message.tool_calls == []


def test_actual_sdk_serializes_responses_request_and_parses_tool_continuation():
    requests = []
    outputs = iter([[reasoning(), function()], [message()]])

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json={"id": "resp_sdk", "object": "response", "created_at": 0,
            "model": PROVIDER_MODEL, "status": "completed", "error": None, "output": next(outputs),
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 2},
                      "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 15}})

    with OpenAI(api_key="synthetic-no-network", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        session = ResponsesCompletionSession(client, observer=lambda **_: None)
        first = session.create(**request())
        final = session.create(**request(messages=tool_history(first), tools=None))
    assert final.model == PROVIDER_MODEL
    assert final.usage.prompt_tokens_details.cached_tokens == 2
    assert requests[1]["input"][-3:] == [reasoning(), function(), {
        "type": "function_call_output", "call_id": "call_a", "output": "Verified synthetic capacity: 250 lb"}]


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "lookup_fixture"}}])
def test_forced_function_contract_does_not_accept_a_plain_text_substitute(choice):
    client, _ = client_with(response([message("Here are guessed facts without using the required tool.")]))
    with pytest.raises(ResponsesContractError, match="required function"):
        ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(tool_choice=choice))


def test_forced_choice_cannot_be_satisfied_by_a_different_available_function():
    other = deepcopy(TOOLS[0])
    other["function"]["name"] = "different_fixture"
    client, _ = client_with(response([function(name="different_fixture")]))
    with pytest.raises(ResponsesContractError, match="unavailable function"):
        ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(
            tools=TOOLS + [other], tool_choice={"type": "function", "function": {"name": "lookup_fixture"}}))


def test_concurrent_request_instances_never_share_encrypted_reasoning():
    both_stored = Barrier(2)

    def complete(**kwargs):
        customer = kwargs["input"][1]["content"]
        previous = [item for item in kwargs["input"] if item.get("type") == "reasoning"]
        if not previous:
            # Identical call IDs/arguments deliberately stress ownership: only
            # the in-memory request's encrypted reasoning differs.
            return response([reasoning(customer), function()])
        assert previous == [reasoning(customer)]
        return response([message("Verified reply for " + customer)])

    client = NS(responses=NS(create=complete))

    def run(customer):
        session = ResponsesCompletionSession(client, observer=lambda **_: None)
        initial = deepcopy(MESSAGES)
        initial[1]["content"] = customer
        first = session.create(**request(messages=initial))
        both_stored.wait(timeout=5)
        final = session.create(**request(messages=tool_history(first, base=initial), tools=None))
        return final.choices[0].message.content

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(run, ["synthetic-A", "synthetic-B"]))
    assert results == ["Verified reply for synthetic-A", "Verified reply for synthetic-B"]


@pytest.mark.parametrize("strict", [None, False, True])
def test_chat_optional_argument_contract_and_explicit_strict_flags_are_preserved(strict):
    tool = deepcopy(TOOLS[0])
    tool["function"]["parameters"]["properties"]["optional_topic"] = {"type": "string"}
    if strict is None:
        tool["function"].pop("strict")
    else:
        tool["function"]["strict"] = strict
    original = deepcopy(tool)
    client, calls = client_with(response([message()]))
    ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(tools=[tool]))
    forwarded = calls[0]["tools"][0]
    assert forwarded["strict"] is (False if strict is None else strict)
    assert forwarded["parameters"] == original["function"]["parameters"]
    assert "optional_topic" not in forwarded["parameters"]["required"]
    assert tool == original


def test_current_application_tools_keep_their_existing_runtime_contracts():
    from app.agent_tools import TOOL_SCHEMAS

    original = deepcopy(TOOL_SCHEMAS)
    client, calls = client_with(response([message()]))
    ResponsesCompletionSession(client, observer=lambda **_: None).create(**request(tools=TOOL_SCHEMAS))
    assert len(calls[0]["tools"]) == len(original)
    for source, forwarded in zip(original, calls[0]["tools"]):
        assert forwarded["parameters"] == source["function"]["parameters"]
        assert forwarded["strict"] is source["function"].get("strict", False)
    assert TOOL_SCHEMAS == original
