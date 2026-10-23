"""Request-local Responses bridge for the existing customer tool loop.

The bridge keeps encrypted reasoning and exact function call identifiers only
in memory for one HTTP request. It does not store a provider conversation or
change the application's tools, schemas, guards or execution policy.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable


class ResponsesContractError(ValueError):
    """The request or provider output cannot satisfy the existing tool contract."""


def _value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _plain(item: Any) -> Any:
    if isinstance(item, dict):
        return {key: _plain(value) for key, value in item.items() if value is not None}
    if isinstance(item, (list, tuple)):
        return [_plain(value) for value in item]
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    if isinstance(item, SimpleNamespace):
        return _plain(vars(item))
    return item


def _signature(calls: list[Any]) -> tuple[tuple[str, str, str], ...]:
    result = []
    for call in calls:
        function = _value(call, "function", {})
        call_id = _value(call, "id")
        name = _value(function, "name")
        arguments = _value(function, "arguments")
        if not all(isinstance(value, str) and value for value in (call_id, name, arguments)):
            raise ResponsesContractError("Invalid function call contract")
        result.append((call_id, name, arguments))
    if len({entry[0] for entry in result}) != len(result):
        raise ResponsesContractError("Duplicate function call identifiers")
    return tuple(result)


def _tools(schemas: Any) -> list[dict[str, Any]]:
    if not schemas:
        return []
    if not isinstance(schemas, list):
        raise ResponsesContractError("Function tools must be a list")
    result = []
    for schema in schemas:
        if not isinstance(schema, dict) or schema.get("type") != "function":
            raise ResponsesContractError("Only existing function tools are supported")
        function = schema.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ResponsesContractError("Invalid function tool schema")
        if set(function) - {"name", "description", "parameters", "strict"}:
            raise ResponsesContractError("Unsupported function schema field")
        # Responses defaults to strict validation; Chat does not. Preserve the
        # source contract when strict was omitted without rewriting optional
        # properties, required fields or additionalProperties.
        mapped = {"type": "function", **deepcopy(function)}
        mapped.setdefault("strict", False)
        result.append(mapped)
    return result


class ResponsesCompletionSession:
    """One request's Chat-shaped view of stateless Responses function calling."""

    def __init__(self, client: Any, observer: Callable[..., None] | None = None):
        self._client = client
        self._observer = observer
        self._outputs: dict[tuple[tuple[str, str, str], ...], list[dict[str, Any]]] = {}

    def _observe(self, model: str, response: Any = None, error: Exception | None = None) -> None:
        try:
            try:
                from .model_runtime import observe_model_call
            except ImportError:
                from model_runtime import observe_model_call
            observe_model_call(role="agent", requested_model=model, response=response,
                               error=error, observer=self._observer)
        except Exception:
            # Metadata observation must not mask a provider failure or disrupt
            # delivery of an otherwise valid customer answer.
            pass

    def _input(self, messages: Any) -> list[dict[str, Any]]:
        if not isinstance(messages, list) or not messages:
            raise ResponsesContractError("A nonempty message list is required")
        result: list[dict[str, Any]] = []
        pending: set[str] = set()
        replayed: set[tuple[tuple[str, str, str], ...]] = set()
        for message in messages:
            role = _value(message, "role")
            content = _value(message, "content")
            calls = _value(message, "tool_calls") or []
            if role == "assistant" and calls:
                if pending:
                    raise ResponsesContractError("Function output is missing")
                signature = _signature(calls)
                if signature in replayed or signature not in self._outputs:
                    raise ResponsesContractError("Function call does not belong to this request")
                result.extend(deepcopy(self._outputs[signature]))
                pending.update(entry[0] for entry in signature)
                replayed.add(signature)
            elif role == "tool":
                call_id = _value(message, "tool_call_id")
                if call_id not in pending or not isinstance(content, str):
                    raise ResponsesContractError("Function output does not match a pending call")
                result.append({"type": "function_call_output", "call_id": call_id, "output": content})
                pending.remove(call_id)
            elif role in {"system", "developer", "user", "assistant"}:
                if pending:
                    raise ResponsesContractError("Function output is missing")
                if not isinstance(content, str):
                    raise ResponsesContractError("Only text chat messages are supported")
                result.append({"role": role, "content": content})
            else:
                raise ResponsesContractError("Unsupported message role")
        if pending:
            raise ResponsesContractError("Function output is missing")
        return result

    def create(self, **chat_kwargs: Any) -> Any:
        """Call Responses with the existing loop's supported Chat arguments.

        ``store=True``, server-side continuation IDs and unknown request fields
        are rejected. Callers must keep this same instance for tool continuation.
        """
        options = dict(chat_kwargs)
        model = options.pop("model", None)
        if not isinstance(model, str) or not model.strip():
            raise ResponsesContractError("A configured model is required")
        messages = options.pop("messages", None)
        if options.pop("store", False) is not False:
            raise ResponsesContractError("Provider conversation storage is disabled")
        if options.pop("stream", False) is not False:
            raise ResponsesContractError("The tool loop requires completed responses")
        budget = options.pop("max_completion_tokens", 3000)
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 32768:
            raise ResponsesContractError("Invalid completion token budget")
        effort = options.pop("reasoning_effort", "medium")
        if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ResponsesContractError("Unsupported reasoning effort")
        if effort == "none" and model.startswith(("gpt-6.1-sol", "gpt-6-astra")):
            raise ResponsesContractError("This model requires reasoning")
        verbosity = options.pop("verbosity", "low")
        if verbosity not in {"low", "medium", "high"}:
            raise ResponsesContractError("Unsupported verbosity")
        schemas = _tools(options.pop("tools", None))
        choice = options.pop("tool_choice", None)
        parallel = options.pop("parallel_tool_calls", False)
        if parallel is not False:
            raise ResponsesContractError("Parallel function execution is disabled")
        safety_identifier = options.pop("safety_identifier", None)
        if options:
            raise ResponsesContractError("Unsupported Responses bridge request option")

        request: dict[str, Any] = {
            "model": model, "input": self._input(messages),
            "store": False, "include": ["reasoning.encrypted_content"],
            "max_output_tokens": budget, "reasoning": {"effort": effort},
            "text": {"verbosity": verbosity},
        }
        if safety_identifier is not None:
            request["safety_identifier"] = safety_identifier
        if schemas:
            request["tools"] = schemas
            request["parallel_tool_calls"] = False
            if isinstance(choice, dict):
                function = choice.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if choice.get("type") != "function" or name not in {tool["name"] for tool in schemas}:
                    raise ResponsesContractError("Forced function must be an available tool")
                request["tool_choice"] = {"type": "function", "name": name}
            elif choice is not None:
                if choice not in {"auto", "none", "required"}:
                    raise ResponsesContractError("Unsupported tool choice")
                request["tool_choice"] = choice
        elif choice is not None and choice not in ("none", "auto"):
            raise ResponsesContractError("A function cannot be forced without tools")

        response = None
        try:
            response = self._client.responses.create(**request)
            allowed_tools = {tool["name"] for tool in schemas} if choice != "none" else set()
            if isinstance(choice, dict):
                allowed_tools.intersection_update({request["tool_choice"]["name"]})
            converted = self._convert(response, allowed_tools=allowed_tools)
            if (isinstance(choice, dict) or choice == "required") and not converted.choices[0].message.tool_calls:
                raise ResponsesContractError("Provider did not produce the required function call")
        except Exception as error:
            self._observe(model, response=response, error=error)
            raise
        self._observe(model, response=response)
        return converted

    def _convert(self, response: Any, *, allowed_tools: set[str]) -> Any:
        if _value(response, "status") != "completed" or _value(response, "error") is not None:
            raise ResponsesContractError("Provider response did not complete")
        output = _value(response, "output") or []
        if not isinstance(output, list):
            raise ResponsesContractError("Invalid provider output")
        content: list[str] = []
        calls: list[Any] = []
        replay: list[dict[str, Any]] = []
        for item in output:
            plain = _plain(item)
            if not isinstance(plain, dict):
                raise ResponsesContractError("Invalid provider output item")
            kind = plain.get("type")
            if kind == "function_call":
                if plain.get("status") not in {None, "completed"}:
                    raise ResponsesContractError("Provider function call did not complete")
                if plain.get("name") not in allowed_tools:
                    raise ResponsesContractError("Provider called an unavailable function")
                calls.append(SimpleNamespace(id=plain.get("call_id"), type="function",
                    function=SimpleNamespace(name=plain.get("name"), arguments=plain.get("arguments"))))
            elif kind == "message":
                if plain.get("status") not in {None, "completed"}:
                    raise ResponsesContractError("Provider message did not complete")
                parts = plain.get("content", [])
                if not isinstance(parts, list):
                    raise ResponsesContractError("Invalid provider message content")
                for part in parts:
                    if not isinstance(part, dict):
                        raise ResponsesContractError("Invalid provider message content")
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        content.append(part["text"])
                    elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                        content.append(part["refusal"])
                    else:
                        raise ResponsesContractError("Unsupported provider message content")
            elif kind == "reasoning":
                if plain.get("status") not in {None, "completed"}:
                    raise ResponsesContractError("Provider reasoning did not complete")
            else:
                raise ResponsesContractError("Unsupported provider output item")
            replay.append(plain)
        if not calls and not any(text.strip() for text in content):
            raise ResponsesContractError("Provider response contained no usable output")
        if calls:
            signature = _signature(calls)
            self._outputs[signature] = deepcopy(replay)
        usage = _value(response, "usage")
        mapped_usage = None
        if usage is not None:
            details = _value(usage, "input_tokens_details")
            mapped_usage = SimpleNamespace(
                prompt_tokens=_value(usage, "input_tokens", 0),
                completion_tokens=_value(usage, "output_tokens", 0),
                total_tokens=_value(usage, "total_tokens", 0),
                prompt_tokens_details=SimpleNamespace(cached_tokens=_value(details, "cached_tokens", 0)),
                completion_tokens_details=_value(usage, "output_tokens_details"),
            )
        return SimpleNamespace(
            id=_value(response, "id"), model=_value(response, "model"), usage=mapped_usage,
            choices=[SimpleNamespace(index=0, finish_reason="tool_calls" if calls else "stop",
                message=SimpleNamespace(role="assistant", content="".join(content), tool_calls=calls))],
        )
