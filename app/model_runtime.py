"""Model-compatible request parameters and metadata-only call observation.

This module does not choose new models, create network clients or retry calls.
Callers retain their deterministic validation, timeout and failure behavior.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)
ModelRole = Literal["agent", "router"]
Observer = Callable[..., None]


def _safe_model_id(value: Any) -> str | None:
    if (not isinstance(value, str) or value.startswith(("sk-", "sess-"))
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}", value)
        or "://" in value):
        return None
    return value


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def completion_options(
    model: str,
    *,
    role: ModelRole = "agent",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Build conservative Chat Completions options for one configured role."""
    if role not in {"agent", "router"}:
        raise ValueError("role must be agent or router")
    if _safe_model_id(model) is None:
        raise ValueError("a valid configured model identifier is required")
    default_budget = 3000 if role == "agent" else 2048
    try:
        budget = int(max_output_tokens) if max_output_tokens is not None else default_budget
    except (TypeError, ValueError, OverflowError):
        budget = default_budget
    reasoning = _is_reasoning_model(model)
    options: dict[str, Any] = {
        "store": False,
        "max_completion_tokens": max(512 if reasoning else 32, min(budget, 16384)),
    }
    if reasoning:
        default_effort = "medium" if role == "agent" else "low"
        allowed = {"low", "medium", "high"}
        # The original GPT-5 family does not share newer models' `none` mode.
        if model.lower().startswith(("gpt-6-sol", "gpt-6-luna", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6")):
            allowed.add("none")
        effort = str(reasoning_effort or default_effort).strip().lower()
        options["reasoning_effort"] = effort if effort in allowed else default_effort
    else:
        options["temperature"] = temperature if temperature is not None else (0.1 if role == "agent" else 0)
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    return options


def _error_code(error: Exception) -> str:
    """Classify without copying provider errors, bodies, URLs or customer text."""
    name = type(error).__name__.lower()
    status = getattr(error, "status_code", None)
    if "timeout" in name:
        return "timeout"
    if status == 401 or "authentication" in name:
        return "authentication"
    if status == 403 or "permission" in name:
        return "permission"
    if status == 429 or "ratelimit" in name:
        return "rate_limit"
    if status == 404:
        return "provider_unavailable"
    if status in {400, 422} or "badrequest" in name:
        return "invalid_request"
    if "connection" in name:
        return "connection"
    if isinstance(status, int) and status >= 500:
        return "provider_error"
    return "unexpected_error"


def _usable_output(response: Any) -> bool:
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        return bool(getattr(message, "content", None) or getattr(message, "tool_calls", None))
    # Responses API callers can use the same observation boundary.
    if getattr(response, "output_text", None):
        return True
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            return True
        if getattr(item, "type", None) == "message" and any(
            getattr(part, "text", None) or getattr(part, "refusal", None)
            for part in (getattr(item, "content", None) or [])
        ):
            return True
    return False


def observe_model_call(
    *,
    role: ModelRole,
    requested_model: str,
    response: Any = None,
    error: Exception | None = None,
    observer: Observer | None = None,
) -> None:
    """Observe actual provider identity separately from requested configuration.

    Observers receive only allowlisted metadata. They must never receive the
    response object, prompts, tools, arguments or exception text.
    """
    try:
        safe_requested_model = _safe_model_id(requested_model)
        if safe_requested_model is None:
            return
        if observer is None:
            try:
                from ai_quality import record_model_call
            except ImportError:
                from app.ai_quality import record_model_call
            observer = record_model_call
        succeeded = error is None and response is not None and _usable_output(response)
        provider_model = getattr(response, "model", None) if response is not None else None
        observer(
            role=role,
            requested_model=safe_requested_model,
            provider_model=_safe_model_id(provider_model),
            succeeded=succeeded,
            error_code=_error_code(error) if error is not None else (None if succeeded else "empty_response"),
        )
    except Exception:
        # Observability must never break a working response or mask its error.
        logger.warning("Model-call observation unavailable")


def create_completion(
    client: Any,
    *,
    model: str,
    role: ModelRole = "agent",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    observer: Observer | None = None,
    **request_kwargs: Any,
) -> Any:
    """Execute one compatible Chat call, preserving caller exception handling."""
    options = completion_options(model, role=role, max_output_tokens=max_output_tokens,
                                 reasoning_effort=reasoning_effort, temperature=temperature, json_mode=json_mode)
    # Never allow old call-site parameters to reintroduce unsupported settings.
    reserved = {"store", "max_tokens", "max_completion_tokens", "verbosity"}
    if reserved.intersection(request_kwargs):
        raise ValueError("pass runtime parameters through the named model_runtime options")
    if request_kwargs.get("tools"):
        if model.lower().startswith(("gpt-6.1-sol", "gpt-6-astra")):
            raise ValueError("This model requires the Responses API for tool calling")
        if model.lower().startswith(("gpt-6-sol", "gpt-6-luna")):
            options["reasoning_effort"] = "none"
    try:
        response = client.chat.completions.create(model=model, **options, **request_kwargs)
    except Exception as error:
        observe_model_call(role=role, requested_model=model, error=error, observer=observer)
        raise
    observe_model_call(role=role, requested_model=model, response=response, observer=observer)
    return response
