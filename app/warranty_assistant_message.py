"""
Shared assistant-message enrichment for warranty workflow nodes.

Used by the dedicated warranty API (``warranty_router``) and the general
chat agent tools (``agent_tools``) so both surfaces deliver the same
Freshdesk-backed step/terminal messages.
"""

from __future__ import annotations

from typing import Any, Optional


def build_assistant_message_bundle(
    *,
    engine,
    ticket,
    node: Optional[dict],
) -> dict[str, Any]:
    """
    Return ``assistant_message`` plus optional step/terminal enrichment dicts.

    Mirrors the enrichment branch in ``warranty_router._serialize_ticket_state``.
    """
    terminal_enrichment: Optional[dict[str, Any]] = None
    step_enrichment: Optional[dict[str, Any]] = None
    assistant_message: Optional[str] = None

    if not node or ticket is None:
        return {
            "assistant_message": None,
            "terminal_enrichment": None,
            "step_enrichment": None,
        }

    if node.get("type") == "terminal":
        from warranty_terminal_enrichment import build_terminal_enrichment  # noqa: WPS433

        terminal_enrichment = build_terminal_enrichment(engine, ticket, node)
        if terminal_enrichment:
            assistant_message = (
                str(terminal_enrichment.get("message") or "").strip() or None
            )
    elif node.get("node_id"):
        from warranty_error_code_gate import (  # noqa: WPS433
            build_gate_assistant_message,
            is_gate_node,
        )

        if is_gate_node(str(node.get("node_id") or "")):
            assistant_message = build_gate_assistant_message(ticket, node)
    if assistant_message is None and node.get("type") != "terminal":
        from warranty_step_enrichment import build_step_enrichment  # noqa: WPS433

        step_enrichment = build_step_enrichment(engine, ticket, node)
        if step_enrichment:
            assistant_message = (
                str(step_enrichment.get("message") or "").strip() or None
            )

    if assistant_message:
        from warranty_knowledge import sanitize_customer_answer

        assistant_message = sanitize_customer_answer(assistant_message)
        if step_enrichment:
            step_enrichment = dict(step_enrichment)
            step_enrichment["message"] = assistant_message
            step_enrichment["summary"] = sanitize_customer_answer(str(step_enrichment.get("summary") or ""))
            step_enrichment["tips"] = [tip for tip in step_enrichment.get("tips", []) if sanitize_customer_answer(str(tip)) == tip]
        if terminal_enrichment:
            terminal_enrichment = dict(terminal_enrichment)
            terminal_enrichment["message"] = assistant_message
            diagnosis = terminal_enrichment.get("diagnosis")
            if isinstance(diagnosis, dict):
                diagnosis = dict(diagnosis)
                diagnosis["summary"] = sanitize_customer_answer(str(diagnosis.get("summary") or ""))
                diagnosis["steps"] = [step for step in diagnosis.get("steps", []) if sanitize_customer_answer(str(step)) == step]
                terminal_enrichment["diagnosis"] = diagnosis

    return {
        "assistant_message": assistant_message,
        "terminal_enrichment": terminal_enrichment,
        "step_enrichment": step_enrichment,
    }
