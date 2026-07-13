"""
Persist and reuse free-text intake from smart-start / natural-start / chat tools.

The intake summary is prepended to knowledge search queries so early workflow
steps can surface Freshdesk tips before the customer has clicked many buttons.
"""

from __future__ import annotations

from typing import Any, Optional

COL_MODEL_CONFIRMED = "model_confirmed"


def mark_model_confirmed(ticket) -> None:
    if ticket is None or not hasattr(ticket, "set_collected"):
        return
    ticket.set_collected(COL_MODEL_CONFIRMED, "1")


def is_model_confirmed(ticket) -> bool:
    if ticket is None:
        return False
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    return str(collected.get(COL_MODEL_CONFIRMED) or "") == "1"


def needs_model_confirmation(ticket) -> bool:
    model = str(getattr(ticket, "model_name", "") or "").strip()
    if not model:
        return False
    return not is_model_confirmed(ticket)


def build_model_confirmation_message(model_name: str) -> str:
    display = (model_name or "your chair").strip()
    return (
        f"I have **{display}** as your chair model.\n\n"
        "Is that correct? Tap **Yes, that's my model** below, or type the correct model name."
    )


def persist_intake_summary(
    ticket,
    *,
    summary: str = "",
    raw_message: str = "",
) -> None:
    """Store a customer-safe one-line intake on the ticket for later enrichment."""
    if ticket is None or not hasattr(ticket, "set_collected"):
        return

    summary_text = str(summary or "").strip()
    raw_text = str(raw_message or "").strip()
    primary = summary_text or raw_text
    if not primary:
        return

    ticket.set_collected("intake_summary", primary[:500])
    if raw_text and raw_text != primary:
        ticket.set_collected("intake_raw_message", raw_text[:800])

    try:
        from warranty_error_code_gate import capture_error_code_from_intake  # noqa: WPS433

        capture_error_code_from_intake(ticket, raw_text or primary)
    except Exception:
        pass


def get_intake_summary(ticket) -> str:
    if ticket is None:
        return ""
    collected: dict = {}
    if hasattr(ticket, "get_collected"):
        collected = ticket.get_collected() or {}
    elif isinstance(ticket, dict):
        collected = ticket.get("collected_data") or {}
    return str(
        collected.get("intake_summary") or collected.get("intake_raw_message") or ""
    ).strip()


def enrich_path_text(path_text: str, ticket) -> str:
    """Prepend stored intake text when it is not already in the path."""
    intake = get_intake_summary(ticket)
    base = str(path_text or "").strip()
    if not intake:
        return base
    if intake.lower() in base.lower():
        return base
    return f"{intake} {base}".strip()


def intake_aware_step_summary(
    *,
    ticket,
    turns,
    summary: str,
    max_turns: int = 5,
) -> str:
    """
    Personalize early step enrichment with what the customer said at intake.
    """
    intake = get_intake_summary(ticket)
    body = str(summary or "").strip()
    if not intake or len(turns or ()) > max_turns:
        return body
    if intake.lower() in body.lower():
        return body
    return f"You mentioned: **{intake}**. {body}"


def try_side_question_for_ticket(engine, ticket_id: str, answer: str) -> Optional[str]:
    """Shared side-question handler for REST API and chat agent tools."""
    from warranty_side_questions import try_answer_side_question  # noqa: WPS433

    node = engine.get_current_node(ticket_id)
    ticket = engine.get_ticket(ticket_id)
    if not node or ticket is None:
        return None

    return try_answer_side_question(
        node=node,
        answer=answer,
        model_name=str(getattr(ticket, "model_name", None) or ""),
        issue_type=str(getattr(ticket, "issue_type", None) or ""),
        turns=engine.get_turns(ticket_id),
        ticket=ticket,
    )
