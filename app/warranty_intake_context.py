"""
Persist and reuse free-text intake from smart-start / natural-start / chat tools.

The intake summary is prepended to knowledge search queries so early workflow
steps can surface Freshdesk tips before the customer has clicked many buttons.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

COL_MODEL_CONFIRMED = "model_confirmed"

_MODEL_NOISE = frozenset(
    {"osaki", "titan", "os", "massage", "chair", "model", "the", "my", "is"}
)
_MODEL_DEPENDENT_COLLECTED_KEYS = (
    "error_code",
    "error_code_gate_completed",
    "pending_terminal",
    "fonz_meaning",
    "fonz_parts_internal",
    "fonz_severity",
    "fonz_lookup_failed",
    "fonz_category_aligned",
)


def _model_identity(value: str) -> str:
    """Comparable model key that ignores brands and display punctuation."""
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return "".join(token for token in tokens if token not in _MODEL_NOISE)


def _strip_previous_model(text: str, previous_model: str) -> str:
    """Remove a superseded model name while retaining any symptom description."""
    value = str(text or "").strip()
    if not value:
        return ""

    previous_identity = _model_identity(previous_model)
    if previous_identity and _model_identity(value) == previous_identity:
        return ""

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", str(previous_model or "").lower())
        if token not in _MODEL_NOISE
    ]
    if tokens:
        flexible = r"[\s_\-/]*".join(re.escape(token) for token in tokens)
        value = re.sub(
            rf"(?<![a-z0-9]){flexible}(?![a-z0-9])",
            " ",
            value,
            flags=re.I,
        )

    value = re.sub(
        r"\b(?:my|the)?\s*(?:chair\s*)?model\s*(?:is|was|:|-)?\s*",
        " ",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip(" ,;:-.()[]")
    meaningful = [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in _MODEL_NOISE
    ]
    return value if meaningful else ""


@lru_cache(maxsize=1)
def _known_model_labels() -> tuple[str, ...]:
    """Model labels used to clean stale context from sessions created pre-fix."""
    try:
        from fonz_warranty_data import load_model_diagnostic_records  # noqa: WPS433

        labels = {
            str(row.get("model") or "").strip()
            for row in load_model_diagnostic_records()
            if str(row.get("model") or "").strip()
        }
        return tuple(
            sorted(
                (label for label in labels if len(_model_identity(label)) >= 5),
                key=len,
                reverse=True,
            )
        )
    except Exception:
        return ()


def _sanitize_intake_for_current_model(text: str, current_model: str) -> str:
    value = str(text or "").strip()
    current_identity = _model_identity(current_model)
    if not value or not current_identity:
        return value

    allowed = {current_identity}
    try:
        from model_families import resolve_family_canonical  # noqa: WPS433

        canonical = resolve_family_canonical(current_model)
        if canonical:
            allowed.add(_model_identity(canonical))
    except Exception:
        pass

    for label in _known_model_labels():
        identity = _model_identity(label)
        if identity in allowed or identity not in _model_identity(value):
            continue
        value = _strip_previous_model(value, label)
        if not value:
            break
    return value


def reconcile_model_change(ticket, previous_model: str, current_model: str) -> bool:
    """
    Remove context that belongs to a superseded chair model.

    Troubleshooting text is preserved when possible, but old model references,
    captured error codes, and Fonz lookup results are cleared so a corrected
    model can never inherit a diagnosis from the previous one.
    """
    previous = str(previous_model or "").strip()
    current = str(current_model or "").strip()
    if not previous or not current:
        return False
    if _model_identity(previous) == _model_identity(current):
        return False
    if ticket is None or not hasattr(ticket, "set_collected"):
        return False

    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    for key in ("intake_summary", "intake_raw_message"):
        cleaned = _strip_previous_model(str(collected.get(key) or ""), previous)
        ticket.set_collected(key, cleaned)

    for key in _MODEL_DEPENDENT_COLLECTED_KEYS:
        ticket.set_collected(key, "")
    ticket.set_collected(COL_MODEL_CONFIRMED, "")
    return True


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
    current_model = str(getattr(ticket, "model_name", "") or "").strip()
    for key in ("intake_summary", "intake_raw_message"):
        value = _sanitize_intake_for_current_model(
            str(collected.get(key) or ""),
            current_model,
        )
        if value:
            return value
    return ""


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
