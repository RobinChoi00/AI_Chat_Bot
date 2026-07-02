"""
Answer customer side questions without advancing the warranty workflow.

When a customer asks about specs, dimensions, or general troubleshooting while
the flowchart expects a menu answer or delivery detail, we respond from the
product catalog / knowledge base and re-show the current question.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from delivery_intake import (
    DeliverySpecQuestion,
    detect_delivery_spec_question,
    fetch_delivery_spec_answer,
    is_plausible_email,
    is_plausible_order_id,
    is_plausible_tracking_number,
)
from warranty_intake_context import enrich_path_text
from warranty_knowledge import contextual_search_knowledge
from warranty_self_help import (
    _collect_fallback_hints,
    _friendly_match_summary,
    build_path_text,
    infer_defect_category_from_turns,
)
from warranty_step_enrichment import _pick_step_tips, format_step_message

_FAQ_START_RE = re.compile(
    r"^\s*(what|how|can|could|does|do|is|are|will|where|when|why|"
    r"give|tell|show|please)\b",
    re.IGNORECASE,
)

_QUESTION_HINT_RE = re.compile(
    r"\b(what|how|give|tell|show|size|dimension|measure|length|width|height|"
    r"please|help|where|when|why|can you|could you|do you|is there|the box|"
    r"shipping box|carton|crate|package|doorway|door|weight|heavy|fit)\b",
    re.IGNORECASE,
)

_SKIP_SIDE_NODE_IDS = frozenset({"root"})

_DELIVERY_TEXT_NODES = frozenset(
    {"delivery_get_name", "delivery_get_tracking_number"}
)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _node_prompt(node: dict) -> str:
    return str(node.get("prompt") or "").strip()


def _reprompt_current(node: dict) -> str:
    prompt = _node_prompt(node)
    if prompt:
        return prompt
    return "Please choose one of the options above to continue."


def _reprompt_order_or_email() -> str:
    return (
        "To look up your delivery, please enter your **order number** "
        "(for example `#12345` or `OSKUS11308`) or the **email address** "
        "used at checkout."
    )


def _reprompt_tracking_number() -> str:
    return (
        "Please enter your carrier **tracking number** "
        "(usually 8–40 letters and numbers, such as `1Z999AA10123456784`)."
    )


def _build_spec_side_answer(
    *,
    model_name: str,
    spec: DeliverySpecQuestion,
    reprompt: str,
) -> str:
    parts: list[str] = []
    if model_name:
        answer = fetch_delivery_spec_answer(model_name, spec)
        if answer:
            parts.append(answer)
        else:
            parts.append(
                f"I don't have exact **{spec.title}** for **{model_name}** in our "
                "catalog right now. Our warranty team can confirm the numbers when "
                "they review your case."
            )
    else:
        parts.append(
            f"I can look up **{spec.title}** once we know your chair model. "
            "Please confirm your model at the start of this chat if you haven't yet."
        )
    parts.append(reprompt)
    return "\n\n".join(parts)


def _looks_like_faq_question(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if "?" in raw:
        return True
    if _FAQ_START_RE.search(raw):
        return True
    if _QUESTION_HINT_RE.search(raw) and len(raw.split()) >= 4:
        return True
    return False


def _matches_option_heuristic(node: dict, text: str) -> bool:
    if node.get("type") != "question":
        return False
    from warranty_nlp import _heuristic_option_match  # noqa: WPS433

    return _heuristic_option_match(list(node.get("options") or []), text) is not None


def _looks_like_valid_workflow_answer(node: dict, text: str) -> bool:
    """Return True when the text should be treated as a normal workflow answer."""
    node_id = str(node.get("node_id") or "")
    raw = (text or "").strip()
    if not raw:
        return False

    if node.get("type") == "question":
        options = list(node.get("options") or [])
        keys = {_normalize(str(o.get("answer_key") or "")) for o in options}
        if _normalize(raw) in keys:
            return True
        if _matches_option_heuristic(node, raw):
            return True
        return False

    if node_id in _DELIVERY_TEXT_NODES:
        from warranty_email import extract_email  # noqa: WPS433

        if node_id == "delivery_get_name":
            embedded = extract_email(raw)
            if embedded and is_plausible_email(embedded):
                return True
            return is_plausible_email(raw) or is_plausible_order_id(raw)

        if node_id == "delivery_get_tracking_number":
            return is_plausible_tracking_number(raw)

    if node.get("type") == "question_text" and node_id == "install_model":
        if _looks_like_faq_question(raw):
            return False
        return len(raw) >= 2 and not raw.endswith("?")

    return False


def _should_handle_side_question(node: dict, text: str) -> bool:
    node_id = str(node.get("node_id") or "")
    if node.get("type") == "terminal" or node_id in _SKIP_SIDE_NODE_IDS:
        return False
    if _looks_like_valid_workflow_answer(node, text):
        return False
    if detect_delivery_spec_question(text):
        return True
    return _looks_like_faq_question(text)


def _format_knowledge_side_answer(
    *,
    node: dict,
    query_text: str,
    model_name: str,
    issue_type: str,
    turns: Sequence[Any],
    ticket=None,
) -> Optional[str]:
    path_text = enrich_path_text(
        f"{query_text} {build_path_text(turns)}".strip(),
        ticket,
    )
    defect_category = infer_defect_category_from_turns(turns)
    matches = contextual_search_knowledge(
        path_text=path_text,
        issue_type=issue_type,
        defect_category=defect_category,
        model_name=model_name,
        limit=2,
    )
    node_id = str(node.get("node_id") or "")
    fallback = _collect_fallback_hints(turns, node_id)
    tips = _pick_step_tips(matches, fallback)

    if not matches and not tips:
        return None

    if matches:
        summary = _friendly_match_summary(
            matches,
            defect_category=defect_category,
            model_name=model_name,
            issue_type=issue_type or "warranty",
        )
    else:
        model_display = (model_name or "your chair").strip()
        summary = (
            f"Here's what we can share about your **{model_display}** before we "
            "continue with the next step."
        )

    return format_step_message(
        base_prompt=_reprompt_current(node),
        summary=summary,
        tips=tips,
    )


def try_answer_side_question(
    *,
    node: dict,
    answer: str,
    model_name: str = "",
    issue_type: str = "",
    turns: Optional[Sequence[Any]] = None,
    ticket=None,
) -> Optional[str]:
    """
    If ``answer`` is a side question, return a customer-facing reply plus the
    current workflow re-prompt. Returns ``None`` when normal submit should run.
    """
    text = (answer or "").strip()
    if not text or not node:
        return None
    if not _should_handle_side_question(node, text):
        return None

    node_id = str(node.get("node_id") or "")
    spec = detect_delivery_spec_question(text)
    if spec:
        if node_id == "delivery_get_name":
            reprompt = _reprompt_order_or_email()
        elif node_id == "delivery_get_tracking_number":
            reprompt = _reprompt_tracking_number()
        else:
            reprompt = _reprompt_current(node)
        return _build_spec_side_answer(
            model_name=model_name,
            spec=spec,
            reprompt=reprompt,
        )

    kb_answer = _format_knowledge_side_answer(
        node=node,
        query_text=text,
        model_name=model_name,
        issue_type=issue_type,
        turns=turns or (),
        ticket=ticket,
    )
    if kb_answer:
        return kb_answer

    if _looks_like_faq_question(text):
        return (
            "I don't have a specific answer for that in our records yet, but we "
            "can keep going with your warranty case.\n\n"
            f"{_reprompt_current(node)}"
        )

    return None
