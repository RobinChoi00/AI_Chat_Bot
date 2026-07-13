"""
warranty_step_enrichment.py
===========================
Freshdesk / Q&A context for *non-terminal* workflow steps.

Terminal nodes already get rich diagnosis via ``build_terminal_enrichment``.
Customers who click answer buttons previously saw only the raw flowchart
``prompt`` until the final step — this module adds a short, knowledge-backed
intro (summary + 1–2 tips) before the next question so button-driven flows
feel as informed as free-text intake.

Design contract
---------------
- Knowledge from ``warranty_self_help`` (Freshdesk JSON, Q&A CSV, Auto-Check).
- Optional LLM paraphrase via ``warranty_step_paraphrase`` when OPENAI_API_KEY
  is set (warmer tone; workflow question kept verbatim).
- Never promise replacement, dispatch, or approval on intermediate steps.
- Always preserve the original workflow ``prompt`` verbatim at the end so
  branching logic stays deterministic.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from warranty_intake_context import enrich_path_text, intake_aware_step_summary
from warranty_knowledge import KnowledgeEntry, contextual_search_knowledge
from warranty_self_help import (
    _collect_fallback_hints,
    _friendly_match_summary,
    build_path_text,
    infer_defect_category_from_turns,
)
from warranty_step_paraphrase import paraphrase_step_message

logger = logging.getLogger(__name__)

# First menu — no prior answers to contextualise yet.
_SKIP_NODE_IDS = frozenset({"root"})


def _turn_text(turn: Any, field: str) -> str:
    if isinstance(turn, dict):
        return str(turn.get(field) or "")
    return str(getattr(turn, field, "") or "")


def _fonz_entry_from_hit(hit: dict[str, Any]) -> KnowledgeEntry:
    from warranty_knowledge import KnowledgeEntry, _extract_customer_steps, _infer_category  # noqa: WPS433

    code = str(hit.get("error_code") or "")
    meaning = str(hit.get("meaning") or "").strip()
    troubleshooting = str(hit.get("troubleshooting") or "").strip()
    steps = _extract_customer_steps(troubleshooting, meaning)
    if not steps and troubleshooting:
        steps = (troubleshooting[:220],)
    return KnowledgeEntry(
        source="fonz_error_code",
        category=_infer_category(f"{meaning} {troubleshooting}"),
        title=f"{hit.get('model')} — error {code}",
        diagnostic=meaning[:300] or f"Error code {code}.",
        customer_steps=steps,
    )


def _fonz_match_from_ticket(ticket, model_name: str) -> Optional[KnowledgeEntry]:
    """Use collected error code or intake text on the ticket."""
    from error_code_lookup import (  # noqa: WPS433
        extract_error_codes_from_text,
        lookup_error_code,
    )
    from warranty_intake_context import get_intake_summary  # noqa: WPS433

    if ticket is None:
        return None

    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    code_raw = str(collected.get("error_code") or "").strip()
    if code_raw:
        hit = lookup_error_code(model_name, code_raw)
        if hit:
            return _fonz_entry_from_hit(hit)

    intake = get_intake_summary(ticket)
    for text in (intake, code_raw):
        if not text:
            continue
        for code in extract_error_codes_from_text(text):
            hit = lookup_error_code(model_name, code)
            if hit:
                return _fonz_entry_from_hit(hit)
    return None


def _fonz_match_from_turns(model_name: str, turns: list) -> Optional[KnowledgeEntry]:
    from error_code_lookup import extract_error_codes_from_text, lookup_error_code  # noqa: WPS433
    from warranty_knowledge import KnowledgeEntry, _extract_customer_steps, _infer_category  # noqa: WPS433

    for turn in reversed(turns):
        for field in ("answer_label", "answer", "user_message", "label", "customer_answer"):
            text = _turn_text(turn, field)
            for code in extract_error_codes_from_text(text):
                hit = lookup_error_code(model_name, code)
                if not hit:
                    continue
                return _fonz_entry_from_hit(hit)
    return None


def _pick_step_tips(
    matches: list[KnowledgeEntry],
    fallback: tuple[str, ...],
    *,
    max_tips: int = 2,
) -> list[str]:
    tips: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        key = text.lower().strip()
        if not key or key in seen:
            return
        seen.add(key)
        tips.append(text.strip())

    for entry in matches:
        if entry.source in ("freshdesk", "freshdesk_kb", "fonz_error_code"):
            for step in entry.customer_steps[:max_tips]:
                _add(step)
            if tips:
                break

    if len(tips) < max_tips:
        for entry in matches:
            if entry.source in ("freshdesk", "freshdesk_kb", "fonz_error_code"):
                continue
            for step in entry.customer_steps[:1]:
                _add(step)
            if len(tips) >= max_tips:
                break

    for hint in fallback:
        _add(hint)
        if len(tips) >= max_tips:
            break

    return tips[:max_tips]


def format_step_message(
    *,
    base_prompt: str,
    summary: str,
    tips: list[str],
) -> str:
    parts: list[str] = []
    if summary.strip():
        parts.append(summary.strip())
    if tips:
        parts.append("\n\n**From similar support cases:**")
        for idx, tip in enumerate(tips, start=1):
            parts.append(f"{idx}. {tip}")
    parts.append(f"\n\n{base_prompt.strip()}")
    return "\n".join(parts)


def build_step_enrichment(
    engine,
    ticket,
    node: Optional[dict],
) -> Optional[dict[str, Any]]:
    """
    Return an enriched assistant message for the current *non-terminal* node,
    or ``None`` when we lack enough context / knowledge to add value.
    """
    if not node or node.get("type") == "terminal":
        return None

    node_id = str(node.get("node_id") or "")
    if node_id in _SKIP_NODE_IDS:
        return None

    base_prompt = str(node.get("prompt") or "").strip()
    if not base_prompt:
        return None

    ticket_id = str(getattr(ticket, "ticket_id", "") or "")
    turns = engine.get_turns(ticket_id)
    if len(turns) < 1:
        return None

    issue_type = str(getattr(ticket, "issue_type", "") or "").lower()
    if not issue_type:
        # Model-only or issue-type menu — don't infer symptoms from KB yet.
        return None

    # Delivery/installation mid-flow prompts are self-explanatory (tracking, order
    # lookup, setup). Generic KB search here often pulls unrelated defect tickets
    # (Voice PCB, footrest, etc.) because defect_category is unset on these paths.
    if issue_type in ("delivery", "installation"):
        return None

    model_name = str(getattr(ticket, "model_name", "") or "")
    path_text = enrich_path_text(build_path_text(turns), ticket)
    defect_category = infer_defect_category_from_turns(turns)

    fonz_entry = _fonz_match_from_ticket(ticket, model_name) or _fonz_match_from_turns(
        model_name,
        turns,
    )

    matches = contextual_search_knowledge(
        path_text=path_text,
        issue_type=issue_type,
        defect_category=defect_category,
        model_name=model_name,
        limit=2,
    )
    if fonz_entry:
        matches = [fonz_entry] + [m for m in matches if m.title != fonz_entry.title]
        matches = matches[:2]

    fallback = _collect_fallback_hints(turns, node_id)
    tips = _pick_step_tips(matches, fallback)

    if not matches and not tips:
        return None

    if matches:
        summary = _friendly_match_summary(
            matches,
            defect_category=defect_category,
            model_name=model_name,
            issue_type=issue_type,
        )
    elif tips:
        model_display = (model_name or "your chair").strip()
        summary = (
            f"Based on what you've told us about your **{model_display}**, "
            "here is a quick note before the next question."
        )
    else:
        return None

    summary = intake_aware_step_summary(
        ticket=ticket,
        turns=turns,
        summary=summary,
    )

    message = format_step_message(
        base_prompt=base_prompt,
        summary=summary,
        tips=tips,
    )

    message, paraphrased = paraphrase_step_message(
        message,
        base_prompt=base_prompt,
        model_name=model_name,
        node_id=node_id,
        options=list(node.get("options") or []),
    )

    return {
        "message": message,
        "phase": "workflow_step",
        "sources": [entry.source for entry in matches[:2]],
        "top_match": matches[0].title if matches else None,
        "tips": tips,
        "paraphrased": paraphrased,
    }
