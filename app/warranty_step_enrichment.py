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

from warranty_knowledge import KnowledgeEntry, search_knowledge
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
        if entry.source in ("freshdesk", "freshdesk_kb"):
            for step in entry.customer_steps[:max_tips]:
                _add(step)
            if tips:
                break

    if len(tips) < max_tips:
        for entry in matches:
            if entry.source in ("freshdesk", "freshdesk_kb"):
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
    model_name = str(getattr(ticket, "model_name", "") or "")
    path_text = build_path_text(turns)
    defect_category = infer_defect_category_from_turns(turns)

    matches = search_knowledge(
        path_text=path_text,
        defect_category=defect_category,
        model_name=model_name,
        limit=2,
    )
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
