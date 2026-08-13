"""
warranty_step_enrichment.py
===========================
Verified copy for *non-terminal* workflow steps.

Terminal nodes already get diagnosis via ``build_terminal_enrichment``.
Mid-flow customer text is the flowchart prompt plus node-locked hints
(and an exact error-code hit when the customer provided a code).

Design contract
---------------
- Do not put fuzzy Freshdesk / Q&A “similar case” tips in the customer message.
- Do not LLM-paraphrase customer copy (facts stay as drafted).
- Never promise replacement, dispatch, or approval on intermediate steps.
- Always preserve the original workflow ``prompt`` verbatim at the end so
  branching logic stays deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from warranty_intake_context import intake_aware_step_summary
from warranty_knowledge import (
    KnowledgeEntry,
    is_presentable_match_title,
)
from warranty_self_help import (
    _collect_fallback_hints,
    _friendly_match_summary,
    category_fallback_hints,
    infer_defect_category_from_turns,
)

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


# Prefer curated sources over raw Freshdesk ticket threads in mid-flow tips,
# unless a Freshdesk hit is strongly similar to what the customer described.
# Solutions (KB) sit above generic Q&A so published DIY guides surface first.
_STEP_SOURCE_PRIORITY: dict[str, int] = {
    "fonz_error_code": 0,
    "freshdesk_kb": 1,
    "freshdesk_qa": 1,
    "qa_csv": 2,
    "auto_check": 3,
    "fault_judgment": 3,
    "freshdesk": 5,
}

_DEFAULT_TIP_HEADING = "**What you can try:**"
_FRESHDESK_SIMILAR_SOURCES = frozenset({"freshdesk", "freshdesk_kb", "freshdesk_qa"})
_FRESHDESK_KB_SOURCES = frozenset({"freshdesk_kb", "freshdesk_qa"})
# Keyword overlap score from warranty_knowledge._score_entry; above this we
# treat a Freshdesk hit as "close enough" to lead the tip block.
# Published KB can lead at a softer overlap; raw ticket threads stay stricter.
_FRESHDESK_SIMILARITY_THRESHOLD = 5.0
_FRESHDESK_KB_SIMILARITY_THRESHOLD = 3.0


def _step_source_rank(entry: KnowledgeEntry) -> int:
    return _STEP_SOURCE_PRIORITY.get(entry.source, 10)


def _entry_relevance(entry: KnowledgeEntry, path_text: str, category: Optional[str]) -> float:
    from warranty_knowledge import _score_entry, _token_set  # noqa: WPS433

    return _score_entry(entry, _token_set(path_text or ""), category)


def _freshdesk_similarity_leader(
    matches: list[KnowledgeEntry],
    *,
    path_text: str,
    defect_category: Optional[str],
) -> Optional[KnowledgeEntry]:
    """Return the best Freshdesk/KB hit when it clearly overlaps the customer path."""
    best_kb: Optional[KnowledgeEntry] = None
    best_kb_score = 0.0
    best_ticket: Optional[KnowledgeEntry] = None
    best_ticket_score = 0.0
    for entry in matches:
        if entry.source not in _FRESHDESK_SIMILAR_SOURCES:
            continue
        if not entry.customer_steps:
            continue
        score = _entry_relevance(entry, path_text, defect_category)
        if entry.source in _FRESHDESK_KB_SOURCES:
            if score > best_kb_score:
                best_kb = entry
                best_kb_score = score
        elif score > best_ticket_score:
            best_ticket = entry
            best_ticket_score = score
    if best_kb is not None and best_kb_score >= _FRESHDESK_KB_SIMILARITY_THRESHOLD:
        return best_kb
    if best_ticket is not None and best_ticket_score >= _FRESHDESK_SIMILARITY_THRESHOLD:
        return best_ticket
    return None


def _usable_step_text(text: str) -> bool:
    from warranty_knowledge import _is_customer_safe_step  # noqa: WPS433

    return _is_customer_safe_step(text)


def _pick_step_tips(
    matches: list[KnowledgeEntry],
    fallback: tuple[str, ...],
    *,
    max_tips: int = 2,
    prefer_freshdesk: bool = False,
) -> list[str]:
    tips: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if not _usable_step_text(text):
            return
        key = text.lower().strip()
        if not key or key in seen:
            return
        seen.add(key)
        tips.append(text.strip())

    if prefer_freshdesk:
        ordered = sorted(
            matches,
            key=lambda entry: (
                0 if entry.source in _FRESHDESK_SIMILAR_SOURCES else 1,
                _step_source_rank(entry),
            ),
        )
    else:
        ordered = sorted(matches, key=_step_source_rank)

    curated_sources = frozenset(
        {
            "fonz_error_code",
            "qa_csv",
            "auto_check",
            "fault_judgment",
            "freshdesk_kb",
            "freshdesk_qa",
        }
    )

    if prefer_freshdesk:
        for entry in ordered:
            if entry.source in _FRESHDESK_SIMILAR_SOURCES:
                for step in entry.customer_steps[:max_tips]:
                    _add(step)
                if tips:
                    break

    if len(tips) < max_tips:
        for entry in ordered:
            if entry.source in curated_sources:
                for step in entry.customer_steps[:max_tips]:
                    _add(step)
                if tips:
                    break

    if len(tips) < max_tips:
        for entry in ordered:
            if entry.source == "freshdesk":
                for step in entry.customer_steps[:max_tips]:
                    _add(step)
                if tips:
                    break

    for hint in fallback:
        key = hint.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        tips.append(hint.strip())
        if len(tips) >= max_tips:
            break

    return tips[:max_tips]


def _aggregate_customer_text(turns: list) -> str:
    parts: list[str] = []
    for turn in turns:
        for field in ("customer_answer", "answer_label", "user_message", "answer"):
            text = _turn_text(turn, field).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


_TITLE_STOP_WORDS = frozenset(
    {"chair", "issue", "problem", "massage", "osaki", "titan", "error", "code", "light"}
)


def _title_grounded_in_customer_context(title: str, customer_text: str) -> bool:
    """True when the KB subject plausibly matches what the customer actually said."""
    title_norm = (title or "").strip().lower().rstrip(".")
    blob = (customer_text or "").lower()
    if not title_norm or not blob:
        return False
    if len(title_norm) >= 10 and title_norm in blob:
        return True
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", title_norm)
        if len(token) >= 4 and token not in _TITLE_STOP_WORDS
    ]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in blob)
    return hits >= max(1, (len(tokens) + 1) // 2)


def _step_enrichment_summary(
    matches: list[KnowledgeEntry],
    *,
    defect_category: Optional[str],
    model_name: str,
    issue_type: str,
    turns: list,
) -> str:
    """
    Customer-facing intro before the next workflow question.

    Prefer category-based wording for button-only flows. Only cite a KB/Q&A
    subject when the customer text supports it (avoids unrelated titles like
    "Red blinking light" after they only tapped Power issue).
    """
    customer_text = _aggregate_customer_text(turns)
    grounded: list[KnowledgeEntry] = []
    for entry in matches:
        if not is_presentable_match_title(entry.title):
            continue
        if entry.source == "fonz_error_code":
            grounded.append(entry)
            continue
        if entry.source in {"qa_csv", "auto_check"} and _title_grounded_in_customer_context(
            entry.title,
            customer_text,
        ):
            grounded.append(entry)
            continue
        if entry.source == "freshdesk":
            # Freshdesk ticket subjects are often generic; category summary is safer.
            continue

    if grounded:
        return _friendly_match_summary(
            grounded,
            defect_category=defect_category,
            model_name=model_name,
            issue_type=issue_type,
        )

    return _friendly_match_summary(
        [],
        defect_category=defect_category,
        model_name=model_name,
        issue_type=issue_type,
    )


def format_step_message(
    *,
    base_prompt: str,
    summary: str,
    tips: list[str],
    tip_heading: str = _DEFAULT_TIP_HEADING,
) -> str:
    parts: list[str] = []
    if summary.strip():
        parts.append(summary.strip())
    if tips:
        parts.append(f"\n\n{tip_heading.strip()}")
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
    defect_category = infer_defect_category_from_turns(turns)

    # Only an exact customer-provided error code may add KB steps. Fuzzy
    # Freshdesk / Q&A hits stay out of the customer message.
    fonz_entry = _fonz_match_from_ticket(ticket, model_name) or _fonz_match_from_turns(
        model_name,
        turns,
    )
    matches = [fonz_entry] if fonz_entry else []

    fallback = _collect_fallback_hints(turns, node_id)
    if not fallback and defect_category:
        fallback = category_fallback_hints(defect_category)
    tips = _pick_step_tips(matches, fallback, prefer_freshdesk=False)

    if not tips:
        return None

    if fonz_entry:
        summary = _step_enrichment_summary(
            matches,
            defect_category=defect_category,
            model_name=model_name,
            issue_type=issue_type,
            turns=turns,
        )
    else:
        model_display = (model_name or "your chair").strip()
        summary = (
            f"Based on what you've told us about your **{model_display}**, "
            "here is a quick note before the next question."
        )

    summary = intake_aware_step_summary(
        ticket=ticket,
        turns=turns,
        summary=summary,
    )

    message = format_step_message(
        base_prompt=base_prompt,
        summary=summary,
        tips=tips,
        tip_heading=_DEFAULT_TIP_HEADING,
    )

    return {
        "message": message,
        "phase": "workflow_step",
        "sources": [fonz_entry.source] if fonz_entry else [],
        "top_match": None,
        "tips": tips,
        "paraphrased": False,
        "similar_symptom_match": False,
    }
