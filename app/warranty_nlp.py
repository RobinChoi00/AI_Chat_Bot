"""
Natural-language helpers for the warranty workflow (Phase 1 hybrid).

Maps free-text customer messages to flowchart answer_keys while keeping the
deterministic WarrantyEngine as the source of truth for branching and records.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ISSUE_TYPES = frozenset({"installation", "delivery", "defect"})

_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "delivery": (
        "deliver",
        "delivery",
        "shipping",
        "shipped",
        "tracking",
        "package",
        "fedex",
        "ups",
        "usps",
        "box",
        "carrier",
        "arrived damaged",
    ),
    "installation": (
        "install",
        "installation",
        "assembly",
        "assemble",
        "setup",
        "set up",
        "put together",
        "manual",
    ),
    "defect": (
        "defect",
        "broken",
        "malfunction",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't turn on",
        "wont turn on",
        "fault",
        "repair",
        "remote",
        "power",
        "recline",
        "massage",
        "air",
        "inflate",
    ),
}

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|correct|affirmative|i do|i have|it was|it is)\b",
    re.I,
)
_NO_RE = re.compile(
    r"\b(no|nope|nah|negative|i don't|i dont|don't have|dont have|never|not really)\b",
    re.I,
)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _keyword_issue_type(text: str) -> Optional[str]:
    """Cheap keyword vote before calling the LLM."""
    norm = _normalize(text)
    scores = {key: 0 for key in _ISSUE_KEYWORDS}
    for issue, words in _ISSUE_KEYWORDS.items():
        for word in words:
            if word in norm:
                scores[issue] += 1
    best = max(scores, key=lambda k: scores[k])
    if scores[best] <= 0:
        return None
    tied = [k for k, v in scores.items() if v == scores[best]]
    if len(tied) > 1:
        return None
    return best


def _heuristic_option_match(options: list[dict], text: str) -> Optional[str]:
    """Match obvious yes/no or label fragments without an LLM call."""
    norm = _normalize(text)
    if not norm:
        return None

    for opt in options:
        label = _normalize(str(opt.get("label", "")))
        key = str(opt.get("answer_key", ""))
        key_norm = _normalize(key)
        if key_norm and norm == key_norm:
            return key
        if label and norm == label:
            return key
        # Require a substantial label phrase in the user's text — avoids
        # matching short fragments like "no" or "air" to the wrong option.
        if label and len(label) >= 12 and label in norm:
            return key

    keys = [str(o.get("answer_key", "")) for o in options]
    has_yes = any(k.startswith("yes") or k == "yes" for k in keys)
    has_no = any(k.startswith("no") or k == "no" for k in keys)
    if has_yes and has_no:
        if _YES_RE.search(norm) and not _NO_RE.search(norm):
            for opt in options:
                key = str(opt.get("answer_key", ""))
                if key.startswith("yes") or key == "yes":
                    return key
        if _NO_RE.search(norm) and not _YES_RE.search(norm):
            for opt in options:
                key = str(opt.get("answer_key", ""))
                if key.startswith("no") or key == "no":
                    return key

    if "tracking" in norm:
        for opt in options:
            key = str(opt.get("answer_key", ""))
            if "tracking" in key or "tracking" in _normalize(str(opt.get("label", ""))):
                if _NO_RE.search(norm) and key.startswith("no"):
                    return key
                if _YES_RE.search(norm) and key.startswith("has"):
                    return key

    return None


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        from config import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT
    except ImportError:
        return None

    return OpenAI(
        api_key=api_key,
        timeout=float(OPENAI_REQUEST_TIMEOUT),
        max_retries=int(OPENAI_MAX_RETRIES),
    )


def _llm_json(prompt: str, *, task: str = "mapping") -> Optional[dict[str, Any]]:
    client = _openai_client()
    if client is None:
        return None

    from config import ROUTER_MODEL

    system = (
        "You are a strict classifier for a warranty workflow. "
        "You NEVER invent facts, outcomes, repair steps, or new options. "
        "You ONLY pick from the allowed values given in the user message. "
        "If the message is ambiguous, off-topic, or you are not confident, "
        'return null for the target field and confidence="low". '
        "Reply with JSON only."
    )
    if task == "issue_type":
        system += (
            " Map the message to installation, delivery, or defect ONLY when clearly indicated."
        )

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.warning("warranty_nlp LLM call failed: %s", exc)
        return None


def _accept_llm_choice(
    parsed: Optional[dict[str, Any]],
    field: str,
    valid_values: list[str],
) -> Optional[str]:
    """Accept an LLM choice only when confident and within the allowed set."""
    if not parsed:
        return None

    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if confidence != "high":
        return None

    value = str(parsed.get(field, "")).strip()
    if value.lower() in ("null", "none", ""):
        return None
    if value in valid_values:
        return value
    return None


def _format_option_bullets(options: list[dict], *, limit: int = 8) -> str:
    lines: list[str] = []
    for opt in options[:limit]:
        label = str(opt.get("label") or opt.get("answer_key") or "").strip()
        if label:
            lines.append(f"• **{label}**")
    return "\n".join(lines)


def suggest_closest_option(options: list[dict], text: str) -> Optional[dict]:
    """Heuristic best-guess option for did-you-mean clarifying (does not auto-submit)."""
    norm = _normalize(text)
    if not norm or not options:
        return None

    best: Optional[dict] = None
    best_score = 0
    for opt in options:
        label = _normalize(str(opt.get("label") or ""))
        key = _normalize(str(opt.get("answer_key") or ""))
        score = 0
        if key and len(key) >= 3 and key in norm:
            score += 4
        if label and len(label) >= 8 and label in norm:
            score += 5
        for word in norm.split():
            if len(word) >= 4 and word in label:
                score += 2
        if score > best_score:
            best_score = score
            best = opt

    if best_score >= 3:
        return best
    return None


def build_clarifying_workflow_message(node: dict, user_text: str) -> str:
    """Customer-facing re-prompt when free text did not map to a menu option."""
    prompt = str(node.get("prompt") or "").strip()
    options = list(node.get("options") or [])
    trimmed = (user_text or "").strip()

    if trimmed:
        lead = (
            f'I wasn\'t fully sure how **"{trimmed[:120]}"** maps to the choices below.'
        )
    else:
        lead = "I want to make sure I pick the right next step for you."

    parts = [lead]
    closest = suggest_closest_option(options, trimmed)
    if closest:
        label = str(closest.get("label") or closest.get("answer_key") or "").strip()
        if label:
            parts.append(f'Did you mean **{label}**? Tap that option below, or rephrase.')

    bullets = _format_option_bullets(options)
    if bullets:
        parts.append("Please tap one of these, or rephrase to match one of them:")
        parts.append(bullets)
    elif prompt:
        parts.append("Please answer the question below.")
    if prompt:
        parts.append(prompt)
    return "\n\n".join(p for p in parts if p)


_ISSUE_TYPE_LABELS: tuple[tuple[str, str], ...] = (
    ("installation", "Setup & installation"),
    ("delivery", "Delivery & tracking"),
    ("defect", "Warranty / defect"),
)


def build_clarifying_issue_type_message(
    user_text: str,
    *,
    model_name: str = "",
) -> str:
    """Re-prompt when issue type could not be inferred from free text."""
    trimmed = (user_text or "").strip()
    parts: list[str] = []
    if trimmed:
        parts.append(
            f'I couldn\'t tell whether **"{trimmed[:120]}"** is installation, '
            "delivery, or a product defect."
        )
    if model_name:
        parts.append(f"For your **{model_name}**, what type of issue can we help with?")
    else:
        parts.append("What type of issue can we help with?")
    parts.append("Choose one of these, or describe your issue a bit more specifically:")
    for _key, label in _ISSUE_TYPE_LABELS:
        parts.append(f"• **{label}**")
    return "\n\n".join(parts)


def interpret_issue_type(user_text: str) -> Optional[str]:
    """
    Map natural language to installation | delivery | defect.
    Returns None when the intent is unclear.
    """
    text = user_text.strip()
    if not text:
        return None

    keyword = _keyword_issue_type(text)
    if keyword:
        return keyword

    prompt = (
        "Classify this customer warranty message into exactly one issue_type.\n"
        f'Message: "{text}"\n\n'
        "Valid issue_type values:\n"
        '- "installation" — setup, assembly, how to install\n'
        '- "delivery" — shipping, tracking, box damage on arrival\n'
        '- "defect" — product malfunction, broken parts, not working\n\n'
        "Rules:\n"
        "- Pick only when the message clearly fits ONE category.\n"
        "- If unclear or mixed, return issue_type=null and confidence=low.\n"
        '- Return JSON: {"issue_type":"installation|delivery|defect|null","confidence":"high|low"}'
    )
    parsed = _llm_json(prompt, task="issue_type")
    if not parsed:
        return None

    issue = _accept_llm_choice(parsed, "issue_type", list(_ISSUE_TYPES))
    return issue


def interpret_warranty_answer(node: dict, user_text: str) -> Optional[str]:
    """
    Map natural language to an answer_key for the current workflow node.

    For question_text nodes, returns the trimmed user text unchanged.
    For option nodes, returns a valid answer_key or None.
    """
    text = user_text.strip()
    if not text:
        return None

    node_type = node.get("type")
    if node_type == "question_text":
        return text

    options = node.get("options") or []
    if not options:
        return None

    heuristic = _heuristic_option_match(options, text)
    if heuristic:
        return heuristic

    valid_keys = [str(o.get("answer_key", "")) for o in options]
    option_lines = [
        f'- answer_key="{o.get("answer_key")}" label="{o.get("label", "")}"'
        for o in options
    ]
    prompt = (
        "Pick the single best matching answer_key for the customer's message.\n\n"
        f'Question: "{node.get("prompt", "")}"\n'
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\n"
        f'Customer message: "{text}"\n\n'
        f"Valid answer_keys ONLY: {valid_keys}\n"
        "Rules:\n"
        "- Choose exactly one answer_key from Valid answer_keys when clearly matched.\n"
        "- Do NOT invent keys, do NOT guess between close options.\n"
        "- If ambiguous or unrelated, return answer_key=null and confidence=low.\n"
        '- Return JSON: {"answer_key":"<one valid key or null>","confidence":"high|low"}'
    )
    parsed = _llm_json(prompt)
    if not parsed:
        return None

    return _accept_llm_choice(parsed, "answer_key", valid_keys)
