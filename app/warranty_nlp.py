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
        "airbag",
        "footrest",
        "foot rest",
        "not inflating",
        "won't inflate",
        "wont inflate",
        "calf",
        "legrest",
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
_SUGGEST_CONFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|correct|right|exactly|yes please|"
    r"that'?s (it|the one|right|correct)|yes that'?s (it|right))\s*[.!]?\s*$",
    re.I,
)
_SUGGEST_REJECT_RE = re.compile(
    r"^\s*(no|nope|nah|not that|wrong|neither)\s*[.!]?\s*$",
    re.I,
)
_STEPS_DONE_RE = re.compile(
    r"\b(tried (all )?(the )?steps|i('ve| have) tried|watched the (guide|video)|"
    r"completed (the )?(prep|steps|setup|guide)|done with (the )?steps|"
    r"all the steps|i('ve| have) (watched|done|finished))\b",
    re.I,
)
_UNABLE_RE = re.compile(
    r"\b(can'?t (safely )?(do|complete|try)|cannot (safely )?(do|complete)|"
    r"unable to|too unsafe|don'?t have (the )?(photos|paperwork|video))\b",
    re.I,
)
_NEED_TEAM_RE = re.compile(
    r"\b(need help|please help|send (a )?(tech|technician)|warranty team|"
    r"call me|contact me|submit (my )?(case|claim)|file a claim)\b",
    re.I,
)
_STILL_BROKEN_RE = re.compile(
    r"\b(still (broken|there|not working|happening)|not (fixed|working)|"
    r"didn'?t work|doesn'?t work|issue is still)\b",
    re.I,
)
_WORKING_NOW_RE = re.compile(
    r"\b(working now|it'?s working|it'?s fixed|fixed now|"
    r"problem (is )?gone|resolved)\b",
    re.I,
)
_SETUP_DONE_RE = re.compile(
    r"\b(set up now|all set up|assembled|"
    r"install(ation)? (is )?(done|complete)|chair is (ready|set up))\b",
    re.I,
)
_ALL_SET_RE = re.compile(
    r"\b(all set|no (more )?help needed|i'?m good|no thanks)\b",
    re.I,
)
_COME_BACK_RE = re.compile(
    r"\b(come back|not yet|later|i'?ll wait)\b",
    re.I,
)
_DELIVERY_SUBMIT_RE = re.compile(
    r"\b(submit|send (the|this|my)? ?case|file (the |a )?(claim|case))\b",
    re.I,
)

PENDING_SUGGESTED_KEY = "pending_suggested_answer_key"
PENDING_SUGGESTED_LABEL = "pending_suggested_label"


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

    words = [word for word in norm.split() if len(word) >= 5]
    if words:
        matches: list[str] = []
        for opt in options:
            label = _normalize(str(opt.get("label", "")))
            key = str(opt.get("answer_key", ""))
            key_norm = _normalize(key)
            if any(word in label or word in key_norm for word in words):
                matches.append(key)
        if len(matches) == 1:
            return matches[0]

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

    if best_score >= 2:
        return best
    return None


def option_label_for(node: dict, answer_key: str) -> str:
    key = str(answer_key or "").strip()
    for opt in node.get("options") or []:
        if str(opt.get("answer_key") or "") == key:
            return str(opt.get("label") or key).strip() or key
    return key


def public_option(opt: dict) -> dict[str, str]:
    key = str(opt.get("answer_key") or "").strip()
    label = str(opt.get("label") or key).strip() or key
    return {"answer_key": key, "label": label}


def node_has_yes_no_options(node: dict) -> bool:
    keys = [str(o.get("answer_key") or "") for o in (node.get("options") or [])]
    has_yes = any(k == "yes" or k.startswith("yes") for k in keys)
    has_no = any(k == "no" or k.startswith("no") for k in keys)
    return has_yes and has_no


def is_suggestion_confirmation(text: str) -> bool:
    return bool(_SUGGEST_CONFIRM_RE.match((text or "").strip()))


def is_suggestion_rejection(text: str) -> bool:
    return bool(_SUGGEST_REJECT_RE.match((text or "").strip()))


def build_mapped_ack(label: str) -> str:
    pretty = str(label or "").strip()
    if not pretty:
        return "Got it."
    return f"Got it — **{pretty}**."


def build_clarifying_workflow_message(
    node: dict,
    user_text: str,
    *,
    closest: Optional[dict] = None,
) -> str:
    """Customer-facing re-prompt when free text did not map to a menu option."""
    prompt = str(node.get("prompt") or "").strip()
    options = list(node.get("options") or [])
    trimmed = (user_text or "").strip()
    suggested = closest if closest is not None else suggest_closest_option(options, trimmed)

    if trimmed:
        lead = (
            f'I wasn\'t fully sure how **"{trimmed[:120]}"** maps to the choices below.'
        )
    else:
        lead = "I want to make sure I pick the right next step for you."

    parts = [lead]
    if suggested:
        label = str(suggested.get("label") or suggested.get("answer_key") or "").strip()
        if label:
            parts.append(
                f"Did you mean **{label}**? Tap **Yes — {label}** below, "
                "or type **yes** to confirm."
            )
            parts.append("If that’s not it, pick a different option or rephrase.")
            return "\n\n".join(parts)

    if options:
        parts.append("Please tap one of the options below, or rephrase to match one of them.")
    elif prompt:
        parts.append("Please answer the question below.")
    if prompt:
        parts.append(prompt)
    return "\n\n".join(p for p in parts if p)


def build_intent_confirmation_message(
    node: dict,
    mapped_key: str,
    user_text: str,
) -> str:
    """Ask the customer to tap the matched option — never auto-advance on guess."""
    options = list(node.get("options") or [])
    label = str(mapped_key or "").strip()
    for opt in options:
        if str(opt.get("answer_key") or "") == mapped_key:
            label = str(opt.get("label") or mapped_key).strip()
            break

    trimmed = (user_text or "").strip()
    parts: list[str] = []
    if trimmed:
        parts.append(
            f'Just to confirm — for **"{trimmed[:120]}"**, did you mean **{label}**?'
        )
    else:
        parts.append(f"Just to confirm — did you mean **{label}**?")
    parts.append(
        "Please tap that option below to continue. I won't choose and move forward for you."
    )
    bullets = _format_option_bullets(options)
    if bullets:
        parts.append(bullets)
    return "\n\n".join(parts)


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


def build_suggested_issue_type_message(issue_type: str, user_text: str = "") -> str:
    """Ask the customer to confirm an inferred issue type by tapping a button."""
    label = issue_type
    for key, pretty in _ISSUE_TYPE_LABELS:
        if key == issue_type:
            label = pretty
            break
    trimmed = (user_text or "").strip()
    parts: list[str] = []
    if trimmed:
        parts.append(
            f'Based on **"{trimmed[:120]}"**, this sounds like **{label}**.'
        )
    else:
        parts.append(f"This sounds like **{label}**.")
    parts.append(
        f"Please tap **{label}** below to confirm. I won't start that path until you choose."
    )
    parts.append("Or pick a different option if I misunderstood:")
    for _key, pretty in _ISSUE_TYPE_LABELS:
        parts.append(f"• **{pretty}**")
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


def interpret_troubleshooting_outcome(
    user_text: str,
    *,
    issue_type: str = "",
    previous_outcome: str = "",
) -> Optional[str]:
    """
    Map clear free text to a terminal troubleshooting outcome.

    Bare yes/no is never mapped — those conflict across install/delivery/defect.
    """
    text = _normalize(user_text)
    if not text:
        return None
    if text in {"yes", "yeah", "yep", "yup", "no", "nope", "nah"}:
        return None

    at_outcome = (previous_outcome or "").strip().lower() == "steps_completed"
    issue = (issue_type or "").strip().lower()

    if not at_outcome:
        if _UNABLE_RE.search(text) or _NEED_TEAM_RE.search(text):
            return "unable_to_attempt"
        if _STEPS_DONE_RE.search(text):
            return "steps_completed"
        return None

    if issue == "delivery":
        if _DELIVERY_SUBMIT_RE.search(text) or _NEED_TEAM_RE.search(text):
            return "unresolved"
        if _COME_BACK_RE.search(text) or _ALL_SET_RE.search(text):
            return "resolved"
        return None

    if issue == "installation":
        if _NEED_TEAM_RE.search(text) or _STILL_BROKEN_RE.search(text):
            return "unresolved"
        if _SETUP_DONE_RE.search(text) or _ALL_SET_RE.search(text):
            return "resolved"
        return None

    if _NEED_TEAM_RE.search(text) or _STILL_BROKEN_RE.search(text):
        return "unresolved"
    if _WORKING_NOW_RE.search(text) or _ALL_SET_RE.search(text):
        return "resolved"
    return None
