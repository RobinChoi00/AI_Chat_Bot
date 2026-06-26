"""
AI-assisted case summaries for warranty team notifications.

Summaries are for operator readability only — they do NOT drive workflow
branching, DIY steps, or customer-facing repair promises.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

_SKIP_ANSWER_KEYS = frozenset({"warranty", "model_name"})

_PROMISE_RE = re.compile(
    r"\b("
    r"replace(?:ment|d)?|refund(?:ed|s)?|dispatch(?:ed)?|send a tech|"
    r"technician|approved|will ship|ship(?:ped|ping)?|repair or replace|"
    r"compensation|free part|we will send|parts will be"
    r")\b",
    re.I,
)

_SUMMARY_STOPWORDS = frozenset({
    "about", "after", "before", "being", "been", "chair", "could",
    "customer", "during", "issue", "model", "noted", "problem",
    "reported", "reports", "stated", "states", "their", "them",
    "there", "these", "those", "through", "warranty", "which",
    "while", "would",
})

_MIN_FACT_TOKEN_LEN = 5
_MIN_FACT_MATCH_RATIO = 0.6


def contains_promise_language(text: str) -> bool:
    """True when text includes repair/dispatch/refund style promises."""
    return bool(_PROMISE_RE.search(text or ""))


def _turn_field(turn: Any, name: str) -> str:
    value = getattr(turn, name, None)
    if value is None and isinstance(turn, dict):
        value = turn.get(name)
    return str(value or "").strip()


def _format_turns_for_prompt(turns: Sequence[Any]) -> str:
    lines: list[str] = []
    for turn in turns or []:
        node_id = _turn_field(turn, "node_id")
        prompt = _turn_field(turn, "node_prompt")
        answer_key = _turn_field(turn, "answer_key")
        answer = _turn_field(turn, "customer_answer")
        if not any((node_id, prompt, answer_key, answer)):
            continue
        chunk = f"[{node_id}]"
        if prompt:
            chunk += f" Q: {prompt}"
        if answer_key:
            chunk += f" (key={answer_key})"
        if answer:
            chunk += f" A: {answer}"
        lines.append(chunk)
    return "\n".join(lines)


def build_transcript_corpus(
    *,
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
) -> str:
    """Lowercased text of all workflow facts available to the summarizer."""
    chunks: list[str] = [
        issue_type.replace("_", " "),
        model_name,
        terminal_node_id.replace("_", " "),
    ]
    for turn in turns or []:
        chunks.extend(
            [
                _turn_field(turn, "node_id").replace("_", " "),
                _turn_field(turn, "node_prompt"),
                _turn_field(turn, "answer_key").replace("_", " "),
                _turn_field(turn, "customer_answer"),
            ]
        )
    return " ".join(part for part in chunks if part).lower()


def _summary_fact_tokens(summary: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[a-z0-9][a-z0-9-]{3,}", (summary or "").lower()):
        token = raw.strip("-")
        if len(token) < 4:
            continue
        if token in _SUMMARY_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def validate_llm_summary_against_transcript(
    *,
    summary: str,
    suggested_subject: str,
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
) -> tuple[bool, str]:
    """
    Reject LLM output that promises service or cites facts outside the transcript.

    Returns (ok, reject_reason).
    """
    summary = (summary or "").strip()
    suggested_subject = (suggested_subject or "").strip()

    if contains_promise_language(summary):
        return False, "promise_in_summary"
    if contains_promise_language(suggested_subject):
        return False, "promise_in_subject"

    model = (model_name or "").strip()
    if len(model) >= 3:
        model_lower = model.lower()
        if model_lower not in summary.lower() and model_lower not in suggested_subject.lower():
            return False, "model_not_in_output"

    corpus = build_transcript_corpus(
        issue_type=issue_type,
        model_name=model_name,
        turns=turns,
        terminal_node_id=terminal_node_id,
    )
    if not corpus.strip():
        return False, "empty_transcript"

    fact_tokens = _summary_fact_tokens(summary)
    if not fact_tokens:
        return False, "no_fact_tokens"

    matched = sum(1 for token in fact_tokens if token in corpus)
    required = max(1, math.ceil(len(fact_tokens) * _MIN_FACT_MATCH_RATIO))
    if matched < required:
        return False, "facts_not_in_transcript"

    return True, ""


def sanitize_email_subject(subject: str, *, fallback: str) -> str:
    """Drop subjects that contain promise language."""
    cleaned = (subject or "").strip()
    if not cleaned or contains_promise_language(cleaned):
        return fallback
    return cleaned[:120]


def format_case_summary_section_header(source: str) -> str:
    if source == "llm":
        return "--- Case summary (AI-generated — verify workflow below) ---"
    if source == "provided":
        return "--- Case summary ---"
    return "--- Case summary (from workflow) ---"


def format_case_summary_for_email(summary: str, source: str) -> str:
    """Format summary body for team email (disclaimer only for LLM source)."""
    text = (summary or "").strip()
    if source == "llm":
        return f"{text}\n\n(AI-generated — verify workflow below.)"
    return text


def build_deterministic_case_summary(
    *,
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
) -> str:
    """Rule-based summary when LLM is unavailable or low confidence."""
    parts: list[str] = []
    if model_name:
        parts.append(f"Model: {model_name}.")
    if issue_type:
        parts.append(f"Issue type: {issue_type}.")

    keys: list[str] = []
    notes: list[str] = []
    for turn in turns or []:
        key = _turn_field(turn, "answer_key")
        if key and key not in _SKIP_ANSWER_KEYS:
            keys.append(key)
        answer = _turn_field(turn, "customer_answer")
        if answer and "@" not in answer and len(answer) > 4:
            notes.append(answer[:160])

    if keys:
        parts.append("Path: " + " → ".join(keys[-8:]) + ".")
    if notes:
        parts.append(f'Latest customer note: "{notes[-1]}".')
    if terminal_node_id:
        parts.append(f"Terminal node: {terminal_node_id}.")

    text = " ".join(parts).strip()
    return text or "Warranty chat completed; see workflow steps below for details."


def suggested_subject_from_summary(
    *,
    issue_type: str,
    model_name: str,
    summary: str,
    ticket_id: str = "",
) -> str:
    model = (model_name or "Chair").strip()
    issue = (issue_type or "warranty").strip().replace("_", " ")
    first_sentence = summary.split(".")[0].strip()
    if len(first_sentence) > 72:
        first_sentence = first_sentence[:69] + "..."
    if first_sentence and first_sentence.lower() not in ("warranty chat completed",):
        base = f"{model} — {first_sentence}"
    else:
        base = f"{model} — {issue} case"
    if ticket_id:
        base = f"{base} ({ticket_id})"
    return sanitize_email_subject(base, fallback=f"{model} — {issue} case")[:120]


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


def _llm_case_summary(
    *,
    issue_type: str,
    model_name: str,
    turns: Sequence[Any],
    terminal_node_id: str,
) -> Optional[dict[str, str]]:
    client = _openai_client()
    if client is None:
        return None

    from config import ROUTER_MODEL

    transcript = _format_turns_for_prompt(turns)
    if not transcript.strip():
        return None

    prompt = (
        "Write a brief internal case summary for the warranty support team.\n\n"
        f"Issue type: {issue_type or 'unknown'}\n"
        f"Model: {model_name or 'unknown'}\n"
        f"Terminal node: {terminal_node_id or 'unknown'}\n\n"
        "Workflow transcript:\n"
        f"{transcript}\n\n"
        "Rules:\n"
        "- 2 to 3 complete sentences in English.\n"
        "- Include model, symptom area, and key customer answers.\n"
        "- Do NOT promise repair, replacement, parts, refunds, or technician dispatch.\n"
        "- Do NOT invent facts not present in the transcript.\n"
        "- suggested_subject: one short line (max 80 chars) suitable for a ticket subject.\n"
        'Return JSON: {"summary":"...","suggested_subject":"...","confidence":"high|low"}'
    )

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize warranty chat transcripts for internal support staff only. "
                        "Never add repair outcomes or promises. Use only facts from the transcript."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        if str(parsed.get("confidence", "low")).strip().lower() != "high":
            return None
        summary = str(parsed.get("summary", "")).strip()
        if len(summary) < 20:
            return None
        subject = str(parsed.get("suggested_subject", "")).strip()[:120]

        ok, reason = validate_llm_summary_against_transcript(
            summary=summary,
            suggested_subject=subject,
            issue_type=issue_type,
            model_name=model_name,
            turns=turns,
            terminal_node_id=terminal_node_id,
        )
        if not ok:
            logger.info("warranty_summary LLM output rejected: %s", reason)
            return None

        return {"summary": summary, "suggested_subject": subject}
    except Exception as exc:
        logger.warning("warranty_summary LLM call failed: %s", exc)
        return None


def summarize_warranty_case(
    *,
    issue_type: str = "",
    model_name: str = "",
    turns: Optional[Sequence[Any]] = None,
    terminal_node_id: str = "",
    use_llm: bool = True,
) -> dict[str, str]:
    """
    Build a team-facing case summary.

    Returns dict with keys: summary, suggested_subject, source (llm|deterministic).
    """
    turns = turns or []
    fallback = build_deterministic_case_summary(
        issue_type=issue_type,
        model_name=model_name,
        turns=turns,
        terminal_node_id=terminal_node_id,
    )
    subject_fallback = suggested_subject_from_summary(
        issue_type=issue_type,
        model_name=model_name,
        summary=fallback,
    )

    if use_llm:
        llm = _llm_case_summary(
            issue_type=issue_type,
            model_name=model_name,
            turns=turns,
            terminal_node_id=terminal_node_id,
        )
        if llm:
            summary = llm["summary"]
            subject = sanitize_email_subject(
                llm.get("suggested_subject") or "",
                fallback=subject_fallback,
            )
            if subject == subject_fallback and llm.get("suggested_subject"):
                subject = suggested_subject_from_summary(
                    issue_type=issue_type,
                    model_name=model_name,
                    summary=summary,
                )
            return {
                "summary": summary,
                "suggested_subject": subject,
                "source": "llm",
            }

    return {
        "summary": fallback,
        "suggested_subject": subject_fallback,
        "source": "deterministic",
    }
