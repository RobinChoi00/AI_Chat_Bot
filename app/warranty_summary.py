"""
AI-assisted case summaries for warranty team notifications.

Summaries are for operator readability only — they do NOT drive workflow
branching, DIY steps, or customer-facing repair promises.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

_SKIP_ANSWER_KEYS = frozenset({"warranty", "model_name"})


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
    return base[:120]


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
                        "Never add repair outcomes or promises."
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
        if re.search(
            r"\b(replace|refund|dispatch|send a tech|approved|will ship)\b",
            summary,
            re.I,
        ):
            return None
        subject = str(parsed.get("suggested_subject", "")).strip()[:120]
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
            subject = llm.get("suggested_subject") or suggested_subject_from_summary(
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
