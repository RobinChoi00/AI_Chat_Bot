"""
Free-text intake for the warranty workflow.

The customer can describe their issue in one line of free text at the start of
the warranty session. This module turns that free text into a sequence of
flowchart answer_keys, which the WarrantyEngine then auto-submits to fast-
forward through the multiple-choice questions.

Design contract
---------------
- LLM picks ONLY from valid answer_keys present in the live flowchart — no
  free invention. Each candidate is re-validated against the engine before
  being submitted.
- We never auto-submit `question_text` answers (model name, tracking number,
  email/order, etc.) — those are PII / specific data and the customer must
  type them explicitly.
- Confidence gate: only "high" confidence answers are auto-submitted. Anything
  lower is dropped silently (workflow continues as a normal multiple-choice).
- Graceful no-op: if no OPENAI_API_KEY, malformed JSON, or zero high-confidence
  picks, this function returns an empty extraction and the workflow behaves
  identically to today.
- This module does NOT call WarrantyEngine itself; the caller (main.py route)
  does the actual submit_answer loop so the engine remains the single source
  of state truth.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_FREE_TEXT_LEN = 1000


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


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


def _router_model() -> str:
    try:
        from config import ROUTER_MODEL  # type: ignore
        if ROUTER_MODEL:
            return ROUTER_MODEL
    except Exception:
        pass
    return os.environ.get("OPENAI_ROUTER_MODEL", "gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Flowchart helpers
# ---------------------------------------------------------------------------


def _collect_choice_keys(nodes: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build a flat list of {answer_key, node_id, label, prompt} for every
    *choice* (question) option in the flowchart. We exclude question_text
    nodes — those need user-typed values, not picklist matching.
    """
    catalog: list[dict[str, str]] = []
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("type") != "question":
            continue
        prompt = str(node.get("prompt") or "")
        for opt in node.get("options", []) or []:
            key = str(opt.get("answer_key") or "")
            label = str(opt.get("label") or "")
            if not key:
                continue
            catalog.append(
                {
                    "answer_key": key,
                    "node_id": node_id,
                    "label": label,
                    "prompt": prompt,
                }
            )
    return catalog


def _format_catalog_for_prompt(catalog: list[dict[str, str]]) -> str:
    """Compact, one-line-per-option, grouped by node."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in catalog:
        grouped.setdefault(row["node_id"], []).append(row)

    lines: list[str] = []
    for node_id, rows in grouped.items():
        prompt = rows[0]["prompt"]
        lines.append(f"\n[node:{node_id}] {prompt}")
        for r in rows:
            lines.append(f"  - {r['answer_key']}: {r['label']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a strict intake classifier for an Osaki/Titan massage chair "
    "warranty workflow. You translate the customer's free-text description "
    "into a SEQUENCE of allowed answer_keys from the flowchart picklist "
    "below.\n\n"
    "Rules:\n"
    "1. Pick ONLY from the answer_keys provided. Never invent new keys.\n"
    "2. The first key MUST be 'warranty' (this is a warranty bot).\n"
    "3. After 'warranty', include an issue_type key ('installation', "
    "'delivery', or 'defect') ONLY when the customer's words clearly fit "
    "one category. If unclear, stop after 'warranty' — do NOT guess defect.\n"
    "4. After that, only include further answer_keys you are clearly "
    "confident about from the customer's words. It is OK and preferred to "
    "stop early; the workflow will ask the missing questions normally.\n"
    "5. If the customer message is ONLY a chair model name with NO issue "
    "description (e.g. \"Maestro\", \"OS-4000T\"), return "
    'answer_keys: ["warranty"], set model_name, confidence: high, and stop '
    "— do NOT pick defect/air/power or any symptom branch.\n"
    "6. Each answer_key belongs to one node — keys are not interchangeable "
    "between nodes. Pick at most one key per node.\n"
    "7. Return JSON only, no prose, in the schema:\n"
    "{\n"
    '  "answer_keys": ["warranty", "defect", "air", "footrest"],\n'
    '  "model_name": "OS-4000T" or null,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "summary": "one short English sentence describing what you understood"\n'
    "}\n"
    "Use confidence=high ONLY when the customer's words clearly map to the "
    "picked answer_keys. If you are guessing, use medium or low — the caller "
    "will then drop the extraction."
)


def extract_workflow_prefill(
    *,
    free_text: str,
    nodes: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a dict like:
      {
        "answer_keys": ["warranty", "defect", "air", "footrest"],
        "model_name": "OS-4000T" or "",
        "confidence": "high",
        "summary": "Footrest air not inflating on OS-4000T.",
        "source": "llm" | "empty",
      }
    On any failure or low-confidence result, returns an "empty" dict with
    answer_keys=[] so the caller falls back to the normal flow.
    """
    empty: dict[str, Any] = {
        "answer_keys": [],
        "model_name": "",
        "confidence": "low",
        "summary": "",
        "source": "empty",
    }

    text = (free_text or "").strip()
    if not text:
        return empty
    if len(text) > _MAX_FREE_TEXT_LEN:
        text = text[:_MAX_FREE_TEXT_LEN]

    try:
        from product_catalog import looks_like_model_only  # noqa: WPS433

        model_only = looks_like_model_only(text)
    except ImportError:
        model_only = None

    if model_only:
        return {
            "answer_keys": ["warranty"],
            "model_name": model_only,
            "confidence": "high",
            "summary": f"Chair model: {model_only}.",
            "source": "model_only",
        }

    client = _openai_client()
    if client is None:
        return empty

    catalog = _collect_choice_keys(nodes)
    if not catalog:
        return empty
    valid_keys = {r["answer_key"] for r in catalog}
    valid_keys.add("warranty")

    user_msg = (
        f"Customer message:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Allowed answer_keys grouped by question (pick at most one per node):\n"
        f"{_format_catalog_for_prompt(catalog)}\n\n"
        "Now produce the JSON described in the system message. "
        "Stop the answer_keys list early if you are not confident about the "
        "next step."
    )

    try:
        response = client.chat.completions.create(
            model=_router_model(),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as exc:
        logger.warning("warranty_intake LLM call failed: %s", exc)
        return empty

    content = (response.choices[0].message.content or "").strip()
    if not content:
        return empty

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("warranty_intake JSON decode failed: %s", exc)
        return empty
    if not isinstance(parsed, dict):
        return empty

    confidence = str(parsed.get("confidence", "low")).strip().lower()
    raw_keys = parsed.get("answer_keys") or []
    if not isinstance(raw_keys, list):
        raw_keys = []

    answer_keys: list[str] = []
    for k in raw_keys:
        key = str(k).strip()
        if not key:
            continue
        if key not in valid_keys:
            logger.info("warranty_intake dropping unknown key=%s", key)
            continue
        answer_keys.append(key)

    if confidence != "high" or not answer_keys:
        return empty
    if answer_keys[0] != "warranty":
        answer_keys.insert(0, "warranty")

    model_name = str(parsed.get("model_name") or "").strip()
    if model_name.lower() in ("null", "none", "unknown", "n/a"):
        model_name = ""

    summary = str(parsed.get("summary") or "").strip()

    return {
        "answer_keys": answer_keys,
        "model_name": model_name,
        "confidence": confidence,
        "summary": summary,
        "source": "llm",
    }


# ---------------------------------------------------------------------------
# Apply extracted keys against a live engine
# ---------------------------------------------------------------------------


def apply_prefill_to_engine(
    *,
    engine,
    ticket_id: str,
    nodes: dict[str, Any],
    answer_keys: list[str],
) -> dict[str, Any]:
    """
    Walk the engine forward, submitting each answer_key in order — but only
    when the current node is a 'question' that actually offers that key as a
    valid option. The moment a key doesn't fit, we stop (the workflow then
    asks that question normally, which is the safe default).

    Returns a dict:
      {
        "applied": ["warranty", "defect", "air"],
        "skipped": ["footrest"],
        "stopped_reason": "no_match" | "question_text" | "terminal" | "done",
        "final_node": <last engine state>,
      }
    """
    applied: list[str] = []
    skipped: list[str] = []
    stopped_reason = "done"
    final_node: Optional[dict[str, Any]] = engine.get_current_node(ticket_id)

    for key in answer_keys:
        current = engine.get_current_node(ticket_id)
        if not current:
            stopped_reason = "no_match"
            break
        if current.get("type") == "terminal":
            stopped_reason = "terminal"
            break
        if current.get("type") != "question":
            stopped_reason = "question_text"
            break
        options = current.get("options") or []
        valid_keys_here = {str(o.get("answer_key") or "") for o in options}
        if key not in valid_keys_here:
            skipped.append(key)
            stopped_reason = "no_match"
            break
        try:
            result = engine.submit_answer(ticket_id, key)
        except Exception as exc:
            logger.info("warranty_intake submit_answer failed at %s: %s", key, exc)
            stopped_reason = "no_match"
            break
        applied.append(key)
        final_node = result.get("next_node") or engine.get_current_node(ticket_id)
        if result.get("is_terminal"):
            stopped_reason = "terminal"
            break

    if not final_node:
        final_node = engine.get_current_node(ticket_id)

    return {
        "applied": applied,
        "skipped": skipped,
        "stopped_reason": stopped_reason,
        "final_node": final_node,
    }
