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
  identically to today — except a small keyword prefill path may still
  advance obvious cases (e.g. footrest air) without the LLM.
- This module does NOT call WarrantyEngine itself; the caller (main.py route)
  does the actual submit_answer loop so the engine remains the single source
  of state truth.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_FREE_TEXT_LEN = 1000

_AIR_WORDS = (
    "air",
    "airbag",
    "air bag",
    "inflate",
    "inflating",
    "inflation",
    "no air",
    "won't inflate",
    "wont inflate",
    "not inflate",
    "doesn't inflate",
    "doesnt inflate",
)
_FOOTREST_WORDS = (
    "footrest",
    "foot rest",
    "legrest",
    "leg rest",
    "calf",
    "calves",
    "ottoman",
)
_EXTEND_WORDS = ("extend", "extension", "telescop", "won't extend", "wont extend")
_ROLLER_WORDS = ("roller", "rollers", "knead")
_POWER_WORDS = (
    "won't turn on",
    "wont turn on",
    "no power",
    "not turning on",
    "dead",
    "fuse",
    "back switch",
)
_REMOTE_WORDS = ("remote", "controller", "tablet", "screen blank")
_RECLINE_WORDS = ("recline", "zero g", "zero-g", "zerog", "lay flat")
_ROLLING_WORDS = ("massage head", "rollers stuck", "rolling", "kneading", "mechanism")
_HEAT_WORDS = ("heat", "heating", "won't heat", "wont heat", "too hot", "not warm")
_VOICE_WORDS = ("voice", "alexa", "hey osaki", "microphone", "ghost voice")
_COSMETIC_WORDS = ("scratch", "tear", "ripped", "cosmetic", "leather damage", "seam")
_DELIVERY_WORDS = (
    "delivery",
    "shipping",
    "tracking",
    "fedex",
    "ups",
    "carrier",
    "in transit",
)
_INSTALL_WORDS = (
    "install",
    "installation",
    "assembly",
    "assemble",
    "setup",
    "set up",
    "put together",
)

_DEFECT_TYPE_KEYS = frozenset(
    {
        "air",
        "cosmetic",
        "remote",
        "rolling",
        "power",
        "recline",
        "footrest",
        "heat",
        "voice",
    }
)

_ERROR_CODE_PHRASE_RE = re.compile(
    r"\b(?:error\s*)?code\s*(?:is|:|#|-)?\s*"
    r"(?:[A-Za-z]{0,3}\s*\d+(?:\.\d+)?|[A-Za-z]{1,4})\b",
    re.I,
)
_FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "on",
        "my",
        "is",
        "error",
        "code",
        "showing",
        "shows",
        "display",
        "screen",
        "chair",
        "model",
        "osaki",
        "titan",
        "please",
        "help",
        "with",
        "have",
        "getting",
        "got",
        "see",
        "sees",
        "says",
    }
)


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
    "6. If the customer mentions an error/code (e.g. \"hiro error code 68\", "
    "\"code C6\") WITHOUT also describing a clear symptom (won't turn on, "
    "footrest stuck, air not inflating, etc.), return only "
    'answer_keys: ["warranty"] (optionally with model_name). Do NOT invent '
    "power/air/remote/footrest/rolling/recline/heat/voice from an error "
    "code alone. Never write summaries like \"indicating a power-related "
    "defect\" unless the customer explicitly described that symptom.\n"
    "7. Each answer_key belongs to one node — keys are not interchangeable "
    "between nodes. Pick at most one key per node.\n"
    "8. Return JSON only, no prose, in the schema:\n"
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


def _fonz_entries_for_code(code: str) -> list[dict[str, Any]]:
    from fonz_warranty_data import load_error_code_records, normalize_error_code  # noqa: WPS433

    wanted = normalize_error_code(code)
    if not wanted:
        return []
    out: list[dict[str, Any]] = []
    for entry in load_error_code_records():
        if normalize_error_code(str(entry.get("error_code") or "")) != wanted:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def _score_fonz_model_match(text: str, entry: dict[str, Any]) -> int:
    """Score how well intake text mentions this Fonz model row."""
    text_l = (text or "").lower()
    text_compact = re.sub(r"[^a-z0-9]+", "", text_l)
    model = str(entry.get("model") or "").lower()
    model_key = str(entry.get("model_key") or "").lower()
    model_compact = re.sub(r"[^a-z0-9]+", "", model)
    key_compact = re.sub(r"[^a-z0-9]+", "", model_key)
    score = 0
    if model_compact and len(model_compact) >= 4 and model_compact in text_compact:
        score += 5
    if key_compact and len(key_compact) >= 4 and key_compact in text_compact:
        score += 5
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+", text_l)
        if len(t) >= 3 and t not in _FILLER_WORDS
    ]
    for token in tokens:
        if token in {"error", "code"}:
            continue
        if token in model or token in model_key.replace("-", " "):
            score += 3
        if key_compact.startswith(token) or model_compact.startswith(token):
            score += 2
        if token in key_compact or token in model_compact:
            score += 1
    return score


def _extract_catalog_model(text: str) -> str:
    """Resolve model via Shopify product catalog only (may be empty in prod)."""
    try:
        from product_catalog import resolve_model_name  # noqa: WPS433
    except ImportError:
        return ""

    from error_code_lookup import extract_error_codes_from_text  # noqa: WPS433
    from fonz_warranty_data import normalize_error_code  # noqa: WPS433

    codes = {
        normalize_error_code(c)
        for c in (extract_error_codes_from_text(text) or [])
        if c
    }
    cleaned = _ERROR_CODE_PHRASE_RE.sub(" ", text)
    for code in codes:
        if not code:
            continue
        cleaned = re.sub(rf"\b{re.escape(code)}\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    if cleaned:
        resolved = resolve_model_name(cleaned)
        if resolved:
            return resolved
    for word in re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", text):
        lower = word.lower()
        if lower in _FILLER_WORDS:
            continue
        if normalize_error_code(word) in codes:
            continue
        resolved = resolve_model_name(word)
        if resolved:
            return resolved
    return ""


def _match_fonz_entry_for_code(text: str, code: str) -> Optional[dict[str, Any]]:
    """Resolve model+code using catalog when possible, else Fonz model-name tokens."""
    from error_code_lookup import lookup_error_code  # noqa: WPS433

    catalog_model = _extract_catalog_model(text)
    if catalog_model:
        hit = lookup_error_code(catalog_model, code)
        if hit:
            return hit

    candidates = _fonz_entries_for_code(code)
    if not candidates:
        return None
    if len(candidates) == 1:
        return dict(candidates[0])

    scored = [(_score_fonz_model_match(text, entry), entry) for entry in candidates]
    scored = [(s, e) for s, e in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda row: (-row[0], -len(str(row[1].get("model") or ""))))
    best = scored[0][0]
    top = [e for s, e in scored if s == best]
    return dict(top[0])


def _extract_model_from_intake(text: str) -> str:
    """Best-effort chair model from mixed intake like 'hiro error code 68'."""
    catalog = _extract_catalog_model(text)
    if catalog:
        return catalog

    from error_code_lookup import extract_error_codes_from_text  # noqa: WPS433

    # Prod hosts may lack the Shopify catalog CSV — fall back to Fonz model names.
    for code in extract_error_codes_from_text(text) or []:
        hit = _match_fonz_entry_for_code(text, code)
        if hit:
            return str(hit.get("model") or "").strip()
    return ""


def _customer_safe_fonz_summary(entry: dict[str, Any], *, model_name: str) -> str:
    code = str(entry.get("error_code") or "").strip() or "?"
    model = (model_name or str(entry.get("model") or "your chair")).strip()
    meaning = str(entry.get("meaning") or "").strip()
    # First sentence only; drop internal part lists.
    short = meaning.split("\n")[0].strip()
    short = re.split(r"(?<=[.!?])\s+", short)[0].strip()
    if len(short) > 140:
        short = short[:137].rstrip() + "..."
    if short:
        return f"Error code {code} on {model}: {short}"
    return f"Customer reported error code {code} on {model}."


def _looks_like_model_and_code_only(text: str) -> bool:
    """True when the message is essentially model + error code, no symptom."""
    from error_code_lookup import extract_error_codes_from_text  # noqa: WPS433
    from fonz_warranty_data import normalize_error_code  # noqa: WPS433

    codes = {
        normalize_error_code(c)
        for c in (extract_error_codes_from_text(text) or [])
        if c
    }
    if not codes:
        return False
    remainder = _ERROR_CODE_PHRASE_RE.sub(" ", text)
    for code in codes:
        remainder = re.sub(rf"\b{re.escape(code)}\b", " ", remainder, flags=re.I)
    tokens = [
        t
        for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", remainder.lower())
        if t not in _FILLER_WORDS and len(t) > 1
    ]
    if not tokens:
        return True

    try:
        from product_catalog import resolve_model_name  # noqa: WPS433
    except ImportError:
        resolve_model_name = None  # type: ignore[assignment]

    for token in tokens:
        if resolve_model_name is not None and resolve_model_name(token):
            continue
        matched = False
        for code in codes:
            for entry in _fonz_entries_for_code(code):
                if _score_fonz_model_match(token, entry) > 0:
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue
        return False
    return True


def _fonz_prefill_from_text(text: str) -> Optional[dict[str, Any]]:
    """
    When intake mentions an error code, route from Fonz — never LLM guesswork.

    Returns a full prefill dict, or None if no usable Fonz hit / no code.
    """
    from error_code_lookup import (  # noqa: WPS433
        entry_workflow_category,
        extract_error_codes_from_text,
        knowledge_category_to_defect_key,
    )

    codes = extract_error_codes_from_text(text)
    if not codes:
        return None

    hit: Optional[dict[str, Any]] = None
    used_code = ""
    for code in codes:
        hit = _match_fonz_entry_for_code(text, code)
        if hit:
            used_code = code
            break

    model_name = _extract_model_from_intake(text)
    if hit and not model_name:
        model_name = str(hit.get("model") or "").strip()

    if not hit:
        # Known pattern: model + code only, but Fonz miss → still don't guess.
        if _looks_like_model_and_code_only(text):
            code_display = codes[0]
            model_display = model_name or "their chair"
            return {
                "answer_keys": ["warranty"],
                "model_name": model_name,
                "confidence": "high",
                "summary": (
                    f"Customer reported error code {code_display} on "
                    f"{model_display}."
                ),
                "source": "fonz_code_only",
            }
        return None

    category = entry_workflow_category(hit)
    defect_key = knowledge_category_to_defect_key(
        category,
        meaning=str(hit.get("meaning") or ""),
        troubleshooting=str(hit.get("troubleshooting") or ""),
    )
    summary = _customer_safe_fonz_summary(hit, model_name=model_name)
    answer_keys = ["warranty", "defect"]
    if defect_key in _DEFECT_TYPE_KEYS:
        answer_keys.append(defect_key)

    logger.info(
        "warranty_intake Fonz prefill code=%s model=%s category=%s defect_key=%s",
        used_code,
        model_name,
        category,
        defect_key,
    )
    return {
        "answer_keys": answer_keys,
        "model_name": model_name,
        "confidence": "high",
        "summary": summary,
        "source": "fonz",
    }


def _sanitize_llm_error_code_guess(
    *,
    text: str,
    answer_keys: list[str],
    summary: str,
) -> tuple[list[str], str]:
    """Drop invented defect-type keys when the message is only model + code."""
    from error_code_lookup import extract_error_codes_from_text  # noqa: WPS433

    if not extract_error_codes_from_text(text):
        return answer_keys, summary
    if not _looks_like_model_and_code_only(text):
        return answer_keys, summary

    kept = [k for k in answer_keys if k not in _DEFECT_TYPE_KEYS and k != "defect"]
    if not kept or kept[0] != "warranty":
        kept = ["warranty"] + [k for k in kept if k != "warranty"]
    model = _extract_model_from_intake(text)
    codes = extract_error_codes_from_text(text)
    code_display = codes[0] if codes else "?"
    model_display = model or "their chair"
    safe_summary = (
        f"Customer reported error code {code_display} on {model_display}."
    )
    return kept, safe_summary


def _norm_intake(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _normalize_air_footrest_keys(answer_keys: list[str]) -> list[str]:
    """
    Rewrite overloaded footrest/air sequences into the flowchart-safe path:
    warranty → defect → air → footrest
    """
    if not answer_keys:
        return answer_keys
    keys = list(answer_keys)
    if "defect" not in keys:
        return keys

    # Prefer air-location path when both air and footrest appear as defect siblings.
    if "air" in keys and "footrest" in keys:
        out: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key == "footrest" and "air" not in seen:
                # footrest before air → treat as air location after air
                continue
            if key not in seen:
                out.append(key)
                seen.add(key)
        if "air" in out and "footrest" not in out:
            # Insert footrest after air when it was dropped above.
            air_idx = out.index("air")
            out.insert(air_idx + 1, "footrest")
        return out

    # On mechanical footrest branch, map air-ish keys to air_not_inflating.
    if "footrest" in keys:
        mapped: list[str] = []
        for key in keys:
            if key in {"air", "air_not_inflating"}:
                mapped.append("air_not_inflating")
            else:
                mapped.append(key)
        # de-dupe while preserving order
        deduped: list[str] = []
        seen2: set[str] = set()
        for key in mapped:
            if key in seen2:
                continue
            deduped.append(key)
            seen2.add(key)
        return deduped

    return keys


def _keyword_workflow_prefill(text: str) -> Optional[dict[str, Any]]:
    """
    Deterministic high-confidence prefill for common symptom phrases.

    Returns None when the text is too ambiguous for a keyword path.
    """
    norm = _norm_intake(text)
    if not norm:
        return None

    has_air = _has_any(norm, _AIR_WORDS)
    has_foot = _has_any(norm, _FOOTREST_WORDS)
    has_extend = _has_any(norm, _EXTEND_WORDS)
    has_roller = _has_any(norm, _ROLLER_WORDS)

    # Footrest air is the highest-value / most ambiguous case.
    if has_air and has_foot:
        return {
            "answer_keys": ["warranty", "defect", "air", "footrest"],
            "model_name": _extract_model_from_intake(text),
            "confidence": "high",
            "summary": "Footrest air not inflating.",
            "source": "keyword",
        }

    if has_air and not has_foot:
        return {
            "answer_keys": ["warranty", "defect", "air"],
            "model_name": _extract_model_from_intake(text),
            "confidence": "high",
            "summary": "Air inflation issue.",
            "source": "keyword",
        }

    if has_foot and has_extend and not has_air:
        return {
            "answer_keys": ["warranty", "defect", "footrest"],
            "model_name": _extract_model_from_intake(text),
            "confidence": "high",
            "summary": "Footrest extension issue.",
            "source": "keyword",
        }

    if has_foot and has_roller and not has_air:
        return {
            "answer_keys": ["warranty", "defect", "footrest"],
            "model_name": _extract_model_from_intake(text),
            "confidence": "high",
            "summary": "Footrest roller issue.",
            "source": "keyword",
        }

    if has_foot and not has_air:
        return {
            "answer_keys": ["warranty", "defect", "footrest"],
            "model_name": _extract_model_from_intake(text),
            "confidence": "high",
            "summary": "Footrest issue.",
            "source": "keyword",
        }

    defect_checks: list[tuple[tuple[str, ...], str, str]] = [
        (_POWER_WORDS, "power", "Power issue."),
        (_REMOTE_WORDS, "remote", "Remote / controller issue."),
        (_RECLINE_WORDS, "recline", "Recline issue."),
        (_ROLLING_WORDS, "rolling", "Massage mechanism issue."),
        (_HEAT_WORDS, "heat", "Heating issue."),
        (_VOICE_WORDS, "voice", "Voice control issue."),
        (_COSMETIC_WORDS, "cosmetic", "Cosmetic damage."),
    ]
    for words, key, summary in defect_checks:
        if _has_any(norm, words):
            return {
                "answer_keys": ["warranty", "defect", key],
                "model_name": _extract_model_from_intake(text),
                "confidence": "high",
                "summary": summary,
                "source": "keyword",
            }

    # Issue-type only (no defect category) — still better than landing cold.
    if _has_any(norm, _DELIVERY_WORDS) and not _has_any(norm, _INSTALL_WORDS):
        return {
            "answer_keys": ["warranty", "delivery"],
            "model_name": _extract_model_from_intake(text),
            "confidence": "high",
            "summary": "Delivery / shipping help.",
            "source": "keyword",
        }
    if _has_any(norm, _INSTALL_WORDS) and not _has_any(norm, _DELIVERY_WORDS):
        return {
            "answer_keys": ["warranty", "installation"],
            "model_name": _extract_model_from_intake(text),
            "confidence": "high",
            "summary": "Setup / installation help.",
            "source": "keyword",
        }

    return None


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
        "source": "llm" | "fonz" | "keyword" | "empty",
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

    # Error-code intake: Fonz lookup beats LLM guessing (Hiro 68 ≠ power).
    fonz_hit = _fonz_prefill_from_text(text)
    if fonz_hit is not None:
        return fonz_hit

    # Deterministic keyword path before LLM — works offline / when API fails.
    keyword_hit = _keyword_workflow_prefill(text)
    if keyword_hit is not None:
        return keyword_hit

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
    answer_keys, summary = _sanitize_llm_error_code_guess(
        text=text,
        answer_keys=answer_keys,
        summary=summary,
    )
    if not answer_keys:
        return empty

    answer_keys = _normalize_air_footrest_keys(answer_keys)

    return {
        "answer_keys": answer_keys,
        "model_name": model_name or _extract_model_from_intake(text),
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
