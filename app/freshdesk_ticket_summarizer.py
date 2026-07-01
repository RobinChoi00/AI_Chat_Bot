"""
freshdesk_ticket_summarizer.py
==============================
Rescue Freshdesk tickets whose agent replies don't yield extractable steps.

The regex-based ``_extract_customer_steps`` in ``warranty_knowledge`` throws
out any ticket where the agent's reply is prose, multi-paragraph, missing
customer-action words, or contains PII markers. For a busy support inbox that
is often 70-90% of the pulled tickets — data we already paid Freshdesk API
calls to fetch, then dropped on the floor.

This module runs a small LLM (default ``gpt-4o-mini``, override via
``WARRANTY_FRESHDESK_SUMMARY_MODEL``) over those "salvageable" tickets and
produces 3-5 customer-safe steps + a short summary.

Design contract
---------------
- Feature-flagged: ``WARRANTY_FRESHDESK_LLM_SUMMARY`` (default on when
  OPENAI_API_KEY is set; explicit "0"/"off" disables).
- Cache by SHA-256 of ``subject|question|answer`` in a JSON sidecar
  (``data/freshdesk_summaries.json``) so re-syncing costs nothing on
  unchanged tickets.
- Steps must pass the same ``_is_customer_safe_step`` guard used by the
  regex path — the LLM never gets to inject PII, promises, or dispatch
  language into the knowledge base.
- Graceful no-op on any failure (missing API key, JSON parse error, empty
  output) — the loader falls back to the regex path unchanged.

Public API
----------
    summarize_ticket(subject, question, answer) -> Optional[SummarizedTicket]
    load_summary_cache() / save_summary_cache(...)
    summarize_missing_tickets(tickets, *, progress=None) -> stats dict
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = _PROJECT_ROOT / "data" / "freshdesk_summaries.json"

_MODEL_ENV = "WARRANTY_FRESHDESK_SUMMARY_MODEL"
_FLAG_ENV = "WARRANTY_FRESHDESK_LLM_SUMMARY"
_DEFAULT_MODEL = "gpt-4o-mini"

_MAX_ANSWER_CHARS = 6000
_MAX_TOKENS = int(os.environ.get("WARRANTY_FRESHDESK_SUMMARY_MAX_TOKENS", "500"))
_MAX_STEPS = 5
_MIN_STEPS = 1


_SYSTEM_PROMPT = (
    "You extract customer-safe troubleshooting steps from Osaki/Titan massage "
    "chair support tickets. The agent's reply may be long, informal, or "
    "contain internal notes.\n"
    "Return ONLY a compact JSON object with these keys:\n"
    '  "summary": one short sentence describing what the customer was asking '
    "(no promises, no dispatch language).\n"
    '  "category": one of "power", "remote", "air", "mech", "footrest", '
    '"heat", "voice", "misc", or "general". Pick the best fit.\n'
    '  "steps": array of 3-5 short imperative sentences the customer can '
    "safely do at home (verify, check, unplug, reconnect, toggle, "
    "reboot, etc.). NEVER include phrases like 'we will replace', 'send a "
    "tech', 'ship a part', 'refund', 'compensation', or anything that "
    "promises warranty action.\n"
    "  NEVER include phone numbers, addresses, order IDs, or customer PII.\n"
    "  Keep each step under 200 characters.\n"
    "If the reply has NO customer-safe DIY content (pure admin notes, only "
    'shipping updates, etc.) return {"summary":"","category":"general","steps":[]}.'
)


@dataclass
class SummarizedTicket:
    summary: str
    category: str
    steps: tuple[str, ...]
    model: str = ""
    created_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = list(self.steps)
        return d

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SummarizedTicket":
        return cls(
            summary=str(data.get("summary") or ""),
            category=str(data.get("category") or "general"),
            steps=tuple(str(s) for s in (data.get("steps") or []) if str(s).strip()),
            model=str(data.get("model") or ""),
            created_at=float(data.get("created_at") or 0.0),
        )


# ---------------------------------------------------------------------------
# Feature flag + model
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    flag = os.environ.get(_FLAG_ENV, "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _model_name() -> str:
    return os.environ.get(_MODEL_ENV, _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


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


# ---------------------------------------------------------------------------
# Content hashing + cache
# ---------------------------------------------------------------------------


def content_hash(subject: str, question: str, answer: str) -> str:
    """Stable key so a re-sync doesn't re-summarize identical tickets."""
    blob = "\u241f".join((subject or "").strip() for subject in (subject, question, answer))
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()


def load_summary_cache(path: Optional[Path] = None) -> dict[str, SummarizedTicket]:
    p = path or _CACHE_PATH
    if not p.is_file():
        return {}
    try:
        with p.open(encoding="utf-8") as handle:
            raw = json.load(handle) or {}
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        str(key): SummarizedTicket.from_json(value)
        for key, value in raw.items()
        if isinstance(value, dict)
    }


def save_summary_cache(
    cache: dict[str, SummarizedTicket], path: Optional[Path] = None
) -> None:
    p = path or _CACHE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value.to_json() for key, value in cache.items()}
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Safety filter — reuse warranty_knowledge guards
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _filter_safe_steps(candidates: Iterable[str]) -> tuple[str, ...]:
    try:
        from warranty_knowledge import _is_customer_safe_step  # type: ignore

        guard = _is_customer_safe_step
    except ImportError:  # pragma: no cover — module always resolvable in prod
        def guard(step: str) -> bool:
            return bool(step and 12 <= len(step) <= 220)

    out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        text = re.sub(r"^\s*\d+[\).\]]\s*", "", str(raw or "").strip())
        text = re.sub(r"^[-•\*]\s*", "", text).strip()
        if not text:
            continue
        if not guard(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text[0].upper() + text[1:] if text else text)
        if len(out) >= _MAX_STEPS:
            break
    return tuple(out)


def _parse_llm_payload(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...\n{tail}"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def summarize_ticket(
    subject: str,
    question: str,
    answer: str,
    *,
    client: Any = None,
) -> Optional[SummarizedTicket]:
    """
    Return a summarized ticket, or ``None`` when the LLM produced nothing
    usable (or the feature is disabled).
    """
    if not is_enabled() and client is None:
        return None

    q = _clip((question or "").strip(), _MAX_ANSWER_CHARS)
    a = _clip((answer or "").strip(), _MAX_ANSWER_CHARS)
    if not q and not a:
        return None

    user_content = (
        f"SUBJECT: {(subject or '').strip()}\n"
        f"CUSTOMER QUESTION:\n{q}\n\n"
        f"AGENT REPLY:\n{a}"
    )

    llm = client or _openai_client()
    if llm is None:
        return None

    try:
        completion = llm.chat.completions.create(
            model=_model_name(),
            max_tokens=_MAX_TOKENS,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = ""
        try:
            text = completion.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Freshdesk summarize failed: %s", exc)
        return None

    parsed = _parse_llm_payload(text)
    if not parsed:
        return None

    steps = _filter_safe_steps(parsed.get("steps") or [])
    if len(steps) < _MIN_STEPS:
        return None

    return SummarizedTicket(
        summary=str(parsed.get("summary") or "").strip(),
        category=str(parsed.get("category") or "general").strip().lower(),
        steps=steps,
        model=_model_name(),
        created_at=time.time(),
    )


def summarize_missing_tickets(
    tickets: Iterable[dict[str, Any]],
    *,
    cache: Optional[dict[str, SummarizedTicket]] = None,
    cache_path: Optional[Path] = None,
    client: Any = None,
    progress: Optional[Callable[[int, int], None]] = None,
    only_when_no_steps: bool = True,
    step_extractor: Optional[Callable[[str, str], tuple[str, ...]]] = None,
) -> dict[str, Any]:
    """
    Summarize every ticket whose agent reply doesn't yield customer steps.

    Parameters
    ----------
    tickets : iterable of dicts (as saved in ``freshdesk_tickets.json``)
    only_when_no_steps : if True (default), skip tickets whose reply already
        produces >=1 safe step via the regex path — saves LLM cost.
    step_extractor : optional callable ``(question, answer) -> tuple[str]``
        used to check whether the regex path would succeed. Defaults to the
        knowledge loader's ``_extract_customer_steps``.

    Returns
    -------
    dict with ``processed``, ``rescued``, ``skipped``, ``cached``, ``errors``.
    """
    if step_extractor is None:
        try:
            from warranty_knowledge import _extract_customer_steps  # type: ignore

            step_extractor = _extract_customer_steps
        except ImportError:  # pragma: no cover
            def step_extractor(_q: str, _a: str) -> tuple[str, ...]:
                return ()

    if cache is None:
        cache = load_summary_cache(cache_path)

    stats = {"processed": 0, "rescued": 0, "skipped": 0, "cached": 0, "errors": 0}
    tickets_list = list(tickets)
    total = len(tickets_list)
    llm = client or _openai_client()
    if llm is None and is_enabled():
        logger.warning("Freshdesk summarizer enabled but OpenAI client unavailable.")

    for idx, ticket in enumerate(tickets_list, start=1):
        subject = str(ticket.get("subject") or "")
        question = str(ticket.get("question") or "")
        answer = str(ticket.get("answer") or "")

        if only_when_no_steps:
            existing = step_extractor(question, answer)
            if existing:
                stats["skipped"] += 1
                if progress:
                    progress(idx, total)
                continue

        key = content_hash(subject, question, answer)
        if key in cache and cache[key].steps:
            stats["cached"] += 1
            if progress:
                progress(idx, total)
            continue

        summary = summarize_ticket(subject, question, answer, client=llm)
        stats["processed"] += 1
        if summary is None:
            stats["errors"] += 1
        else:
            cache[key] = summary
            stats["rescued"] += 1

        if progress:
            progress(idx, total)

    save_summary_cache(cache, cache_path)
    return stats
