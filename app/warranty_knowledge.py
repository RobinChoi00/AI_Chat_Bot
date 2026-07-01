"""
Unified warranty troubleshooting knowledge base.

Sources (merged at load time):
  - raw_data/Warranty Daily Report - Q&A.csv
  - data/freshdesk_tickets.json  (Freshdesk API ETL output)
  - raw_data/Auto-Check.csv      (error-code troubleshooting steps)

Used by the warranty workflow to suggest customer-safe steps BEFORE asking for email.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_QA_PATH = _PROJECT_ROOT / "raw_data" / "Warranty Daily Report - Q&A.csv"
_FRESHDESK_PATH = _PROJECT_ROOT / "data" / "freshdesk_tickets.json"
_FRESHDESK_KB_PATH = _PROJECT_ROOT / "data" / "freshdesk_solutions.json"
_AUTOCHECK_PATH = _PROJECT_ROOT / "raw_data" / "Auto-Check.csv"

_DEFECT_CATEGORY_MAP: dict[str, str] = {
    "power": "power",
    "remote": "remote",
    "air": "air",
    "rolling": "mech",
    "recline": "mech",
    "footrest": "footrest",
    "cosmetic": "misc",
    "heat": "heat",
    "voice": "voice",
}

_QA_CATEGORY_MAP: dict[str, str] = {
    "Category - Power": "power",
    "Category - Remote": "remote",
    "Category - Air": "air",
    "Category - Mech": "mech",
    "Category - Footrest": "footrest",
    "Category - Heat": "heat",
    "Category - Misc.": "misc",
}

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "power": ("power", "fuse", "outlet", "plug", "clicking", "turn on", "pcb", "surge", "switch"),
    "remote": ("remote", "tablet", "controller", "bluetooth", "pair"),
    "air": ("air", "airbag", "inflate", "hissing", "compressor", "footrest hose", "no air"),
    "mech": ("mech", "roller", "massage", "rolling", "knead", "track"),
    "footrest": ("footrest", "calf", "leg rest"),
    "heat": ("heat", "heated", "warming"),
    "misc": ("cosmetic", "voice", "speaker", "software", "bluetooth"),
    "voice": ("voice", "command", "microphone", "mic", "alexa", "hey", "ghost", "false trigger", "random voice"),
}

_INTERNAL_MARKERS = (
    "replace ",
    "send tech",
    "dispatch",
    "send replacement",
    "send a tech",
    "admin",
    "change main pcb",
    "change the main pcb",
    "ship ",
    "warranty team will arrange",
)

_PII_OR_ADMIN_MARKERS = (
    "customer address",
    "phone number",
    "description of issue:",
    "information we need",
    "proof of purchase",
    "serial number:",
    "order number",
    "customer name:",
    "place of purchase",
    "service location:",
    "tracking id",
    "merged into ticket",
    "your ticket case #",
    "original message",
    "sent from my",
    " wrote:",
    "http://",
    "https://",
    "888-848-2630",
    "8888482630",
    "ota world",
    "non-refundable",
    "warranty agreement",
    "note: please",
    "ticket #",
    "fedex (tracking",
)

_BOILERPLATE_MARKERS = (
    "qualified technician inspect",
    "we strongly suggest having",
    "with that being said",
    "we always recommend having",
    "parts purchased from our service",
    "unauthorized sources",
    "facebook marketplace",
    "courtesy email to inform",
)

_MAX_CUSTOMER_STEP_LEN = 220
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

_CUSTOMER_ACTION_WORDS = (
    "check",
    "verify",
    "ensure",
    "reconnect",
    "unplug",
    "toggle",
    "test",
    "remove",
    "inspect",
    "try",
    "confirm",
    "look for",
    "make sure",
    "power cycle",
    "press and hold",
    "disconnect",
)


@dataclass(frozen=True)
class KnowledgeEntry:
    source: str
    category: str
    title: str
    diagnostic: str
    customer_steps: tuple[str, ...]


def map_workflow_defect_category(defect_category: Optional[str]) -> Optional[str]:
    """Map workflow answer_key (e.g. rolling) to knowledge category (e.g. mech)."""
    if not defect_category:
        return None
    return _DEFECT_CATEGORY_MAP.get(defect_category.lower(), defect_category)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _normalize(text)) if len(t) > 2}


def _infer_category(text: str) -> str:
    lower = _normalize(text)
    best = "general"
    best_score = 0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best = cat
    return best if best_score > 0 else "general"


def _is_internal(text: str) -> bool:
    lower = (text or "").lower()
    return any(marker in lower for marker in _INTERNAL_MARKERS)


def _is_customer_safe_step(text: str) -> bool:
    chunk = (text or "").strip()
    if len(chunk) < 12 or len(chunk) > _MAX_CUSTOMER_STEP_LEN:
        return False
    if _is_internal(chunk):
        return False
    lower = chunk.lower()
    if any(marker in lower for marker in _PII_OR_ADMIN_MARKERS):
        return False
    if any(marker in lower for marker in _BOILERPLATE_MARKERS):
        return False
    if _PHONE_RE.search(chunk) or _ZIP_RE.search(chunk) or _EMAIL_RE.search(chunk):
        return False
    if not any(word in lower for word in _CUSTOMER_ACTION_WORDS):
        return False
    return True


def _clean_freshdesk_title(subject: str, question: str = "") -> str:
    raw = (subject or question or "").strip()
    raw = re.sub(r"^(?:reopened:\s*)+", "", raw, flags=re.I)
    raw = re.sub(r"^(?:re|fwd):\s*", "", raw, flags=re.I).strip()
    raw = re.sub(r"^ticket\s+#\d+\s*--\s*", "", raw, flags=re.I).strip()
    if len(raw) > 80:
        raw = raw[:77] + "..."
    return raw or "Support case"


def _is_usable_freshdesk_answer(answer: str) -> bool:
    lower = (answer or "").lower().strip()
    if len(lower) < 20:
        return False
    if "merged into ticket" in lower:
        return False
    if lower.startswith("this ticket is closed"):
        return False
    if lower.startswith("expedite shipping"):
        return False
    return True


def _extract_customer_steps(*texts: str) -> tuple[str, ...]:
    steps: list[str] = []
    seen: set[str] = set()
    for blob in texts:
        if not blob:
            continue
        for chunk in re.split(r"[\n.;]+", blob):
            chunk = chunk.strip()
            chunk = re.sub(r"^\d+[\).\]]\s*", "", chunk)
            if not _is_customer_safe_step(chunk):
                continue
            key = _normalize(chunk)
            if key in seen:
                continue
            seen.add(key)
            steps.append(chunk[0].upper() + chunk[1:] if chunk else chunk)
    return tuple(steps[:4])


def _load_qa_entries() -> list[KnowledgeEntry]:
    if not _QA_PATH.is_file():
        return []

    entries: list[KnowledgeEntry] = []
    current_qa_category = ""

    with _QA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            col0 = (row[0] if len(row) > 0 else "").strip()
            if col0.startswith("Category -"):
                current_qa_category = col0
                continue
            if not col0 or col0.lower() in {"n/a", "nan"}:
                continue
            diagnostic = (row[1] if len(row) > 1 else "").strip()
            solution = (row[2] if len(row) > 2 else "").strip()
            steps = _extract_customer_steps(diagnostic, solution)
            if not steps and not diagnostic:
                continue
            category = _QA_CATEGORY_MAP.get(current_qa_category, "general")
            if "voice control" in col0.lower() or col0.lower().startswith("voice "):
                category = "voice"
            entries.append(
                KnowledgeEntry(
                    source="qa_csv",
                    category=category,
                    title=col0,
                    diagnostic=diagnostic,
                    customer_steps=steps or (diagnostic,) if diagnostic else (),
                )
            )
    return entries


def _load_freshdesk_summary_cache() -> dict:
    """
    Return the LLM-rescued Freshdesk summaries produced by
    ``freshdesk_ticket_summarizer.summarize_missing_tickets``.

    Loaded lazily so that a stale cache never breaks knowledge loading.
    """
    try:
        from freshdesk_ticket_summarizer import content_hash, load_summary_cache  # type: ignore
    except ImportError:
        return {}
    try:
        return load_summary_cache()
    except Exception:  # noqa: BLE001
        return {}


def _load_freshdesk_entries() -> list[KnowledgeEntry]:
    if not _FRESHDESK_PATH.is_file():
        return []

    try:
        with _FRESHDESK_PATH.open(encoding="utf-8") as handle:
            tickets = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    # Optional LLM-rescued steps for tickets whose regex extraction failed.
    summaries = _load_freshdesk_summary_cache()
    try:
        from freshdesk_ticket_summarizer import content_hash  # type: ignore
    except ImportError:
        content_hash = None  # type: ignore[assignment]

    entries: list[KnowledgeEntry] = []
    for ticket in tickets:
        question = str(ticket.get("question") or "").strip()
        answer = str(ticket.get("answer") or "").strip()
        subject = str(ticket.get("subject") or "").strip()
        if not _is_usable_freshdesk_answer(answer):
            continue
        if not question and not subject:
            continue
        blob = f"{subject} {question}"
        steps = _extract_customer_steps(answer)
        category_hint: str | None = None
        summary_diagnostic: str = ""
        # ``freshdesk`` regardless of rescue path so downstream filters that
        # already look for ``source == "freshdesk"`` keep working; the LLM
        # rescue only shows up in sync stats via the summary cache size.
        source = "freshdesk"

        if not steps and summaries and content_hash is not None:
            key = content_hash(subject, question, answer)
            cached = summaries.get(key)
            if cached and cached.steps:
                steps = tuple(cached.steps)
                category_hint = (cached.category or "").strip().lower() or None
                summary_diagnostic = cached.summary

        if not steps:
            continue

        category = category_hint or _infer_category(blob)
        title = _clean_freshdesk_title(subject, question)
        diagnostic_text = summary_diagnostic or question
        entries.append(
            KnowledgeEntry(
                source=source,
                category=category,
                title=title,
                diagnostic=diagnostic_text[:300],
                customer_steps=steps,
            )
        )
    return entries


def _load_autocheck_entries() -> list[KnowledgeEntry]:
    if not _AUTOCHECK_PATH.is_file():
        return []

    entries: list[KnowledgeEntry] = []
    with _AUTOCHECK_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            code = (row[0] or "").strip()
            phenomenon = (row[1] or "").strip()
            description = (row[2] or "").strip()
            troubleshooting = (row[3] or "").strip()
            if not code or code.lower() in {"code no.", "nan"}:
                continue
            steps = _extract_customer_steps(troubleshooting, description)
            if not steps:
                continue
            blob = f"{phenomenon} {description}"
            entries.append(
                KnowledgeEntry(
                    source="auto_check",
                    category=_infer_category(blob),
                    title=phenomenon or f"Error {code}",
                    diagnostic=description[:300],
                    customer_steps=steps,
                )
            )
    return entries


def _load_freshdesk_kb_entries() -> list[KnowledgeEntry]:
    """
    Load Freshdesk Solutions (Knowledge Base) articles as knowledge entries.

    KB articles are typically already customer-facing and structured, so we
    accept the whole description as a single "step blob" and let the same
    ``_extract_customer_steps`` guard turn it into safe bullets.
    """
    if not _FRESHDESK_KB_PATH.is_file():
        return []

    try:
        with _FRESHDESK_KB_PATH.open(encoding="utf-8") as handle:
            articles = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    entries: list[KnowledgeEntry] = []
    for article in articles or []:
        title = str(article.get("title") or "").strip()
        description = str(article.get("description_text") or "").strip()
        if not title or not description:
            continue
        steps = _extract_customer_steps(description)
        if not steps:
            # KB article without imperative bullets — keep the description as
            # a single-step diagnostic so semantic search can still find it.
            trimmed = re.sub(r"\s+", " ", description)[:_MAX_CUSTOMER_STEP_LEN]
            if not trimmed:
                continue
            steps = (trimmed,)
        blob = f"{title} {description[:400]}"
        entries.append(
            KnowledgeEntry(
                source="freshdesk_kb",
                category=_infer_category(blob),
                title=title[:80] + ("..." if len(title) > 80 else ""),
                diagnostic=description[:300],
                customer_steps=steps,
            )
        )
    return entries


@lru_cache(maxsize=1)
def load_knowledge_entries() -> tuple[KnowledgeEntry, ...]:
    combined: list[KnowledgeEntry] = []
    combined.extend(_load_qa_entries())
    combined.extend(_load_freshdesk_entries())
    combined.extend(_load_freshdesk_kb_entries())
    combined.extend(_load_autocheck_entries())
    return tuple(combined)


def clear_knowledge_cache() -> None:
    """Invalidate cached knowledge after freshdesk_tickets.json is updated."""
    load_knowledge_entries.cache_clear()


def _score_entry(
    entry: KnowledgeEntry,
    path_tokens: set[str],
    category: Optional[str],
) -> float:
    blob = f"{entry.title} {entry.diagnostic} {' '.join(entry.customer_steps)}"
    blob_tokens = _token_set(blob)
    overlap = len(path_tokens & blob_tokens)
    if overlap == 0:
        return 0.0
    score = float(overlap)
    if category and entry.category == category:
        score += 3.0
    elif category and entry.category == "general":
        score += 0.5
    if entry.source == "qa_csv":
        score += 1.5
    elif entry.source == "auto_check":
        score += 1.0
    elif entry.source == "freshdesk_kb":
        # KB articles are curated help content and usually more precise
        # than raw ticket threads — give them a small bump.
        score += 1.2
    if entry.customer_steps:
        score += 1.0
    return score


# Hybrid scoring weights — keyword token overlap vs cosine similarity.
# Keyword scores are capped at ~10, cosine is in [0, 1], so we scale cosine
# up before mixing. Toggle the whole semantic layer off with
# WARRANTY_SEMANTIC_SEARCH=0.
_HYBRID_SEMANTIC_WEIGHT = 6.0
_SEMANTIC_TOP_K = 12


def _semantic_enabled() -> bool:
    import os

    flag = os.environ.get("WARRANTY_SEMANTIC_SEARCH", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def search_knowledge(
    *,
    path_text: str,
    defect_category: Optional[str] = None,
    model_name: str = "",
    limit: int = 3,
) -> list[KnowledgeEntry]:
    entries = load_knowledge_entries()
    if not entries:
        return []

    path_tokens = _token_set(path_text)
    if model_name:
        path_tokens |= _token_set(model_name)

    category = _DEFECT_CATEGORY_MAP.get((defect_category or "").lower())

    # Keyword scoring (existing path)
    keyword_scores: dict[int, float] = {}
    for idx, entry in enumerate(entries):
        score = _score_entry(entry, path_tokens, category)
        if score > 0:
            keyword_scores[idx] = score

    # Semantic layer (optional, graceful fallback)
    semantic_scores: dict[int, float] = {}
    if _semantic_enabled() and path_text.strip():
        try:
            from warranty_embeddings import semantic_search

            query = path_text
            if model_name:
                query = f"{path_text} {model_name}"
            sem_results = semantic_search(query, top_k=_SEMANTIC_TOP_K, category=category)
            if sem_results:
                entry_to_idx = {id(e): i for i, e in enumerate(entries)}
                for sim, entry in sem_results:
                    idx = entry_to_idx.get(id(entry))
                    if idx is None:
                        continue
                    semantic_scores[idx] = sim
        except Exception:
            semantic_scores = {}

    # Hybrid merge — both maps may be empty; keyword-only is the safety net.
    combined: list[tuple[float, KnowledgeEntry]] = []
    candidate_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())
    for idx in candidate_ids:
        kw = keyword_scores.get(idx, 0.0)
        sem = semantic_scores.get(idx, 0.0)
        score = kw + sem * _HYBRID_SEMANTIC_WEIGHT
        if score < 2.0 and sem < 0.35:
            continue
        combined.append((score, entries[idx]))

    combined.sort(key=lambda x: x[0], reverse=True)
    seen_titles: set[str] = set()
    results: list[KnowledgeEntry] = []
    for _score, entry in combined:
        key = _normalize(entry.title)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        results.append(entry)
        if len(results) >= limit:
            break
    return results
