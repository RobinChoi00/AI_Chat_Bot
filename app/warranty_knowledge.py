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
    "air": ("air", "airbag", "inflate", "hissing", "compressor"),
    "mech": ("mech", "roller", "massage", "rolling", "knead", "track"),
    "footrest": ("footrest", "calf", "leg rest"),
    "heat": ("heat", "heated", "warming"),
    "misc": ("cosmetic", "voice", "speaker", "software", "bluetooth"),
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


def _extract_customer_steps(*texts: str) -> tuple[str, ...]:
    steps: list[str] = []
    seen: set[str] = set()
    for blob in texts:
        if not blob:
            continue
        for chunk in re.split(r"[\n.;]+", blob):
            chunk = chunk.strip()
            chunk = re.sub(r"^\d+[\).\]]\s*", "", chunk)
            if len(chunk) < 12:
                continue
            if _is_internal(chunk):
                continue
            lower = chunk.lower()
            if not any(word in lower for word in _CUSTOMER_ACTION_WORDS):
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


def _load_freshdesk_entries() -> list[KnowledgeEntry]:
    if not _FRESHDESK_PATH.is_file():
        return []

    try:
        with _FRESHDESK_PATH.open(encoding="utf-8") as handle:
            tickets = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    entries: list[KnowledgeEntry] = []
    for ticket in tickets:
        question = str(ticket.get("question") or "").strip()
        answer = str(ticket.get("answer") or "").strip()
        subject = str(ticket.get("subject") or "").strip()
        if not question and not subject:
            continue
        blob = f"{subject} {question}"
        steps = _extract_customer_steps(answer, question)
        if not steps:
            continue
        category = _infer_category(blob)
        title = subject or question[:80]
        entries.append(
            KnowledgeEntry(
                source="freshdesk",
                category=category,
                title=title,
                diagnostic=question[:300],
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


@lru_cache(maxsize=1)
def load_knowledge_entries() -> tuple[KnowledgeEntry, ...]:
    combined: list[KnowledgeEntry] = []
    combined.extend(_load_qa_entries())
    combined.extend(_load_freshdesk_entries())
    combined.extend(_load_autocheck_entries())
    return tuple(combined)


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
    if entry.source == "freshdesk":
        score += 0.5
    if entry.customer_steps:
        score += 1.0
    return score


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
    scored: list[tuple[float, KnowledgeEntry]] = []
    for entry in entries:
        score = _score_entry(entry, path_tokens, category)
        if score >= 2.0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen_titles: set[str] = set()
    results: list[KnowledgeEntry] = []
    for _score, entry in scored:
        key = _normalize(entry.title)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        results.append(entry)
        if len(results) >= limit:
            break
    return results
