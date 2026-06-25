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
        if not _is_usable_freshdesk_answer(answer):
            continue
        if not question and not subject:
            continue
        blob = f"{subject} {question}"
        steps = _extract_customer_steps(answer)
        if not steps:
            continue
        category = _infer_category(blob)
        title = _clean_freshdesk_title(subject, question)
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
