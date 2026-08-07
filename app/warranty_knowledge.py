"""
Unified warranty troubleshooting knowledge base.

Sources (merged at load time):
  - raw_data/Warranty Daily Report - Q&A.csv
  - data/freshdesk_tickets.json  (Freshdesk API ETL output)
  - raw_data/Auto-Check.csv      (error-code troubleshooting steps)
  - raw_data/fault_judgment.csv  (massage-chair fault judgment manual)
  - data/fonz_error_codes.json   (Fonz All-in-one Warranty List)

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
_FAULT_JUDGMENT_PATH = _PROJECT_ROOT / "raw_data" / "fault_judgment.csv"
_FONZ_ERROR_PATH = _PROJECT_ROOT / "data" / "fonz_error_codes.json"

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
    "footrest": (
        "footrest",
        "calf",
        "calves",
        "leg rest",
        "legrest",
        "leg-rest",
        "extend",
        "extension",
        "telescopic",
        "telescop",
        "count sensor",
        "push rod",
        "legrest extend",
        "calf extension",
    ),
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
    "refer to:",
    "refer to page",
    "the customer",
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

# Agent follow-up / logistics — not actionable DIY for mid-flow enrichment.
_LOGISTICS_OR_FOLLOWUP_MARKERS = (
    "technician will",
    "technician to",
    "tech will",
    "in-house tech",
    "our technician",
    "we've asked the technician",
    "we have asked the technician",
    "we will dispatch",
    "we'll dispatch",
    "reach out",
    "contact you to",
    "contact you within",
    "return visit",
    "set up a visit",
    "schedule a visit",
    "arrange a visit",
    "prioritize your repair",
    "prioritize the repair",
    "follow up with you",
    "follow-up with you",
    "within 24 hours",
    "within 48 hours",
    "business days",
    "warranty inquiry form",
    "received a message from",
    "via warranty inquiry",
)

_UNHELPFUL_MATCH_TITLE_MARKERS = (
    "warranty inquiry form",
    "received a message from",
    "via warranty inquiry",
    "contact form",
    "web form",
)

_MAX_CUSTOMER_STEP_LEN = 220
_MIN_CUSTOMER_STEP_LEN = 10
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
# Manual CSVs often use "1. check … 2. replace …" (sometimes without a space).
_NUMBERED_ITEM_RE = re.compile(r"(?:(?<=^)|(?<=\s))(\d{1,2})[\.\)\:]\s*")
_CONNECTION_HINT_RE = re.compile(
    r"\b(connector|wire|wiring|plug|cable|poor[- ]contact|loose|disconnect|"
    r"hose|tracheal|socket)\b",
    re.I,
)
_BUTTON_HINT_RE = re.compile(r"\b(button|stuck|side panel|keys?)\b", re.I)
_SENSOR_HINT_RE = re.compile(r"\b(sensor|counting|encoder|limit|hall)\b", re.I)

_CUSTOMER_ACTION_WORDS = (
    "check",
    "verify",
    "ensure",
    "reconnect",
    "reseat",
    "unplug",
    "plug",
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
    "press",
    "hold",
    "disconnect",
    "secure",
    "reset",
    "clean",
    "tighten",
    "adjust",
    "detect",
)


@dataclass(frozen=True)
class KnowledgeEntry:
    source: str
    category: str
    title: str
    diagnostic: str
    customer_steps: tuple[str, ...]


def _model_signature(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    noise = {"osaki", "titan", "os", "massage", "chair", "model"}
    return "".join(token for token in tokens if token not in noise)


@lru_cache(maxsize=1)
def _known_model_signatures() -> frozenset[str]:
    """Known Fonz identifiers used to detect explicit cross-model matches."""
    try:
        from fonz_warranty_data import load_model_diagnostic_records  # noqa: WPS433

        values = {
            _model_signature(str(row.get("model") or ""))
            for row in load_model_diagnostic_records()
        }
        return frozenset(value for value in values if len(value) >= 5)
    except Exception:
        return frozenset()


def _entry_explicit_model_signatures(entry: KnowledgeEntry) -> set[str]:
    flat = _model_signature(
        f"{entry.title} {entry.diagnostic} {' '.join(entry.customer_steps)}"
    )
    if not flat:
        return set()
    return {signature for signature in _known_model_signatures() if signature in flat}


def _entry_matches_requested_model(entry: KnowledgeEntry, model_name: str) -> bool:
    """Reject knowledge that explicitly names a different chair model."""
    if not str(model_name or "").strip():
        return True
    explicit = _entry_explicit_model_signatures(entry)
    if not explicit:
        return True

    allowed = {_model_signature(model_name)}
    try:
        from model_families import resolve_family_canonical  # noqa: WPS433

        canonical = resolve_family_canonical(model_name)
        if canonical:
            allowed.add(_model_signature(canonical))
    except Exception:
        pass
    allowed.discard("")
    return bool(explicit & allowed)


def map_workflow_defect_category(defect_category: Optional[str]) -> Optional[str]:
    """Map workflow answer_key (e.g. rolling) to knowledge category (e.g. mech)."""
    if not defect_category:
        return None
    return _DEFECT_CATEGORY_MAP.get(defect_category.lower(), defect_category)


# Categories that must not leak across defect paths (remote tips ≠ air tips).
_SCOPED_DEFECT_CATEGORIES = frozenset(_CATEGORY_KEYWORDS.keys())


def entry_allowed_for_category(
    entry: KnowledgeEntry,
    category: Optional[str],
) -> bool:
    """
    Hard gate for defect-scoped search/enrichment.

    When the customer picked a topic (e.g. remote), drop rows tagged as a
    different defect family (e.g. air). ``general`` rows are kept only when
    their text does not clearly belong to another scoped category.
    """
    if not category:
        return True
    wanted = (category or "").strip().lower()
    entry_cat = (entry.category or "").strip().lower() or "general"
    if entry_cat == wanted:
        return True
    if entry_cat in _SCOPED_DEFECT_CATEGORIES:
        return False
    if entry_cat == "general":
        blob = f"{entry.title} {entry.diagnostic} {' '.join(entry.customer_steps)}"
        inferred = _infer_category(blob)
        if inferred in _SCOPED_DEFECT_CATEGORIES and inferred != wanted:
            return False
        return True
    return False


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
    if len(chunk) < _MIN_CUSTOMER_STEP_LEN or len(chunk) > _MAX_CUSTOMER_STEP_LEN:
        return False
    if _is_internal(chunk):
        return False
    lower = chunk.lower()
    if any(marker in lower for marker in _PII_OR_ADMIN_MARKERS):
        return False
    if any(marker in lower for marker in _BOILERPLATE_MARKERS):
        return False
    if any(marker in lower for marker in _LOGISTICS_OR_FOLLOWUP_MARKERS):
        return False
    if _PHONE_RE.search(chunk) or _ZIP_RE.search(chunk) or _EMAIL_RE.search(chunk):
        return False
    if not any(word in lower for word in _CUSTOMER_ACTION_WORDS):
        return False
    return True


def is_presentable_match_title(title: str) -> bool:
    """Hide generic web-form / intake ticket subjects from customer UI."""
    lower = (title or "").lower().strip()
    if not lower:
        return False
    return not any(marker in lower for marker in _UNHELPFUL_MATCH_TITLE_MARKERS)


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


def _normalize_manual_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "").replace("\xa0", " ").replace("\u3000", " ")).strip()


def _split_step_chunks(blob: str) -> list[str]:
    """Split troubleshooting text into candidate step chunks.

    Manual CSVs often pack numbered steps on one line
    (``1.check … 2.replace …``) without newlines.
    """
    text = _normalize_manual_text(blob)
    if not text:
        return []

    matches = list(_NUMBERED_ITEM_RE.finditer(text))
    if len(matches) >= 2 or (len(matches) == 1 and matches[0].start() <= 2):
        chunks: list[str] = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip(" \n\t;")
            if chunk:
                chunks.append(chunk)
        if chunks:
            return chunks

    chunks = []
    for chunk in re.split(r"[\n;]+", text):
        chunk = chunk.strip()
        chunk = re.sub(r"^\d+[\).\]]\s*", "", chunk)
        if chunk:
            # Keep period-separated clauses only when they look like actions.
            if ". " in chunk and any(w in chunk.lower() for w in _CUSTOMER_ACTION_WORDS):
                for piece in re.split(r"(?<=[a-z0-9\)])\.\s+", chunk):
                    piece = piece.strip()
                    if piece:
                        chunks.append(piece)
            else:
                chunks.append(chunk)
    return chunks


def _format_step(chunk: str) -> str:
    chunk = chunk.strip()
    if not chunk:
        return chunk
    return chunk[0].upper() + chunk[1:]


def _extract_customer_steps(*texts: str) -> tuple[str, ...]:
    steps: list[str] = []
    seen: set[str] = set()
    for blob in texts:
        if not blob:
            continue
        for chunk in _split_step_chunks(blob):
            if not _is_customer_safe_step(chunk):
                continue
            key = _normalize(chunk)
            if key in seen:
                continue
            seen.add(key)
            steps.append(_format_step(chunk))
    return tuple(steps[:4])


def _fallback_manual_steps(*texts: str) -> tuple[str, ...]:
    """Customer-safe DIY when a tech manual only lists part replacements."""
    blob = " ".join(_normalize_manual_text(t) for t in texts if t)
    if not blob:
        return ()
    fallbacks: list[str] = []
    if _CONNECTION_HINT_RE.search(blob):
        fallbacks.append(
            "Check that all visible connectors and cables are firmly seated."
        )
        fallbacks.append("Power cycle the chair, then test the same function again.")
    elif _BUTTON_HINT_RE.search(blob):
        fallbacks.append(
            "Check whether any side-panel or remote buttons are stuck down."
        )
        fallbacks.append("Power cycle the chair, then retry the same button once.")
    elif _SENSOR_HINT_RE.search(blob):
        fallbacks.append(
            "Power cycle the chair and retry the function once to confirm the symptom."
        )
    else:
        fallbacks.append(
            "Power cycle the chair once, then note exactly when the issue repeats."
        )
    return tuple(fallbacks[:3])


def _customer_steps_from_manual(*texts: str) -> tuple[str, ...]:
    steps = _extract_customer_steps(*texts)
    if steps:
        return steps
    return _fallback_manual_steps(*texts)


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
                steps = tuple(
                    s for s in cached.steps if _is_customer_safe_step(str(s))
                )
                category_hint = (cached.category or "").strip().lower() or None
                summary_diagnostic = cached.summary

        if not steps:
            continue

        steps = tuple(s for s in steps if _is_customer_safe_step(s))
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
    with _AUTOCHECK_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            code = _normalize_manual_text(row[0] or "")
            phenomenon = _normalize_manual_text(row[1] or "")
            description = _normalize_manual_text(row[2] or "")
            troubleshooting = _normalize_manual_text(row[3] or "")
            if not code or code.lower() in {"code no.", "nan"}:
                continue
            if not phenomenon and not troubleshooting and not description:
                continue
            steps = _customer_steps_from_manual(
                troubleshooting, description, phenomenon
            )
            blob = f"{phenomenon} {description} error {code}"
            entries.append(
                KnowledgeEntry(
                    source="auto_check",
                    category=_infer_category(blob),
                    title=phenomenon or f"Error {code}",
                    diagnostic=(description or phenomenon)[:300],
                    customer_steps=steps,
                )
            )
    return entries


def _load_fault_judgment_entries() -> list[KnowledgeEntry]:
    """Load Massage chair fault judgment manual into customer-safe knowledge."""
    if not _FAULT_JUDGMENT_PATH.is_file():
        return []

    entries: list[KnowledgeEntry] = []
    with _FAULT_JUDGMENT_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        # Skip title / status preamble rows (same offset as master_ingester).
        for _ in range(5):
            next(reader, None)
        header = next(reader, None)
        if not header:
            return []

        for row in reader:
            if len(row) < 4:
                continue
            code = _normalize_manual_text(row[0] or "")
            # Section labels like "Lower mechanism alarm number"
            if not code or not code.isdigit():
                continue
            phenomenon = _normalize_manual_text(row[1] or "")
            description = _normalize_manual_text(row[2] or "")
            troubleshooting = _normalize_manual_text(row[3] or "")
            if not phenomenon and not troubleshooting:
                continue
            steps = _customer_steps_from_manual(
                troubleshooting, description, phenomenon
            )
            blob = f"{phenomenon} {description} fault {code}"
            entries.append(
                KnowledgeEntry(
                    source="fault_judgment",
                    category=_infer_category(blob),
                    title=phenomenon or f"Fault {code}",
                    diagnostic=(description or phenomenon)[:300],
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


def _load_fonz_error_entries() -> list[KnowledgeEntry]:
    if not _FONZ_ERROR_PATH.is_file():
        return []

    try:
        with _FONZ_ERROR_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    records = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []

    entries: list[KnowledgeEntry] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or "").strip()
        code = str(row.get("error_code") or "").strip()
        meaning = str(row.get("meaning") or "").strip()
        troubleshooting = str(row.get("troubleshooting") or "").strip()
        if not model or not code:
            continue
        steps = _extract_customer_steps(troubleshooting, meaning)
        if not steps and troubleshooting:
            trimmed = troubleshooting.strip()
            if len(trimmed) >= _MIN_CUSTOMER_STEP_LEN and not _is_internal(trimmed):
                steps = (trimmed[:_MAX_CUSTOMER_STEP_LEN],)
        if not steps:
            steps = _fallback_manual_steps(troubleshooting, meaning)
        blob = f"{meaning} {troubleshooting} error {code}"
        entries.append(
            KnowledgeEntry(
                source="fonz_error_code",
                category=_infer_category(blob),
                title=f"{model} — error {code}",
                diagnostic=meaning[:300] or f"Error code {code} on {model}.",
                customer_steps=steps,
            )
        )
    return entries


@lru_cache(maxsize=1)
def load_knowledge_entries() -> tuple[KnowledgeEntry, ...]:
    combined: list[KnowledgeEntry] = []
    combined.extend(_load_fonz_error_entries())
    combined.extend(_load_qa_entries())
    combined.extend(_load_freshdesk_entries())
    combined.extend(_load_freshdesk_kb_entries())
    combined.extend(_load_autocheck_entries())
    combined.extend(_load_fault_judgment_entries())
    return tuple(combined)


def clear_knowledge_cache() -> None:
    """Invalidate cached knowledge after freshdesk_tickets.json is updated."""
    load_knowledge_entries.cache_clear()
    _known_model_signatures.cache_clear()
    try:
        from error_code_lookup import clear_error_code_cache  # noqa: WPS433

        clear_error_code_cache()
    except ImportError:
        pass
    try:
        from warranty_embeddings import clear_embedding_cache  # noqa: WPS433

        clear_embedding_cache()
    except ImportError:
        pass


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
    elif entry.source == "fault_judgment":
        score += 1.0
    elif entry.source == "fonz_error_code":
        score += 2.5
    elif entry.source == "freshdesk_kb":
        # KB articles are curated help content and usually more precise
        # than raw ticket threads — give them a small bump.
        score += 1.2
    elif entry.source == "freshdesk":
        # Raw ticket threads often contain logistics follow-ups, not DIY steps.
        score -= 0.8
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
        entry = entries[idx]
        if not _entry_matches_requested_model(entry, model_name):
            continue
        if category and not entry_allowed_for_category(entry, category):
            continue
        kw = keyword_scores.get(idx, 0.0)
        sem = semantic_scores.get(idx, 0.0)
        score = kw + sem * _HYBRID_SEMANTIC_WEIGHT
        if score < 2.0 and sem < 0.35:
            continue
        combined.append((score, entry))

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


_DELIVERY_TOPIC_RE = re.compile(
    r"\b(delivery|shipping|tracking|carrier|freight|fedex|ups|usps|"
    r"shipment|in transit|unboxing|box damage|signed cleared|"
    r"delivery receipt|expedite|warehouse|order status)\b",
    re.I,
)

_INSTALL_TOPIC_RE = re.compile(
    r"\b(install|installation|assembly|setup|video|footrest hose|"
    r"air hose|white glove|wg install)\b",
    re.I,
)

_REPAIR_ONLY_MARKERS = (
    "voice pcb",
    "main pcb",
    "remote fuse",
    "actuator",
    "heating element",
    "massage mechanism",
    "footrest mechanism",
    "recline motor",
    "ghost voice",
    "false trigger",
    "blown fuse",
)

_REPAIR_CATEGORIES = frozenset({"power", "remote", "air", "mech", "footrest", "heat", "voice"})


def _entry_text_blob(entry: KnowledgeEntry) -> str:
    return " ".join((entry.title, entry.diagnostic, " ".join(entry.customer_steps)))


def _filter_delivery_entries(
    entries: list[KnowledgeEntry],
    limit: int,
) -> list[KnowledgeEntry]:
    filtered: list[KnowledgeEntry] = []
    for entry in entries:
        blob = _entry_text_blob(entry).lower()
        if _DELIVERY_TOPIC_RE.search(blob):
            filtered.append(entry)
            continue
        if any(marker in blob for marker in _REPAIR_ONLY_MARKERS):
            continue
        if entry.category in _REPAIR_CATEGORIES:
            continue
        if entry.category == "general" and ("ship" in blob or "delivery" in blob):
            filtered.append(entry)
    return filtered[:limit]


def _filter_installation_entries(
    entries: list[KnowledgeEntry],
    limit: int,
) -> list[KnowledgeEntry]:
    filtered: list[KnowledgeEntry] = []
    for entry in entries:
        blob = _entry_text_blob(entry).lower()
        if _INSTALL_TOPIC_RE.search(blob):
            filtered.append(entry)
            continue
        if entry.category == "air" and any(
            token in blob for token in ("hose", "footrest", "install")
        ):
            filtered.append(entry)
            continue
        if any(marker in blob for marker in _REPAIR_ONLY_MARKERS):
            if not _INSTALL_TOPIC_RE.search(blob):
                continue
        if entry.category in {"power", "remote", "voice", "heat"} and not _INSTALL_TOPIC_RE.search(
            blob
        ):
            continue
        if entry.category == "general":
            filtered.append(entry)
    return filtered[:limit]


def contextual_search_knowledge(
    *,
    path_text: str,
    issue_type: str = "",
    defect_category: Optional[str] = None,
    model_name: str = "",
    limit: int = 3,
) -> list[KnowledgeEntry]:
    """
    Issue-type-aware KB search so delivery/installation paths do not pull
    unrelated defect tickets, and defect paths do not search before category
    is known.
    """
    issue = (issue_type or "").strip().lower()

    if issue == "delivery":
        query = f"delivery shipping tracking order status carrier freight {path_text}"
        raw = search_knowledge(
            path_text=query,
            defect_category=None,
            model_name=model_name,
            limit=max(limit * 3, 6),
        )
        return _filter_delivery_entries(raw, limit)

    if issue == "installation":
        query = f"installation assembly setup footrest air hose video {path_text}"
        raw = search_knowledge(
            path_text=query,
            defect_category="air",
            model_name=model_name,
            limit=max(limit * 2, 4),
        )
        return _filter_installation_entries(raw, limit)

    if issue == "defect":
        if not defect_category:
            return []
        return search_knowledge(
            path_text=path_text,
            defect_category=defect_category,
            model_name=model_name,
            limit=limit,
        )

    if not defect_category:
        return []
    return search_knowledge(
        path_text=path_text,
        defect_category=defect_category,
        model_name=model_name,
        limit=limit,
    )
