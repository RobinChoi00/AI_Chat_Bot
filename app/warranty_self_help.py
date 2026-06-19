"""
Customer-safe troubleshooting hints from raw_data/Warranty Daily Report - Q&A.csv.

Used at defect workflow terminals before asking for email — DIY checks only,
not internal part-replacement instructions.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

_QA_PATH = Path(__file__).resolve().parent.parent / "raw_data" / "Warranty Daily Report - Q&A.csv"

_DEFECT_CATEGORY_MAP: dict[str, str] = {
    "power": "Category - Power",
    "remote": "Category - Remote",
    "air": "Category - Air",
    "rolling": "Category - Mech",
    "recline": "Category - Mech",
    "footrest": "Category - Footrest",
    "cosmetic": "Category - Misc.",
    "heat": "Category - Heat",
}

_INTERNAL_SOLUTION_MARKERS = (
    "replace ",
    "send tech",
    "dispatch",
    "send replacement",
    "send a tech",
    "admin",
    "pcb",
    "main board",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _normalize(text)) if len(t) > 2}


@lru_cache(maxsize=1)
def _load_qa_entries() -> tuple[tuple[str, str, str, str], ...]:
    if not _QA_PATH.is_file():
        return ()

    entries: list[tuple[str, str, str, str]] = []
    current_category = ""

    with _QA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # title row
        for row in reader:
            if not row:
                continue
            col0 = (row[0] if len(row) > 0 else "").strip()
            if col0.startswith("Category -"):
                current_category = col0
                continue
            if not col0 or col0.lower() in {"n/a", "nan"}:
                continue
            diagnostic = (row[1] if len(row) > 1 else "").strip()
            solution = (row[2] if len(row) > 2 else "").strip()
            if not diagnostic and not solution:
                continue
            entries.append((current_category, col0, diagnostic, solution))

    return tuple(entries)


def _customer_safe_solution(solution: str) -> Optional[str]:
    text = (solution or "").strip()
    if not text:
        return None
    lower = text.lower()
    if any(marker in lower for marker in _INTERNAL_SOLUTION_MARKERS):
        return None
    if len(text) > 220:
        text = text[:217].rstrip() + "…"
    return text


def _score_entry(
    issue_tokens: set[str],
    path_tokens: set[str],
    issue: str,
    diagnostic: str,
    solution: str,
) -> float:
    blob = _normalize(f"{issue} {diagnostic} {solution}")
    blob_tokens = _token_set(blob)
    overlap = len((issue_tokens | path_tokens) & blob_tokens)
    if overlap == 0:
        return 0.0
    score = float(overlap)
    if issue_tokens & _token_set(issue):
        score += 2.0
    return score


def find_defect_self_help(
    *,
    defect_category: Optional[str],
    path_text: str,
    model_name: str = "",
) -> Optional[str]:
    """
    Return formatted self-help text for defect terminals, or None if no match.
    """
    category_label = _DEFECT_CATEGORY_MAP.get((defect_category or "").lower())
    if not category_label:
        return None

    entries = _load_qa_entries()
    if not entries:
        return None

    path_tokens = _token_set(path_text)
    if model_name:
        path_tokens |= _token_set(model_name)

    scored: list[tuple[float, str, str, str]] = []
    for cat, issue, diagnostic, solution in entries:
        if cat != category_label:
            continue
        issue_tokens = _token_set(issue)
        score = _score_entry(issue_tokens, path_tokens, issue, diagnostic, solution)
        if score >= 2.0:
            scored.append((score, issue, diagnostic, solution))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    lines: list[str] = ["Here are a few things you can try before we escalate your case:"]

    for _score, issue, diagnostic, solution in scored[:2]:
        lines.append(f"\n• **{issue}**")
        if diagnostic:
            lines.append(f"  Check: {diagnostic}")
        safe = _customer_safe_solution(solution)
        if safe:
            lines.append(f"  Try: {safe}")

    if len(lines) <= 1:
        return None

    lines.append(
        "\nIf these steps don't resolve the issue, leave your email below "
        "and our warranty team will follow up."
    )
    return "\n".join(lines)


def infer_defect_category_from_turns(turns) -> Optional[str]:
    """Read defect_problem_type answer_key from stored workflow turns."""
    for turn in turns:
        key = str(getattr(turn, "answer_key", "") or "")
        if key in _DEFECT_CATEGORY_MAP:
            return key
    return None


def build_path_text(turns) -> str:
    parts: list[str] = []
    for turn in turns:
        for attr in ("customer_answer", "node_prompt", "node_id"):
            val = str(getattr(turn, attr, "") or "").strip()
            if val:
                parts.append(val)
    return " ".join(parts)
