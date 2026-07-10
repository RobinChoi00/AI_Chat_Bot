"""
Exact lookup for Fonz error-code records (model + code).

Used by warranty workflow enrichment and ``get_repair_help`` before FAISS RAG.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

from fonz_warranty_data import (
    infer_workflow_category,
    load_error_code_records,
    load_model_diagnostic_records,
    normalize_error_code,
    normalize_model_key,
)

_ERROR_CODE_RE = re.compile(
    r"\berror\s*code\s*([A-Za-z]{0,3}\s*\d+(?:\.\d+)?|[A-Za-z]{1,4})\b"
    r"|\b([A-Za-z]{0,3}\d+(?:\.\d+)?)\b",
    re.I,
)


def clear_error_code_cache() -> None:
    _load_index.cache_clear()
    get_model_diagnostic.cache_clear()


@lru_cache(maxsize=1)
def _load_index() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_model_code: dict[tuple[str, str], dict[str, Any]] = {}
    by_code: dict[str, list[dict[str, Any]]] = {}

    for entry in load_error_code_records():
        model_key = str(entry.get("model_key") or normalize_model_key(str(entry.get("model") or "")))
        code = normalize_error_code(str(entry.get("error_code") or ""))
        if not model_key or not code:
            continue
        enriched = dict(entry)
        if not enriched.get("workflow_category"):
            enriched["workflow_category"] = infer_workflow_category(
                str(enriched.get("meaning") or ""),
                str(enriched.get("troubleshooting") or ""),
            )
        by_model_code[(model_key, code)] = enriched
        by_code.setdefault(code, []).append(enriched)

    return by_model_code, by_code


def _resolve_model_key(model_name: str) -> str:
    raw = (model_name or "").strip()
    if not raw:
        return ""
    try:
        from product_catalog import resolve_model_name  # noqa: WPS433

        resolved = resolve_model_name(raw)
        if resolved:
            return normalize_model_key(resolved)
    except Exception:
        pass
    return normalize_model_key(raw)


def extract_error_codes_from_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()

    for match in _ERROR_CODE_RE.finditer(text):
        token = normalize_error_code(match.group(1) or match.group(2) or "")
        if not token or token in seen:
            continue
        # Skip obvious non-codes.
        if token in {"THE", "AND", "FOR", "NOT", "YES", "NO"}:
            continue
        if re.fullmatch(r"\d+", token) and len(token) > 3:
            continue
        seen.add(token)
        found.append(token)

    return found


def lookup_error_code(
    model_name: Optional[str],
    error_code: str,
) -> Optional[dict[str, Any]]:
    """
    Return the best matching Fonz record for ``model_name`` + ``error_code``.

    When ``model_name`` is omitted, returns the sole match for that code if unique.
    """
    code = normalize_error_code(error_code)
    if not code:
        return None

    by_model_code, by_code = _load_index()
    model_key = _resolve_model_key(model_name or "")

    if model_key:
        hit = by_model_code.get((model_key, code))
        if hit:
            return dict(hit)
        # Fuzzy model: key contained in catalog name or vice versa.
        for (mk, ck), entry in by_model_code.items():
            if ck != code:
                continue
            if mk in model_key or model_key in mk:
                return dict(entry)

    matches = by_code.get(code) or []
    if len(matches) == 1:
        return dict(matches[0])
    if len(matches) > 1 and model_key:
        for entry in matches:
            mk = str(entry.get("model_key") or "")
            if mk in model_key or model_key in mk:
                return dict(entry)
    return None


@lru_cache(maxsize=256)
def get_model_diagnostic(model_name: str) -> Optional[dict[str, Any]]:
    model_key = _resolve_model_key(model_name)
    if not model_key:
        return None
    for row in load_model_diagnostic_records():
        mk = str(row.get("model_key") or normalize_model_key(str(row.get("model") or "")))
        if mk == model_key or mk in model_key or model_key in mk:
            return dict(row)
    return None


def format_repair_help(entry: dict[str, Any]) -> str:
    model = entry.get("model") or "your chair"
    code = entry.get("error_code") or "?"
    lines = [
        f"FONZ_ERROR_LOOKUP match for **{model}** error code **{code}**:",
        f"\n**Meaning:** {entry.get('meaning', '').strip()}",
    ]
    steps = (entry.get("troubleshooting") or "").strip()
    if steps:
        lines.append(f"\n**Suggested steps:** {steps}")
    parts = (entry.get("parts_required") or "").strip()
    if parts:
        lines.append(f"\n**Parts (internal reference):** {parts}")
    return "\n".join(lines)


def entry_workflow_category(entry: dict[str, Any]) -> str:
    cat = str(entry.get("workflow_category") or "").strip()
    if cat:
        return cat
    return infer_workflow_category(
        str(entry.get("meaning") or ""),
        str(entry.get("troubleshooting") or ""),
    )


def code_aligns_with_defect(entry: dict[str, Any], defect_type: str) -> bool:
    if not defect_type:
        return False
    try:
        from warranty_knowledge import map_workflow_defect_category  # noqa: WPS433

        mapped = map_workflow_defect_category(defect_type.lower())
    except Exception:
        mapped = defect_type.lower()
    cat = entry_workflow_category(entry)
    if cat == mapped:
        return True
    if cat == defect_type.lower():
        return True
    return False


def _model_key_matches(model_key: str, entry: dict[str, Any]) -> bool:
    mk = str(entry.get("model_key") or normalize_model_key(str(entry.get("model") or "")))
    if not mk or not model_key:
        return False
    return mk == model_key or mk in model_key or model_key in mk


def list_model_error_codes(
    model_name: str,
    *,
    workflow_category: Optional[str] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Codes for a model, optionally filtered by workflow category."""
    model_key = _resolve_model_key(model_name)
    if not model_key:
        return []

    try:
        from warranty_knowledge import map_workflow_defect_category  # noqa: WPS433

        mapped = (
            map_workflow_defect_category(workflow_category.lower())
            if workflow_category
            else None
        )
    except Exception:
        mapped = (workflow_category or "").lower() or None

    by_model_code, _ = _load_index()
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for (mk, _code), entry in by_model_code.items():
        if not _model_key_matches(model_key, {**entry, "model_key": mk}):
            continue
        code = str(entry.get("error_code") or "")
        if not code or code in seen:
            continue
        if mapped:
            cat = entry_workflow_category(entry)
            if cat != mapped and cat != (workflow_category or "").lower():
                continue
        seen.add(code)
        matches.append(dict(entry))

    matches.sort(key=lambda row: str(row.get("error_code") or ""))
    return matches[: max(1, limit)]


def suggest_error_codes_for_ticket(
    model_name: str,
    defect_type: str,
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Soft terminal hints when the customer did not enter a code."""
    if not model_name:
        return []
    matches = list_model_error_codes(
        model_name,
        workflow_category=defect_type or None,
        limit=limit,
    )
    if not matches and defect_type:
        matches = list_model_error_codes(model_name, limit=limit)
    return matches


def parse_pick_answer_key(answer_key: str) -> Optional[str]:
    key = str(answer_key or "").strip()
    if key.startswith("pick_"):
        return normalize_error_code(key[5:])
    return None
