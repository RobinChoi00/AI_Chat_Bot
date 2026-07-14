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

_EXPLICIT_ERROR_CODE_RE = re.compile(
    r"\b(?:error\s*)?code\s*(?:is|:|#|-)?\s*"
    r"([A-Za-z]{0,3}\s*\d+(?:\.\d+)?|[A-Za-z]{1,4})\b",
    re.I,
)
_STANDALONE_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,3}\d{1,3}(?:\.\d+)?|\d{1,3})(?![A-Za-z0-9])",
    re.I,
)
_NUMERIC_CODE_PREFIX_RE = re.compile(
    r"(?:\b(?:error\s*)?code\b|"
    r"\b(?:display|screen)\s+(?:shows?|reads?)|"
    r"\b(?:shows?|reads?)\s*)\s*(?:is|:|#|-)?\s*$",
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
    """Extract deliberate error-code mentions without treating every number as a code."""
    if not text or not text.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        token = normalize_error_code(raw)
        if not token or token in seen:
            return
        # Skip obvious non-codes.
        if token in {"THE", "AND", "FOR", "NOT", "YES", "NO"}:
            return
        if re.fullmatch(r"\d+", token) and len(token) > 3:
            return
        seen.add(token)
        found.append(token)

    explicit_spans: list[tuple[int, int]] = []
    for match in _EXPLICIT_ERROR_CODE_RE.finditer(text):
        _add(match.group(1) or "")
        explicit_spans.append(match.span())

    stripped = text.strip()
    code_only = bool(
        re.fullmatch(
            r"(?:error\s*)?(?:code\s*)?(?:is|:|#|-)?\s*"
            r"(?:[A-Za-z]{1,3}\d{1,3}(?:\.\d+)?|\d{1,3})",
            stripped,
            flags=re.I,
        )
    )
    for match in _STANDALONE_CODE_RE.finditer(text):
        if any(start <= match.start() < end for start, end in explicit_spans):
            continue
        token = match.group(1) or ""
        has_letters = bool(re.search(r"[A-Za-z]", token))
        prefix = text[max(0, match.start() - 32) : match.start()]
        if not has_letters and not code_only and not _NUMERIC_CODE_PREFIX_RE.search(prefix):
            continue
        _add(token)

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
        from model_families import resolve_family_canonical  # noqa: WPS433

        family_model = resolve_family_canonical(model_name or "")
        if family_model:
            family_key = _resolve_model_key(family_model)
            if family_key:
                hit = by_model_code.get((family_key, code))
                if hit:
                    out = dict(hit)
                    out["family_fallback"] = True
                    out["family_canonical"] = family_model
                    return out

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


def _collect_codes_for_model_key(
    model_key: str,
    *,
    mapped: Optional[str],
    workflow_category: Optional[str],
    limit: int,
) -> list[dict[str, Any]]:
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

    matches = _collect_codes_for_model_key(
        model_key,
        mapped=mapped,
        workflow_category=workflow_category,
        limit=limit,
    )
    if matches:
        return matches

    from model_families import resolve_family_canonical  # noqa: WPS433

    family_model = resolve_family_canonical(model_name)
    if not family_model:
        return []

    family_key = _resolve_model_key(family_model)
    if not family_key or family_key == model_key:
        return []

    family_matches = _collect_codes_for_model_key(
        family_key,
        mapped=mapped,
        workflow_category=workflow_category,
        limit=limit,
    )
    if not family_matches:
        return []

    out: list[dict[str, Any]] = []
    for row in family_matches:
        tagged = dict(row)
        tagged["family_fallback"] = True
        tagged["family_canonical"] = family_model
        out.append(tagged)
    return out


def suggest_error_codes_for_ticket(
    model_name: str,
    defect_type: str,
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Soft terminal hints when the customer did not enter a code."""
    if model_name:
        matches = list_model_error_codes(
            model_name,
            workflow_category=defect_type or None,
            limit=limit,
        )
        if not matches and defect_type:
            matches = list_model_error_codes(model_name, limit=limit)
        return matches

    return suggest_category_error_codes(defect_type, limit=limit)


def suggest_category_error_codes(
    defect_type: str,
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Category-only Fonz hints when chair model is not yet known."""
    if not defect_type:
        return []

    try:
        from warranty_knowledge import map_workflow_defect_category  # noqa: WPS433

        mapped = map_workflow_defect_category(defect_type.lower())
    except Exception:
        mapped = defect_type.lower()

    by_model_code, _ = _load_index()
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for (_mk, _code), entry in by_model_code.items():
        code = str(entry.get("error_code") or "")
        if not code or code in seen:
            continue
        cat = entry_workflow_category(entry)
        if cat != mapped and cat != defect_type.lower():
            continue
        seen.add(code)
        matches.append(dict(entry))

    matches.sort(key=lambda row: str(row.get("error_code") or ""))
    return matches[: max(1, limit)]


def parse_pick_answer_key(answer_key: str) -> Optional[str]:
    key = str(answer_key or "").strip()
    if key.startswith("pick_"):
        return normalize_error_code(key[5:])
    return None
