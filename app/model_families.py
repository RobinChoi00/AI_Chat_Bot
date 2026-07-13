"""
Platform family map for Fonz error-code fallback.

When a model has no direct Fonz rows, lookup can use the family's canonical
model (one hop) — e.g. Hamilton LE → Allure.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fonz_warranty_data import normalize_model_key

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FAMILIES_PATH = _PROJECT_ROOT / "data" / "model_families.json"


@lru_cache(maxsize=1)
def _member_to_canonical() -> dict[str, str]:
    if not _FAMILIES_PATH.is_file():
        return {}

    try:
        payload = json.loads(_FAMILIES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    mapping: dict[str, str] = {}
    for family in payload.get("families") or []:
        canonical = str(family.get("canonical") or "").strip()
        if not canonical:
            continue
        canonical_key = normalize_model_key(canonical)
        for member in family.get("members") or []:
            name = str(member or "").strip()
            if not name:
                continue
            mapping[normalize_model_key(name)] = canonical
        mapping.setdefault(canonical_key, canonical)
    return mapping


def resolve_family_canonical(model_name: str) -> Optional[str]:
    """Return the family canonical display name, or None if unmapped."""
    raw = (model_name or "").strip()
    if not raw:
        return None

    mapping = _member_to_canonical()
    if not mapping:
        return None

    try:
        from product_catalog import resolve_model_name  # noqa: WPS433

        resolved = resolve_model_name(raw) or raw
    except Exception:
        resolved = raw

    key = normalize_model_key(resolved)
    canonical = mapping.get(key)
    if not canonical:
        for member_key, member_canonical in mapping.items():
            if member_key in key or key in member_key:
                canonical = member_canonical
                break
    if not canonical:
        return None
    if normalize_model_key(canonical) == key:
        return None
    return canonical


def clear_model_family_cache() -> None:
    _member_to_canonical.cache_clear()
