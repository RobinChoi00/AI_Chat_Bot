"""
Lookup installation video URLs by chair model name.

Uses data/install_videos.json with series + exact model keys.
Falls back to config REPAIR_MANUAL_URL when no match is found.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from config import REPAIR_MANUAL_URL
from product_catalog import resolve_model_name

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "install_videos.json"


def _normalize_key(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"\bos-", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


@lru_cache(maxsize=1)
def _load_catalog() -> dict:
    if not _DATA_PATH.is_file():
        return {"default": {"url": REPAIR_MANUAL_URL, "label": "Installation guides"}}
    with _DATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _entry_to_result(entry: dict, *, resolved: str, match: str, fallback_url: str) -> dict[str, Any]:
    videos = entry.get("videos")
    result: dict[str, Any] = {
        "url": str(entry.get("url") or fallback_url),
        "label": str(entry.get("label") or resolved),
        "match": match,
    }
    if isinstance(videos, list) and videos:
        result["videos"] = [
            {
                "url": str(item.get("url") or fallback_url),
                "label": str(item.get("label") or resolved),
            }
            for item in videos
            if isinstance(item, dict)
        ]
    return result


def lookup_install_video(model_name: str) -> dict[str, Any]:
    """
    Return {"url": str, "label": str, "match": "model|series|default", "videos"?: list}.
    """
    catalog = _load_catalog()
    default = catalog.get("default") or {}
    fallback_url = str(default.get("url") or REPAIR_MANUAL_URL)
    fallback_label = str(default.get("label") or "Installation guide")

    raw = (model_name or "").strip()
    if not raw:
        return {"url": fallback_url, "label": fallback_label, "match": "default"}

    resolved = resolve_model_name(raw) or raw
    key = _normalize_key(resolved)

    models: dict = catalog.get("models") or {}
    if key in models:
        return _entry_to_result(
            models[key], resolved=resolved, match="model", fallback_url=fallback_url
        )

    for model_key, entry in models.items():
        if model_key in key or key in model_key:
            return _entry_to_result(
                entry, resolved=resolved, match="model", fallback_url=fallback_url
            )

    series: dict = catalog.get("series") or {}
    for series_key, entry in series.items():
        sk = _normalize_key(series_key)
        if sk and sk in key:
            return _entry_to_result(
                entry,
                resolved=resolved,
                match="series",
                fallback_url=fallback_url,
            )

    return {"url": fallback_url, "label": fallback_label, "match": "default"}
