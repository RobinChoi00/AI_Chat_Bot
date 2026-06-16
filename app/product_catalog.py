"""
Lightweight product title index from Shopify export.

Used to normalize customer-entered model names during warranty intake
(install_model node) without an LLM call.
"""

from __future__ import annotations

import csv
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CSV_PATH = _PROJECT_ROOT / "raw_data" / "products_export.csv"

_NOISE = frozenset(
    {"massage", "chair", "the", "my", "model", "is", "a", "an", "osaki", "titan", "os", "usa"}
)


def _normalize_key(text: str) -> str:
    s = (text or "").lower()
    s = s.replace("massage chair", "")
    s = re.sub(r"\bos-", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s.strip()


def _is_base_warranty(option3: str) -> bool:
    w = (option3 or "").lower()
    return "parts/labor" in w and "free" in w and "extended" not in w


def _is_massage_chair(row: dict) -> bool:
    typ = str(row.get("Type", "") or "").lower()
    cat = str(row.get("Product Category", "") or "").lower()
    if typ == "massage chair":
        return True
    return "massage chairs" in cat or "massage chair" in cat


@lru_cache(maxsize=1)
def load_catalog_titles() -> tuple[tuple[str, str], ...]:
    """
    Return (normalized_key, title) pairs — one entry per product handle.
    """
    if not _CSV_PATH.is_file():
        logger.warning("product_catalog: %s not found", _CSV_PATH)
        return ()

    by_handle: dict[str, dict] = {}
    with _CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            h = (row.get("Handle") or "").strip()
            if not h:
                continue
            title = (row.get("Title") or "").strip()
            if title:
                by_handle.setdefault(h, dict(row))
                by_handle[h]["Title"] = title
                for k, v in row.items():
                    if k not in by_handle[h] or not str(by_handle[h].get(k, "")).strip():
                        if str(v or "").strip():
                            by_handle[h][k] = v
            elif h in by_handle:
                # variant row — keep accumulating option rows for base price pick
                pass

    entries: list[tuple[str, str]] = []
    with _CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        current_handle = ""
        current_title = ""
        for row in reader:
            h = (row.get("Handle") or "").strip()
            if h:
                current_handle = h
            title = (row.get("Title") or "").strip()
            if title:
                current_title = title
            if not current_handle or not current_title:
                continue
            if not _is_massage_chair(row if title else by_handle.get(current_handle, row)):
                continue
            key = _normalize_key(current_title)
            if key and (key, current_title) not in entries:
                entries.append((key, current_title))

    logger.info("product_catalog: loaded %d massage chair titles", len(entries))
    return tuple(entries)


def resolve_model_name(raw: str) -> Optional[str]:
    """
    Map free-text model input to the closest catalog Title, or None.
    """
    text = (raw or "").strip()
    if len(text) < 2:
        return None

    catalog = load_catalog_titles()
    if not catalog:
        return None

    norm = _normalize_key(text)
    if not norm:
        return None

    for key, title in catalog:
        if norm == key:
            return title

    for key, title in catalog:
        if len(key) >= 5 and (key in norm or norm in key):
            return title

    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _NOISE and len(t) >= 3]
    if not tokens:
        return None

    best_title: Optional[str] = None
    best_score = 0
    for key, title in catalog:
        score = sum(1 for t in tokens if t in key)
        if score > best_score and score >= max(2, len(tokens) - 1):
            best_score = score
            best_title = title

    return best_title
