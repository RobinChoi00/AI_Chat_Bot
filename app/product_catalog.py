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


def _is_base_warranty_delivery(row: dict) -> bool:
    warranty = str(row.get("Option3 Value", "") or "").lower()
    delivery = str(row.get("Option2 Value", "") or "").lower()
    if "parts/labor" not in warranty or "free" not in warranty:
        return False
    if "extended" in warranty:
        return False
    return "curbside" in delivery or delivery in ("", "n/a")


@lru_cache(maxsize=1)
def load_catalog_base_prices() -> dict[str, float]:
    """
    Return normalized title key → base variant price (USD) from Shopify export.
    """
    if not _CSV_PATH.is_file():
        logger.warning("product_catalog: %s not found", _CSV_PATH)
        return {}

    by_handle: dict[str, list[dict]] = {}
    with _CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        current_handle = ""
        current_title = ""
        current_row: dict = {}
        for row in reader:
            h = (row.get("Handle") or "").strip()
            if h:
                current_handle = h
                current_row = dict(row)
            title = (row.get("Title") or "").strip()
            if title:
                current_title = title
                current_row = dict(row)
            if not current_handle or not current_title:
                continue
            if not _is_massage_chair(current_row):
                continue
            merged = dict(current_row)
            merged["Title"] = current_title
            merged["Handle"] = current_handle
            by_handle.setdefault(current_handle, []).append(merged)

    prices: dict[str, float] = {}
    for _handle, rows in by_handle.items():
        title = str(rows[0].get("Title", "")).strip()
        if not title:
            continue
        base_rows = [r for r in rows if _is_base_warranty_delivery(r)]
        candidates = base_rows or rows
        best_price: Optional[float] = None
        for row in candidates:
            raw = str(row.get("Variant Price", "") or "").replace(",", "").strip()
            if not raw:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if 500 <= val <= 50000 and (best_price is None or val < best_price):
                best_price = val
        if best_price is not None:
            prices[_normalize_key(title)] = best_price

    logger.info("product_catalog: loaded %d base prices", len(prices))
    return prices


def resolve_catalog_price(title_or_text: str) -> Optional[float]:
    """Look up canonical base price for a product title or free-text model."""
    text = (title_or_text or "").strip()
    if len(text) < 2:
        return None

    prices = load_catalog_base_prices()
    if not prices:
        return None

    norm = _normalize_key(text)
    if norm in prices:
        return prices[norm]

    resolved = resolve_model_name(text)
    if resolved:
        key = _normalize_key(resolved)
        if key in prices:
            return prices[key]

    best_price: Optional[float] = None
    best_len = 0
    for key, price in prices.items():
        if len(key) >= 5 and (key in norm or norm in key):
            if len(key) > best_len:
                best_len = len(key)
                best_price = price
    return best_price


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
