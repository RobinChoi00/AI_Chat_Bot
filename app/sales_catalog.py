"""
sales_catalog.py
================
Sales-oriented product façade on top of ``product_catalog`` and the raw
Shopify export CSV.

Everything here returns *deterministic* data pulled from the CSV — no live
Shopify calls, no LLM. That guarantees the Sales AI never invents a price or
a spec, which is the single biggest guardrail we owe customers.

The Shopify export includes rich metafields we can safely surface:
    - Massage Mechanism (2D / 3D / 4D)
    - Track Type (S-Track / L-Track / SL-Track)
    - Zero Gravity, Heating, Airbag, Foot/Calf Roller
    - Number of Massage Styles, Auto Program count

Recommendation is intentionally rule-based (height/weight band → track type,
budget → price band). We do not personalize beyond what the CSV supports.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from product_catalog import (
    _normalize_key,
    format_model_display_name,
    load_catalog_base_prices,
    load_catalog_titles,
    resolve_model_name,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CSV_PATH = _PROJECT_ROOT / "raw_data" / "products_export.csv"


# ---------------------------------------------------------------------------
# Model index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductSpecs:
    """Facts we're willing to state publicly for a given chair."""

    handle: str
    title: str
    display_name: str
    vendor: str
    status: str  # "active" | "draft" | "archived"
    price_usd: Optional[float]
    massage_mechanism: str  # "2D" | "3D" | "4D" | ""
    track_type: str          # "S-Track" | "L-Track" | "SL-Track" | ""
    zero_gravity: str
    heating: str
    airbag: str
    foot_roller: str
    auto_programs: str
    massage_styles: str

    def as_public_dict(self) -> dict:
        return {
            "model": self.display_name,
            "vendor": self.vendor,
            "price_usd": self.price_usd,
            "in_catalog": self.status.lower() == "active",
            "specs": {
                "massage_mechanism": self.massage_mechanism,
                "track_type": self.track_type,
                "zero_gravity": self.zero_gravity,
                "heating": self.heating,
                "airbag": self.airbag,
                "foot_roller": self.foot_roller,
                "auto_programs": self.auto_programs,
                "massage_styles": self.massage_styles,
            },
        }


def _mech(value: str) -> str:
    v = (value or "").upper()
    if "4D" in v:
        return "4D"
    if "3D" in v:
        return "3D"
    if "2D" in v:
        return "2D"
    return ""


def _track(value: str) -> str:
    v = (value or "").upper()
    if "SL" in v:
        return "SL-Track"
    if v.startswith("L") or "L-TRACK" in v or " L " in f" {v} ":
        return "L-Track"
    if v.startswith("S") or "S-TRACK" in v:
        return "S-Track"
    return ""


def _is_massage_chair(row: dict) -> bool:
    typ = str(row.get("Type", "") or "").lower()
    cat = str(row.get("Product Category", "") or "").lower()
    return typ == "massage chair" or "massage chair" in cat


@lru_cache(maxsize=1)
def load_product_index() -> tuple[ProductSpecs, ...]:
    """One record per Shopify handle, deduplicated across variant rows."""
    if not _CSV_PATH.is_file():
        logger.warning("sales_catalog: %s not found", _CSV_PATH)
        return ()

    prices = load_catalog_base_prices()

    by_handle: dict[str, dict] = {}
    with _CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        current_handle = ""
        for row in reader:
            h = (row.get("Handle") or "").strip()
            if h:
                current_handle = h
            if not current_handle:
                continue
            merged = by_handle.setdefault(current_handle, {})
            for key, value in row.items():
                if key is None:
                    continue
                if not merged.get(key) and str(value or "").strip():
                    merged[key] = value

    entries: list[ProductSpecs] = []
    for handle, row in by_handle.items():
        title = (row.get("Title") or "").strip()
        if not title:
            continue
        if not _is_massage_chair(row):
            continue

        display = format_model_display_name(title)
        norm = _normalize_key(title)
        price = prices.get(norm)
        specs = ProductSpecs(
            handle=handle,
            title=title,
            display_name=display,
            vendor=(row.get("Vendor") or "").strip(),
            status=(row.get("Status") or "").strip(),
            price_usd=price,
            massage_mechanism=_mech(
                row.get("Massage Mechanism (product.metafields.custom.massage_mechanism)", "")
            ),
            track_type=_track(
                row.get("Track Type (product.metafields.custom.track_type)", "")
            ),
            zero_gravity=str(
                row.get("Zero Gravity (product.metafields.custom.zero_gravity)", "") or ""
            ).strip(),
            heating=str(row.get("Heating (product.metafields.custom.heating)", "") or "").strip(),
            airbag=str(row.get("Airbag (product.metafields.custom.airbag)", "") or "").strip(),
            foot_roller=str(
                row.get("Foot Roller (product.metafields.custom.foot_roller)", "") or ""
            ).strip(),
            auto_programs=str(
                row.get("Auto Program (product.metafields.custom.auto_program)", "") or ""
            ).strip(),
            massage_styles=str(
                row.get(
                    "Number of Massage Styles (product.metafields.custom.number_of_massage_styles)",
                    "",
                )
                or ""
            ).strip(),
        )
        entries.append(specs)

    logger.info("sales_catalog: loaded %d products", len(entries))
    return tuple(entries)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def resolve_product(text: str) -> Optional[ProductSpecs]:
    """Find the single best ProductSpecs record for a free-text model input."""
    resolved_title = resolve_model_name(text or "")
    if not resolved_title:
        return None
    resolved_key = _normalize_key(resolved_title)

    for record in load_product_index():
        if _normalize_key(record.display_name) == resolved_key:
            return record
        if _normalize_key(record.title).startswith(resolved_key):
            return record
    # Loose fallback — first record whose display name shares the resolved key.
    for record in load_product_index():
        if resolved_key and resolved_key in _normalize_key(record.title):
            return record
    return None


def list_active_products() -> list[ProductSpecs]:
    return [p for p in load_product_index() if p.status.lower() == "active"]


# ---------------------------------------------------------------------------
# Recommendation & comparison
# ---------------------------------------------------------------------------


_TALL_MIN_IN = 74  # 6'2"+ → prefer L-Track / SL-Track & 4D
_TALL_HINT_RE = re.compile(r"\b(tall|large|big\s+guy|nba|basketball)\b", re.I)
_PETITE_HINT_RE = re.compile(r"\b(petite|small|short|tiny|wife|mom|mother|elder(?:ly)?)\b", re.I)
_BACK_HINT_RE = re.compile(r"\b(back|spine|lower\s+back|sciatica|posture)\b", re.I)
_NECK_HINT_RE = re.compile(r"\b(neck|shoulder|shoulders|traps)\b", re.I)
_FEET_HINT_RE = re.compile(r"\b(feet|foot|calf|legs?)\b", re.I)
_BUDGET_RE = re.compile(r"\$?\s*(\d{3,5})(?:\s*(?:k|,000))?\b", re.I)


@dataclass
class RecommendationRequest:
    height_in: Optional[int] = None
    weight_lb: Optional[int] = None
    budget_usd: Optional[int] = None
    focus_areas: list[str] = field(default_factory=list)  # ["back", "neck", "feet"]
    free_text: str = ""


def parse_recommendation_hints(text: str) -> RecommendationRequest:
    """Extract height/weight/budget hints from free text without an LLM."""
    raw = (text or "").strip()
    req = RecommendationRequest(free_text=raw)
    if not raw:
        return req

    ft_in = re.search(r"(\d)\s*(?:'|ft|feet)\s*(\d{1,2})?\s*(?:\"|in|inches)?", raw, re.I)
    if ft_in:
        feet = int(ft_in.group(1))
        inches = int(ft_in.group(2) or 0)
        req.height_in = feet * 12 + inches

    cm_match = re.search(r"(\d{3})\s*cm\b", raw, re.I)
    if cm_match and not req.height_in:
        req.height_in = int(round(int(cm_match.group(1)) / 2.54))

    weight_match = re.search(r"(\d{2,3})\s*(?:lb|lbs|pounds?)\b", raw, re.I)
    if weight_match:
        req.weight_lb = int(weight_match.group(1))
    else:
        kg_match = re.search(r"(\d{2,3})\s*kg\b", raw, re.I)
        if kg_match:
            req.weight_lb = int(round(int(kg_match.group(1)) * 2.20462))

    budget = _BUDGET_RE.search(raw)
    if budget:
        raw_num = budget.group(1)
        num = int(raw_num)
        if raw.lower().find(raw_num + "k") != -1 or "$" + raw_num + "k" in raw.lower():
            num *= 1000
        if num < 1000 and (raw.lower().endswith("k") or "budget" in raw.lower()):
            num *= 1000
        if 500 <= num <= 50000:
            req.budget_usd = num

    if _TALL_HINT_RE.search(raw) and not req.height_in:
        req.height_in = 76
    if _PETITE_HINT_RE.search(raw) and not req.height_in:
        req.height_in = 62

    focus = []
    if _BACK_HINT_RE.search(raw):
        focus.append("back")
    if _NECK_HINT_RE.search(raw):
        focus.append("neck")
    if _FEET_HINT_RE.search(raw):
        focus.append("feet")
    req.focus_areas = focus
    return req


def _score(product: ProductSpecs, req: RecommendationRequest) -> float:
    if product.status.lower() != "active":
        return -1.0
    score = 0.0

    if req.height_in and req.height_in >= _TALL_MIN_IN:
        if product.track_type == "SL-Track":
            score += 3
        elif product.track_type == "L-Track":
            score += 2
        elif product.track_type == "S-Track":
            score -= 1

    if req.weight_lb and req.weight_lb >= 250:
        # Heavier users benefit from 3D/4D + full airbag coverage.
        if product.massage_mechanism in ("3D", "4D"):
            score += 2

    if "back" in req.focus_areas:
        if product.track_type in ("L-Track", "SL-Track"):
            score += 2
        if product.massage_mechanism in ("3D", "4D"):
            score += 1

    if "neck" in req.focus_areas and product.massage_mechanism in ("3D", "4D"):
        score += 1

    if "feet" in req.focus_areas and "yes" in product.foot_roller.lower():
        score += 1

    if req.budget_usd and product.price_usd:
        if product.price_usd <= req.budget_usd:
            score += 1.5
        elif product.price_usd <= req.budget_usd * 1.15:
            score += 0.5
        else:
            score -= 2.0

    # Small tie-breaker: prefer higher tier (4D > 3D > 2D) when no signal.
    if product.massage_mechanism == "4D":
        score += 0.3
    elif product.massage_mechanism == "3D":
        score += 0.2

    return score


def recommend(req: RecommendationRequest, limit: int = 3) -> list[ProductSpecs]:
    """Return up to ``limit`` best matches for a recommendation request."""
    ranked = [
        (product, _score(product, req))
        for product in load_product_index()
    ]
    ranked = [pair for pair in ranked if pair[1] > 0]
    ranked.sort(key=lambda pair: (-pair[1], (pair[0].price_usd or 1e9)))
    return [product for product, _ in ranked[: max(1, min(limit, 5))]]


def compare(a_text: str, b_text: str) -> Optional[dict]:
    """Structured comparison between two models — deterministic, no LLM."""
    left = resolve_product(a_text)
    right = resolve_product(b_text)
    if not left or not right:
        return None
    return {
        "left": left.as_public_dict(),
        "right": right.as_public_dict(),
        "diff": {
            "price_delta_usd": (
                None
                if left.price_usd is None or right.price_usd is None
                else round(right.price_usd - left.price_usd, 2)
            ),
            "mechanism": (left.massage_mechanism, right.massage_mechanism),
            "track": (left.track_type, right.track_type),
        },
    }
