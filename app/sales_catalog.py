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

Recommendation is intentionally rule-based (height/weight → track/mechanism
fit, then one pick each from Value $2–3k / Mid $4–6k / Premium $8k+). We do
not personalize beyond what the CSV supports.
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
    raw = (text or "").strip()
    if len(raw) < 2:
        return None

    # Explicit aliases for case-workbook / sales names that diverge from Shopify titles.
    alias = _CASE_MODEL_ALIASES.get(_normalize_key(raw))
    if alias:
        raw = alias

    resolved_title = resolve_model_name(raw)
    if not resolved_title:
        resolved_title = raw
    resolved_key = _normalize_key(resolved_title)
    if not resolved_key:
        return None

    products = load_product_index()
    for record in products:
        if _normalize_key(record.display_name) == resolved_key:
            return record
        if _normalize_key(record.title) == resolved_key:
            return record
        if _normalize_key(record.handle) == resolved_key:
            return record

    for record in products:
        title_key = _normalize_key(record.title)
        display_key = _normalize_key(record.display_name)
        if resolved_key and (
            title_key.startswith(resolved_key)
            or display_key.startswith(resolved_key)
            or resolved_key in title_key
            or resolved_key in display_key
        ):
            return record

    # Token overlap: require most distinctive tokens (ignore brand noise).
    tokens = [t for t in re.findall(r"[a-z0-9]+", raw.lower()) if len(t) >= 3]
    tokens = [t for t in tokens if t not in {"osaki", "titan", "massage", "chair", "pro", "the"}]
    if len(tokens) >= 1:
        best: Optional[ProductSpecs] = None
        best_score = 0
        for record in products:
            hay = f"{record.title} {record.display_name} {record.handle}".lower()
            score = sum(1 for t in tokens if t in hay)
            if score > best_score and score >= max(1, len(tokens) - 1):
                best = record
                best_score = score
        if best is not None:
            return best
    return None


# Case workbook / sales shorthand → Shopify-facing name hints.
_CASE_MODEL_ALIASES: dict[str, str] = {
    _normalize_key("Osaki aI 4D Yoga Flex"): "Osaki 4D Yoga Flex",
    _normalize_key("Grande XL-Big and Tall"): "Titan Grande XL",
    _normalize_key("OS-3D AI Vito"): "Osaki OS-3D AI Vito",
    _normalize_key("Titan Rejuv 4D"): "Titan Rejūv 4D",
    _normalize_key("Ventura 3D"): "Osaki Ventura 3D",
}


def list_active_products() -> list[ProductSpecs]:
    return [p for p in load_product_index() if p.status.lower() == "active"]


# ---------------------------------------------------------------------------
# Recommendation & comparison
# ---------------------------------------------------------------------------


_TALL_MIN_IN = 74  # 6'2"+ → prefer L-Track / SL-Track & 4D
_TALL_HINT_RE = re.compile(r"\b(tall|large|big\s+guy|nba|basketball)\b", re.I)
_PETITE_HINT_RE = re.compile(r"\b(petite|small|short|tiny|wife|mom|mother|elder(?:ly)?)\b", re.I)
_BACK_HINT_RE = re.compile(
    r"\b("
    r"back|spine|lower\s+back|sciatica|posture|"
    r"glute|glutes|buttock|buttocks|hamstring|hamstrings"
    r")\b",
    re.I,
)
_NECK_HINT_RE = re.compile(r"\b(neck|shoulder|shoulders|traps)\b", re.I)
_FEET_HINT_RE = re.compile(r"\b(feet|foot|calf|calves|legs?)\b", re.I)
# "$5k" / "5k" / "under 5k"
_BUDGET_K_RE = re.compile(r"(?:\$|usd\s*)?\s*(\d{1,2})\s*k\b", re.I)
# "$6,000", "under 5000", "budget 6000", "6000$"
_BUDGET_NUM_RE = re.compile(
    r"(?:\$)\s*(\d{1,3}(?:,\d{3})+|\d{3,5})\b|"
    r"\b(?:under|around|about|near|budget(?:\s*(?:of|is|around|under)?)?|max|up\s+to|below)"
    r"\s*(?:of\s+)?(?:\$)?\s*(\d{1,3}(?:,\d{3})+|\d{3,5})\b|"
    r"\b(\d{3,5})\s*\$|"
    r"\b(\d{3,5})\s*(?:usd|dollars?)\b",
    re.I,
)


@dataclass
class RecommendationRequest:
    height_in: Optional[int] = None
    weight_lb: Optional[int] = None
    budget_usd: Optional[int] = None
    focus_areas: list[str] = field(default_factory=list)  # ["back", "neck", "feet"]
    free_text: str = ""


def _parse_budget_usd(raw: str) -> Optional[int]:
    """Pull a chair budget from free text; ignore weight/height numbers."""
    if not raw:
        return None
    k_match = _BUDGET_K_RE.search(raw)
    if k_match:
        num = int(k_match.group(1)) * 1000
        if 500 <= num <= 50000:
            return num

    candidates: list[int] = []
    for match in _BUDGET_NUM_RE.finditer(raw):
        token = next((g for g in match.groups() if g), None)
        if not token:
            continue
        num = int(token.replace(",", ""))
        start, end = match.span()
        tail = raw[end : end + 10].lower()
        # "220 lb" / "95 kg" must never become a budget.
        if re.match(r"\s*(?:lb|lbs|pounds?|kg)\b", tail):
            continue
        if 500 <= num <= 50000:
            candidates.append(num)
    if not candidates:
        return None
    # Prefer the largest plausible chair price when several numbers appear.
    return max(candidates)


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

    req.budget_usd = _parse_budget_usd(raw)

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
        # Treat a stated dollar amount as a *target*, not a hard ceiling that
        # dumps every cheaper chair into the same bucket. "$6,000 chair" should
        # surface ~$5–6k picks, not the $1,999 entry models.
        ratio = product.price_usd / req.budget_usd
        proximity = 1.0 - abs(product.price_usd - req.budget_usd) / req.budget_usd
        if 0.70 <= ratio <= 1.05:
            score += 3.0 + 2.0 * max(0.0, proximity)
        elif 0.50 <= ratio < 0.70:
            score += 1.5
        elif 1.05 < ratio <= 1.15:
            score += 1.0
        elif ratio < 0.50:
            score -= 1.5
        else:
            score -= 2.0

    # Small tie-breaker: prefer higher tier (4D > 3D > 2D) when no signal.
    if product.massage_mechanism == "4D":
        score += 0.3
    elif product.massage_mechanism == "3D":
        score += 0.2

    return score


def _recommend_sort_key(
    pair: tuple[ProductSpecs, float],
    req: RecommendationRequest,
) -> tuple:
    product, score = pair
    price = product.price_usd if product.price_usd is not None else 1e9
    if req.budget_usd:
        # Closer to the stated budget wins ties.
        return (-score, abs(price - req.budget_usd), price)
    # No budget → cheaper first among equal scores (browse-friendly).
    return (-score, price)


# Fallback catalog shelves — aligned with chat Value / Mid / Premium bands.
_PRICE_TIERS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("Value (under ~$3k)", 1500, 3499),
    ("Mid-range (~$5–7k)", 5000, 6999),
    ("Premium ($7k+)", 7000, None),
)


def price_tier_label(price_usd: Optional[float]) -> Optional[str]:
    if price_usd is None:
        return None
    for label, lo, hi in _PRICE_TIERS:
        if price_usd >= lo and (hi is None or price_usd <= hi):
            return label
    return None


def _fit_request(req: RecommendationRequest) -> RecommendationRequest:
    """Body-fit scoring only — budget must not collapse all tiers to one band."""
    return RecommendationRequest(
        height_in=req.height_in,
        weight_lb=req.weight_lb,
        focus_areas=list(req.focus_areas or []),
        free_text=req.free_text,
        budget_usd=None,
    )


def _recommend_across_tiers(req: RecommendationRequest) -> list[ProductSpecs]:
    """Pick the best active chair in each Value / Mid / Premium price shelf."""
    fit_req = _fit_request(req)
    products = [
        p
        for p in load_product_index()
        if p.status.lower() == "active" and p.price_usd is not None
    ]
    picked: list[ProductSpecs] = []
    used: set[str] = set()
    for _label, lo, hi in _PRICE_TIERS:
        band = [
            p
            for p in products
            if p.handle not in used
            and p.price_usd is not None
            and p.price_usd >= lo
            and (hi is None or p.price_usd <= hi)
        ]
        if not band:
            continue
        band.sort(key=lambda p: (-_score(p, fit_req), p.price_usd or 0))
        best = band[0]
        picked.append(best)
        used.add(best.handle)
    return picked


def recommend(req: RecommendationRequest, limit: int = 3) -> list[ProductSpecs]:
    """Return up to ``limit`` best matches for a recommendation request.

    Default (limit ≥ 3): one Value (under ~$3k), one Mid (~$5–7k), one Premium ($7k+)
    pick so shoppers see a clear good / better / best spread.
    """
    limit = max(1, min(limit, 5))
    if limit >= 3:
        tiered = _recommend_across_tiers(req)
        if tiered:
            return tiered[:limit]

    ranked = [
        (product, _score(product, req))
        for product in load_product_index()
    ]
    ranked = [pair for pair in ranked if pair[1] > 0]
    ranked.sort(key=lambda pair: _recommend_sort_key(pair, req))
    return [product for product, _ in ranked[:limit]]

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
            "zero_gravity": (left.zero_gravity, right.zero_gravity),
            "heating": (left.heating, right.heating),
            "foot_roller": (left.foot_roller, right.foot_roller),
        },
    }
