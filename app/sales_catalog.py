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
import html
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from catalog_identity import generation_normalized as _generation_normalized, sentence_product as _sentence_product

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
    colors: tuple[str, ...] = ()
    bluetooth_speakers: str = ""
    body_scanning: str = ""
    feature_sources: tuple[tuple[str, str], ...] = ()

    def as_public_dict(self) -> dict:
        public = {
            "model": self.display_name,
            "catalog_handle": self.handle,
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
                "bluetooth_speakers": self.bluetooth_speakers,
                "body_scanning": self.body_scanning,
            },
        }
        if self.feature_sources:
            public["feature_sources"] = dict(self.feature_sources)
        if self.colors:
            public["colors"] = list(self.colors)
        from sales_spec_index import lookup_fit_spec
        fit = lookup_fit_spec(self.display_name)
        public['fit'] = {
            'maximum_user_weight_lb': fit.max_user_lb if fit else None,
            'doorway_assembled_in': fit.door_asm_in if fit else None,
            'doorway_disassembled_in': fit.door_dis_in if fit else None,
            'wall_clearance_in': fit.wall_clearance_in if fit else None,
            'minimum_user_height_in': None,
            'maximum_user_height_in': None,
        }
        return public


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


_COLOR_OPTION_NAMES = frozenset({"color", "colour"})
_COLOR_CANON = {
    "black": "Black",
    "brown": "Brown",
    "dark brown": "Dark Brown",
    "white": "White",
    "beige": "Beige",
    "grey": "Gray",
    "gray": "Gray",
    "ivory": "Ivory",
    "espresso": "Espresso",
    "charcoal": "Charcoal",
    "taupe": "Taupe",
}
_COLOR_RE = re.compile(
    r"\b(dark\s+brown|black|brown|white|beige|grey|gray|ivory|espresso|charcoal|taupe)\b",
    re.I,
)
_TYPO_TOKEN = {
    "tital": "titan",
    "titen": "titan",
    "tytan": "titan",
    "osacki": "osaki",
    "oskai": "osaki",
    "maesto": "maestro",
    "maetro": "maestro",
    "paragun": "paragon",
    "paragonn": "paragon",
}
_LOOKUP_FILLER = frozenset(
    {
        "the",
        "a",
        "an",
        "in",
        "for",
        "with",
        "please",
        "color",
        "colour",
        "available",
        "model",
        "chair",
        "massage",
        "do",
        "does",
        "did",
        "you",
        "your",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "get",
        "got",
        "want",
        "this",
        "that",
        "any",
        "some",
        "one",
        "it",
        "is",
        "are",
        "was",
        "were",
    }
)


def rewrite_shopper_typos(text: str) -> str:
    """Fix a few storefront typos (Tital → Titan) without fuzzy-matching Jupiter."""

    def _repl(match: re.Match[str]) -> str:
        word = match.group(0)
        mapped = _TYPO_TOKEN.get(word.lower())
        if not mapped:
            return word
        if word.isupper():
            return mapped.upper()
        if word[:1].isupper():
            return mapped[:1].upper() + mapped[1:]
        return mapped

    return re.sub(r"[A-Za-z]+", _repl, text or "")


def parse_asked_color(text: str) -> Optional[str]:
    """Return a canonical color when the shopper named one — never Black Friday."""
    raw = text or ""
    if re.search(r"\bblack\s+friday\b", raw, re.I):
        raw = re.sub(r"\bblack\s+friday\b", " ", raw, flags=re.I)
    match = _COLOR_RE.search(raw)
    if not match:
        return None
    return _COLOR_CANON.get(match.group(1).lower())


def looks_like_color_question(text: str) -> bool:
    return parse_asked_color(text) is not None


def _strip_lookup_noise(text: str) -> str:
    raw = rewrite_shopper_typos(text or "")
    raw = re.sub(r"\bblack\s+friday\b", " ", raw, flags=re.I)
    raw = _COLOR_RE.sub(" ", raw)
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", raw)
        if token.lower() not in _LOOKUP_FILLER
    ]
    return " ".join(tokens)


def _row_colors(row: dict) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for index in (1, 2, 3):
        name = str(row.get(f"Option{index} Name") or "").strip().lower()
        value = str(row.get(f"Option{index} Value") or "").strip()
        if name not in _COLOR_OPTION_NAMES or not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(value)
    return tuple(found)


def _feature_from_description(description: str, pattern: str) -> str:
    """Read explicit feature statements; missing text never establishes 'No'."""
    text = re.sub(r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>", "", description or "", flags=re.I | re.S)
    text = re.sub(r"</?(?:li|p|h[1-6]|div|br)\b[^>]*>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    observed: set[str] = set()
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        for match in re.finditer(pattern, line, re.I):
            before = line[max(0, match.start() - 50):match.start()]
            after = line[match.end():match.end() + 45]
            if line.endswith("?") or re.search(r"\b(?:optional|external|separate)\s*$", before, re.I):
                continue
            negative = bool(re.search(r"\b(?:no|without|does not (?:have|include)|doesn't (?:have|include)|not equipped with)\s*(?:(?:any|built[ -]?in)\s*)?$", before, re.I)
                            or re.match(r"\s+(?:is |are )?(?:not (?:included|available|supported)|unavailable)\b", after, re.I))
            observed.add("No" if negative else "Yes")
    return next(iter(observed)) if len(observed) == 1 else ""


def _published_features(row: dict) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    raw_bluetooth = str(row.get("Bluetooth Speaker (product.metafields.custom.bluetooth_speaker)") or "").strip()
    bluetooth = {"yes": "Yes", "true": "Yes", "1": "Yes", "no": "No", "false": "No", "0": "No"}.get(raw_bluetooth.lower(), "")
    source = "Shopify export: Bluetooth Speaker metafield"
    described_bluetooth = _feature_from_description(row.get("Body (HTML)", ""), r"\bbluetooth\s+(?:(?:sound|audio)\s+)?speakers?\b")
    if bluetooth and described_bluetooth and bluetooth != described_bluetooth:
        bluetooth = ""
    elif not bluetooth:
        bluetooth = described_bluetooth
        source = "Shopify export: product description"
    body_scan = _feature_from_description(row.get("Body (HTML)", ""), r"\bbody[\s-]+scan(?:ning|ner)?\b")
    sources = []
    if bluetooth:
        sources.append(("bluetooth_speakers", source))
    if body_scan:
        sources.append(("body_scanning", "Shopify export: product description"))
    return bluetooth, body_scan, tuple(sources)


def product_feature_answer(product: ProductSpecs, query: str) -> str | None:
    """Answer requested published features without turning missing data into a fact."""
    lines = []
    if re.search(r"\bcolou?rs?\b", query or "", re.I):
        if product.colors:
            lines.append(f"- **Listed colors**: {', '.join(product.colors)}. Checkout confirms current availability.")
        else:
            lines.append("- **Colors**: not verified in this catalog. Sales can confirm the options for this exact model.")
    for pattern, label, value in (
        (r"\bbluetooth\b|\bspeakers?\b", "Bluetooth speakers", product.bluetooth_speakers),
        (r"\bbody[\s-]+scan(?:ning|ner)?\b|\bscan(?:s|ning)?\b.{0,30}\bbody\b|\bbody\b.{0,24}\bscanned\b", "Body scanning", product.body_scanning),
        (r"\b(?:mobile\s+)?app(?:lication)?\b|\b(?:smartphone|phone)\b.{0,24}\b(?:control|remote)\b", "App control", ""),
        (r"\bvoice[\s-]+(?:control|commands?|assistant)\b", "Voice control", ""),
    ):
        if re.search(pattern, query or "", re.I):
            if value:
                lines.append(f"- **{label}**: {value}, according to the published catalog.")
            else:
                lines.append(f"- **{label}**: not verified in this catalog. Sales can confirm this feature for the exact model.")
    return f"**{product.display_name}**\n" + "\n".join(lines) if lines else None


def is_public_browse_pick(product: ProductSpecs) -> bool:
    """True for storefront chairs we may show as anonymous price examples."""
    if product.status.lower() != "active" or product.price_usd is None:
        return False
    hay = f"{product.title} {product.display_name} {product.handle}".lower()
    if "costco" in hay:
        return False
    if "quest-3d" in hay or "quest 3d" in hay or hay.endswith("quest 3d"):
        return False
    return True


def color_is_listed(product: ProductSpecs, color: str) -> list[str]:
    wanted = (color or "").strip().lower()
    if not wanted:
        return []
    return [item for item in product.colors if wanted in item.lower()]


@lru_cache(maxsize=1)
def load_product_index() -> tuple[ProductSpecs, ...]:
    """One record per Shopify handle, deduplicated across variant rows."""
    if not _CSV_PATH.is_file():
        logger.warning("sales_catalog: %s not found", _CSV_PATH)
        return ()

    prices = load_catalog_base_prices()

    by_handle: dict[str, dict] = {}
    colors_by_handle: dict[str, list[str]] = {}
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
            bucket = colors_by_handle.setdefault(current_handle, [])
            seen_colors = {item.lower() for item in bucket}
            for color in _row_colors(row):
                if color.lower() in seen_colors:
                    continue
                seen_colors.add(color.lower())
                bucket.append(color)

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
        bluetooth, body_scanning, feature_sources = _published_features(row)
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
            colors=tuple(colors_by_handle.get(handle) or ()),
            bluetooth_speakers=bluetooth,
            body_scanning=body_scanning,
            feature_sources=feature_sources,
        )
        entries.append(specs)

    logger.info("sales_catalog: loaded %d products", len(entries))
    return tuple(entries)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def resolve_product(text: str) -> Optional[ProductSpecs]:
    """Find a single product identity without substituting another generation."""
    raw = _strip_lookup_noise(text or "")
    if len(raw) < 2:
        return None

    # Explicit aliases for case-workbook / sales names that diverge from Shopify titles.
    alias = _CASE_MODEL_ALIASES.get(_normalize_key(raw))
    if alias:
        raw = alias

    products = load_product_index()
    raw_keys = {_normalize_key(raw), _normalize_key(text or "")}
    exact = [record for record in products if raw_keys & {
        _normalize_key(record.display_name), _normalize_key(record.title), _normalize_key(record.handle),
    }]
    if len(exact) == 1:
        return exact[0]
    named, has_named_identity = _sentence_product(rewrite_shopper_typos(alias or text), products)
    if has_named_identity:
        return named
    # A version explicitly supplied by the customer is a hard identity limit.
    if re.search(r"\b(?:le\s*|v(?:ersion)?\s*)\d+(?:\.\d+)?\b", text or "", re.I):
        return None

    resolved_title = resolve_model_name(raw)
    if not resolved_title:
        resolved_title = raw
    resolved_key = _normalize_key(resolved_title)
    if not resolved_key:
        return None

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
    tokens = [
        t
        for t in tokens
        if t not in {"osaki", "titan", "massage", "chair", "pro", "the"} | _LOOKUP_FILLER
    ]
    if len(tokens) >= 1:
        scored: list[tuple[int, ProductSpecs]] = []
        need = max(1, len(tokens) - 1)
        for record in products:
            hay_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    f"{record.title} {record.display_name} {record.handle}".lower(),
                )
            )
            score = sum(1 for t in tokens if t in hay_tokens)
            if score >= need:
                scored.append((score, record))
        if scored:
            best_score = max(item[0] for item in scored)
            top = [record for score, record in scored if score == best_score]
            if "xl" in tokens:
                xl_hits = [
                    record
                    for record in top
                    if "xl" in f"{record.title} {record.display_name} {record.handle}".lower()
                ]
                if xl_hits:
                    top = xl_hits
            if len(top) == 1:
                return top[0]
            # Prefer the shortest display name among remaining ties (Grande XL
            # over a longer XL family member when scores match).
            top.sort(key=lambda rec: (len(rec.display_name), rec.display_name))
            return top[0]
    return None


# Case workbook / sales shorthand → Shopify-facing name hints.
_CASE_MODEL_ALIASES: dict[str, str] = {
    _normalize_key("LE 2.0"): "Osaki 4D Maestro LE 2.0",
    _normalize_key("Osaki aI 4D Yoga Flex"): "Osaki 4D Yoga Flex",
    _normalize_key("Grande XL-Big and Tall"): "Titan Grande XL",
    _normalize_key("Grande XL"): "Titan Grande XL",
    _normalize_key("Titan XL 3D"): "Titan Grande XL",
    _normalize_key("Tital XL 3D"): "Titan Grande XL",
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
        if 0 <= num <= 50000:
            return num

    candidates: list[int] = []
    for match in _BUDGET_NUM_RE.finditer(raw):
        token = next((g for g in match.groups() if g), None)
        if not token:
            continue
        num = int(token.replace(",", ""))
        start, end = match.span()
        tail = raw[end : end + 32].lower()
        # "220 lb" / "95 kg" must never become a budget.
        if re.match(r"\s*(?:lbs?|pounds?|kgs?|kilograms?|cm|centimet(?:er|re)s?|inches|inch|feet|ft)\b", tail):
            continue
        if 0 <= num <= 50000:
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

    from sales_constraints import extract_constraints, height_measurements, without_height_measurements

    heights = height_measurements(raw)
    if heights:
        req.height_in = heights[-1][2]

    exact = extract_constraints(raw)
    if exact.get('weight_lb'):
        req.weight_lb = int(round(float(exact['weight_lb'])))
    if 'budget_max_usd' in exact:
        req.budget_usd = int(float(exact['budget_max_usd'])) if exact['budget_max_usd'] else None
    else:
        req.budget_usd = _parse_budget_usd(raw)

    # Bare "tall" / "petite" is not a measured height. Ask instead of guessing
    # Extra Tall (6'3"+) or Petite (<5'4").

    focus = []
    focus_text = without_height_measurements(raw)
    if _BACK_HINT_RE.search(focus_text):
        focus.append("back")
    if _NECK_HINT_RE.search(focus_text):
        focus.append("neck")
    if _FEET_HINT_RE.search(focus_text):
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
        # Eligible products have already passed the hard ceiling. Within
        # that set, prefer prices nearer the requested budget.
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
    ("Value (under ~$3k)", 0, 2999),
    ("Mid-range (~$3–7k)", 3000, 6999),
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
    """Rank body preferences within shelves already filtered by hard limits."""
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
    from sales_constraints import extract_constraints, product_constraint_issues
    constraints = extract_constraints(req.free_text)
    if req.budget_usd is not None:
        constraints['budget_max_usd'] = str(req.budget_usd)
    if req.weight_lb is not None:
        constraints['weight_lb'] = str(req.weight_lb)
    products = [
        p
        for p in load_product_index()
        if p.status.lower() == "active" and p.price_usd is not None
        and not product_constraint_issues(p, constraints)
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

    from sales_constraints import extract_constraints, product_constraint_issues
    constraints = extract_constraints(req.free_text)
    if req.budget_usd is not None:
        constraints['budget_max_usd'] = str(req.budget_usd)
    if req.weight_lb is not None:
        constraints['weight_lb'] = str(req.weight_lb)
    ranked = [
        (product, _score(product, req))
        for product in load_product_index()
        if not product_constraint_issues(product, constraints)
    ]
    ranked = [pair for pair in ranked if pair[1] > 0]
    ranked.sort(key=lambda pair: _recommend_sort_key(pair, req))
    return [product for product, _ in ranked[:limit]]

def compare_products(left: ProductSpecs, right: ProductSpecs) -> dict:
    """Structured spec/price diff for two already-resolved catalog rows."""
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


def compare(a_text: str, b_text: str) -> Optional[dict]:
    """Structured comparison between two models — deterministic, no LLM."""
    left = resolve_product(a_text)
    right = resolve_product(b_text)
    if not left or not right:
        return None
    return compare_products(left, right)
