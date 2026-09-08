"""
Nickname matching for sales compare.

Shoppers type Maestro, not the Shopify title. This module finds *active*
store chairs from those short names, splits A vs B without requiring a
full title, and scores a published-fit pick between two resolved chairs.

It never invents a model that is not in the current Shopify export.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from sales_catalog import (
    ProductSpecs,
    RecommendationRequest,
    _score,
    list_active_products,
    parse_recommendation_hints,
)
from sales_spec_index import doorway_ok, lookup_fit_spec, wall_ok, weight_ok

_BRAND_PREFIXES = (
    "osaki platinum ",
    "osaki japan ",
    "osaki pro ",
    "osaki ",
    "titan premium ",
    "titan pro ",
    "titan ",
    "ador ",
    "amamedic ",
    "os-pro ",
    "os-3d ",
    "os-4d ",
    "os ",
    "tp-",
    "tp ",
    "pro-",
    "pro ",
    "premium ",
)

_BRAND_TOKENS = frozenset(
    {
        "osaki",
        "titan",
        "ador",
        "amamedic",
        "massage",
        "chair",
        "chairs",
        "the",
        "pro",
        "premium",
        "tp",
        "os",
        "usa",
        "platinum",
        "japan",
        "series",
        "model",
        "models",
        "ti",
    }
)

_MECH_TOKENS = frozenset({"2d", "3d", "4d", "5d", "le", "ii", "xl", "plus", "max"})

_VAGUE_COMPACT = frozenset(
    {
        "osaki",
        "titan",
        "ador",
        "2d",
        "3d",
        "4d",
        "5d",
        "le",
        "ii",
        "chair",
        "massagechair",
    }
)

_TRAILING_FLUFF_RE = re.compile(
    r"[\s,]+(?:"
    r"and\s+recommend\b.*|"
    r"recommend(?:\s+which)?\b.*|"
    r"which\s+is\s+better\b.*|"
    r"which\s+one\b.*|"
    r"please|thanks|thank\s+you|"
    r"비교.*|추천.*"
    r")\s*$",
    re.IGNORECASE,
)

_LEADING_FLUFF_RE = re.compile(
    r"^(?:please\s+|compare\s+|comparison\s+|비교(?:해(?:줘|요|주세요)?)?\s*|"
    r"the\s+|which\s+is\s+better[,]?\s*)+",
    re.IGNORECASE,
)

_VS_RE = re.compile(
    r"\s*\b(?:vs\.?|versus|compared\s+to|compared\s+with)\b\s*",
    re.IGNORECASE,
)

_DIFF_BETWEEN_RE = re.compile(
    r"\bdifference\s+between\s+(.+)",
    re.IGNORECASE,
)

_COMPARE_PAIR_RE = re.compile(
    r"^(?:compare|comparison|비교)\s+(.+?)\s+"
    r"(?:and|with|to|랑|와|이랑|하고)\s+(.+)$",
    re.IGNORECASE,
)

_KO_PAIR_RE = re.compile(r"^(.+?)\s*(?:이랑|랑|와|하고)\s+(.+)$")

_OR_RE = re.compile(r"\s+\bor\b\s+", re.IGNORECASE)

_AND_RE = re.compile(r"\s+and\s+", re.IGNORECASE)

_BIG_AND_TALL_RE = re.compile(r"\bbig\s+and\s+tall\b", re.IGNORECASE)

_WHICH_OF_PAIR_RE = re.compile(
    r"\b("
    r"which\s+one|"
    r"recommend\s+which|"
    r"which\s+(?:should|do)\s+i|"
    r"둘\s*중|"
    r"어느\s*(?:거|것|쪽)"
    r")\b",
    re.IGNORECASE,
)

_HEIGHT_IN = {
    'Petite (<5\'4")': 62,
    'Average (5\'4"–5\'11")': 68,
    'Tall (6\'0"–6\'2")': 73,
    'Extra Tall (6\'3"+)': 76,
}

_WEIGHT_LB = {
    "≤180 lb": 170,
    "181–220 lb": 200,
    "221–260 lb": 240,
    "261–300 lb": 280,
    "301+ lb": 320,
}


@dataclass(frozen=True)
class ModelLookup:
    query: str
    matches: tuple[ProductSpecs, ...]
    vague: bool = False

    @property
    def unique(self) -> Optional[ProductSpecs]:
        if len(self.matches) == 1:
            return self.matches[0]
        return None


@dataclass(frozen=True)
class _NameCard:
    product: ProductSpecs
    tokens: tuple[str, ...]
    family: str
    specific: str
    aliases: frozenset[str]


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _tokens(text: str) -> tuple[str, ...]:
    raw = (text or "").lower().replace("2.0", "20")
    out: list[str] = []
    for token in re.findall(r"[a-z0-9]+", raw):
        if token in _BRAND_TOKENS:
            continue
        if len(token) < 2:
            continue
        out.append(token)
    return tuple(out)


def _family_key(tokens: tuple[str, ...]) -> str:
    for token in tokens:
        if token not in _MECH_TOKENS and len(token) >= 3:
            return token
    return tokens[0] if tokens else ""


def _strip_brand_prefixes(name: str) -> list[str]:
    """Successively peel Osaki / OS-Pro / Titan prefixes for nickname aliases."""
    found = [name]
    current = name
    changed = True
    while changed and current:
        changed = False
        lower = current.lower().lstrip()
        for prefix in _BRAND_PREFIXES:
            if lower.startswith(prefix):
                current = current[len(prefix) :].strip(" -")
                if current and current not in found:
                    found.append(current)
                changed = True
                break
    return found


def _aliases_for(product: ProductSpecs) -> frozenset[str]:
    aliases = set()
    for raw in (product.display_name, product.title, product.handle.replace("-", " ")):
        for variant in _strip_brand_prefixes(raw):
            compact = _compact(variant)
            if compact:
                aliases.add(compact)
    tokens = _tokens(product.display_name)
    if tokens:
        aliases.add("".join(tokens))
        family = _family_key(tokens)
        if family:
            aliases.add(family)
    return frozenset(a for a in aliases if a)


@lru_cache(maxsize=1)
def _name_index() -> tuple[_NameCard, ...]:
    cards: list[_NameCard] = []
    for product in list_active_products():
        tokens = _tokens(product.display_name)
        cards.append(
            _NameCard(
                product=product,
                tokens=tokens,
                family=_family_key(tokens),
                specific="".join(tokens),
                aliases=_aliases_for(product),
            )
        )
    return tuple(cards)


def product_by_handle(handle: str) -> Optional[ProductSpecs]:
    key = (handle or "").strip().lower()
    if not key:
        return None
    for product in list_active_products():
        if product.handle.lower() == key:
            return product
    return None


def clean_compare_term(text: str) -> str:
    raw = (text or "").strip(" ?,.")
    raw = _TRAILING_FLUFF_RE.sub("", raw)
    raw = _LEADING_FLUFF_RE.sub("", raw)
    return raw.strip(" ?,.")


def _protect_big_and_tall(text: str) -> str:
    return _BIG_AND_TALL_RE.sub("big+tall", text)


def _unprotect_big_and_tall(text: str) -> str:
    return re.sub(r"big\+tall", "Big and Tall", text, flags=re.IGNORECASE)


def _looks_model_term(text: str) -> bool:
    cleaned = clean_compare_term(text)
    if len(cleaned) < 2:
        return False
    tokens = _tokens(cleaned)
    if not tokens:
        return False
    if set(tokens) <= _MECH_TOKENS:
        return False
    return True


def _pair(left: str, right: str) -> Optional[tuple[str, str]]:
    left = clean_compare_term(_unprotect_big_and_tall(left))
    right = clean_compare_term(_unprotect_big_and_tall(right))
    if _looks_model_term(left) and _looks_model_term(right):
        return left, right
    return None


def split_compare_terms(message: str) -> Optional[tuple[str, str]]:
    """Pull two model hints from shopper phrasing, including short nicknames."""
    raw = clean_compare_term(message or "")
    if not raw:
        return None
    protected = _protect_big_and_tall(raw)

    vs_parts = _VS_RE.split(protected, maxsplit=1)
    if len(vs_parts) == 2:
        pair = _pair(vs_parts[0], vs_parts[1])
        if pair:
            return pair

    diff = _DIFF_BETWEEN_RE.search(protected)
    if diff:
        and_parts = _AND_RE.split(diff.group(1), maxsplit=1)
        if len(and_parts) == 2:
            pair = _pair(and_parts[0], and_parts[1])
            if pair:
                return pair

    compare_and = _COMPARE_PAIR_RE.match(protected)
    if compare_and:
        pair = _pair(compare_and.group(1), compare_and.group(2))
        if pair:
            return pair

    ko = _KO_PAIR_RE.match(raw)
    if ko:
        pair = _pair(ko.group(1), ko.group(2))
        if pair:
            return pair

    and_parts = _AND_RE.split(protected, maxsplit=1)
    if len(and_parts) == 2:
        pair = _pair(and_parts[0], and_parts[1])
        if pair:
            return pair

    or_parts = _OR_RE.split(protected, maxsplit=1)
    if len(or_parts) == 2:
        pair = _pair(or_parts[0], or_parts[1])
        if pair:
            return pair
    return None


def lookup_shop_models(text: str) -> ModelLookup:
    """Active Shopify chairs that match a shopper nickname.

    Zero matches: not on the current store catalog (or too vague).
    One match: safe to compare.
    Several matches: ask which family member.
    """
    query = clean_compare_term(text)
    compact = _compact(query)
    tokens = _tokens(query)
    if len(query) < 2:
        return ModelLookup(query=query, matches=(), vague=True)
    if compact in _VAGUE_COMPACT or (tokens and set(tokens) <= _MECH_TOKENS):
        return ModelLookup(query=query, matches=(), vague=True)
    if not tokens:
        return ModelLookup(query=query, matches=(), vague=True)

    cards = _name_index()
    exact = [card.product for card in cards if compact in card.aliases]
    if exact:
        return ModelLookup(query=query, matches=tuple(exact), vague=False)

    scored: list[tuple[int, ProductSpecs]] = []
    wanted = set(tokens)
    for card in cards:
        have = set(card.tokens)
        if not wanted <= have:
            continue
        extra = len(have - wanted)
        scored.append((extra, card.product))
    if not scored:
        return ModelLookup(query=query, matches=(), vague=False)
    best_extra = min(item[0] for item in scored)
    top = tuple(item[1] for item in scored if item[0] == best_extra)
    return ModelLookup(query=query, matches=top, vague=False)


def short_model_label(product: ProductSpecs, *, limit: int = 28) -> str:
    name = (product.display_name or "").strip()
    for prefix in ("Osaki ", "Titan ", "Ador ", "AmaMedic "):
        if name.startswith(prefix):
            name = name[len(prefix) :].strip()
            break
    if len(name) <= limit:
        return name
    return name[: max(1, limit - 1)].rstrip() + "…"


def is_which_of_pair(message: str) -> bool:
    return bool(_WHICH_OF_PAIR_RE.search(message or ""))


def looks_like_compare_pair(message: str) -> bool:
    """True when the message names two catalog nicknames, even without 'vs'."""
    req = parse_recommendation_hints(message or "")
    if req.height_in and req.weight_lb:
        return False
    pair = split_compare_terms(message or "")
    if pair is None:
        return False
    left = lookup_shop_models(pair[0])
    right = lookup_shop_models(pair[1])
    if left.vague or right.vague:
        return False
    return bool(left.matches or right.matches)


def request_from_prefs(prefs: Optional[dict]) -> RecommendationRequest:
    data = prefs or {}
    height = str(data.get("height") or "").strip()
    weight = str(data.get("weight") or "").strip()
    goal = str(data.get("goal") or "").strip().lower()
    focus: list[str] = []
    if "neck" in goal or "shoulder" in goal:
        focus.append("neck")
    if "back" in goal or "hip" in goal or "seat" in goal:
        focus.append("back")
    if "foot" in goal or "calf" in goal:
        focus.append("feet")
    return RecommendationRequest(
        height_in=_HEIGHT_IN.get(height),
        weight_lb=_WEIGHT_LB.get(weight),
        focus_areas=focus,
        free_text=goal,
    )


def pair_fit_scores(
    left: ProductSpecs,
    right: ProductSpecs,
    prefs: Optional[dict],
) -> tuple[float, float]:
    """Higher is a better published fit. Catalog facts only."""
    rec = dict((prefs or {}).get("recommend_prefs") or prefs or {})
    req = request_from_prefs(rec)
    weight = str(rec.get("weight") or "").strip()
    space = str(rec.get("space") or "").strip()
    doorway = rec.get("doorway_in")
    try:
        door_limit = float(doorway) if doorway and str(doorway) != "skip" else None
    except (TypeError, ValueError):
        door_limit = None
    mode = str(rec.get("doorway_fit") or "assembled").strip().lower() or "assembled"

    def _one(product: ProductSpecs) -> float:
        score = _score(product, req)
        name = product.display_name
        if weight:
            score += 5.0 if weight_ok(name, weight) else -8.0
        if door_limit is not None:
            score += 3.0 if doorway_ok(name, limit_in=door_limit, mode=mode) else -8.0
        if space:
            score += 2.0 if wall_ok(name, space) else -4.0
        return score

    return _one(left), _one(right)


def pair_fit_reasons(
    winner: ProductSpecs,
    other: ProductSpecs,
    prefs: Optional[dict],
) -> list[str]:
    rec = dict((prefs or {}).get("recommend_prefs") or prefs or {})
    reasons: list[str] = []
    weight = str(rec.get("weight") or "").strip()
    if weight and weight_ok(winner.display_name, weight) and not weight_ok(
        other.display_name, weight
    ):
        spec = lookup_fit_spec(winner.display_name)
        if spec and spec.max_user_lb is not None:
            reasons.append(
                f"**{winner.display_name}** lists a **{spec.max_user_lb:g} lb** max user weight."
            )
    if winner.massage_mechanism and winner.massage_mechanism != other.massage_mechanism:
        reasons.append(
            f"Mechanism: **{winner.massage_mechanism}** vs {other.massage_mechanism or '—'}."
        )
    if winner.track_type and winner.track_type != other.track_type:
        reasons.append(
            f"Track: **{winner.track_type}** vs {other.track_type or '—'}."
        )
    goal = str(rec.get("goal") or "").strip()
    if goal:
        reasons.append(f"Scored against **{goal}** plus the published specs.")
    return reasons[:4]
