"""Shared product-name identity rules independent of catalog loaders and prices."""
from __future__ import annotations

import re
import unicodedata
from typing import Protocol, Sequence, TypeVar


class CatalogName(Protocol):
    handle: str
    display_name: str


P = TypeVar("P", bound=CatalogName)
_IDENTITY_NOISE = frozenset({
    "osaki", "titan", "amamedic", "kanji", "os", "op", "tp", "ti", "pro", "ai", "massage", "chair", "chairs",
    "platinum", "premium", "series", "official", "warranty", "with", "free", "and", "big", "tall", "2d", "3d", "4d", "5d",
})


def _identity_words(text: str) -> list[tuple[str, int, int]]:
    # Decimal generations and short identity tokens such as LE, XL and II matter.
    value = "".join(c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c)).lower()
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"[a-z0-9]+(?:\.[0-9]+)?", value)
            if m.group() not in _IDENTITY_NOISE]


def generation_normalized(text: str) -> str:
    return re.sub(r"\bmaestro\s+le\s*(?:v(?:ersion)?\s*)?2(?:\.0)?\b(?![.\d])", "Maestro LE 2.0", text or "", flags=re.I)


def _named_product_matches(text: str, products: Sequence[P]) -> list[tuple[int, int, int, P]]:
    words = _identity_words(text)
    tokens = [word for word, _, _ in words]
    matches = []
    for product in products:
        name = re.sub(r"\([^)]*\)", " ", generation_normalized(product.display_name))
        wanted = [word for word, _, _ in _identity_words(name)]
        if not wanted:
            continue
        for offset in range(len(tokens) - len(wanted) + 1):
            if tokens[offset:offset + len(wanted)] != wanted:
                continue
            start, end = words[offset][1], words[offset + len(wanted) - 1][2]
            before = text[max(0, start - 70):start].replace("-", " ")
            if re.search(r"\b(?:not|rather than|instead of|other than)\s+(?:(?:the|a|an|old|older|original|previous|osaki|titan|os|pro|[2345]d)\s+)*$", before, re.I):
                continue
            # An explicit later generation cannot fall back to an older LE.
            if re.match(r"\s*(?:[2345]d\s*)?(?:le\s*)?(?:v(?:ersion)?\s*)?\d+(?:\.\d+)?\b(?!\s*(?:years?|months?|days?|dollars?|lb|pounds?)\b)", text[end:], re.I):
                continue
            matches.append((start, end, len(wanted), product))
    return matches


def sentence_product(text: str, products: Sequence[P]) -> tuple[P | None, bool]:
    """Return one named identity; true means legacy fuzzy matching must stop."""
    normalized = generation_normalized(text)
    matches = _named_product_matches(normalized, products)
    # Contained family names are subordinate to a more specific generation.
    matches = [item for item in matches if not any(
        other[0] <= item[0] and other[1] >= item[1] and other[2] > item[2]
        for other in matches
    )]
    identities = {item[3].handle: item[3] for item in matches}
    if len(identities) != 1:
        return None, bool(identities)
    found = next(iter(identities.values()))
    # Do not silently discard an unknown alternate model such as OS-4000T.
    for alternative in re.split(r"\b(?:or|versus|vs\.?)\b", normalized, flags=re.I):
        if re.search(r"\b(?:os|tp|rt)[-\s]*\d+[a-z]*\b", alternative, re.I):
            if not _named_product_matches(alternative, (found,)):
                return None, True
    return found, True
