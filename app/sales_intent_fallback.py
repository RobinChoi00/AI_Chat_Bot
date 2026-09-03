"""
sales_intent_fallback.py
========================
Second-chance routing for messages the regex classifier labels ``unclear``.

Production measurement (81 sessions, 410 turns) showed ``unclear`` was the
second most common label at 27% of all turns — the single largest source of
customer dissatisfaction. Those shoppers were asking answerable questions in
phrasing the regexes did not cover.

Two layers run in order, and both only ever choose an existing intent *label*:

1. ``rule_fallback`` — offline and deterministic. Catches a named catalog
   model, browse/choose-help phrasing, and "tell me more" phrasing.
2. ``llm_fallback`` — optional, gated on ``SALES_INTENT_LLM`` plus an API key.
   Classifies into the same closed label set.

The critical safety property: neither layer ever writes customer-facing text.
Every reply is still produced by the deterministic builders in
``sales_agent``, so the bot cannot invent a price, a spec, or a delivery date
no matter what the model returns. A failure at either layer leaves the
original ``unclear`` result untouched.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Optional

from sales_intent import (
    INTENT_CANCEL_REFUND,
    INTENT_COMPARE,
    INTENT_DISCOUNT,
    INTENT_ETA_SHIPPING,
    INTENT_GREETING,
    INTENT_HUMAN,
    INTENT_ORDER_STATUS,
    INTENT_PARTS_TECHNICIAN,
    INTENT_PRICE,
    INTENT_RECOMMEND,
    INTENT_SPECS,
    INTENT_STOCK,
    INTENT_WARRANTY_REDIRECT,
    HANDOFF_INTENTS,
    SalesIntent,
)

logger = logging.getLogger(__name__)

# Labels the fallback layers are allowed to return. Deliberately excludes
# ``unclear`` (the caller already has that) and ``intensity`` (too easily
# confused with a recommend request — see the body-fit override upstream).
_ALLOWED_LABELS = frozenset(
    {
        INTENT_CANCEL_REFUND,
        INTENT_COMPARE,
        INTENT_DISCOUNT,
        INTENT_ETA_SHIPPING,
        INTENT_GREETING,
        INTENT_HUMAN,
        INTENT_ORDER_STATUS,
        INTENT_PARTS_TECHNICIAN,
        INTENT_PRICE,
        INTENT_RECOMMEND,
        INTENT_SPECS,
        INTENT_STOCK,
        INTENT_WARRANTY_REDIRECT,
    }
)

# Catalog tokens that are ordinary English or accessory words. A shopper
# typing these means the word, not the model, so they must not resolve a
# product ("do you have a cover?", "any deal?", "is it a recliner?").
_AMBIGUOUS_MODEL_TOKENS = frozenset(
    {
        "bundle",
        "card",
        "cleaner",
        "cover",
        "deal",
        "escape",
        "fence",
        "flagship",
        "grand",
        "grande",
        "haven",
        "japan",
        "made",
        "master",
        "mech",
        "package",
        "platinum",
        "premium",
        "reader",
        "recliner",
        "relax",
        "signature",
        "solo",
        "tall",
        "therabed",
        "ultra",
        "ultima",
        "vending",
        "yoga",
    }
)

_BRAND_NOISE = frozenset(
    {
        "osaki",
        "titan",
        "massage",
        "chair",
        "chairs",
        "series",
        "model",
        "models",
    }
)

# "Help me pick" phrasing — the shopper wants a recommendation but never used
# the word "recommend".
_BROWSE_HELP_RE = re.compile(
    r"("
    r"help\s+me\s+(?:pick|choose|decide|find|select)|"
    r"(?:not|don'?t)\s+(?:sure|know)\s+(?:which|what|where)|"
    r"which\s+one|"
    r"too\s+many\s+(?:options|choices|models)|"
    r"what\s+(?:do\s+you|models?\s+do\s+you)\s+(?:have|carry|sell|offer)|"
    r"what'?s?\s+available|"
    r"show\s+me\s+(?:your|the|some|all)?\s*(?:chairs?|models?|options?|catalog)|"
    r"see\s+(?:your|the|all)\s+(?:chairs?|models?|options?|catalog)|"
    r"(?:full\s+)?(?:catalog|line\s?up|product\s+list)|"
    r"what\s+should\s+i\s+(?:get|buy|pick|choose)|"
    r"looking\s+(?:for|to\s+buy)\s+(?:a|an|the|some)?\s*(?:chair|massager|recliner)|"
    r"first\s+time\s+buyer|"
    r"new\s+to\s+(?:this|massage\s+chairs?)|"
    r"where\s+(?:do|should)\s+i\s+start|"
    r"어떤\s*(?:거|것|모델|의자)|골라|고르"
    r")",
    re.IGNORECASE,
)

# "Tell me more" phrasing — the shopper wants product detail.
_TELL_ME_MORE_RE = re.compile(
    r"("
    r"tell\s+me\s+(?:more|about)|"
    r"more\s+(?:info(?:rmation)?|details?)|"
    r"how\s+does\s+it\s+work|"
    r"what\s+does\s+it\s+do|"
    r"what'?s?\s+(?:it|this)\s+like|"
    r"any\s+(?:info|details)|"
    r"details?\s+please|"
    r"자세히|알려줘|설명"
    r")",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _model_token_index() -> dict[str, str]:
    """Map distinctive catalog token → canonical display name."""
    try:
        from sales_catalog import load_product_index
    except ImportError:  # pragma: no cover — catalog is always present in app
        return {}

    index: dict[str, str] = {}
    for product in load_product_index():
        name = product.display_name or product.title
        for token in re.findall(r"[a-z0-9]+", (name or "").lower()):
            if len(token) < 4:
                continue
            if token in _BRAND_NOISE or token in _AMBIGUOUS_MODEL_TOKENS:
                continue
            # First product wins so the token maps to a stable name.
            index.setdefault(token, name)
    return index


_SHORT_DISAMBIGUATORS = frozenset({"le", "xl", "xt", "4d", "3d", "2d", "v2"})


def named_model_in_text(text: str) -> Optional[str]:
    """Return a catalog model name when the message clearly names one.

    Distinctive tokens ("maestro", "highpointe") only answer "is a model
    named here?". The actual name comes from ``resolve_product``, which
    already knows that bare "Maestro" is the 4D. Short suffixes like LE
    are kept on the phrase so "Maestro LE" still resolves correctly.
    """
    index = _model_token_index()
    if not index:
        return None
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    phrases: list[str] = []
    for i, word in enumerate(words):
        if word not in index:
            continue
        chunk = [word]
        for nxt in words[i + 1 : i + 3]:
            if nxt in _SHORT_DISAMBIGUATORS or nxt in index:
                chunk.append(nxt)
            else:
                break
        if len(chunk) > 1:
            phrases.append(" ".join(chunk))
        phrases.append(word)
    if not phrases:
        return None

    try:
        from sales_catalog import resolve_product
    except ImportError:  # pragma: no cover
        return _model_token_index().get(phrases[-1])

    distinctive = [w for w in words if w in index or w in _SHORT_DISAMBIGUATORS]
    best_name: Optional[str] = None
    best_score = -1
    seen: set[str] = set()
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        product = resolve_product(phrase)
        if product is None:
            continue
        hay = f"{product.display_name} {product.title}".lower()
        score = sum(1 for token in distinctive if token in hay)
        if score > best_score:
            best_score = score
            best_name = product.display_name or product.title
    return best_name


def rule_fallback(text: str) -> Optional[SalesIntent]:
    """Deterministic second pass. Returns ``None`` when still unclear."""
    raw = (text or "").strip()
    if not raw:
        return None

    # A named model is the strongest signal available — answer about that
    # chair instead of showing the menu again.
    model = named_model_in_text(raw)
    if model:
        return SalesIntent(
            label=INTENT_SPECS,
            confidence="medium",
            matched_terms=(model.lower(),),
        )

    if _BROWSE_HELP_RE.search(raw):
        return SalesIntent(label=INTENT_RECOMMEND, confidence="medium")

    if _TELL_ME_MORE_RE.search(raw):
        return SalesIntent(label=INTENT_SPECS, confidence="medium")

    return None


def llm_enabled() -> bool:
    return os.environ.get("SALES_INTENT_LLM", "1") == "1"


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        from config import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT
    except ImportError:  # pragma: no cover — optional dependency
        return None

    return OpenAI(
        api_key=api_key,
        # Routing must never stall a chat turn, so cap well below the app-wide
        # default and skip retries.
        timeout=min(float(OPENAI_REQUEST_TIMEOUT), 8.0),
        max_retries=min(int(OPENAI_MAX_RETRIES), 1),
    )


_LLM_PROMPT = """You route messages in a massage-chair storefront sales chat.

Pick the single best label for the CUSTOMER MESSAGE:

- price: asking what something costs
- stock: asking if something is available to buy
- specs: asking what a chair has, does, or is like
- recommend: wants help choosing a chair, or describes their body/needs/budget
- compare: weighing two or more chairs against each other
- discount: asking for a deal, promo, price match, or financing terms
- eta_shipping: asking about delivery timing, cost, or destinations
- order_status: asking where an existing order or package is
- warranty_redirect: a chair they own is broken, faulty, or needs service
- parts_technician: wants replacement parts or a service visit
- cancel_refund: wants to cancel, return, or refund an order
- human: explicitly wants a person
- greeting: only a greeting or thanks
- unclear: genuinely cannot tell, or unrelated to massage chairs

Reply with JSON only: {"label": "<label>", "confidence": "high"|"low"}
Use confidence "high" only when the message clearly fits the label.

CUSTOMER MESSAGE:
"""


def llm_fallback(text: str) -> Optional[SalesIntent]:
    """Classify with an LLM into the closed label set, or return ``None``.

    The model picks a route only. It never produces customer-facing copy, so
    a wrong answer can misroute a turn but can never state a wrong fact.
    """
    raw = (text or "").strip()
    if not raw or not llm_enabled():
        return None

    client = _openai_client()
    if client is None:
        return None

    try:
        from config import ROUTER_MODEL

        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0,
            max_tokens=32,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict intent classifier for a sales chat. "
                        "Never answer the customer — only return the label JSON."
                    ),
                },
                {"role": "user", "content": f"{_LLM_PROMPT}{raw[:1000]}"},
            ],
        )
        parsed = json.loads((response.choices[0].message.content or "").strip())
    except Exception as exc:  # pragma: no cover — network/parse side-effects
        logger.warning("sales intent LLM fallback failed: %s", exc)
        return None

    if not isinstance(parsed, dict):
        return None
    label = str(parsed.get("label") or "").strip().lower()
    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence != "high" or label not in _ALLOWED_LABELS:
        return None

    return SalesIntent(
        label=label,
        confidence="medium",
        handoff=label in HANDOFF_INTENTS,
        matched_terms=("llm_fallback",),
    )


def resolve_unclear(text: str) -> Optional[SalesIntent]:
    """Run both fallback layers in order; ``None`` means genuinely unclear."""
    ruled = rule_fallback(text)
    if ruled is not None:
        return ruled
    return llm_fallback(text)


def clear_caches() -> None:
    _model_token_index.cache_clear()
