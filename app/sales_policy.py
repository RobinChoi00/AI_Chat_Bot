"""
sales_policy.py
===============
Factual answers to pre-purchase policy questions.

A shopper asking "what's your return policy?" or "how long is the warranty?"
was being told to email the Warranty Department, and "how much is shipping?"
was routed the same way. Those are ordinary sales questions with published
answers, and handing them to a human is the wrong outcome.

Source of truth
---------------
Every fact below is quoted from the storefront's own published policy pages,
which are linked in each answer so the customer can verify:

  /policies/refund-policy    – 30-day window, RMA, restocking fee
  /policies/shipping-policy  – curbside vs White Glove, assembly, fees
  /pages/warranty            – 3-year standard coverage and extensions

What this module deliberately will NOT do
-----------------------------------------
- Quote a delivery date or transit time. The published policy says carriers
  coordinate a window and that exact times are unavailable, so promising one
  would be inventing a fact.
- Quote a shipping price, an APR, or a financing term. None are published.
- Answer for a customer who already owns a chair or has an order in flight —
  those still route to the warranty and order-status paths, which is why
  ``is_post_purchase`` gates every lookup.
"""

from __future__ import annotations

import re
from typing import Optional

from store_config import get_storefront_base_url

TOPIC_RETURNS = "returns"
TOPIC_WARRANTY_TERMS = "warranty_terms"
TOPIC_SHIPPING = "shipping"
TOPIC_RESTRICTED_REGION = "restricted_region"
TOPIC_WHITE_GLOVE = "white_glove"
TOPIC_FINANCING = "financing"
TOPIC_SHOWROOM = "showroom"

POLICY_TOPICS = (
    TOPIC_RETURNS,
    TOPIC_WARRANTY_TERMS,
    TOPIC_SHIPPING,
    TOPIC_RESTRICTED_REGION,
    TOPIC_WHITE_GLOVE,
    TOPIC_FINANCING,
    TOPIC_SHOWROOM,
)

# Destinations we do not deliver to. Sourced from the same business rule the
# warranty scope gate already enforces (``warranty_scope.py``), so both bots
# tell a shopper the same thing.
_RESTRICTED_REGION_RE = re.compile(
    r"\b(hawaii|hawaiian|honolulu|alaska|alaskan|anchorage|guam|guamanian)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Ownership detection — the line between "shopper" and "existing customer"
# ---------------------------------------------------------------------------

_POST_PURCHASE_RE = re.compile(
    r"("
    r"\bmy\s+(?:chair|order|unit|purchase|delivery|shipment|package|tracking)\b|"
    r"\bi\s+(?:bought|ordered|purchased|received|got)\b|"
    r"\bi'?ve\s+(?:bought|ordered|purchased|received)\b|"
    r"\balready\s+(?:bought|ordered|purchased|own|received|paid)\b|"
    r"\bwe\s+(?:bought|ordered|purchased|received)\b|"
    r"\bwhen\s+(?:will|does)\s+(?:it|mine|my)\b|"
    r"\bwhere(?:'?s|\s+is)\s+my\b|"
    r"\border\s*(?:#|number)\b|"
    r"\btracking\b|"
    r"\bit\s+(?:arrived|came|shipped)\b|"
    r"\bunder\s+warranty\b|"
    r"\bclaim\b|"
    r"\bi\s+own\b|"
    r"구매했|주문했|받았"
    r")",
    re.IGNORECASE,
)


def is_post_purchase(text: str) -> bool:
    """True when the message is about a chair or order the customer has."""
    return bool(_POST_PURCHASE_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        TOPIC_WHITE_GLOVE,
        re.compile(
            r"("
            r"white\s*glove|"
            r"(?:do\s+you|can\s+you|will\s+you)\s+(?:assemble|set\s*it\s*up|install)|"
            r"\bassembl(?:y|e|ed|ing)\b|"
            r"set\s*[\-\s]?up\s+(?:service|fee|cost|included)|"
            r"put\s+it\s+together|"
            r"carry\s+(?:it\s+)?(?:up|upstairs)|stairs?\s+(?:carry|fee)|"
            r"bring\s+it\s+(?:in|inside|upstairs)|"
            r"take\s+away\s+(?:my\s+)?old|haul\s+away"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_RETURNS,
        re.compile(
            r"("
            r"return\s+polic|refund\s+polic|"
            r"(?:can|could)\s+i\s+return|"
            r"(?:if|what\s+if)\s+i\s+(?:don'?t|do\s+not)\s+like|"
            r"change\s+my\s+mind|"
            r"restocking(?:\s+fee)?|"
            r"money\s+back|"
            r"trial\s+period|try\s+it\s+(?:at\s+home|for\s+\d+)|"
            r"\brma\b|"
            r"return\s+window|how\s+long\s+(?:do\s+i\s+have\s+)?to\s+return"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_WARRANTY_TERMS,
        re.compile(
            r"("
            r"how\s+long\s+is\s+the\s+warranty|"
            r"warranty\s+(?:length|period|term|coverage|cover|included|come)|"
            r"(?:what|which)\s+warranty|"
            r"(?:is|does)\s+(?:there|it)\s+(?:a\s+)?(?:come\s+with\s+a\s+)?warranty|"
            r"come\s+with\s+(?:a\s+)?warranty|"
            r"extended\s+warranty|"
            r"years?\s+warranty|warranty\s+years?|"
            r"what'?s?\s+covered"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_FINANCING,
        re.compile(
            r"("
            r"financ(?:e|ing)|"
            r"monthly\s+payment|pay\s+monthly|payment\s+plan|"
            r"pay\s+over\s+time|installments?|"
            r"affirm|klarna|afterpay|"
            r"lease\s+to\s+own|rent\s+to\s+own|"
            r"can\s+i\s+split\s+the\s+(?:cost|payment)|"
            r"할부"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_SHOWROOM,
        re.compile(
            r"("
            r"showroom|show\s+room|"
            r"(?:do\s+you\s+have\s+a|is\s+there\s+a|nearest)\s+(?:store|shop|location)|"
            r"store\s+(?:near|location|address|hours)|"
            r"(?:try|test|sit\s+in|see)\s+(?:it|one|them|a\s+chair)\s+(?:in\s+person|out|first)?|"
            r"in\s+person|"
            r"visit\s+(?:you|your|the)|come\s+see|"
            r"where\s+are\s+you\s+located|your\s+address|"
            r"(?:business|store|opening)\s+hours|what\s+are\s+your\s+hours|"
            r"매장|쇼룸"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_SHIPPING,
        re.compile(
            r"("
            r"shipping\s+(?:cost|fee|price|rate|polic|charge)|"
            r"how\s+much\s+(?:is|for|does)\s+(?:the\s+)?(?:shipping|delivery)|"
            r"(?:free|paid)\s+(?:shipping|delivery)|"
            r"(?:shipping|delivery)\s+(?:is\s+)?free|"
            r"is\s+(?:shipping|delivery)\s+(?:free|included|extra)|"
            r"(?:do|can)\s+you\s+(?:ship|deliver)\s+to|"
            r"delivery\s+(?:polic|process|option|method)|"
            r"curbside|"
            r"how\s+(?:is|does)\s+it\s+(?:delivered|ship|arrive)|"
            r"who\s+delivers"
            r")",
            re.IGNORECASE,
        ),
    ),
)


def detect_topic(text: str) -> Optional[str]:
    """Return the pre-purchase policy topic in this message, if any."""
    raw = (text or "").strip()
    if not raw or is_post_purchase(raw):
        return None
    # Checked ahead of the generic shipping answer: the published shipping
    # policy says nothing about these destinations, so describing curbside
    # service would imply we deliver there when we do not.
    if _RESTRICTED_REGION_RE.search(raw):
        return TOPIC_RESTRICTED_REGION
    for topic, pattern in _TOPIC_PATTERNS:
        if pattern.search(raw):
            return topic
    return None


# ---------------------------------------------------------------------------
# Answers — every claim traceable to a published page
# ---------------------------------------------------------------------------


def _policy_url(domain: str, path: str) -> str:
    return f"{get_storefront_base_url(domain).rstrip('/')}/{path.lstrip('/')}"


def _returns_answer(domain: str) -> str:
    url = _policy_url(domain, "policies/refund-policy")
    return (
        "**Returns — 30 days from delivery**\n\n"
        "- You have **30 days after receiving** the chair to request a return or exchange.\n"
        "- Keep the **original packaging** — the chair must come back in new, resellable condition.\n"
        "- A **Return Merchandise Authorization (RMA)** has to be approved by us first.\n"
        "- **Shipping both ways is the customer's cost**, and this also applies to a "
        "cancellation once the order has already shipped.\n"
        "- Non-Titan-brand items carry a **20% restocking fee**. A White Glove fee is non-refundable.\n\n"
        f"Full policy: {url}"
    )


def _warranty_terms_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/warranty")
    return (
        "**Warranty — 3 years standard**\n\n"
        "Coverage for defects in parts, workmanship, or structural defects for the "
        "first **three (3) years** of ownership:\n\n"
        "- **Year 1** — parts *and* labor at no cost to you.\n"
        "- **Year 2** — parts at no cost to you.\n"
        "- **Structural framework** — 3 years on selected products.\n\n"
        "Extended **4-year and 5-year** plans are available at purchase, which add "
        "labor coverage in the later years.\n\n"
        "The warranty is non-transferable and proof of purchase is required for any claim.\n\n"
        f"Full terms and exclusions: {url}"
    )


def _shipping_answer(domain: str) -> str:
    url = _policy_url(domain, "policies/shipping-policy")
    return (
        "**Delivery — standard curbside included**\n\n"
        "- Delivered to the **curb or driveway** of your address, and a **signature is required**.\n"
        "- The carrier contacts you to arrange a **delivery window**. Exact time-of-day "
        "requests aren't available, so I can't promise a specific date here.\n"
        "- **Assembly is not included** with standard curbside — the chair ships with "
        "instructions and the tools you need. White Glove delivery with assembly is "
        "available for an added fee.\n"
        "- Please **inspect the packaging before signing**. Note any visible damage on "
        "the delivery receipt and tell us right away.\n"
        "- Measure your **doorways, hallways, and stairs** before ordering — I can check "
        "a specific chair against your doorway width if you tell me the measurement.\n\n"
        f"Full policy: {url}\n\n"
        "Shipping cost and coverage depend on your address, and we deliver within "
        "the continental US. Share your **email** and a specialist will confirm "
        "both for where you are."
    )


def _restricted_region_answer(domain: str) -> str:
    url = _policy_url(domain, "policies/shipping-policy")
    return (
        "**We don't deliver to Hawaii, Alaska, or Guam.**\n\n"
        "Our delivery network covers the continental US only, so I can't get a "
        "chair to those locations.\n\n"
        "If you have a **continental US shipping address** — a freight forwarder or "
        "a friend or family member's address — we can deliver there and you'd arrange "
        "the rest. A specialist can talk through that option with you.\n\n"
        f"Shipping policy: {url}"
    )


def _white_glove_answer(domain: str) -> str:
    url = _policy_url(domain, "policies/shipping-policy")
    return (
        "**White Glove delivery and assembly**\n\n"
        "- Standard curbside delivery **does not include assembly**. White Glove adds "
        "in-home delivery plus assembly for an extra fee.\n"
        "- The standard team is **two delivery professionals**, and the basic fee covers "
        "a short on-site window for delivery and assembly.\n"
        "- Crews **cannot move your furniture, remove doors, or haul away an old chair** — "
        "the path needs to be clear and accessible before they arrive.\n"
        "- Extra charges can apply for a **third person for stair carry ($120–$300)**, "
        "additional on-site time, or a redelivery attempt if the chair can't get in.\n"
        "- If a door or staircase is too narrow to complete delivery, an attempt fee may apply — "
        "so measuring first matters. Tell me your **doorway width** and I'll check fit.\n\n"
        f"Full policy: {url}"
    )


def _financing_answer(domain: str) -> str:
    return (
        "**Pay over time**\n\n"
        "**Affirm** is available at checkout on our store, so you can split the cost into "
        "monthly payments. You'll see the exact monthly amount, term length, and any "
        "interest on the checkout page once you pick a chair — those depend on the "
        "purchase amount and Affirm's approval, so I won't quote a rate here.\n\n"
        "Want a specialist to walk through the options? Share your **email** and "
        "they'll reach out."
    )


def _showroom_answer(domain: str) -> str:
    from sales_cta import showroom_address

    return (
        "**Visit our showroom**\n\n"
        f"{showroom_address()}\n\n"
        "You're welcome to come try the chairs in person. Please **call ahead** so we can "
        "confirm which models are on the floor and that a specialist is free for you.\n\n"
        "If you'd rather narrow it down first, tell me your height, weight, and what you "
        "want the chair to help with, and I'll shortlist a few to try."
    )


_ANSWERS = {
    TOPIC_RETURNS: _returns_answer,
    TOPIC_WARRANTY_TERMS: _warranty_terms_answer,
    TOPIC_SHIPPING: _shipping_answer,
    TOPIC_RESTRICTED_REGION: _restricted_region_answer,
    TOPIC_WHITE_GLOVE: _white_glove_answer,
    TOPIC_FINANCING: _financing_answer,
    TOPIC_SHOWROOM: _showroom_answer,
}


def policy_answer(topic: str, domain: str = "") -> Optional[str]:
    """Render the published answer for a pre-purchase policy topic."""
    builder = _ANSWERS.get(topic)
    return builder(domain) if builder else None
