"""
sales_intent.py
===============
Deterministic intent router for the Sales AI (Tidio) chat.

Design philosophy (aligned with warranty_scope):
  1. Confirm-before-advance. Guardrail intents (cancel, warranty, discount, ETA
     promise, parts/technician) are detected up front and *never* fall through
     to the sales answer path.
  2. Facts only. Anything that would require negotiation, promising a delivery
     date, quoting an unofficial discount, or diagnosing a defect exits the
     AI branch and routes to a human or the warranty chat.
  3. Buttons over free text. Ambiguous inputs are labeled ``unclear`` so the
     caller can present quick-reply buttons instead of guessing.

The classifier is rule-based on purpose — it must be reproducible in tests,
runnable offline, and safe to expose to a public storefront.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Intent labels — kept small on purpose. Everything unclassified maps to
# ``unclear`` so the UI can show a quick-reply menu.
# ---------------------------------------------------------------------------

INTENT_WARRANTY_REDIRECT = "warranty_redirect"
INTENT_CANCEL_REFUND = "cancel_refund"
INTENT_PARTS_TECHNICIAN = "parts_technician"
INTENT_DISCOUNT = "discount"
INTENT_ETA_SHIPPING = "eta_shipping"
INTENT_ORDER_STATUS = "order_status"
INTENT_PRICE = "price"
INTENT_STOCK = "stock"
INTENT_RECOMMEND = "recommend"
INTENT_COMPARE = "compare"
INTENT_SPECS = "specs"
INTENT_INTENSITY = "intensity"
INTENT_HUMAN = "human"
INTENT_GREETING = "greeting"
INTENT_UNCLEAR = "unclear"

# Intents that must NEVER be answered by the AI — always hand off / redirect.
HANDOFF_INTENTS = frozenset(
    {
        INTENT_WARRANTY_REDIRECT,
        INTENT_CANCEL_REFUND,
        INTENT_PARTS_TECHNICIAN,
        INTENT_DISCOUNT,
        INTENT_ETA_SHIPPING,
        INTENT_ORDER_STATUS,  # tracking / "where's my order" → Warranty
        INTENT_HUMAN,
    }
)

# Route to Warranty chat (not sales human). Discount stays with sales human.
WARRANTY_ROUTE_INTENTS = frozenset(
    {
        INTENT_WARRANTY_REDIRECT,
        INTENT_CANCEL_REFUND,
        INTENT_PARTS_TECHNICIAN,
        INTENT_ETA_SHIPPING,
        INTENT_ORDER_STATUS,
    }
)


# ---------------------------------------------------------------------------
# Guardrail patterns
# ---------------------------------------------------------------------------

_GREETING_EN_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|"
    r"thanks|thank\s+you|thx|bye|goodbye)[!.?\s]*$",
    re.IGNORECASE,
)
# Korean greetings: word boundaries don't fire between CJK chars.
_GREETING_KO_RE = re.compile(r"(안녕|반가워|고마워|고맙)")


def _is_greeting(text: str) -> bool:
    if _GREETING_EN_RE.match(text):
        return True
    return bool(_GREETING_KO_RE.search(text))

# --- Warranty / post-purchase support (Tidio Sales AI must redirect these) ---
_WARRANTY_DEFECT_VERBS = (
    r"(?:stuck|broken|leak(?:ing)?|not\s+working|won'?t\s+\w+|"
    r"stopped\s+\w+|failing|fails|failed|dead|died|"
    r"loose|noisy|smell(?:s|ing)?)"
)
_WARRANTY_REDIRECT_RE = re.compile(
    r"\b("
    r"warranty|defect|malfunction|broken|not\s+working|won'?t\s+(?:turn|power|start|inflate)|"
    r"repair|troubleshoot|error\s*code|err\s*\d+|e\d{1,3}|"
    # part + defect verb (order matters — allows "footrest is stuck", "airbag stopped inflating")
    rf"(?:remote|footrest|airbag|roller|heat|heater|recline|motor|calf\s+massager)\s+"
    rf"(?:(?:is|are|was|were|has|have|keeps?|keep|got|got\s+stuck)\s+)?{_WARRANTY_DEFECT_VERBS}|"
    r"already\s+(?:bought|ordered|purchased|own(?:ed)?)|"
    r"delivered\s+(?:damaged|broken)|damaged\s+(?:on\s+)?arriv|missing\s+part|"
    r"install(?:ation|ing)?\s+(?:help|problem|issue)|assembly\s+(?:help|problem)|"
    r"my\s+chair\s+(?:is|has|won'?t|does\s+not|isn'?t|doesn'?t)|"
    r"보증|워런티|고장|수리|불량"
    r")\b",
    re.IGNORECASE,
)

# --- Parts / technician (post-purchase service; must route to warranty team) ---
_PARTS_TECHNICIAN_RE = re.compile(
    r"\b("
    r"replacement\s+parts?|spare\s+parts?|need\s+(?:a\s+)?parts?|order\s+(?:a\s+)?parts?|"
    r"send\s+(?:me\s+)?(?:a\s+)?parts?|buy\s+(?:a\s+)?parts?|"
    r"technician|repair\s+(?:tech|person|man|service)|service\s+call|on-?site\s+service|"
    r"send\s+someone|come\s+(?:and\s+)?fix|come\s+(?:out\s+)?to\s+(?:my|our)\s+(?:house|home)|"
    r"labor\s+visit"
    r")\b",
    re.IGNORECASE,
)

# --- Cancel / refund / return — same handling as warranty chat ------------
# English uses word boundaries; Korean tokens are matched without them since
# `\b` doesn't fire between two CJK word characters (e.g. "환불해주세요").
_CANCEL_REFUND_RE = re.compile(
    r"(?:\b(?:"
    r"cancel(?:ling|led)?\s+(?:my\s+)?(?:order|purchase|buy|subscription)|"
    r"(?:want|need|please|how\s+(?:do|can)\s+i)\s+to\s+cancel|"
    r"cancel\s+(?:my\s+)?(?:order|purchase)|order\s+cancel|"
    r"refund(?:\s+my\s+(?:order|purchase|money|payment))?|"
    r"(?:want|need|please)\s+(?:a\s+)?refund|"
    r"return\s+(?:my\s+)?(?:order|chair|purchase|item)|"
    r"undo\s+(?:my\s+)?(?:order|purchase)"
    r")\b)|(?:취소|환불|반품)",
    re.IGNORECASE,
)

# --- Discount / negotiation — AI never invents % ------------------------
_DISCOUNT_RE = re.compile(
    r"\b("
    r"discount|promo(?:tion)?|coupon(?:\s+code)?|promo\s*code|"
    r"any\s+(?:deal|deals|sale|sales|offer|offers)|"
    r"can\s+you\s+(?:do\s+)?(?:any\s+)?better|"
    r"best\s+price|lower\s+price|price\s+match|match\s+(?:a\s+)?price|"
    r"can\s+i\s+get\s+(?:it\s+)?cheaper|"
    r"할인|쿠폰|프로모"
    r")\b",
    re.IGNORECASE,
)

# --- Shipping / ETA — OsakiUSA: never answer; redirect to Warranty chat ---
# Covers delivery dates, free shipping, ship-to region, and freight questions.
_ETA_SHIPPING_RE = re.compile(
    r"\b("
    r"when\s+(?:will|does)\s+(?:it|this|my)\s+(?:arrive|ship|deliver)|"
    r"how\s+(?:long|many\s+days)\s+(?:until|to\s+(?:deliver|arrive|ship))|"
    r"delivery\s+(?:date|time|eta|fee|cost|price|window)|"
    r"estimated\s+(?:delivery|arrival)|"
    r"lead\s+time|shipping\s+(?:time|cost|fee|rate|price|policy)|"
    r"free\s+(?:shipping|delivery)|"
    r"(?:do\s+you|can\s+you|will\s+you)\s+(?:ship|deliver)\b|"
    r"(?:ship|deliver|shipping|delivery)\s+to\b|"
    r"guarantee\s+(?:by|before)|"
    r"before\s+(?:christmas|xmas|new\s*year|thanksgiving|father'?s\s+day|mother'?s\s+day|"
    r"black\s+friday|cyber\s+monday|holidays?)|"
    r"hawaii|alaska|guam"
    r")\b",
    re.IGNORECASE,
)

# --- Order status (post-purchase tracking) — route to order-status tool -----
_ORDER_STATUS_RE = re.compile(
    r"\b("
    r"track(?:ing)?\s+(?:number|my)|where(?:'?s|\s+is)\s+my\s+(?:order|package|chair|shipment)|"
    r"tracking\s+#|order\s+(?:#|number)\s*[a-z0-9]|"
    r"my\s+order\s+(?:status|update)|fedex|ups|usps|"
    r"in\s+transit"
    r")\b",
    re.IGNORECASE,
)

# --- Explicit request for a human ------------------------------------------
_HUMAN_RE = re.compile(
    r"\b("
    r"talk\s+to\s+(?:a\s+)?(?:human|person|representative|rep|agent|sales|someone)|"
    r"speak\s+(?:with|to)\s+(?:a\s+)?(?:human|person|rep|agent|sales|someone)|"
    r"connect\s+me\s+(?:to|with)\s+(?:a\s+)?(?:human|person|rep|agent|sales)|"
    r"call\s+me|phone\s+me|human\s+please|real\s+person|"
    r"사람|상담원|담당자"
    r")\b",
    re.IGNORECASE,
)

# --- Sales sub-intents -----------------------------------------------------
_PRICE_RE = re.compile(
    r"\b("
    r"price|cost|how\s+much|what'?s?\s+the\s+price|how\s+expensive|msrp|list\s+price|"
    r"가격|얼마"
    r")\b",
    re.IGNORECASE,
)

_STOCK_RE = re.compile(
    r"\b("
    r"in\s+stock|out\s+of\s+stock|available|availability|backorder|back\s+order|"
    r"can\s+i\s+(?:buy|order)\s+(?:it|this|one)\s+(?:now|today)|"
    r"do\s+you\s+have\s+(?:it|this|one|the)\s+.*(?:in\s+stock|available)?|"
    r"재고|입고"
    r")\b",
    re.IGNORECASE,
)

_RECOMMEND_RE = re.compile(
    r"\b("
    r"recommend|suggestion|which\s+(?:chair|model)|what\s+(?:chair|model).*should|"
    r"best\s+chair\s+for|good\s+chair\s+for|fit\s+for\s+me|good\s+for\s+(?:tall|short|back|neck)|"
    r"my\s+height|my\s+weight|i\s+am\s+\d+\s*(?:ft|feet|cm|kg|lb|lbs|pounds|inches|'|\"|tall)|"
    r"추천"
    r")\b",
    re.IGNORECASE,
)

_COMPARE_RE = re.compile(
    r"\b("
    r"compare|comparison|difference\s+between|vs\.?|versus|"
    r"which\s+is\s+better|better\s+than|"
    r"비교|차이"
    r")\b",
    re.IGNORECASE,
)

_SPECS_RE = re.compile(
    r"\b("
    r"spec(?:s|ification)?|features?|dimensions?|sizes?|weight\s+capacity|weight\s+limit|"
    r"height\s+range|track\s+type|s-?track|l-?track|zero\s+gravity|"
    r"3d|4d|airbags?|heating|foot\s+rollers?|calf\s+rollers?|bluetooth"
    r")\b",
    re.IGNORECASE,
)

_INTENSITY_RE = re.compile(
    r"\b("
    r"intensity|(?:massage\s+)?strong|(?:massage\s+)?strength|(?:massage\s+)?power(?:ful)?|"
    r"(?:massage\s+)?deep|(?:massage\s+)?hard|(?:massage\s+)?soft|"
    r"gentle\s+massage|firm\s+massage"
    r")\b",
    re.IGNORECASE,
)

# --- Body-fit hints — treat as strong recommend signal -----------------------
# When a message mentions height / weight / body-part cues, the customer is
# almost always asking us to recommend a chair for *their* body (even if the
# same sentence also mentions "strong" or "deep" massage). Without this
# override "I'm 5'5", 200 pounds and prefer strong massage" was being
# labelled `intensity`, so the recommend flow never ran.
_BODY_FIT_RE = re.compile(
    r"("
    # Height patterns: 5'5, 6 ft 2, 6 feet, 6'2", 175 cm
    r"\b\d\s*(?:'|ft|feet|foot)\s*\d{0,2}\s*(?:\"|in|inches)?|"
    r"\b\d{3}\s*cm\b|"
    # Weight patterns: 200 lb, 200 lbs, 200 pounds, 90 kg
    r"\b\d{2,3}\s*(?:lb|lbs|pound|pounds|kg)\b|"
    # "I'm 6", "I am 5'5", "im 200 lb"
    r"\bi(?:\s*am|'m|m)\s+\d+|"
    # Body-part cues typical of "recommend a chair for my …"
    r"\b(?:back|neck|shoulder|shoulders|lower\s+back|hip|hips|calf|calves|"
    r"foot|feet|hamstring|hamstrings|glute|glutes|buttock|buttocks|"
    r"sciatica|scoliosis|posture)\b|"
    r"\bmy\s+(?:height|weight|back|neck|shoulders?|posture|body)\b|"
    r"\b(?:tall|petite|short|large|big\s+guy)\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Korean substring lookup — `\b` doesn't work between CJK chars, so we do a
# plain "in text" check for the few tokens we care about.
# ---------------------------------------------------------------------------

_KOREAN_INTENT_TOKENS: tuple[tuple[str, str], ...] = (
    # 1) Guardrails first — a cancel or warranty message must never fall through.
    ("고장", INTENT_WARRANTY_REDIRECT),
    ("수리", INTENT_WARRANTY_REDIRECT),
    ("불량", INTENT_WARRANTY_REDIRECT),
    ("워런티", INTENT_WARRANTY_REDIRECT),
    ("보증", INTENT_WARRANTY_REDIRECT),
    ("환불", INTENT_CANCEL_REFUND),
    ("취소", INTENT_CANCEL_REFUND),
    ("반품", INTENT_CANCEL_REFUND),
    ("할인", INTENT_DISCOUNT),
    ("쿠폰", INTENT_DISCOUNT),
    ("프로모", INTENT_DISCOUNT),
    ("배송", INTENT_ETA_SHIPPING),
    ("택배", INTENT_ETA_SHIPPING),
    ("상담원", INTENT_HUMAN),
    ("담당자", INTENT_HUMAN),
    ("사람 연결", INTENT_HUMAN),
    # 2) Specific-first for sales sub-intents so a mixed sentence like
    #    "세기가 얼마나 세나요" is labelled intensity rather than price.
    ("세기", INTENT_INTENSITY),
    ("강도", INTENT_INTENSITY),
    ("비교", INTENT_COMPARE),
    ("차이", INTENT_COMPARE),
    ("재고", INTENT_STOCK),
    ("입고", INTENT_STOCK),
    ("추천", INTENT_RECOMMEND),
    ("가격", INTENT_PRICE),
    ("얼마", INTENT_PRICE),
)


def _korean_intent(text: str) -> Optional[str]:
    for token, label in _KOREAN_INTENT_TOKENS:
        if token in text:
            return label
    return None


# ---------------------------------------------------------------------------
# Decision object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalesIntent:
    """Classifier output.

    Attributes
    ----------
    label:
        One of the ``INTENT_*`` constants above.
    confidence:
        ``"high"`` when a strong regex hit, ``"medium"`` for keyword-only
        matches, ``"low"`` for greetings, ``"none"`` for empty input.
    handoff:
        True when the AI must not answer directly — a human or the warranty
        chat has to take over.
    """

    label: str
    confidence: str = "medium"
    handoff: bool = False
    matched_terms: tuple[str, ...] = ()

    @property
    def is_handoff(self) -> bool:
        return self.handoff or self.label in HANDOFF_INTENTS


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _matched(pattern: re.Pattern, text: str) -> tuple[str, ...]:
    return tuple(m.group(0).strip().lower() for m in pattern.finditer(text))


def classify(text: str) -> SalesIntent:
    """Classify a customer message into a sales intent.

    The order below is critical — guardrail intents are checked first so a
    single message like *"my chair is broken and how much is the OS-Pro 4D?"*
    is routed to warranty (not price), which is the safe behavior.
    """
    raw = (text or "").strip()
    if not raw:
        return SalesIntent(label=INTENT_UNCLEAR, confidence="none")

    # 1) Cancel / refund / return → warranty team (silent AI). Highest priority
    #    because a message like "I want to cancel my broken chair" must not
    #    fall into the defect flow.
    hits = _matched(_CANCEL_REFUND_RE, raw)
    if hits:
        return SalesIntent(
            label=INTENT_CANCEL_REFUND,
            confidence="high",
            handoff=True,
            matched_terms=hits,
        )

    # 2) Parts / technician requests. Checked *before* generic warranty so a
    #    phrase like "dispatch a repair tech" is labeled parts_technician
    #    rather than the broader warranty bucket.
    hits = _matched(_PARTS_TECHNICIAN_RE, raw)
    if hits:
        return SalesIntent(
            label=INTENT_PARTS_TECHNICIAN,
            confidence="high",
            handoff=True,
            matched_terms=hits,
        )

    # 3) Warranty / defect / already-purchased problems → warranty chat.
    hits = _matched(_WARRANTY_REDIRECT_RE, raw)
    if hits:
        return SalesIntent(
            label=INTENT_WARRANTY_REDIRECT,
            confidence="high",
            handoff=True,
            matched_terms=hits,
        )

    # 4) Discount / promo → human handoff. AI never quotes a % on its own.
    hits = _matched(_DISCOUNT_RE, raw)
    if hits:
        return SalesIntent(
            label=INTENT_DISCOUNT,
            confidence="high",
            handoff=True,
            matched_terms=hits,
        )

    # 5) ETA / delivery date promise → human handoff (region-based).
    hits = _matched(_ETA_SHIPPING_RE, raw)
    if hits:
        return SalesIntent(
            label=INTENT_ETA_SHIPPING,
            confidence="high",
            handoff=True,
            matched_terms=hits,
        )

    # 6) Explicit human request.
    hits = _matched(_HUMAN_RE, raw)
    if hits:
        return SalesIntent(
            label=INTENT_HUMAN,
            confidence="high",
            handoff=True,
            matched_terms=hits,
        )

    # 7) Order status (post-purchase tracking) — Warranty chat, not sales.
    hits = _matched(_ORDER_STATUS_RE, raw)
    if hits:
        return SalesIntent(
            label=INTENT_ORDER_STATUS,
            confidence="high",
            handoff=True,
            matched_terms=hits,
        )

    # 8) Greetings — friendly but no auto-recommendation.
    if _is_greeting(raw):
        return SalesIntent(label=INTENT_GREETING, confidence="low")

    # 9) Sales sub-intents — priority: price > stock > compare > recommend
    #    > intensity > specs. This mirrors what customers actually ask when
    #    they combine signals ("how much is the OS-Pro Maestro?" → price).
    price_hits = _matched(_PRICE_RE, raw)
    stock_hits = _matched(_STOCK_RE, raw)
    compare_hits = _matched(_COMPARE_RE, raw)
    recommend_hits = _matched(_RECOMMEND_RE, raw)
    body_fit_hits = _matched(_BODY_FIT_RE, raw)
    intensity_hits = _matched(_INTENSITY_RE, raw)
    specs_hits = _matched(_SPECS_RE, raw)

    if price_hits:
        return SalesIntent(label=INTENT_PRICE, confidence="high", matched_terms=price_hits)
    if stock_hits:
        return SalesIntent(label=INTENT_STOCK, confidence="high", matched_terms=stock_hits)
    if compare_hits:
        return SalesIntent(label=INTENT_COMPARE, confidence="high", matched_terms=compare_hits)
    # Body-fit cues (height/weight/body-part) win over intensity — a customer
    # who shares physical details wants a recommendation, not a lecture on
    # massage strength.
    if recommend_hits or body_fit_hits:
        return SalesIntent(
            label=INTENT_RECOMMEND,
            confidence="high",
            matched_terms=recommend_hits + body_fit_hits,
        )
    if intensity_hits:
        return SalesIntent(
            label=INTENT_INTENSITY, confidence="high", matched_terms=intensity_hits
        )
    if specs_hits:
        return SalesIntent(label=INTENT_SPECS, confidence="medium", matched_terms=specs_hits)

    # Korean fallback — needed because `\b` doesn't fire between CJK word chars.
    korean_label = _korean_intent(raw)
    if korean_label is not None:
        return SalesIntent(
            label=korean_label,
            confidence="medium",
            handoff=korean_label in HANDOFF_INTENTS,
        )

    return SalesIntent(label=INTENT_UNCLEAR, confidence="low")


# ---------------------------------------------------------------------------
# Handoff copy — kept next to the classifier so refusals are consistent.
# ---------------------------------------------------------------------------


_WARRANTY_CHAT_REDIRECT = (
    "Hi there\n"
    "\n"
    "Thank you for reaching out to us.\n"
    "\n"
    "Please forward your email to our Warranty Department to create a "
    "service ticket promptly.\n"
    "\n"
    "Warranty Service Email: service@osakititan.com\n"
    "\n"
    "Phone: 1-888-848-2630 ext.3\n"
    "\n"
    "You can also submit ticket directly using below link:\n"
    "\n"
    "https://titanchair.freshdesk.com/support/home\n"
    "\n"
    "They are all in charge of technical supports and replacement parts.\n"
    "\n"
    "Thank you."
)


def handoff_message(intent: SalesIntent) -> Optional[str]:
    """Return the safe, non-committal reply for a handoff intent.

    OsakiUSA Sales (Tidio) policy:
      - Never explain discount or shipping policy in this chat.
      - Warranty / cancel / refund / return / shipping / tracking / delivery /
        parts / technician requests get the Warranty Department contact
        (email / phone / Freshdesk) — the Sales AI must never handle these.
      - Discount / explicit human request → silent handoff to sales human
        (email capture, no policy talk).
    """
    label = intent.label
    if label in (
        INTENT_WARRANTY_REDIRECT,
        INTENT_CANCEL_REFUND,
        INTENT_PARTS_TECHNICIAN,
        INTENT_ETA_SHIPPING,
        INTENT_ORDER_STATUS,
    ):
        return _WARRANTY_CHAT_REDIRECT
    if label == INTENT_DISCOUNT:
        # No promo %, no "current offers" language — just connect a human.
        return (
            "I'll connect you with our sales team. "
            "Please share your **email** and they will follow up."
        )
    if label == INTENT_HUMAN:
        return (
            "I'll connect you with our sales team. "
            "Please share your **email** (and optionally a phone number)."
        )
    return None
