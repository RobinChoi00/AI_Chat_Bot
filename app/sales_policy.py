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
Every fact below is quoted from the storefront's own published policy pages
or from the sales team's written rule, and each answer links the page so
the customer can verify:

  /pages/sales-policy        – 30-day returns, both-way freight, White Glove
  /pages/shipping-handling   – curbside vs White Glove, lead times
  /pages/warranty            – 3-year standard coverage and extensions

Hawaii / Alaska / Guam
----------------------
Sales ships to Hawaii and Alaska; the customer pays freight, quoted by the
carrier for that model and address. Guam is not served. Do not invent a
dollar amount.

What this module deliberately will NOT do
-----------------------------------------
- Quote a calendar delivery date. "Up to 2 weeks" / "up to 3 weeks" is the
  published ceiling; exact day-of-week requests stay unavailable.
- Quote a shipping price, an APR, or a financing term.
- Answer for a customer who already owns a chair or has an order in flight —
  those still route to the warranty and order-status paths, which is why
  ``is_post_purchase`` gates every lookup.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Optional

from store_config import get_storefront_base_url

TOPIC_RETURNS = "returns"
TOPIC_WARRANTY_TERMS = "warranty_terms"
TOPIC_TECHNICAL_SERVICE = "technical_service"
TOPIC_SHIPPING = "shipping"
TOPIC_REMOTE_SHIPPING = "remote_shipping"
TOPIC_RESTRICTED_REGION = "restricted_region"
TOPIC_WHITE_GLOVE = "white_glove"
TOPIC_FINANCING = "financing"
TOPIC_SHOWROOM = "showroom"
TOPIC_MECHANISM = "mechanism"
TOPIC_REFURBISHED = "refurbished"
TOPIC_TRADE_IN = "trade_in"
TOPIC_LEASE = "lease"
TOPIC_INTERNATIONAL = "international"
TOPIC_LIMITS = "limits"

POLICY_TOPICS = (
    TOPIC_RETURNS,
    TOPIC_WARRANTY_TERMS,
    TOPIC_TECHNICAL_SERVICE,
    TOPIC_SHIPPING,
    TOPIC_REMOTE_SHIPPING,
    TOPIC_RESTRICTED_REGION,
    TOPIC_INTERNATIONAL,
    TOPIC_WHITE_GLOVE,
    TOPIC_FINANCING,
    TOPIC_REFURBISHED,
    TOPIC_TRADE_IN,
    TOPIC_LEASE,
    TOPIC_LIMITS,
    TOPIC_SHOWROOM,
    TOPIC_MECHANISM,
)

# Guam is not served. Hawaii and Alaska are served with customer-paid freight.
_RESTRICTED_REGION_RE = re.compile(
    r"\b(guam|guamanian)\b|괌",
    re.IGNORECASE,
)
_REMOTE_REGION_RE = re.compile(
    r"\b(hawaii|hawaiian|honolulu|alaska|alaskan|anchorage)\b|하와이|알래스카",
    re.IGNORECASE,
)
_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_ZIP_WORD_RE = re.compile(r"\bzip(?:\s*code)?\b|\bpostal\s+code\b", re.IGNORECASE)


def _topic_from_us_zip(text: str) -> Optional[str]:
    """Map a US zip to Guam / HI-AK freight when the shopper only sent digits."""
    found: list[str] = []
    for match in _US_ZIP_RE.finditer(text or ""):
        zip_code = int(match.group(1))
        if 96910 <= zip_code <= 96932:
            found.append(TOPIC_RESTRICTED_REGION)
        elif 96701 <= zip_code <= 96898 or 99501 <= zip_code <= 99950:
            found.append(TOPIC_REMOTE_SHIPPING)
    if TOPIC_RESTRICTED_REGION in found:
        return TOPIC_RESTRICTED_REGION
    if TOPIC_REMOTE_SHIPPING in found:
        return TOPIC_REMOTE_SHIPPING
    return None


_INTERNATIONAL_RE = re.compile(
    r"("
    r"\b(canada|canadian|mexico|mexican|overseas|international)\b|"
    r"\b(united\s+kingdom|\buk\b|europe|european|australia|australian)\b|"
    r"ship(?:ping)?\s+to\s+(?:canada|mexico|europe|uk|britain|australia)"
    r")",
    re.IGNORECASE,
)

_LEASE_QUESTION_RE = re.compile(
    r"\b(?:lease|rent)[\s-]+to[\s-]+own\b|\b(?:lease|rent)(?:\s*/\s*|\s+or\s+)(?:lease|rent)\b|"
    r"\b(?:do you|can i|can you|could i|do you offer|looking to|want to)\s+(?:lease|rent)\b|"
    r"\b(?:lease|rent)\s+(?:a\s+|the\s+|your\s+)?(?:massage\s+)?(?:chairs?|one|it)\b|"
    r"\b(?:leasing|rental)\s+(?:options?|program|plans?)\b|\b(?:chair|equipment)\s+(?:lease|rental)\b",
    re.I,
)
_TECHNICAL_SERVICE_RE = re.compile(
    r"\b(?:tech(?:nical)?\s+(?:service|support)|warranty\s+service|repair\s+service)\b|"
    r"\b(?:how|who|where)\b[^?!.;]{0,60}\b(?:service|repair)\s+(?:it|the\s+(?:chair|unit))\b",
    re.I,
)
_NEGATED_SERVICE_PREFIX_RE = re.compile(
    r"\b(?:not|never|no|don'?t|doesn'?t|do\s+not|does\s+not)\s+"
    r"(?:(?:need|want|offer|request|look(?:ing)?|ask(?:ing)?|shop(?:ping)?)\s+(?:for\s+)?)?"
    r"(?:(?:any|the|a)\s+)?$", re.I,
)
_PROSPECTIVE_CLAUSE_RE = re.compile(r"\b(?:if|whether|once|in\s+case|suppose|hypothetically)\b", re.I)
_EXPLICIT_NEW_PURCHASE_RE = re.compile(
    r"\b(?:before\s+(?:i|we)\s+(?:buy|purchase|order)|shopping\s+for|"
    r"(?:(?:a\s+)?(?:second|new)|another)\s+(?:massage\s+)?chair)\b", re.I,
)
_LOCAL_DEALER_RE = re.compile(
    r"\b(?:near|nearby|nearest|closest|local|around|(?:my|this)\s+area|zip(?:\s*code)?)\b|"
    r"\b(?:dealers?|retailers?|showrooms?)\s+in\b|"
    r"\b(?:i\s+am|i['’]m|we\s+are|we['’]re|i\s+live|we\s+live)\s+(?:in|from)\b",
    re.I,
)
_DEALER_LOCATOR_URL = "https://osakichair.com/pages/find-your-osaki-store"


# ---------------------------------------------------------------------------
# Ownership detection — the line between "shopper" and "existing customer"
# ---------------------------------------------------------------------------

_POST_PURCHASE_RE = re.compile(
    r"("
    r"\bmy\s+(?:chair|order|unit|purchase|delivery|shipment|package|tracking)\b|"
    r"\bi\s+(?:(?:just|already|recently)\s+)?(?:bought|ordered|purchased|received|got)\b|"
    r"\bi(?:'ve|\s+have)\s+(?:(?:just|already|recently)\s+)?(?:bought|ordered|purchased|received)\b|"
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
    r"\bexisting\s+order\b|\b(?:ordered|purchased|delivered)\s+chair\b|"
    r"구매했|주문했|받았"
    r")",
    re.IGNORECASE,
)

_CURRENT_CHAIR_EVIDENCE_RE = re.compile(
    r"\b(?:my|our)\s+(?:existing\s+|current\s+|old\s+)?(?:massage\s+)?(?:chair|unit)\b|"
    r"\b(?:i|we)(?:\s+already\s+|\s+|['’]ve\s+)(?:have|own|received|bought|purchased)\s+"
    r"(?:(?:just|already)\s+)?(?:it|(?:this|that|the|a|an|my|our)\s+(?:massage\s+)?(?:chair|unit))\b|"
    r"\b(?:i|we)(?:['’]ve|\s+have)\s+had\s+(?:it|(?:this|that|the|a|an|my|our)\s+"
    r"(?:(?:massage\s+)?(?:chair|unit)|osaki|titan))\b|"
    r"\b(?:chair|unit|it)\s+(?:is|was)\s+(?:already\s+)?(?:in|at)\s+"
    r"(?:(?:my|our)\s+(?:house|home|living\s+room)|home)\b",
    re.I,
)
_ACTUAL_CHAIR_FAILURE_RE = re.compile(
    r"\b(?P<subject>(?:(?:this|that|the|my|our)\s+(?:massage\s+)?(?:chair|unit))|it|"
    r"(?:(?:its|the|my|our)\s+)?(?:remote|(?:(?:foot|calf)\s+)?rollers?|footrest|heater|screen))\s+"
    r"(?:has\s+|have\s+)?(?:"
    r"(?:is|are|was|were)\s+(?:(?:still|now|completely)\s+)?(?:broken|faulty|stuck|dead|not\s+working)|"
    r"(?:broke\s+down|broke|malfunctioned|stopped\s+(?:working|turning\s+on|responding)|"
    r"(?:does\s+not|doesn't|do\s+not|don't|won't|will\s+not|no\s+longer)\s+(?:works?|turns?\s+on|responds?)))\b",
    re.I,
)
_OWNER_SUPPORT_REQUEST_RE = re.compile(
    r"\b(?:need|want|seeking|looking\s+for|ask(?:ing)?\s+for|get|contact)\s+(?:the\s+)?"
    r"(?:tech(?:nical)?\s+(?:help|service|support)|warranty\s+service|repair\s+help)\b|"
    r"\b(?:how|where|who)\b[^.!?;]{0,45}\b(?:repair|fix|service)\s+(?:it|this|that|(?:the|my|our)\s+chair)\b",
    re.I,
)


def _clause_before(text: str, start: int) -> str:
    return re.split(r"[.!?;,\n]", text[max(0, start - 100):start])[-1]


def is_prospective_clause_before(text: str, start: int) -> bool:
    """An `if`/`once` clause describes a possible fault, not a report."""
    return bool(_PROSPECTIVE_CLAUSE_RE.search(_clause_before(text, start)))


def _affirmative_matches(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    for match in pattern.finditer(text):
        if not _NEGATED_SERVICE_PREFIX_RE.search(_clause_before(text, match.start())):
            yield match


def _current_matches(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    return (match for match in _affirmative_matches(pattern, text)
            if not is_prospective_clause_before(text, match.start()))


def reports_owned_chair_problem(text: str) -> bool:
    """Distinguish a real fault or owner service request from a policy inquiry.

    A concrete fault of a specific chair is sufficient. A pronoun-only fault
    needs ownership evidence or an actual support request; the conversation
    layer handles shorter follow-ups with an active case.
    """
    raw = (text or "").replace("’", "'")
    has_owner = bool(next(_current_matches(_CURRENT_CHAIR_EVIDENCE_RE, raw), None))
    has_request = bool(next(_current_matches(_OWNER_SUPPORT_REQUEST_RE, raw), None))
    if any(has_owner or has_request or match.group('subject').lower() != 'it'
           for match in _current_matches(_ACTUAL_CHAIR_FAILURE_RE, raw)):
        return True
    return has_owner and has_request


def is_post_purchase(text: str) -> bool:
    """True when the message is about a chair or order the customer has."""
    raw = text or ""
    for match in _POST_PURCHASE_RE.finditer(raw):
        # A future or hypothetical "my chair" is not proof of ownership.
        if match.group().lower() in {'my chair', 'my unit'} and is_prospective_clause_before(raw, match.start()):
            continue
        if match.group().lower() == 'claim' and re.search(
            r"\b(?:not|no)\s+(?:(?:a|an|the)\s+)?(?:(?:repair|warranty)(?:\s+or\s+(?:repair|warranty))?\s+)?$",
            raw[max(0, match.start() - 65):match.start()], re.I,
        ):
            continue
        return True
    return False


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
            r"tablet\s+instal|"
            r"\bassembl(?:y|e|ed|ing)\b|"
            r"set\s*[\-\s]?up\s+(?:service|fee|cost|included)|"
            r"put\s+it\s+together|"
            r"carry\s+(?:it\s+)?(?:up|upstairs)|stairs?\s+(?:carry|fee)|"
            r"(?:third|second|fourth)\s+floor|walk[\s-]?up|"
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
            r"(?:can|could)\s+i\s+return|return\s+(?:shipping|freight|costs?)|"
            r"(?:shipping|freight)\s+(?:both\s+ways|back)|"
            r"(?:if|what\s+if)\s+i\s+(?:don'?t|do\s+not)\s+like|"
            r"change\s+my\s+mind|"
            r"restocking(?:\s+fee)?|"
            r"money\s+back|"
            r"trial\s+period|try\s+it\s+(?:at\s+home|for\s+\d+)|"
            r"\brma\b|"
            r"return\s+window|how\s+long\s+(?:do\s+i\s+have\s+)?to\s+return|"
            r"반품\s*정책|교환\s*정책"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_WARRANTY_TERMS,
        re.compile(
            r"("
            r"how\s+long\s+is\s+the\s+warranty|"
            r"\bwarranty\b[^.!?;]{0,75}\bcover(?:s|ed)?\b|"
            r"\blabor\s+coverage\b|\byears?\s+(?:cover|include)\s+labor\b|"
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
            r"\bapr\b|interest\s+rate|"
            r"can\s+i\s+split\s+the\s+(?:cost|payment)|"
            r"할부"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_REFURBISHED,
        re.compile(
            r"("
            r"refurbish(?:ed)?|open[\s-]?box|scratch\s+and\s+dent|"
            r"used\s+(?:chair|massage)|second[\s-]?hand|"
            r"outlet\s+(?:chair|model)|floor\s+model|"
            r"do\s+you\s+sell\s+used|"
            r"(?:is|are)\s+(?:this|it|they|these)\s+(?:a\s+)?(?:brand[\s-]?)?new(?:\s+chairs?)?|"
            r"brand[\s-]?new\s+chairs?|"
            r"(?:brand[\s-]?)?new\s+or\s+used"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_TRADE_IN,
        re.compile(
            r"("
            r"trade[\s-]?ins?|"
            r"trade\s+(?:in|my|an?)\s+(?:my\s+)?(?:old\s+)?(?:chair|one)|"
            r"buy\s+(?:my|an?)\s+old\s+chair|"
            r"haul\s+away\s+(?:my\s+)?old|"
            r"exchange\s+programs?|"
            r"replace\s+previous\s+models?|"
            r"upgrade\s+(?:program|my\s+(?:old\s+)?(?:chair|model|one))"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_LIMITS,
        re.compile(
            r"("
            r"pet[\s-]?friendly|scratch(?:es|ing)?|"
            r"\b(?:cat|cats|dog|dogs|kitten|puppy)\b|"
            r"gift\s*wrap|gift\s+receipt|gift\s+card|hide\s+(?:the\s+)?price|"
            r"speak\s+spanish|in\s+spanish|espa[nñ]ol|puedo\s+comprar|"
            r"commercial\s+use|\bsalon\b|\bspa\s+use\b|wholesale|dealer\s+pric|"
            r"tax\s+exempt|resale\s+cert|"
            r"affiliate\s+program|influencer\s+program|referral\s+program|"
            r"become\s+an?\s+affiliate|"
            r"\b889\b|government\s+(?:form|contract|purchase)|"
            r"outdoor|patio|porch|"
            r"medical\s+device|fda\s+approv|"
            r"pregnan(?:t|cy)|for\s+(?:kids?|children|toddlers?)|"
            r"safe\s+(?:for\s+)?(?:kids?|children|pregnancy)|"
            r"pacemaker|after\s+(?:a\s+)?(?:hip\s+)?surgery|"
            r"\bhsa\b|\bfsa\b|flex(?:ible)?\s+spend|insurance\s+cover|"
            r"how\s+loud|decibel|\bnois(?:y|e)\b|"
            r"voltage|special\s+outlet|what\s+outlet|"
            r"(?:110|120|220|240)\s*v(?:olts?)?|"
            r"\d{2,3}\s*hz|"
            r"power\s+input|ac\s+(?:power|input|voltage)|"
            r"supports?\s+(?:both\s+)?(?:\d+\s*v|120|240)|"
            r"(?:auto(?:\s+|-)?show|trade\s+show|car\s+show|\bexpo\b)|"
            r"displayed\s+at.{0,48}\b(?:show|expo|event)\b|"
            r"which\s+(?:listed\s+)?product\s+was\s+displayed|"
            r"on\s+(?:the\s+)?(?:show\s+)?floor\s+at|"
            r"paypal|apple\s+pay|google\s+pay|\bvenmo\b|"
            r"(?:amex|american\s+express|visa|mastercard|discover|credit\s+card|debit\s+card|\bcard\b)\s+"
            r"(?:(?:is|was|has\s+been|keeps?\s+getting|is\s+being)\s+)?(?:not\s+going\s+through|declin(?:ed|ing)|won'?t\s+go\s+through|rejected|fail(?:ed|ing))|"
            r"\bcheckout\s+(?:problem|issue|help)\b|"
            r"(?:payment|checkout)\s+(?:won'?t|will\s+not|not)\s+(?:go\s+through|work|process)|"
            r"card\s+declin|"
            r"look\s+up.{0,48}friend|friend.{0,48}look\s+up|(?:my|a)\s+friend\s+has|"
            r"friend'?s\s+(?:chair|model)|"
            r"how\s+easy.{0,24}move|move\s+(?:the\s+)?chair\s+once|"
            r"closer\s+or\s+further\s+from\s+a\s+wall|"
            r"sales\s+tax|how\s+much\s+tax|"
            r"made\s+in|country\s+of\s+origin|"
            r"how\s+long\s+(?:do\s+they|does\s+it|will\s+it)\s+last|lifespan|"
            r"p\.?\s*o\.?\s*box|po\s+box"
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
            r"where\s+(?:(?:can|could)\s+i\s+|to\s+)?(?:try|test|sit\s+in)\b|"
            r"where\s+(?:can|could)\s+i\s+see\s+(?:the|this|a)\s+chair|"
            r"(?:local|nearby|nearest)\s+(?:dealer|retailer)|"
            r"(?:dealer|retailer)s?\s+(?:in|near)\b|"
            r"in\s+person|"
            r"visit\s+(?:you|your|the)|come\s+see|"
            r"where\s+are\s+you\s+located|your\s+address|"
            r"(?:business|store|opening)\s+hours|what\s+are\s+your\s+hours|"
            r"(?:book|schedule|make)\s+(?:a\s+)?(?:showroom\s+)?(?:visit|appointment|tour)|"
            r"open\s+(?:on\s+)?(?:sun(?:day)?s?|sat(?:urday)?s?|today)|"
            r"are\s+you\s+open|"
            r"(?:sunday|saturday)\s+(?:hours|open)|"
            r"(?:can\s+i|do\s+you)\s+pick\s*up|"
            r"\bpickup\b|"
            r"pick\s+up\s+(?:today|now|a\s+chair|from)|"
            r"매장|쇼룸"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_MECHANISM,
        re.compile(
            r"("
            r"(?:what(?:'?s|\s+is)|explain|mean(?:ing)?)\s+"
            r"(?:a\s+|the\s+|an\s+)?(?:2\s*d|3\s*d|4\s*d|5\s*d|dual[\s-]*roller)|"
            r"(?:2\s*d|3\s*d|4\s*d|5\s*d)\s*(?:vs\.?|versus|or)\s*(?:2\s*d|3\s*d|4\s*d|5\s*d)|"
            r"difference\s+between\s+(?:2\s*d|3\s*d|4\s*d|5\s*d)|"
            r"what(?:'?s|\s+is)\s+(?:2\s*d|3\s*d|4\s*d|5\s*d)\s+massage|"
            r"dual[\s-]*roller|"
            r"2\s*d\s+massage|3\s*d\s+massage\s+chairs?|4\s*d\s+massage\s+chairs?|"
            r"5\s*d\s+(?:massage|mechanism)|"
            r"(?:2d|3d|4d|5d).{0,12}차이|차이.{0,12}(?:2d|3d|4d|5d)"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        TOPIC_SHIPPING,
        re.compile(
            r"("
            r"shipping\s+(?:cost|fee|price|rate|polic|charge|time|handle)|"
            r"how\s+long\s+(?:does\s+)?(?:the\s+)?(?:shipping|delivery)|"
            r"how\s+long\s+(?:for|until|to)\s+(?:ship|deliver|arrive)|"
            r"how\s+many\s+weeks|"
            r"how\s+much\s+(?:is|for|does)\s+(?:the\s+)?(?:shipping|delivery)|"
            r"(?:free|paid)\s+(?:shipping|delivery)|"
            r"(?:shipping|delivery)\s+(?:is\s+)?free|"
            r"is\s+(?:shipping|delivery)\s+(?:free|included|extra)|"
            r"(?:do|can)\s+you\s+(?:ship|deliver)\s+to|"
            r"delivery\s+(?:polic|process|option|method|time)|"
            r"curbside|"
            r"how\s+(?:is|does)\s+it\s+(?:delivered|ship|arrive)|"
            r"who\s+delivers|"
            r"takes?\s+(?:to\s+)?(?:ship|deliver|arrive)|"
            r"\bshipping\b|\bdelivery\b|"
            r"배송|택배"
            r")",
            re.IGNORECASE,
        ),
    ),
)


def detect_topic(text: str) -> Optional[str]:
    """Return the pre-purchase policy topic in this message, if any."""
    raw = (text or "").strip().replace("’", "'")
    if not raw or reports_owned_chair_problem(raw):
        return None
    service_question = bool(next(_affirmative_matches(_TECHNICAL_SERVICE_RE, raw), None))
    if is_post_purchase(raw):
        # Owning one chair does not turn a question about another chair's
        # pre-buy service policy into a repair. Actual faults took priority.
        if service_question and next(_affirmative_matches(_EXPLICIT_NEW_PURCHASE_RE, raw), None):
            return TOPIC_TECHNICAL_SERVICE
        return None
    from sales_language import is_medical_scope_question

    if is_medical_scope_question(raw):
        return TOPIC_LIMITS
    # Paying for ownership is different from temporary use with a return.
    # Service coverage is also independent of shipping to that region.
    if _LEASE_QUESTION_RE.search(raw):
        return TOPIC_LEASE
    if service_question:
        return TOPIC_TECHNICAL_SERVICE
    # A location in a showroom question is not a shipping destination.
    if re.search(
        r"\b(?:showroom|local\s+dealer|nearby\s+dealer|nearest\s+dealer|"
        r"dealers?\s+(?:in|near)|where\s+(?:(?:can|could)\s+i\s+|to\s+)?(?:try|test))\b",
        raw,
        re.I,
    ):
        return TOPIC_SHOWROOM
    # Guam is an explicit no. Hawaii/Alaska ship, but freight is quoted and
    # paid by the customer — that must not fall into the generic "curbside
    # included" copy.
    if _RESTRICTED_REGION_RE.search(raw):
        return TOPIC_RESTRICTED_REGION
    if _REMOTE_REGION_RE.search(raw):
        return TOPIC_REMOTE_SHIPPING
    zip_topic = _topic_from_us_zip(raw)
    if zip_topic:
        return zip_topic
    if _ZIP_WORD_RE.search(raw) and _US_ZIP_RE.search(raw):
        return TOPIC_SHIPPING
    if _INTERNATIONAL_RE.search(raw):
        return TOPIC_INTERNATIONAL
    skip_shipping = bool(
        re.search(
            r"shipping\s+weight|boxed\s+(?:size|weight|dimensions)|package\s+weight",
            raw,
            re.I,
        )
    )
    # A mechanism keyword in a model name must not replace a concrete feature
    # question with the generic 2D/3D/4D explainer. A product title such as
    # "4D massage chair" is not a request to define 4D.
    mechanism_definition = bool(re.search(
        r"\b(?:what(?:'s|\s+is)|explain|mean(?:ing)?|difference|versus|\bvs\.?\b)\b",
        raw,
        re.I,
    ))
    concrete_purchase = bool(re.search(
        r"\b(?:in\s+stock|out\s+of\s+stock|lead\s+time|how\s+much|sku|quantity)\b",
        raw,
        re.I,
    ))
    specific_spec = bool(
        re.search(
            r"\bhow\s+(?:heavy|wide|tall|long)|how\s+much.{0,48}\bweigh|"
            r"\b(?:chair|shipping|product)\s+weight|"
            r"\bweight\s+(?:capacity|limit)|\b(?:maximum|minimum|max|min)\s+(?:user\s+)?height|"
            r"\b(?:dimensions?|doorway|foot\s+rollers?|calf\s+rollers?|airbags?|"
            r"bluetooth|wi-fi|upholstery)\b|"
            r"\b(?:does|has|have)\b.{0,100}\b(?:heat|roller|recline|body\s+scan)",
            raw,
            re.I,
        )
    )
    for topic, pattern in _TOPIC_PATTERNS:
        if skip_shipping and topic == TOPIC_SHIPPING:
            continue
        if topic == TOPIC_MECHANISM and (specific_spec or concrete_purchase or not mechanism_definition):
            continue
        if topic == TOPIC_WHITE_GLOVE and specific_spec and not re.search(
            r"\b(?:white[\s-]*glove|delivery|deliver|crew|assembly\s+(?:service|fee|cost)|installation\s+(?:service|fee|cost))\b", raw, re.I,
        ):
            # Assembled/disassembled dimensions describe fit, not a request
            # for an installation or delivery service.
            continue
        if topic == TOPIC_WHITE_GLOVE and re.search(r"(?<!dis)assembled\b", raw, re.I) and re.search(
            r"\b(?:doors?|doorway|opening|entrance|narrower|wider|width|\d{2}(?:\.\d+)?\s*(?:\"|in\b|inch(?:es)?|cm))", raw, re.I,
        ) and not re.search(r"\b(?:white[\s-]*glove|assembly|install\w*|set\s*it\s*up|put\s+it\s+together)\b", raw, re.I):
            continue
        if pattern.search(raw):
            return topic
    return None


def followup_topic(text: str, previous: Optional[str]) -> Optional[str]:
    """Carry a published-policy question across a short topical follow-up."""
    raw = (text or "").strip()
    if not previous or is_post_purchase(raw):
        return None
    patterns = {
        TOPIC_RETURNS: r"\bboth\s+ways\b|\b(?:deduct\w*|restocking|packaging|rma|authorization)\b|\breturn\s+(?:shipping|freight|costs?)\b|\btrip\s+(?:here|back)\b",
        TOPIC_WARRANTY_TERMS: r"\b(?:labor|parts|coverage)\b|\b(?:all|three|3)\s+years\b",
        TOPIC_LEASE: r"\bgive\s+(?:it|the chair)\s+back\b|\breturn\s+(?:it|the chair)\b.{0,40}\b(?:after|at the end)\b|\b(?:own|keep)\b.{0,40}\b(?:after|at the end)\b",
        TOPIC_REMOTE_SHIPPING: r"\b(?:freight|there|that\s+(?:location|address))\b.{0,40}\b(?:free|pay|quote|cost|included)\b|\b(?:free|pay|quote|cost)\b.{0,40}\b(?:there|freight)\b",
        TOPIC_SHOWROOM: r"\b(?:open|closed|hours|sunday|saturday|appointment|visit|confirmed)\b|\b(?:there|floor)\b.{0,35}\b(?:try|today|available)\b|\bbe\s+there\b",
    }
    pattern = patterns.get(previous)
    return previous if pattern and re.search(pattern, raw, re.I) else None


# ---------------------------------------------------------------------------
# Answers — every claim traceable to a published page
# ---------------------------------------------------------------------------


def _policy_url(domain: str, path: str) -> str:
    return f"{get_storefront_base_url(domain).rstrip('/')}/{path.lstrip('/')}"


def _returns_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/sales-policy")
    return (
        "**Returns — 30 days from delivery**\n\n"
        "- You can return a chair **within 30 days of delivery**, for any reason.\n"
        "- **You pay both the original outbound shipping and the return shipping.** "
        "The same applies if you cancel after the order has already shipped.\n"
        "- The chair must come back in **original packaging**, new and resellable.\n"
        "- A **Return Merchandise Authorization (RMA)** has to be approved first.\n"
        "- **White Glove fee is not refundable.**\n"
        "- Non-Titan-brand items carry a **20% restocking fee**.\n\n"
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


def _technical_service_answer(domain: str, message: str = "") -> str:
    # These storefronts publish the same Titan Chair service contact and
    # geographic limitation. Don't infer coverage for an unverified store.
    origin = get_storefront_base_url(domain)
    if origin not in {"https://osakiusa.com", "https://titanchair.com", "https://osakimassagechair.com"}:
        return (
            "For technical support, call **1-888-848-2630 ext. 3**. Ask the service team to confirm "
            "your model's warranty and the options for your ZIP code before purchasing. "
            "I can't confirm coverage or schedule a technician from this chat."
        )
    url = _policy_url(domain, "pages/warranty")
    area = " (including California)" if re.search(r"\b(?:california|los angeles)\b", message or "", re.I) else ""
    return (
        "**How to get technical support**\n\n"
        f"The published massage-chair warranty terms apply in the **48 contiguous U.S. states**{area} and "
        "**Canada**. Coverage still depends on your model and warranty terms.\n\n"
        "Call the service team at **1-888-848-2630 ext. 3**. If you're buying, ask about the service options "
        "for the exact model and ZIP code. If you need warranty service after purchase, have your proof of "
        "purchase ready. The team can advise on next steps; I can't confirm local technician availability, "
        "a visit, or repair costs here.\n\n"
        f"Warranty and service details: {url}"
    )


def _shipping_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/shipping-handling")
    return (
        "**Delivery — curbside or White Glove**\n\n"
        "- Standard **curbside** delivery currently takes **up to 2 weeks**.\n"
        "- **White Glove** (brought inside and assembled) currently takes **up to 3 weeks**.\n"
        "- The carrier contacts you with a **delivery window**. Exact time-of-day "
        "requests aren't available, so I can't promise a specific calendar date.\n"
        "- Curbside is to the **curb or driveway**, and a **signature is required**. "
        "Assembly is not included — the chair ships with instructions and tools.\n"
        "- Please **inspect the packaging before signing**. Note any visible damage on "
        "the delivery receipt and tell us right away.\n"
        "- Measure **doorways, hallways, and stairs** before ordering — tell me a "
        "doorway width and I'll check fit.\n\n"
        f"Full details: {url}\n\n"
        "Shipping cost depends on the model and your address. I won't quote a dollar "
        "amount here. Share your **zip code** and the model, and a specialist will confirm."
    )


def _remote_shipping_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/shipping-handling")
    return (
        "**Hawaii and Alaska — we do ship, you pay freight**\n\n"
        "We deliver to Hawaii and Alaska, but **standard included shipping does not "
        "apply**. You pay the shipping cost.\n\n"
        "The amount **depends on the chair model and the exact address**, so there "
        "isn't a published rate I can quote. Sales gets a quote from the carrier "
        "and then tells you the cost before you order.\n\n"
        "Share your **email, zip code, and the model** you're looking at and a "
        "specialist will request that quote.\n\n"
        f"Delivery options: {url}"
    )


def _restricted_region_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/shipping-handling")
    return (
        "**We don't ship to Guam.**\n\n"
        "Hawaii and Alaska are served (you pay freight, quoted per model and "
        "address), but Guam is outside our delivery network.\n\n"
        f"Delivery options: {url}"
    )


def _white_glove_answer(domain: str, message: str = "") -> str:
    url = _policy_url(domain, "pages/shipping-handling")
    extra = ""
    if re.search(r"tablet", message or "", re.I):
        extra = (
            "\nI won't confirm a separate **tablet-install** service from this "
            "chat. White Glove is in-home delivery plus assembly of the chair.\n"
        )
    return (
        "**White Glove delivery and assembly**\n\n"
        "- Currently takes **up to 3 weeks** (standard curbside is up to 2 weeks).\n"
        "- Standard curbside **does not include assembly**. White Glove adds "
        "in-home delivery plus assembly.\n"
        "- The published **basic White Glove fee is $275**. Extra charges can apply for a "
        "**third person for stair carry ($120–$300)**, additional on-site time, added "
        "distance, or a redelivery attempt if the chair can't get in. "
        "The $275 figure is the published basic fee, not a final quote for every address.\n"
        "- **The White Glove fee is not refundable.**\n"
        "- The standard team is **two delivery professionals**, and the basic fee covers "
        "a short on-site window for delivery and assembly.\n"
        "- Crews **cannot move your furniture, remove doors, or haul away an old chair** — "
        "the path needs to be clear and accessible before they arrive.\n"
        "- If a door or staircase is too narrow to complete delivery, an attempt fee may apply — "
        "so measuring first matters. Tell me your **doorway width** and I'll check fit.\n"
        f"{extra}\n"
        f"Full details: {url}"
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


def local_dealer_requested(message: str) -> bool:
    """Recognize a local demo request without extracting or echoing a location."""
    return bool(_LOCAL_DEALER_RE.search(message or ""))


def _showroom_answer(domain: str, message: str = "") -> str:
    from sales_cta import showroom_address, showroom_hours, showroom_maps_url, showroom_phone

    phone = showroom_phone(domain)
    phone_line = f"- **Call:** {phone}\n" if phone else ""
    extra: list[str] = []
    raw = message or ""
    if local_dealer_requested(raw):
        return (
            "**Find a nearby place to try the chair**\n\n"
            f"Use the official [Osaki Dealer Locator]({_DEALER_LOCATOR_URL}) and search by city or ZIP code. "
            "Call a listed store before you travel and ask whether the exact model is on display to try. "
            "I don't have live dealer floor inventory.\n\n"
            f"The Carrollton showroom ({showroom_address()}) is another option, not a confirmed nearest store. "
            "Confirm model availability there before visiting too."
        )
    if re.search(r"\bsun(?:day)?s?\b", raw, re.I):
        extra.append(
            "Sunday isn't listed in the published hours, so I won't promise "
            "we're open that day."
        )
    window = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday)s?\b(?:\s+(morning|afternoon|evening|midday|noon))?",
        raw, re.I,
    )
    if window and re.search(r"\b(?:prefer\w*|works?|best|would\s+like|want|plan\w*|visit|come|window|available)\b", raw, re.I):
        day = window.group(1).capitalize()
        when = f"{day} {window.group(2).lower()}" if window.group(2) else day
        extra.append(
            f"Noted: your preferred visit window is **{when}**. Compare it with the published hours above, "
            "and share it with **Request a visit** so Sales can confirm a time; it is not booked yet."
        )
    if re.search(r"pick\s*up|\bpickup\b", raw, re.I):
        extra.append(
            "I won't confirm a chair is ready for same-day pickup from this "
            "chat. A specialist can check showroom stock."
        )
    if re.search(r"\b(?:dealer|retailer)s?\b", raw, re.I):
        extra.append(
            f"You can also search the official [Osaki Dealer Locator]({_DEALER_LOCATOR_URL}). "
            "Check with the store for availability of the exact floor model before visiting."
        )
    extra_block = ("\n\n" + " ".join(extra)) if extra else ""
    return (
        "**Visit our showroom**\n\n"
        f"- **Address:** {showroom_address()}\n"
        f"- **Hours:** {showroom_hours()}\n"
        f"{phone_line}"
        f"- **Map:** {showroom_maps_url()}\n\n"
        "Tap **Request a visit** to share your preferred time. "
        "Sales must confirm the appointment and model availability before you travel. "
        "I can't lock a calendar slot from this chat."
        f"{extra_block}"
    )


def _refurbished_answer(_domain: str) -> str:
    return (
        "We sell **new** chairs from the current store catalog. I don't have "
        "refurbished, open-box, or used inventory to quote here.\n\n"
        "A specialist can tell you if anything like that exists outside this "
        "chat — share your **email** if you want that follow-up."
    )


def _trade_in_answer(_domain: str) -> str:
    return (
        "We **don't take trade-ins**, and delivery crews **don't haul away** "
        "an old chair.\n\n"
        "A specialist can still help you choose a new one — share your "
        "**email** if you want that follow-up."
    )


def _international_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/shipping-handling")
    return (
        "**US delivery from this storefront**\n\n"
        "Hawaii and Alaska are served (**you pay freight**, quoted per model and "
        "address). **Guam is not served.**\n\n"
        "I don't have a published Canada / Mexico / overseas rate I can quote "
        "here. A specialist can confirm whether they can quote that address — "
        "share your **email** and the destination.\n\n"
        f"US delivery options: {url}"
    )


def _limits_kind(message: str) -> str:
    raw = message or ""
    from sales_language import is_medical_scope_question

    if is_medical_scope_question(raw):
        return "medical"
    if re.search(
        r"gift\s*wrap|gift\s+receipt|gift\s+card|hide\s+(?:the\s+)?price", raw, re.I
    ):
        return "gift"
    if re.search(
        r"speak\s+spanish|in\s+spanish|espa[nñ]ol|puedo\s+comprar", raw, re.I
    ):
        return "language"
    if re.search(
        r"affiliate\s+program|influencer\s+program|referral\s+program|"
        r"become\s+an?\s+affiliate",
        raw,
        re.I,
    ):
        return "affiliate"
    if re.search(
        r"\b889\b|government\s+(?:form|contract|purchase)",
        raw,
        re.I,
    ):
        return "procurement"
    if re.search(
        r"commercial\s+use|\bsalon\b|\bspa\s+use\b|wholesale|dealer\s+pric|"
        r"tax\s+exempt|resale\s+cert",
        raw,
        re.I,
    ):
        return "commercial"
    if re.search(
        r"look\s+up.{0,48}friend|friend.{0,48}look\s+up|(?:my|a)\s+friend\s+has|"
        r"friend'?s\s+(?:chair|model)",
        raw,
        re.I,
    ):
        return "friend_model"
    if re.search(
        r"how\s+easy.{0,24}move|move\s+(?:the\s+)?chair\s+once|"
        r"closer\s+or\s+further\s+from\s+a\s+wall",
        raw,
        re.I,
    ):
        return "relocate"
    if re.search(r"outdoor|patio|porch", raw, re.I):
        return "outdoor"
    if re.search(
        r"medical\s+device|fda\s+approv|pregnan(?:t|cy)|"
        r"for\s+(?:kids?|children|toddlers?)|safe\s+(?:for\s+)?(?:kids?|children|pregnancy)|"
        r"pacemaker|after\s+(?:a\s+)?(?:hip\s+)?surgery",
        raw,
        re.I,
    ):
        return "medical"
    if re.search(r"\bhsa\b|\bfsa\b|flex(?:ible)?\s+spend|insurance\s+cover", raw, re.I):
        return "benefit_payment"
    if re.search(r"how\s+loud|decibel|\bnois(?:y|e)\b", raw, re.I):
        return "noise"
    if re.search(
        r"(?:auto(?:\s+|-)?show|trade\s+show|car\s+show|\bexpo\b)|"
        r"displayed\s+at.{0,48}\b(?:show|expo|event)\b|"
        r"which\s+(?:listed\s+)?product\s+was\s+displayed|"
        r"on\s+(?:the\s+)?(?:show\s+)?floor\s+at",
        raw,
        re.I,
    ):
        return "event_display"
    if re.search(
        r"voltage|special\s+outlet|what\s+outlet|"
        r"(?:110|120|220|240)\s*v(?:olts?)?|"
        r"\d{2,3}\s*hz|"
        r"power\s+input|ac\s+(?:power|input|voltage)|"
        r"supports?\s+(?:both\s+)?(?:\d+\s*v|120|240)",
        raw,
        re.I,
    ):
        return "power"
    if re.search(
        r"paypal|apple\s+pay|google\s+pay|\bvenmo\b|"
        r"(?:amex|american\s+express|visa|mastercard|discover|credit\s+card|debit\s+card|\bcard\b)\s+"
        r"(?:(?:is|was|has\s+been|keeps?\s+getting|is\s+being)\s+)?(?:not\s+going\s+through|declin(?:ed|ing)|won'?t\s+go\s+through|rejected|fail(?:ed|ing))|"
        r"\bcheckout\s+(?:problem|issue|help)\b|"
        r"(?:payment|checkout)\s+(?:won'?t|will\s+not|not)\s+(?:go\s+through|work|process)|"
        r"card\s+declin",
        raw,
        re.I,
    ):
        if re.search(
            r"not\s+going\s+through|declin|won'?t\s+go\s+through|rejected|fail|"
            r"not\s+(?:go\s+through|work|process)|checkout\s+(?:problem|issue|help)",
            raw,
            re.I,
        ):
            return "checkout_issue"
        return "payments"
    if re.search(r"sales\s+tax|how\s+much\s+tax", raw, re.I):
        return "tax"
    if re.search(r"made\s+in|country\s+of\s+origin", raw, re.I):
        return "origin"
    if re.search(
        r"how\s+long\s+(?:do\s+they|does\s+it|will\s+it)\s+last|\blifespan\b",
        raw,
        re.I,
    ):
        return "lifespan"
    if re.search(r"p\.?\s*o\.?\s*box|po\s+box", raw, re.I):
        return "pobox"
    return "pets"


def _limits_answer(_domain: str, message: str = "") -> str:
    kind = _limits_kind(message)
    if kind == "gift":
        return (
            "I **can't hide a published price** or offer gift wrap from this chat.\n\n"
            "I can send a model link, or a specialist can help with a gift order — "
            "share your **email**."
        )
    if kind == "language":
        return (
            "I can help with basic chair shopping in **Spanish**. Some policy and "
            "technical source text remains in **English**, and I won't guess at "
            "a translation that hasn't been reviewed.\n\n"
            "A specialist can help with those questions; sharing contact details is optional."
        )
    if kind == "affiliate":
        return (
            "I don't have a published **affiliate** or referral program in this "
            "chat.\n\n"
            "A specialist can confirm whether anything exists outside this "
            "storefront — share your **email**."
        )
    if kind == "procurement":
        return (
            "I don't have a published **government-procurement** or form-889 "
            "answer I can quote here.\n\n"
            "A specialist can follow up — share your **email**."
        )
    if kind == "commercial":
        return (
            "The chairs on this storefront are sold as **home** massage chairs. "
            "I don't have a published commercial, salon, or wholesale program here.\n\n"
            "A specialist can confirm — share your **email**."
        )
    if kind == "friend_model":
        return (
            "I can only match chairs on this **store catalog**. I can't look up "
            "a chair someone else owns unless you share the **model name** as it "
            "appears on the chair or the box.\n\n"
            "If you're not sure, tell me your **height** and what you want the "
            "chair to help with."
        )
    if kind == "relocate":
        return (
            "I don't have a published rating for how easy a chair is to scoot "
            "after setup, and I won't invent casters or a move-it-yourself spec.\n\n"
            "Some models list **wall clearance** — tell me a model name and I'll "
            "share that number. White Glove crews don't come back later to "
            "rearrange a room."
        )
    if kind == "checkout_issue":
        return (
            "I can't see the checkout page from this chat, so I won't guess why "
            "a card didn't go through, and I won't list processors from memory.\n\n"
            "A Sales specialist can look at the payment page with you — share your "
            "**email**."
        )
    if kind == "outdoor":
        return (
            "These chairs are sold for **indoor home use**. I won't recommend one "
            "for a patio or outdoor space."
        )
    if kind == "benefit_payment":
        return (
            "I don't have verified **FSA / HSA eligibility or payment acceptance** "
            "for these chairs. Please check with your **plan administrator** "
            "before purchasing; eligibility and any documentation requirements "
            "depend on your plan.\n\n"
            "Sales can confirm the storefront's payment process and available "
            "product documentation. I can't promise reimbursement or coverage."
        )
    if kind == "medical":
        return (
            "I **won't give medical advice**, and these chairs are **not listed as "
            "medical devices**.\n\n"
            "For pregnancy, children, or a medical condition, ask your clinician. "
            "A specialist can still help you shop if a doctor says a home massage "
            "chair is OK."
        )
    if kind == "noise":
        return (
            "I don't have a **published decibel rating** I can quote.\n\n"
            "The Carrollton showroom is the honest way to judge noise, or a "
            "specialist can follow up — share your **email**."
        )
    if kind == "event_display":
        try:
            from config import SALES_PHONE
        except ImportError:  # pragma: no cover
            SALES_PHONE = "+1-888-848-2630 ext. 2"
        return (
            "I don't have a published list of which models were displayed at a "
            "show or event.\n\n"
            f"Please call **Sales at {SALES_PHONE}**. "
            "I won't invent floor inventory from this chat."
        )
    if kind == "power":
        try:
            from config import SALES_PHONE
        except ImportError:  # pragma: no cover
            SALES_PHONE = "+1-888-848-2630 ext. 2"
        return (
            "I don't have a published **voltage / outlet** spec I can quote here.\n\n"
            "This storefront is for US checkout. "
            f"Call **Sales at {SALES_PHONE}** to confirm the plug for your address, "
            "or share your **email** and zip code for a specialist follow-up. "
            "I won't invent a voltage rating."
        )
    if kind == "payments":
        return (
            "Checkout shows the payment methods this storefront accepts. "
            "I can confirm **Affirm** for pay-over-time. I won't list other "
            "processors from memory, and I won't quote a rate.\n\n"
            "A specialist can confirm what's on the payment page — share your "
            "**email**."
        )
    if kind == "tax":
        return (
            "**Sales tax** is calculated at checkout for the ship-to address. "
            "I won't quote a rate from this chat."
        )
    if kind == "origin":
        return (
            "I don't have a published **country-of-origin** line I can quote "
            "per model here.\n\n"
            "A specialist can pull that from the carton or spec sheet — share "
            "your **email** and the model."
        )
    if kind == "lifespan":
        return (
            "I don't have a published **lifespan in years** I can quote. "
            "What I can confirm is the **3-year** standard warranty, with optional "
            "4- and 5-year plans at purchase. Ask about warranty terms if you "
            "want those published details."
        )
    if kind == "pobox":
        return (
            "I won't confirm **PO Box** delivery from this chat. Share your "
            "**email**, zip code, and the model, and a specialist can check "
            "whether that address is serviceable."
        )
    return (
        "I don't have a published **pet-proof** or scratch-resistance spec. "
        "These are indoor home chairs.\n\n"
        "A specialist can talk materials if that's the deciding factor — "
        "share your **email**."
    )


def _lease_answer(domain: str) -> str:
    text = (
        "This storefront does not list a **lease or rental** program for massage chairs. "
        "I can't offer a fixed rental term or a return-at-the-end agreement.\n\n"
        "Paying in installments is a **purchase**, not a lease where you give the chair back. "
        "If temporary use without ownership is essential, purchase financing doesn't meet that need."
    )
    if get_storefront_base_url(domain) in {"https://osakiusa.com", "https://titanchair.com", "https://osakimassagechair.com"}:
        return f"{text}\n\nPublished sales policy: {_policy_url(domain, 'pages/sales-policy')}"
    return text


def _mechanism_answer(_domain: str) -> str:
    return (
        "**2D / 3D / 4D / 5D / Dual Roller**\n\n"
        "- **2D** massages along the X and Y axis (up and down).\n"
        "- **3D** massages on the X, Y, and Z axis (up and down, left and right, "
        "and in and out).\n"
        "- **4D** includes the 3D features with rhythmical, speed-varying massage "
        "patterns for a more lifelike experience.\n"
        "- **5D** combines all 4D features with enhanced AI body scanning, "
        "micro-adjustments, or extra fine-tuning.\n"
        "- **Dual Roller** means two separate mechanisms operate independently — "
        "usually upper back/shoulders and lower back/hips at the same time.\n\n"
        "Tell me a model name and I'll confirm which mechanism that chair uses."
    )


_ANSWERS = {
    TOPIC_RETURNS: _returns_answer,
    TOPIC_WARRANTY_TERMS: _warranty_terms_answer,
    TOPIC_TECHNICAL_SERVICE: _technical_service_answer,
    TOPIC_SHIPPING: _shipping_answer,
    TOPIC_REMOTE_SHIPPING: _remote_shipping_answer,
    TOPIC_RESTRICTED_REGION: _restricted_region_answer,
    TOPIC_INTERNATIONAL: _international_answer,
    TOPIC_WHITE_GLOVE: _white_glove_answer,
    TOPIC_FINANCING: _financing_answer,
    TOPIC_REFURBISHED: _refurbished_answer,
    TOPIC_TRADE_IN: _trade_in_answer,
    TOPIC_LEASE: _lease_answer,
    TOPIC_LIMITS: _limits_answer,
    TOPIC_SHOWROOM: _showroom_answer,
    TOPIC_MECHANISM: _mechanism_answer,
}


def policy_answer(topic: str, domain: str = "", message: str = "") -> Optional[str]:
    """Render the published answer for a pre-purchase policy topic."""
    builder = _ANSWERS.get(topic)
    if builder is None:
        return None
    if topic == TOPIC_SHOWROOM:
        return _showroom_answer(domain, message=message)
    if topic == TOPIC_LIMITS:
        return _limits_answer(domain, message=message)
    if topic == TOPIC_WHITE_GLOVE:
        return _white_glove_answer(domain, message=message)
    if topic == TOPIC_TECHNICAL_SERVICE:
        return _technical_service_answer(domain, message=message)
    return builder(domain)
