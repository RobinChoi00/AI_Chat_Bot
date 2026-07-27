"""
Warranty-chat scope gate — installation, delivery, and defect support only.

The warranty embed is not a general product/sales assistant. Off-topic and
sales/pricing questions get a fixed refusal without KB lookup or LLM answers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|"
    r"thanks|thank\s+you|thx|bye|goodbye)[!.?\s]*$",
    re.IGNORECASE,
)

_WARRANTY_TOPIC_RE = re.compile(
    r"\b("
    r"warranty|install(?:ation|ing)?|setup|set\s*up|assembly|deliver(?:y|ed)?|"
    r"tracking|shipment|fedex|ups|usps|order\s*(?:status|number|#)?|"
    r"defect|malfunction|broken|not\s+working|won'?t\s+(?:turn|power|start|inflate)|"
    r"repair|troubleshoot|fix|error\s*code|err\s*\d+|e\d{1,3}|"
    r"replacement|refund|exchange|rma|compensation|"
    r"damaged|damage|missing\s+part|wrong\s+item|"
    r"remote|controller|footrest|airbag|roller|recline|power|heat|"
    r"evidence|invoice|serial|model\s*number|"
    r"보증|워런티|배송|설치|고장|수리|교환|"
    r"garant[ií]a|entrega|instalaci[oó]n|aver[ií]a|reemplazo"
    r")\b",
    re.IGNORECASE,
)

_SALES_ONLY_RE = re.compile(
    r"\b("
    r"how\s+much|what'?s?\s+the\s+price|price\s+of|cost\s+of|on\s+sale|discount|"
    r"coupon|promo\s*code|financing|affirm|buy\s+now|purchase|"
    r"recommend|suggest|best\s+chair|compare\s+models|which\s+chair\s+should|"
    r"showroom|visit\s+(?:your|the)\s+store|"
    r"specs?\s+only|dimensions?\s+only|tell\s+me\s+about\s+(?:the\s+)?(?:chair|model)\b|"
    r"가격|할인|구매|추천|"
    r"precio|descuento|comprar|recomienda"
    r")\b",
    re.IGNORECASE,
)

# Pre-purchase shipping eligibility (not post-purchase tracking / damage claims).
_RESTRICTED_SHIP_REGION_RE = re.compile(
    r"\b("
    r"hawaii|hawaiian|alaska|alaskan|guam|guamanian|"
    r"honolulu|anchorage"
    r")\b",
    re.IGNORECASE,
)

_RESTRICTED_STATE_CODE_RE = re.compile(
    r"(?:^|[^\w])(?:hi|ak)(?:$|[^\w])",
    re.IGNORECASE,
)

_PRE_PURCHASE_SHIPPING_ASK_RE = re.compile(
    r"\b("
    r"free\s+(?:deliver(?:y|ies)?|shipping)|"
    r"(?:do\s+you|can\s+you|will\s+you|able\s+to)\s+(?:deliver|ship)\b|"
    r"(?:deliver|ship|shipping|delivery)\s+to\b|"
    r"ship\s+to\b|deliver\s+to\b|"
    r"(?:available|eligibility)\s+(?:in|for|to)\b|"
    r"shipping\s+(?:to|for|policy|cost|fee|rate|price)|"
    r"delivery\s+(?:to|for|available|fee|cost)"
    r")\b",
    re.IGNORECASE,
)

_POST_PURCHASE_DELIVERY_RE = re.compile(
    r"\b("
    r"tracking|track(?:ing)?\s+(?:number|my)|fedex|ups|usps|"
    r"in\s+transit|where\s+is\s+my\s+(?:order|chair|package|shipment)|"
    r"damaged|damage|box\s+(?:was\s+)?(?:damage|crushed|opened)|"
    r"missing\s+part|wrong\s+item|signed\s+(?:cleared|damaged)|"
    r"my\s+order\s+(?:#|number)|order\s+(?:#|number)\s*[a-z0-9]"
    r")\b",
    re.IGNORECASE,
)

_OFF_TOPIC_RE = [
    re.compile(
        r"\b(write|generate|create|make)\s+(me\s+)?(a\s+)?"
        r"(poem|story|essay|code|script|song|lyrics|joke)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(python|javascript|typescript|java|c\+\+|react|sql)\b", re.IGNORECASE),
    re.compile(r"\b(recipe|cook|cooking|weather|forecast|bitcoin|crypto)\b", re.IGNORECASE),
    re.compile(r"\b(homework|math\s+problem|capital\s+of|who\s+won)\b", re.IGNORECASE),
    re.compile(r"\b(movie|netflix|anime|celebrity|politics|election)\b", re.IGNORECASE),
]

_SALES_ANSWER_KEYS = frozenset({"sales"})


@dataclass(frozen=True)
class WarrantyScopeDecision:
    in_scope: bool
    reason: str

    @property
    def is_blocked(self) -> bool:
        return not self.in_scope


def build_warranty_scope_refusal(reason: str = "") -> str:
    if (reason or "").strip().lower() == "shipping_policy":
        return (
            "We **do not deliver** to **Hawaii, Alaska, or Guam**.\n\n"
            "This chat is for **warranty support** only — setup, a delivery "
            "problem after purchase, or a chair malfunction.\n\n"
            "For other shipping or sales questions, please check the shipping "
            "policy on our website or contact our sales team."
        )
    return (
        "This chat is for **warranty support** only — installation, delivery, "
        "or a chair malfunction.\n\n"
        "For **sales, pricing, or product recommendations**, please use the "
        "main website chat or call our sales line.\n\n"
        "If you have a warranty issue, describe your chair model and what "
        "you need help with (setup, delivery, or a defect)."
    )


def is_pre_purchase_shipping_policy(text: str) -> bool:
    """True for free-shipping / region-eligibility questions (not claim tracking)."""
    raw = (text or "").strip()
    if not raw:
        return False
    if _POST_PURCHASE_DELIVERY_RE.search(raw):
        return False

    has_region = bool(
        _RESTRICTED_SHIP_REGION_RE.search(raw) or _RESTRICTED_STATE_CODE_RE.search(raw)
    )
    has_shipping_ask = bool(_PRE_PURCHASE_SHIPPING_ASK_RE.search(raw))

    if has_region and has_shipping_ask:
        return True
    # "free delivery?" with no region still belongs to sales/shipping policy.
    if has_shipping_ask and re.search(r"\bfree\s+(?:deliver|shipping)", raw, re.I):
        return True
    # Region + delivery/shipping word without post-purchase signal.
    if has_region and re.search(r"\b(deliver|delivery|ship|shipping)\b", raw, re.I):
        return True
    return False


def _enabled() -> bool:
    return os.environ.get("WARRANTY_SCOPE_GATE_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_sales_workflow_answer(answer: str) -> bool:
    return _normalize(answer) in _SALES_ANSWER_KEYS


def evaluate_warranty_scope(
    text: str,
    *,
    node_id: Optional[str] = None,
    issue_type: Optional[str] = None,
    skip: bool = False,
) -> WarrantyScopeDecision:
    """
    Return whether free text belongs in the warranty embed chat.

    Workflow answer keys and delivery-spec questions are handled by callers.
    """
    if skip or not _enabled():
        return WarrantyScopeDecision(in_scope=True, reason="disabled")

    raw = (text or "").strip()
    if not raw:
        return WarrantyScopeDecision(in_scope=True, reason="empty")

    if _GREETING_RE.match(raw):
        return WarrantyScopeDecision(in_scope=True, reason="greeting")

    if is_sales_workflow_answer(raw):
        return WarrantyScopeDecision(in_scope=False, reason="sales")

    # Pre-purchase HI/AK/Guam / free-shipping questions must not enter the
    # post-purchase delivery flowchart (even though they contain "delivery").
    if is_pre_purchase_shipping_policy(raw):
        return WarrantyScopeDecision(in_scope=False, reason="shipping_policy")

    try:
        from delivery_intake import detect_delivery_spec_question  # noqa: WPS433

        if detect_delivery_spec_question(raw):
            return WarrantyScopeDecision(in_scope=True, reason="delivery_spec")
    except ImportError:
        pass

    normalized = _normalize(raw)

    if any(p.search(raw) for p in _OFF_TOPIC_RE):
        if not _WARRANTY_TOPIC_RE.search(raw):
            return WarrantyScopeDecision(in_scope=False, reason="off_topic")

    if _SALES_ONLY_RE.search(raw) and not _WARRANTY_TOPIC_RE.search(raw):
        return WarrantyScopeDecision(in_scope=False, reason="sales_topic")

    # During an active warranty path, allow short replies that may be workflow answers.
    if issue_type in ("installation", "delivery", "defect") and len(normalized.split()) <= 3:
        return WarrantyScopeDecision(in_scope=True, reason="short_active_flow")

    if node_id and node_id not in ("root", "issue_type") and issue_type:
        if _WARRANTY_TOPIC_RE.search(raw):
            return WarrantyScopeDecision(in_scope=True, reason="warranty_keywords")

    if node_id in (None, "root", "issue_type") or not issue_type:
        if not _WARRANTY_TOPIC_RE.search(raw):
            if _looks_like_general_product_question(raw):
                return WarrantyScopeDecision(in_scope=False, reason="general_product")

    return WarrantyScopeDecision(in_scope=True, reason="allow")


def _looks_like_general_product_question(text: str) -> bool:
    if "?" not in text and not re.search(r"^(what|how|tell|give|show)\b", text, re.I):
        return False
    if _WARRANTY_TOPIC_RE.search(text):
        return False
    product_hint = re.search(
        r"\b(osaki|titan|hypnos|nova|chair|model|spec|dimension|feature|weight)\b",
        text,
        re.I,
    )
    return bool(product_hint) and bool(_SALES_ONLY_RE.search(text) or "spec" in text.lower())


def filter_warranty_menu_options(node: dict) -> list[dict]:
    """Hide sales routing from customer-facing warranty menus."""
    options = list(node.get("options") or [])
    node_id = str(node.get("node_id") or "")
    if node_id != "root":
        return options
    return [opt for opt in options if _normalize(str(opt.get("answer_key") or "")) != "sales"]
