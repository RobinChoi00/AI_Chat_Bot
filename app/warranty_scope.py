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


def build_warranty_scope_refusal() -> str:
    return (
        "This chat is for **warranty support** only — installation, delivery, "
        "or a chair malfunction.\n\n"
        "For **sales, pricing, or product recommendations**, please use the "
        "main website chat or call our sales line.\n\n"
        "If you have a warranty issue, describe your chair model and what "
        "you need help with (setup, delivery, or a defect)."
    )


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
