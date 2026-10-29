"""
Warranty-chat scope gate — setup, defect support, and verified delivery lookup.

The warranty embed is not a general product/sales assistant. Off-topic,
sales/pricing questions get a fixed refusal (sales phone ext. 2) without KB
lookup or LLM answers. Post-purchase tracking uses the verified lookup route.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

try:
    from config import SALES_PHONE, department_phone_directory
except ImportError:  # pragma: no cover — tests still have project root on path
    SALES_PHONE = "+1-888-848-2630 ext. 2"

    def department_phone_directory() -> str:
        return (
            f"**Sales (delivery & orders):** {SALES_PHONE}\n"
            "**Warranty (setup & defects):** +1-888-848-2630 ext. 3"
        )

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

# Post-purchase delivery / tracking enters the verified delivery flow.
_DELIVERY_INQUIRY_RE = re.compile(
    r"("
    r"\btracking\b|"
    r"track(?:ing)?\s+(?:number|my)|"
    r"\bfedex\b|\bups\b|\busps\b|"
    r"in\s+transit|"
    r"where(?:'?s|\s+is)\s+my\s+(?:order|chair|package|shipment)|"
    r"when\s+will\s+it\s+(?:arrive|deliver|ship)|"
    r"how\s+long.{0,40}(?:ship|deliver|arriv)|"
    r"late\s+deliver|"
    r"never\s+arrived|"
    r"marked\s+delivered|"
    r"box\s+(?:was\s+)?(?:damage|crushed|opened)|"
    r"signed\s+(?:cleared|damaged)|"
    r"^(?:delivery|shipping|tracking)[\s!.?]*$"
    r")",
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

# Cancel / refund / return — do not troubleshoot; escalate to warranty team.
_ORDER_CANCEL_RE = re.compile(
    r"\b("
    r"cancel(?:ling|led)?\s+(?:my\s+)?(?:order|purchase|buy|subscription)|"
    r"(?:want|need|please|how\s+(?:do|can)\s+i)\s+to\s+cancel|"
    r"cancel\s+(?:my\s+)?(?:order|purchase)|"
    r"order\s+cancel|"
    r"refund(?:\s+my\s+(?:order|purchase|money|payment))?|"
    r"(?:want|need|please)\s+(?:a\s+)?refund|"
    r"return\s+(?:my\s+)?(?:order|chair|purchase|item)|"
    r"undo\s+(?:my\s+)?(?:order|purchase)|"
    r"취소|환불|반품"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WarrantyScopeDecision:
    in_scope: bool
    reason: str

    @property
    def is_blocked(self) -> bool:
        return not self.in_scope


def is_order_cancel_request(text: str) -> bool:
    """True when the customer wants to cancel, refund, or return an order."""
    raw = re.sub(r"[._]+", " ", (text or "").strip())
    if not raw:
        return False
    return bool(_ORDER_CANCEL_RE.search(raw))


def build_order_cancel_handoff_message() -> str:
    """Minimal confirmation — no cancel policy advice from the bot."""
    return (
        "Your request is saved with this case for **warranty team** review. "
        "An email response or callback time has not been confirmed. "
        "For direct follow-up, use the warranty contact below.\n\n"
        f"{department_phone_directory()}"
    )


def build_warranty_scope_refusal(reason: str = "") -> str:
    phones = department_phone_directory()
    if (reason or "").strip().lower() == "order_cancel":
        return build_order_cancel_handoff_message()
    if (reason or "").strip().lower() == "shipping_policy":
        return (
            "This chat is for **warranty support** after a purchase.\n\n"
            "**Hawaii and Alaska:** we do ship there, but **you pay the freight**. "
            "The cost depends on the model and address — sales gets a carrier quote.\n"
            "**Guam:** we do not ship there.\n\n"
            "For a quote or any other shipping question, please call **sales** "
            f"at {SALES_PHONE}. I won't quote a dollar amount here.\n\n"
            f"{phones}"
        )
    if (reason or "").strip().lower() == "delivery":
        return (
            "I can check a delivery after you provide both your **order number** "
            "and the **email used at checkout**. Carrier estimates are not guaranteed arrival dates.\n\n"
            f"{phones}"
        )
    return (
        "This chat is for **warranty support** only — setup or a chair "
        "malfunction.\n\n"
        "For **sales, pricing, product recommendations, or delivery**, "
        f"please call **sales** at {SALES_PHONE}.\n\n"
        f"{phones}\n\n"
        "If you have a warranty issue, describe your chair model and what "
        "you need help with (setup or a defect)."
    )


def is_delivery_inquiry(text: str) -> bool:
    """True for post-purchase delivery / tracking supported by this chat."""
    raw = (text or "").strip()
    if not raw:
        return False
    if _normalize(raw) == "delivery":
        return True
    if bool(_DELIVERY_INQUIRY_RE.search(raw)):
        return True
    # Vague "shipping help" / "delivery status" still belongs to sales, not
    # the warranty flowchart — unless it's clearly setup/defect or HI/AK policy.
    if is_pre_purchase_shipping_policy(raw):
        return False
    if re.search(
        r"\b(install|installation|setup|assembly|broken|defect|won't\s+turn|"
        r"airbag|footrest|remote|error\s*code)\b",
        raw,
        re.I,
    ):
        return False
    has_ship = bool(
        re.search(r"\b(shipping|shipment|delivery|deliveries|tracking)\b", raw, re.I)
    )
    has_help = bool(
        re.search(
            r"\b(help|status|where|when|late|track|package|order|question|issue)\b",
            raw,
            re.I,
        )
    )
    return has_ship and has_help


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

    # Cancel / refund / return → warranty team handoff (not sales, not defect flow).
    if is_order_cancel_request(raw):
        return WarrantyScopeDecision(in_scope=False, reason="order_cancel")

    if is_sales_workflow_answer(raw):
        return WarrantyScopeDecision(in_scope=False, reason="sales")

    # Pre-purchase HI/AK/Guam / free-shipping questions must not enter the
    # post-purchase delivery flowchart (even though they contain "delivery").
    if is_pre_purchase_shipping_policy(raw):
        return WarrantyScopeDecision(in_scope=False, reason="shipping_policy")

    # Post-purchase tracking is supported through the verified delivery lookup.
    if is_delivery_inquiry(raw):
        return WarrantyScopeDecision(in_scope=True, reason="delivery")

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
    """Keep post-purchase delivery available; hide general sales routing."""
    options = list(node.get("options") or [])
    node_id = str(node.get("node_id") or "")
    if node_id not in ("root", "issue_type"):
        return options
    hidden = {"sales"}
    return [
        opt
        for opt in options
        if _normalize(str(opt.get("answer_key") or "")) not in hidden
    ]


def human_support_reason(text: str) -> Optional[str]:
    """Detect explicit support/callback requests before any workflow mapping."""
    raw = (text or "").strip().replace("’", "'")
    # A request not to call or transfer must not itself cause that action.
    raw = re.sub(
        r"\b(?:don't|do not|never|no need to)\s+(?:call\s+me(?:\s+back)?|phone\s+me|"
        r"(?:connect|transfer)\s+me\s+(?:to|with)\s+(?:a\s+)?(?:human|person|agent))\b|"
        r"\b(?:don't|do not)\s+(?:need|want)\s+(?:a\s+)?(?:human|person|agent|callback)\b", "", raw, flags=re.I,
    )
    if re.search(r"\b(?:call\s+me|call\s+(?:me\s+)?back|callback|phone\s+me)\b|전화\s*(?:해|주세요|부탁)|ll[aá]m(?:e|ame)", raw, re.I):
        return "callback_request"
    if re.search(
        r"\b(?:human|real\s+person|live\s+(?:agent|person)|representative|customer\s+service)\b|"
        r"\b(?:speak|talk|connect)\s+(?:me\s+)?(?:to|with)\s+(?:a\s+)?(?:person|someone|agent|support)\b|"
        r"\b(?:have|get|ask)\s+(?:a\s+)?(?:person|someone|agent|specialist)\s+(?:to\s+)?(?:review|help|assist|contact)\b|"
        r"\b(?:need|want|would like)\s+(?:a\s+)?(?:person|someone|agent|specialist)\s+(?:to\s+)?(?:help|review|assist|contact)\b|"
        r"\b(?:someone|a person)\s+(?:to\s+)?(?:review|help|assist)\s+(?:me|with|this|my)\b|"
        r"상담원|(?:직원|담당자)(?:와|하고|에게|랑)?\s*(?:직접\s*)?(?:연결|상담|도움)|"
        r"hablar\s+con\s+(?:una?\s+)?(?:persona|agente)|"
        r"(?:necesito|quiero)\s+(?:que\s+)?(?:una?\s+)?(?:persona|agente|alguien)\b", raw, re.I
    ):
        return "customer_request"
    # Do not repeat a disassembly instruction once the customer says that
    # they cannot safely attempt it. The current diagnosis stays available for
    # staff review rather than being interpreted as a yes/no workflow answer.
    if re.search(
        r"\b(?:cannot|can't|unable to|not able to)\s+(?:safely\s+)?"
        r"(?:remove|open|disconnect|disassemble|lift|reach|move|perform|complete|try|do)\b[^.!?\n]{0,90}"
        r"\b(?:panel|hose|wire|cable|chair|step|check|test|that|this|it)\b|"
        r"\b(?:cannot|can't|unable to)\s+safely\s+(?:do|try|continue|proceed)\b|"
        r"\b(?:don't|do not)\s+feel\s+safe\s+(?:opening|removing|disconnecting|lifting|doing|trying)\b|"
        r"(?:안전하게\s*)?(?:분해|패널|호스|점검)[^.!?\n]{0,30}(?:못\s*하|할\s*수\s*없)|"
        r"(?:no puedo|no me siento segur[oa])\s+(?:(?:hacer|quitar|abrir|desconectar|realizar)\b|.*\bcon seguridad\b)", raw, re.I,
    ):
        return "customer_request"
    return None


def support_contact(department: str = "warranty", domain: str = "") -> dict[str, str]:
    """Customer contact details from the same verified application constants."""
    from config import SALES_EMAIL_BY_DOMAIN, SUPPORT_BUSINESS_HOURS, WARRANTY_BUSINESS_HOURS, WARRANTY_PHONE, WARRANTY_TEAM_EMAIL

    is_sales = department == "sales"
    phone = SALES_PHONE if is_sales else WARRANTY_PHONE
    base, _, extension = phone.partition(" ext.")
    phone_href = "tel:" + re.sub(r"[^+\d]", "", base)
    if extension.strip().isdigit():
        phone_href += ";ext=" + extension.strip()
    email = next((address for key, address in SALES_EMAIL_BY_DOMAIN.items() if key in (domain or "").lower()), "") if is_sales else WARRANTY_TEAM_EMAIL
    result = {"department": "sales" if is_sales else "warranty", "phone": phone, "phone_href": phone_href,
              "hours": SUPPORT_BUSINESS_HOURS if is_sales else WARRANTY_BUSINESS_HOURS}
    if re.fullmatch(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", email or ""):
        result["email"] = email
    return result
