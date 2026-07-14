"""
Pre-LLM scope gate for the main chat agent.

Blocks clearly off-topic messages before the agent loop runs (zero main-model cost).
Uses fast rules first; optional cheap router LLM only for ambiguous cases.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from app.intent_router import infer_forced_tool
except ImportError:
    from intent_router import infer_forced_tool  # type: ignore

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|"
    r"thanks|thank\s+you|thx|bye|goodbye|see\s+you)[!.?\s]*$",
    re.IGNORECASE,
)

_SHORT_FOLLOW_UP_RE = re.compile(
    r"^(yes|no|yep|nope|ok|okay|sure|price|specs|dimensions|more\s+info|"
    r"that\s+one|go\s+on|continue|please|help)[!.?\s]*$",
    re.IGNORECASE,
)

_IN_SCOPE_KEYWORDS = (
    "massage chair",
    "massage chairs",
    "osaki",
    "titan",
    "hypnos",
    "nova",
    "otamic",
    "soho",
    "maestro",
    "orion",
    "chair",
    "warranty",
    "order",
    "delivery",
    "shipping",
    "tracking",
    "return",
    "refund",
    "exchange",
    "repair",
    "install",
    "assembly",
    "showroom",
    "white glove",
    "promo",
    "discount",
    "coupon",
    "financing",
    "service@",
    "osakititan",
    "888-848",
    "888-501",
)

_CHAIR_CONTEXT_KEYWORDS = _IN_SCOPE_KEYWORDS + (
    "spec",
    "dimension",
    "model",
    "recommend",
    "purchase",
    "buy",
    "price",
    "$",
)

_OUT_OF_SCOPE_PATTERNS = [
    re.compile(
        r"\b(write|generate|create|make)\s+(me\s+)?(a\s+)?"
        r"(poem|story|essay|code|script|song|lyrics|joke|riddle)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(python|javascript|typescript|java|c\+\+|react|sql|html|css)\s+"
        r"(code|function|program|script|app)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(recipe|ingredients|cook|cooking|bake|baking)\b", re.IGNORECASE),
    re.compile(r"\b(weather|forecast|temperature\s+today|rain\s+today)\b", re.IGNORECASE),
    re.compile(
        r"\b(stock\s+market|crypto|bitcoin|ethereum|nft)\s+(price|market|trade)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(who won|world cup|super bowl|nba\s+finals|nfl\s+draft)\b", re.IGNORECASE),
    re.compile(
        r"\b(election|president|politics|democrat|republican|congress)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(homework|solve\s+this\s+equation|math\s+problem)\b", re.IGNORECASE),
    re.compile(
        r"\btranslate\s+.+\s+to\s+(?:korean|spanish|french|japanese|chinese|german)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(capital\s+of|population\s+of|who\s+is\s+the\s+president)\b", re.IGNORECASE),
    re.compile(r"\b(movie|netflix|anime|k-pop|celebrity)\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class ScopeDecision:
    in_scope: bool
    reason: str
    used_llm: bool = False

    @property
    def is_blocked(self) -> bool:
        return not self.in_scope


def build_scope_refusal(user_query: str = "") -> str:
    from config import SUPPORT_BUSINESS_HOURS

    if re.search(r"[가-힣]", user_query or ""):
        return (
            "저는 Osaki와 Titan 마사지 의자 고객지원 전용 상담원입니다. "
            "해당 주제는 도와드릴 수 없지만, 제품·주문·서비스 관련 문의는 기꺼이 안내해 드리겠습니다.\n\n"
            f"운영 시간: {SUPPORT_BUSINESS_HOURS}."
        )
    if re.search(
        r"\b(hola|gracias|silla|precio|pedido|garant[ií]a|necesito|escribe|dame|quiero|puedes|por\s+favor)\b|[¿¡]",
        user_query or "",
        re.IGNORECASE,
    ):
        return (
            "Estoy especializado en soporte para sillas de masaje Osaki y Titan. "
            "No puedo ayudar con ese tema, pero con gusto le ayudo con nuestras sillas, pedidos o servicios.\n\n"
            f"Horario: {SUPPORT_BUSINESS_HOURS}."
        )
    return (
        "I'm specialized in Osaki and Titan massage chair support. "
        "I'm not able to help with that, but I'm happy to assist with anything "
        "about our chairs, orders, or services!\n\n"
        f"Our business hours are {SUPPORT_BUSINESS_HOURS}."
    )


def _message_text(msg: Any) -> str:
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return str(getattr(msg, "content", "") or "")


def _history_blob(chat_history: Sequence[Any] | None, *, max_messages: int = 8) -> str:
    if not chat_history:
        return ""
    parts = [_message_text(m) for m in chat_history[-max_messages:]]
    return "\n".join(p for p in parts if p.strip())


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _has_in_scope_keywords(text: str) -> bool:
    lower = _normalize(text)
    return any(kw in lower for kw in _IN_SCOPE_KEYWORDS)


def _recent_chair_context(chat_history: Sequence[Any] | None) -> bool:
    blob = _history_blob(chat_history).lower()
    return any(kw in blob for kw in _CHAIR_CONTEXT_KEYWORDS)


def _is_greeting_or_closing(text: str) -> bool:
    return bool(_GREETING_RE.match(text.strip()))


def _is_short_follow_up(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) <= 60 and bool(_SHORT_FOLLOW_UP_RE.match(stripped))


def _matches_out_of_scope(text: str) -> bool:
    return any(p.search(text) for p in _OUT_OF_SCOPE_PATTERNS)


def _rule_classify(user_query: str, chat_history: Sequence[Any] | None) -> Optional[bool]:
    """
    Fast scope decision.

    Returns True (in scope), False (out of scope), or None (needs LLM / allow).
    """
    q = (user_query or "").strip()
    if not q:
        return True

    if infer_forced_tool(q):
        return True

    if _is_greeting_or_closing(q):
        return True

    if _has_in_scope_keywords(q):
        return True

    if _recent_chair_context(chat_history) and (
        _is_short_follow_up(q) or len(q) <= 80
    ):
        return True

    if _matches_out_of_scope(q) and not _has_in_scope_keywords(q):
        return False

    return None


def _scope_classifier_enabled() -> bool:
    return os.environ.get("SCOPE_CLASSIFIER_ENABLED", "1") == "1"


def _scope_llm_enabled() -> bool:
    return os.environ.get("SCOPE_CLASSIFIER_LLM", "1") == "1"


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        from config import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT
    except ImportError:
        return None

    return OpenAI(
        api_key=api_key,
        timeout=float(OPENAI_REQUEST_TIMEOUT),
        max_retries=int(OPENAI_MAX_RETRIES),
    )


def _llm_classify(user_query: str, chat_history: Sequence[Any] | None) -> Optional[bool]:
    client = _openai_client()
    if client is None:
        return None

    from config import ROUTER_MODEL

    history_excerpt = _history_blob(chat_history, max_messages=6)
    prompt = (
        "Classify whether this customer message belongs in Osaki/Titan massage chair support chat.\n\n"
        "IN SCOPE: massage chairs, models/specs/pricing, orders, delivery/tracking, warranty, "
        "repair/troubleshooting, returns, showroom/company contact, sales/promotions, greetings, "
        "or short follow-ups when recent chat is about chairs/orders.\n\n"
        "OUT OF SCOPE: unrelated general knowledge, coding, recipes, weather, sports, politics, "
        "other products/brands, creative writing, homework, translation unrelated to chairs.\n\n"
        f"RECENT CHAT:\n{history_excerpt or '(none)'}\n\n"
        f"CUSTOMER MESSAGE:\n{user_query.strip()}\n\n"
        'Reply JSON only: {"in_scope": true|false, "confidence": "high"|"low"}. '
        "If uncertain, prefer in_scope=true with confidence=low."
    )

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict scope classifier. Never answer the customer — only classify."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        confidence = str(parsed.get("confidence", "low")).strip().lower()
        if confidence != "high":
            return None
        return bool(parsed.get("in_scope", True))
    except Exception as exc:
        logger.warning("scope_classifier LLM call failed: %s", exc)
        return None


def evaluate_scope(
    user_query: str,
    chat_history: Sequence[Any] | None = None,
    *,
    skip: bool = False,
) -> ScopeDecision:
    """
    Decide whether to allow the main agent to run.

    When blocked, the caller should return `build_scope_refusal()` without invoking the agent.
    """
    if skip or not _scope_classifier_enabled():
        return ScopeDecision(in_scope=True, reason="disabled")

    ruled = _rule_classify(user_query, chat_history)
    if ruled is True:
        return ScopeDecision(in_scope=True, reason="rule_in_scope")
    if ruled is False:
        return ScopeDecision(in_scope=False, reason="rule_out_of_scope")

    if _scope_llm_enabled():
        llm_result = _llm_classify(user_query, chat_history)
        if llm_result is False:
            return ScopeDecision(in_scope=False, reason="llm_out_of_scope", used_llm=True)
        if llm_result is True:
            return ScopeDecision(in_scope=True, reason="llm_in_scope", used_llm=True)

    return ScopeDecision(in_scope=True, reason="ambiguous_allow")
