"""
Post-processing guards for the main chat agent.

Goal: the LLM may phrase answers, but must not invent facts that did not come
from a tool result in the same turn. Keeps price, repair, and tracking claims
grounded in retrieved data.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

_PRICE_RE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d{2})?"
)
_DISCOUNT_NARRATIVE_RE = re.compile(
    r"(?:originally|was|regular(?:ly)?|compare\s*at|list\s*price|msrp)[^.\n]{0,80}\$\s?\d[\d,]*",
    re.IGNORECASE,
)
_NUMBERED_STEPS_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+[\).:]|step\s+\d+)",
    re.IGNORECASE,
)
_SPEC_NUMBER_CTX_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:inch|inches|\"|\s*in\b|lb|lbs|pound|kg|cm|mm|ft|feet)",
    re.IGNORECASE,
)
_SPEC_NUMBER_REV_RE = re.compile(
    r"(?:doorway|width|height|depth|weight|clearance|maximum\s+user)[^\d]{0,24}(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_REPAIR_BLOCK_TOOLS = frozenset({"get_repair_help", "escalate_to_human"})
_PRICE_BLOCK_TOOLS = frozenset({"search_chair_specs", "recommend_chairs"})
_TRACKING_BLOCK_TOOLS = frozenset({"lookup_order_status"})

_TRACKING_SIGNALS = (
    "current status:", "tracking number:", "carrier:",
    "estimated delivery:", "in preparation", "in transit",
)
_REPAIR_SIGNALS = (
    "installation step", "install the", "assemble the", "assembly step",
    "troubleshooting step", "remove the back", "manual mode",
    "general steps", "follow these steps", "troubleshoot", "assembly",
)
_SPEC_TOPIC_SIGNALS = (
    "inch", "inches", "dimension", "doorway", "weight", "lb", "clearance",
)

_SPANISH_HINTS = re.compile(
    r"\b(hola|gracias|silla|precio|pedido|garant[ií]a|reparar|cu[aá]nto|necesito|escribe|dame|quiero|puedes|por\s+favor)\b|[¿¡]",
    re.IGNORECASE,
)


def _language(text: str) -> str:
    if re.search(r"[가-힣]", text or ""):
        return "ko"
    if _SPANISH_HINTS.search(text or ""):
        return "es"
    return "en"


_MESSAGES = {
    "tracking_input": {
        "en": "I can look up your order status if you share your order number (for example OSKMC1234) and the email used at checkout.",
        "es": "Puedo consultar el estado de su pedido si comparte el número de pedido (por ejemplo, OSKMC1234) y el correo usado en la compra.",
        "ko": "주문 번호(예: OSKMC1234)와 결제할 때 사용한 이메일을 알려주시면 배송 상태를 조회해 드릴 수 있습니다.",
    },
    "repair_unverified": {
        "en": "I don't have verified repair or installation steps for that in our knowledge base yet. Please contact our warranty support team at +1-888-848-2630 ext. 3 — Mon-Fri, 10:00 AM - 6:00 PM CST — and they can walk you through it safely.",
        "es": "Aún no tengo pasos verificados de reparación o instalación para ese caso. Llame al equipo de garantía al +1-888-848-2630 ext. 3 (lun-vie, 10:00 AM-6:00 PM CST) para recibir ayuda segura.",
        "ko": "해당 문제에 대해 확인된 수리·설치 절차가 아직 지식베이스에 없습니다. 안전한 안내를 위해 보증팀 +1-888-848-2630 내선 3번으로 연락해 주세요. 보증팀 운영 시간은 월-금 오전 10시-오후 6시(미 중부 시간)입니다.",
    },
    "price_model": {
        "en": "I can quote a price after I look up the exact model in our catalog. Which chair model are you asking about?",
        "es": "Puedo confirmar el precio después de buscar el modelo exacto en nuestro catálogo. ¿Qué modelo de silla le interesa?",
        "ko": "정확한 모델을 공식 카탈로그에서 확인한 뒤 가격을 안내드릴 수 있습니다. 어떤 의자 모델을 문의하시나요?",
    },
    "spec_fallback": {
        "en": "I want to give you accurate information, so I need to look that up in our official catalog first. Could you share the exact model name from your chair's serial-number sticker? You can also reach our support team at +1-888-848-2630 — business hours Mon-Fri, 9:30 AM - 6:30 PM / Sat, 10:00 AM - 4:00 PM CST.",
        "es": "Quiero darle información exacta y necesito confirmarla en el catálogo oficial. ¿Puede compartir el modelo exacto que aparece en la etiqueta del número de serie? También puede llamar al +1-888-848-2630 (lun-vie 9:30 AM-6:30 PM; sáb 10:00 AM-4:00 PM, hora central).",
        "ko": "정확한 안내를 위해 공식 카탈로그 확인이 필요합니다. 의자 시리얼 번호 스티커에 적힌 정확한 모델명을 알려주시겠어요? +1-888-848-2630으로도 문의할 수 있으며, 운영 시간은 월-금 오전 9:30-오후 6:30, 토 오전 10:00-오후 4:00(미 중부 시간)입니다.",
    },
    "source_unavailable": {
        "en": "I'm sorry, but the authoritative lookup is temporarily unavailable. I don't want to guess. Please try again shortly or contact support at +1-888-848-2630.",
        "es": "Lo siento, la fuente oficial no está disponible temporalmente y no quiero adivinar. Inténtelo de nuevo en unos minutos o llame al +1-888-848-2630.",
        "ko": "죄송하지만 현재 공식 정보 조회가 일시적으로 불가능해 추측해서 안내드릴 수 없습니다. 잠시 후 다시 시도하거나 +1-888-848-2630으로 연락해 주세요.",
    },
}


def _message(key: str, user_query: str) -> str:
    return _MESSAGES[key][_language(user_query)]


def _tool_blob(tool_results: Sequence[str]) -> str:
    return "\n".join(tool_results or [])


def _prices_in_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in _PRICE_RE.finditer(text or ""):
        token = m.group(0).replace(" ", "")
        out.add(token)
        num = token.lstrip("$").replace(",", "")
        try:
            val = float(num)
            out.add(f"${val:,.2f}")
            out.add(f"${int(val):,}" if val == int(val) else f"${val:,.2f}")
        except ValueError:
            pass
    return out


def _strip_ungrounded_prices(response: str, allowed_prices: set[str]) -> str:
    if not allowed_prices:
        return _PRICE_RE.sub("[check current price on our website]", response)

    def repl(match: re.Match[str]) -> str:
        token = match.group(0).replace(" ", "")
        if token in allowed_prices:
            return match.group(0)
        return "[check current price on our website]"

    return _PRICE_RE.sub(repl, response)


def _floats_in_line(line: str) -> set[float]:
    nums: set[float] = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)", line):
        try:
            val = float(m.group(1))
            if val > 0:
                nums.add(val)
        except ValueError:
            pass
    return nums


def _allowed_spec_numbers(blob: str) -> set[float]:
    """Collect numeric spec values present in tool output."""
    allowed: set[float] = set()
    in_authoritative = False

    for line in blob.splitlines():
        if "AUTHORITATIVE SPEC VALUES" in line:
            in_authoritative = True
            continue
        if in_authoritative:
            stripped = line.strip()
            if stripped.startswith("---") or stripped.startswith("Additional context"):
                in_authoritative = False
            elif stripped:
                allowed.update(_floats_in_line(line))

        lower = line.lower()
        if any(kw in lower for kw in _SPEC_TOPIC_SIGNALS):
            allowed.update(_floats_in_line(line))

    return allowed


def _claimed_spec_numbers(text: str) -> set[float]:
    nums: set[float] = set()
    for pattern in (_SPEC_NUMBER_CTX_RE, _SPEC_NUMBER_REV_RE):
        for m in pattern.finditer(text or ""):
            try:
                nums.add(float(m.group(1)))
            except ValueError:
                pass
    return nums


def _number_is_allowed(value: float, allowed: set[float], tolerance: float = 0.08) -> bool:
    for candidate in allowed:
        if candidate <= 0:
            continue
        if abs(value - candidate) <= max(0.5, candidate * tolerance):
            return True
    return False


def _has_ungrounded_spec_numbers(response: str, blob: str) -> bool:
    allowed = _allowed_spec_numbers(blob)
    if not allowed:
        return False

    claims = _claimed_spec_numbers(response)
    if not claims:
        return False

    return any(not _number_is_allowed(n, allowed) for n in claims)


def sanitize_agent_response(
    response: str,
    *,
    tools_called: Iterable[str],
    user_query: str,
    tool_results: Sequence[str] | None = None,
) -> str:
    """
    Scrub or replace hallucinated price / repair / tracking content.

    Returns the (possibly modified) customer-facing string.
    """
    text = (response or "").strip()
    if not text:
        return text

    called = set(tools_called or [])
    blob = _tool_blob(tool_results or [])
    blob_lower = blob.lower()
    lower = text.lower()

    if "tool_error:" in blob_lower:
        return _message("source_unavailable", user_query)

    text = _DISCOUNT_NARRATIVE_RE.sub(
        "Please check the current price on our website.", text
    )

    if not called.intersection(_TRACKING_BLOCK_TOOLS) and any(
        s in lower for s in _TRACKING_SIGNALS
    ):
        return _message("tracking_input", user_query)

    numbered_steps = bool(_NUMBERED_STEPS_RE.search(text))
    repair_like = numbered_steps or any(s in lower for s in _REPAIR_SIGNALS)
    if repair_like and not called.intersection(_REPAIR_BLOCK_TOOLS):
        return _message("repair_unverified", user_query)

    response_prices = _prices_in_text(text)
    if response_prices and not called.intersection(_PRICE_BLOCK_TOOLS):
        text = _PRICE_RE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if len(text) < 40:
            return _message("price_model", user_query)

    if response_prices and called.intersection(_PRICE_BLOCK_TOOLS):
        allowed = _prices_in_text(blob)
        for m in re.finditer(
            r"(?:price|variant price|base price)[^\n$]*\$?([\d,]+(?:\.\d{2})?)",
            blob_lower,
        ):
            raw = m.group(1).replace(",", "")
            allowed.add(f"${raw}")
            try:
                val = float(raw)
                allowed.add(f"${val:,.2f}")
                allowed.add(f"${int(val):,}")
            except ValueError:
                pass
        text = _strip_ungrounded_prices(text, allowed)

    if "search_chair_specs" in called:
        if "no_results" in blob_lower and any(kw in lower for kw in _SPEC_TOPIC_SIGNALS):
            return _message("spec_fallback", user_query)
        if "authoritative spec values" in blob_lower and _has_ungrounded_spec_numbers(text, blob):
            return _message("spec_fallback", user_query)

    return text.strip()
