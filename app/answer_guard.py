"""
Post-processing guards for the main chat agent.

Goal: the LLM may phrase answers, but must not invent facts that did not come
from a tool result in the same turn. Keeps price, repair, and tracking claims
grounded in retrieved data.
"""

from __future__ import annotations

import re
import json
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

_WARRANTY_TOOLS = frozenset({
    "start_warranty_workflow",
    "answer_warranty_question",
    "warranty_start",
    "warranty_answer",
})
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


def conversation_language(text: str, history: Sequence | None = None) -> str:
    """Keep the last explicit user language across numbers, model names and yes/no.

    Assistant text is deliberately ignored: a prior accidental English footer
    must not switch a customer's conversation away from their chosen language.
    """
    current = (text or "").strip()
    explicit = re.search(r"(?:in|speak|reply|answer)\s+(English|Korean|Spanish)\b", current, re.I)
    if explicit:
        return {"english": "en", "korean": "ko", "spanish": "es"}[explicit.group(1).lower()]
    if re.search(r"[가-힣]", current):
        return "ko"
    if _SPANISH_HINTS.search(current) or current.lower() in {"sí", "si", "no sé", "de acuerdo"}:
        return "es"
    # A substantive English sentence may deliberately switch language; short
    # acknowledgements and product codes carry no reliable language evidence.
    if len(re.findall(r"\b[a-zA-Z]{2,}\b", current)) >= 4:
        return "en"
    for message in reversed(history or []):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if role != "user":
            continue
        explicit = re.search(r"(?:in|speak|reply|answer)\s+(English|Korean|Spanish)\b", str(content), re.I)
        if explicit:
            return {"english": "en", "korean": "ko", "spanish": "es"}[explicit.group(1).lower()]
        if re.search(r"[가-힣]", str(content)):
            return "ko"
        if _SPANISH_HINTS.search(str(content)) or str(content).strip().lower() in {"sí", "si"}:
            return "es"
        if len(re.findall(r"\b[a-zA-Z]{2,}\b", str(content))) >= 4:
            return "en"
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
    "recommendation_unverified": {
        "en": "I could not verify a chair matching all your requirements in the current catalog. I will keep those requirements rather than guess a fit. Would you like help from a specialist?",
        "es": "No pude verificar una silla que cumpla todos sus requisitos en el catálogo actual. Mantendré esos requisitos sin adivinar si encaja. ¿Desea ayuda de un especialista?",
        "ko": "현재 카탈로그에서 모든 조건을 충족하는 의자를 확인하지 못했습니다. 조건을 유지하며 적합성을 추측하지 않겠습니다. 전문 상담원의 도움을 받으시겠어요?",
    },
    "unsupported_promise": {
        "en": "Our team needs to confirm that request. I cannot approve warranty coverage, guarantee a delivery date, or promise a medical result here. I can help connect you with the appropriate team.",
        "es": "Nuestro equipo debe confirmar esa solicitud. No puedo aprobar la garantía, garantizar una fecha de entrega ni prometer un resultado médico. Puedo ayudarle a contactar al equipo adecuado.",
        "ko": "해당 요청은 담당 팀의 확인이 필요합니다. 이 상담에서 보증 승인, 배송일 보장 또는 의학적 효과를 약속할 수 없습니다. 적절한 담당 팀에 연결하는 방법을 안내드리겠습니다.",
    },
}


def _message(key: str, user_query: str, language: str | None = None) -> str:
    return _MESSAGES[key][language if language in {"en", "es", "ko"} else _language(user_query)]


def _tool_blob(tool_results: Sequence[str]) -> str:
    return "\n".join(tool_results or [])


def _warranty_locked_message(blob: str) -> str:
    """Return the engine's customer copy from a warranty tool result, if any."""
    if "CUSTOMER_MESSAGE:" in blob:
        after = blob.split("CUSTOMER_MESSAGE:", 1)[1]
        if "INSTRUCTION:" in after:
            after = after.split("INSTRUCTION:", 1)[0]
        text = after.strip()
        if text and not text.lower().startswith("(none"):
            return text
    for line in blob.splitlines():
        if line.startswith("PROMPT:"):
            prompt = line.split("PROMPT:", 1)[1].strip()
            if prompt:
                return prompt
    return ""


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


def _number_is_allowed(value: float, allowed: set[float], tolerance: float = 0.0001) -> bool:
    for candidate in allowed:
        if candidate <= 0:
            continue
        if abs(value - candidate) <= max(0.01, candidate * tolerance):
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


def _model_facts(blob: str) -> list[dict]:
    facts = []
    for line in blob.splitlines():
        if not line.startswith("MODEL_FACTS_JSON: "):
            continue
        try:
            fact = json.loads(line.split(": ", 1)[1])
            if isinstance(fact, dict) and isinstance(fact.get("model"), str) and isinstance(fact.get("specs"), dict):
                facts.append(fact)
        except (ValueError, TypeError):
            continue
    return facts


def _fact_mentions(text: str, facts: list[dict]) -> list[tuple[int, int, dict]]:
    matches = []
    for fact in facts:
        model = fact["model"].strip()
        aliases = {model, re.sub(r"^(?:osaki|titan)\s+", "", model, flags=re.I)}
        for alias in aliases:
            if len(alias) < 4:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
            matches.extend((m.start(), m.end(), fact) for m in re.finditer(pattern, text, re.I))
    # Prefer the complete model name over its shorter alias, and never interpret
    # a shared prefix as a second model in the same character span.
    selected = []
    for item in sorted(matches, key=lambda value: (-(value[1] - value[0]), value[0])):
        if not any(item[0] < other[1] and other[0] < item[1] for other in selected):
            selected.append(item)
    return sorted(selected, key=lambda value: value[0])


def _bound_claim_conflicts(text: str, fact: dict) -> bool:
    if re.search(r"€\s*\d|\b(?:EUR|CAD|GBP)\s*\d|\d[\d,.]*\s*(?:EUR|CAD|GBP)\b", text, re.I) and fact.get("currency") == "USD":
        return True
    prices = _PRICE_RE.findall(text)
    expected_price = fact.get("price_usd")
    for price in prices:
        try:
            if expected_price is None or abs(float(price.replace("$", "").replace(",", "").strip()) - float(expected_price)) > .011:
                return True
        except (TypeError, ValueError):
            return True
    specs = fact["specs"]
    field_patterns = (
        ("max_user_weight_lb", r"(?:max(?:imum)?\s+(?:user\s+)?(?:weight|capacity)|supports?\s+(?:users?\s+)?(?:up\s+to)?)[^\d\n]{0,25}(\d+(?:\.\d+)?)\s*(?:lb|pound)"),
        ("doorway_assembled_in", r"(?:doorway|door\s+width|door\s+clearance)[^\d\n]{0,25}(\d+(?:\.\d+)?)\s*(?:inches|inch|in\b|\")"),
        ("wall_clearance_in", r"wall\s+clearance[^\d\n]{0,25}(\d+(?:\.\d+)?)\s*(?:inches|inch|in\b|\")"),
    )
    for field, pattern in field_patterns:
        for value in re.findall(pattern, text, re.I):
            actual = specs.get(field)
            if actual is None or not _number_is_allowed(float(value), {float(actual)}):
                return True
    mechanisms = re.findall(r"\b[2345]D\b", text, re.I)
    source_mechanism = str(specs.get("massage_mechanism") or "").upper()
    if mechanisms and source_mechanism and any(mechanism.upper() not in source_mechanism for mechanism in mechanisms):
        return True
    foot = str(specs.get("foot_roller") or "").strip().lower()
    if foot:
        claims_yes = bool(re.search(r"(?:has|have|includes?|with)\s+(?:\w+\s+){0,2}foot\s*rollers?", text, re.I))
        claims_no = bool(re.search(r"(?:no|without|does not have|doesn't have)\s+foot\s*rollers?", text, re.I))
        if (claims_yes and foot in {"no", "none", "false"}) or (claims_no and foot in {"yes", "true"}):
            return True
    return False


def _has_model_fact_conflict(response: str, blob: str) -> bool:
    facts = _model_facts(blob)
    if not facts:
        return False
    # A feature-by-model table carries its model identity in the header, not
    # in each value row. Check the column binding before ordinary prose.
    columns: dict[int, dict] = {}
    bound_table_rows: set[str] = set()
    for row in response.splitlines():
        if row.count("|") < 2:
            columns = {}
            continue
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        header = {index: mentions[0][2] for index, cell in enumerate(cells)
                  if (mentions := _fact_mentions(cell, facts)) and len(mentions) == 1}
        if len(header) >= 2:
            columns = header
            bound_table_rows.add(row.strip())
            continue
        if columns and cells:
            bound_table_rows.add(row.strip())
            for index, fact in columns.items():
                if index < len(cells) and _bound_claim_conflicts(f"{cells[0]}: {cells[index]}", fact):
                    return True
    current = facts[0] if len(facts) == 1 else None
    for line in re.split(r"\n|(?<=[.;])\s+|\bwhile\b", response):
        if line.strip() in bound_table_rows:
            continue
        mentions = _fact_mentions(line, facts)
        if not mentions:
            if current is not None and _bound_claim_conflicts(line, current):
                return True
            continue
        for index, (start, _, fact) in enumerate(mentions):
            end = mentions[index + 1][0] if index + 1 < len(mentions) else len(line)
            segment = line[0 if index == 0 else start:end]
            if _bound_claim_conflicts(segment, fact):
                return True
            current = fact
    return False


def sanitize_agent_response(
    response: str,
    *,
    tools_called: Iterable[str],
    user_query: str,
    tool_results: Sequence[str] | None = None,
    language: str | None = None,
) -> str:
    """
    Scrub or replace hallucinated price / repair / tracking content.

    Returns the (possibly modified) customer-facing string.
    """
    text = (response or "").strip()
    tool_names = list(tools_called or [])
    called = set(tool_names)
    blob = _tool_blob(tool_results or [])
    blob_lower = blob.lower()
    paired_results = list(zip(tool_names, tool_results or [])) if len(tool_names) == len(tool_results or []) else []

    if called.intersection(_WARRANTY_TOOLS):
        locked = _warranty_locked_message(_tool_blob([result for name, result in paired_results if name in _WARRANTY_TOOLS]))
        if locked:
            return locked

    # The tracking formatter already produces the complete customer answer.
    # Never let model prose add a carrier, number, date or a status from a
    # different lookup. The marker also covers deterministic not-found errors.
    if "lookup_order_status" in called:
        tracking_blob = _tool_blob([result for name, result in paired_results if name == "lookup_order_status"])
        for line in reversed(tracking_blob.splitlines()):
            if line.startswith("ORDER_LOOKUP_RESULT_JSON: "):
                try:
                    result = json.loads(line.split(": ", 1)[1])
                    customer_message = result.get("customer_message")
                    if isinstance(customer_message, str) and customer_message.strip():
                        return customer_message.strip()
                except (ValueError, TypeError, AttributeError):
                    pass
        if "missing_input:" in tracking_blob.lower():
            return _message("tracking_input", user_query, language)
        return _message("source_unavailable", user_query, language)

    if not text:
        return text

    lower = text.lower()

    if "tool_error:" in blob_lower:
        return _message("source_unavailable", user_query, language)

    if "recommend_chairs" in called and "no_results:" in blob_lower and not _model_facts(blob):
        return _message("recommendation_unverified", user_query, language)
    if "search_chair_specs" in called and "no_results:" in blob_lower and not _model_facts(blob):
        return _message("spec_fallback", user_query, language)

    unsupported_promise = re.search(
        r"\b(?:your|the)\s+(?:claim|replacement|warranty)\s+(?:is|has been)\s+approved\b|"
        r"\b(?:we|i)\s+(?:guarantee|have approved|approved your)\b|"
        r"\b(?:will|guaranteed to)\s+(?:cure|heal)\b|"
        r"\b(?:guaranteed delivery|will (?:arrive|be delivered))\s+tomorrow\b",
        text, re.I,
    )
    if unsupported_promise:
        return _message("unsupported_promise", user_query, language)

    # Historical service correspondence is not the current customer's status.
    # A tool name alone does not authorize closure, shipment or repair promises.
    if re.search(
        r"\bclos(?:e|ing|ed)\b.{0,35}\b(?:ticket|case)\b|"
        r"\b(?:your|the)\s+chair\s+(?:is|has been)\s+(?:fixed|repaired)\b|"
        r"\b(?:i|we)(?:'ll| will)\s+(?:go ahead and\s+)?(?:ship|send|replace|refund|dispatch)\b",
        text, re.I,
    ):
        return _message("unsupported_promise", user_query, language)
    status_claims = re.findall(
        r"\b(?:parts?|replacement|technician|your order)\b[^.\n]{0,70}?"
        r"\b(on (?:the|their) way|shipped|dispatched|scheduled|in transit|delivered)\b",
        text, re.I,
    )
    if status_claims and not (
        "lookup_order_status" in called and "VERIFIED_ORDER_STATUS:" in blob
        and all(claim.lower() in blob_lower for claim in status_claims)
    ):
        return _message("unsupported_promise", user_query, language)

    if _has_model_fact_conflict(text, blob):
        return _message("spec_fallback", user_query, language)

    text = _DISCOUNT_NARRATIVE_RE.sub(
        "Please check the current price on our website.", text
    )

    if not called.intersection(_TRACKING_BLOCK_TOOLS) and any(
        s in lower for s in _TRACKING_SIGNALS
    ):
        return _message("tracking_input", user_query, language)

    repair_actions = bool(re.search(
        r"\b(?:turn\s+(?:off|on)|unplug|reconnect|reset|inspect|remove\s+the\s+(?:back|panel)|check\s+the\s+fuse|retire|cambie|desconecte|enchufe)\b|패널.*제거|퓨즈.*교체", lower,
    ))
    numbered_steps = bool(_NUMBERED_STEPS_RE.search(text)) and (not called.intersection(_PRICE_BLOCK_TOOLS) or repair_actions)
    repair_like = numbered_steps or any(s in lower for s in _REPAIR_SIGNALS)
    verified_repair = "get_repair_help" in called and "VERIFIED_REPAIR_GUIDANCE:" in blob
    if repair_like and not verified_repair and "escalate_to_human" not in called:
        return _message("repair_unverified", user_query, language)

    response_prices = _prices_in_text(text)
    if response_prices and not called.intersection(_PRICE_BLOCK_TOOLS):
        text = _PRICE_RE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if len(text) < 40:
            return _message("price_model", user_query, language)

    if response_prices and called.intersection(_PRICE_BLOCK_TOOLS):
        allowed = _prices_in_text(blob)
        for fact in _model_facts(blob):
            if fact.get("currency") == "USD" and isinstance(fact.get("price_usd"), (int, float)):
                price = float(fact["price_usd"])
                allowed.update({f"${price:,.2f}", f"${price:,.0f}"})
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
            return _message("spec_fallback", user_query, language)
        if "authoritative spec values" in blob_lower and _has_ungrounded_spec_numbers(text, blob):
            return _message("spec_fallback", user_query, language)

    return text.strip()
