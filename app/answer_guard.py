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
_UNIT_PATTERN = r'inches\b|inch\b|in\b|["″]|pounds?\b|lbs?\b|kilograms?\b|kg\b|centimeters?\b|cm\b|millimeters?\b|mm\b|feet\b|foot\b|ft\b|meters?\b|m\b'
_MEASUREMENT_RE = re.compile(
    r"(?<![a-z\d.])(?:(?P<low>\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*)?"
    r"(?P<number>\d+(?:\.\d+)?)\s*-?\s*(?P<unit>" + _UNIT_PATTERN + r")",
    re.I,
)
# Match the named field, not an arbitrary number elsewhere in the tool output.
_FIELD_RE = re.compile(
    r"(?:min(?:imum)?|max(?:imum)?)\s+user\s+height|(?:recommended\s+)?user\s+height(?:\s+range)?|height\s+range|"
    r"maximum\s+user\s+weight|max(?:imum)?\s+(?:user\s+)?(?:weight|capacity)|"
    r"user\s+weight|weight\s+(?:limit|capacity)|supports?\s+(?:users?\s+)?(?:up\s+to)?|capacity|"
    r"minimum\s+doorway|doorway|door\s+(?:width|clearance)|entrance\s+width|"
    r"wall\s+clearance|space[ -]saving|clearance\s+(?:from|to|behind)\s+(?:the\s+)?wall|"
    r"chair\s+weight|weighs?|weight|footrest\s+extension|track\s+(?:length|moving\s+range)|"
    r"(?:inside\s+of\s+)?shoulder\s+width|seat\s+width|width|wide|height|high|length|depth|deep|tall",
    re.I,
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


def _unit_info(unit: str) -> tuple[str, float]:
    unit = unit.lower()
    if unit in {'"', '″', 'in', 'inch', 'inches'}:
        return "length", 1.0
    if unit in {"ft", "feet", "foot"}:
        return "length", 12.0
    if unit in {"cm", "centimeter", "centimeters"}:
        return "length", 1 / 2.54
    if unit in {"mm", "millimeter", "millimeters"}:
        return "length", 1 / 25.4
    if unit in {"m", "meter", "meters"}:
        return "length", 100 / 2.54
    if unit in {"kg", "kilogram", "kilograms"}:
        return "weight", 1 / 0.45359237
    return "weight", 1.0


def _spec_field(label: str) -> str | None:
    """Normalize a field while retaining assembled/reclined distinctions."""
    text = re.sub(r"[^a-z0-9]+", " ", label.lower())
    if re.search(r"doorway|door (?:width|clearance)|entrance width", text):
        return "doorway_disassembled_in" if "disassembled" in text else "doorway_assembled_in"
    if re.search(r"user weight|(?:maximum|max) weight|weight (?:limit|capacity)|capacity|supports?", text):
        return "max_user_weight_lb"
    if re.search(r"wall|space saving", text):
        return "wall_clearance_in"
    if re.search(r"user height|height range", text):
        if re.search(r"\bmin(?:imum)?\b", text):
            return "minimum_user_height_in"
        if re.search(r"\bmax(?:imum)?\b", text):
            return "maximum_user_height_in"
        return "user_height_in"
    if re.search(r"weigh", text):
        return "chair_weight_lb"
    if "footrest" in text and "extension" in text:
        return "footrest_extension_in"
    if "track" in text:
        return "track_moving_range_in" if "moving" in text else "track_length_in"
    if "shoulder" in text and "width" in text:
        return "shoulder_width_in"
    if "seat" in text and "width" in text:
        return "seat_width_in"
    pose = "reclined" if "reclined" in text else "standing"
    if re.search(r"width|wide", text):
        return f"{pose}_width_in"
    if re.search(r"height|high|tall", text):
        return f"{pose}_height_in"
    if re.search(r"length|depth|deep", text):
        return f"{pose}_length_in"
    return None


def _measurements(text: str, default_unit: str | None = None) -> list[tuple[str, str, int, int]]:
    values = []
    for match in _MEASUREMENT_RE.finditer(text):
        if match.group("low"):
            values.append((match.group("low"), match.group("unit"), match.start(), match.end()))
        values.append((match.group("number"), match.group("unit"), match.start(), match.end()))
    if not values and default_unit:
        values = [(match.group(), default_unit, match.start(), match.end())
                  for match in re.finditer(r"(?<![a-z\d.])\d+(?:\.\d+)?", text, re.I)]
    return values


def _claim_field(text: str, start: int, end: int, query: str = "") -> str | None:
    fields = list(_FIELD_RE.finditer(text))
    before = [field for field in fields if field.end() <= start]
    after = [field for field in fields if field.start() >= end]
    # "32 inches wide" binds to width, rather than an earlier field in a sentence.
    field = (after[0] if after and after[0].start() - end <= 3
             and not text[end:after[0].start()].strip()
             else (before[-1] if before else None))
    if field is None:
        field = after[0] if after else None
    if field is None:
        return _spec_field(query)
    context = field.group()
    # A previous field in the same sentence must not change this field's
    # identity ("doorway 32 in and chair weight 285 lb"). Only borrow pose
    # qualifiers from this field's local clause.
    previous = max((item.end() for item in fields if item.end() < field.start()), default=0)
    context_start = max(previous, text.rfind(";", 0, field.start()) + 1,
                        text.rfind("\n", 0, field.start()) + 1)
    if re.search(r"\breclined\b", text[context_start:field.start()], re.I):
        context += " reclined"
    if re.search(r"\bdisassembled\b", text[context_start:field.start()], re.I):
        context += " disassembled"
    # Qualifiers may follow the measurement ("doorway: 28 in disassembled").
    next_field = min((item.start() for item in fields if item.start() > field.end()), default=len(text))
    qualifier = text[field.start():min(next_field, end + 35)]
    if re.search(r"\breclined\b", qualifier, re.I):
        context += " reclined"
    if re.search(r"\bdisassembled\b", qualifier, re.I):
        context += " disassembled"
    return _spec_field(context)


def _measurement_matches(number: str, unit: str, expected: tuple[float, str]) -> bool:
    value, expected_unit = expected
    kind, factor = _unit_info(unit)
    expected_kind, expected_factor = _unit_info(expected_unit)
    if kind != expected_kind:
        return False
    claim = float(number)
    if factor == expected_factor:
        # The same unit must preserve the source value; no percentage allowance.
        return abs(claim - value) < 1e-8
    converted = value * expected_factor / factor
    # Conversion may round to the displayed precision, but never more coarsely
    # than one decimal place. E.g. 32 in -> 81.3 cm, 285 lb -> 129.3 kg.
    decimals = len(number.partition(".")[2])
    tolerance = 0.5 * (10 ** -max(1, decimals)) + 1e-8
    return abs(claim - converted) <= tolerance


def _model_facts(blob: str) -> list[dict]:
    facts = []
    current = None
    in_authoritative = False
    for line in blob.splitlines():
        heading = re.match(r"\s*--- Result \d+: (.*?) ---\s*$", line)
        if heading:
            current = {"model": heading.group(1), "specs": {}, "_measurements": {}}
            facts.append(current)
            in_authoritative = False
            continue
        if line.startswith("MODEL_FACTS_JSON: "):
            try:
                fact = json.loads(line.split(": ", 1)[1])
                if isinstance(fact, dict) and isinstance(fact.get("model"), str) and isinstance(fact.get("specs"), dict):
                    if current is None or current["model"].lower() != fact["model"].lower():
                        current = fact
                        facts.append(current)
                    else:
                        current.update(fact)
            except (ValueError, TypeError):
                continue
        elif re.match(r"\s*BASE PRICE \(USD\):", line, re.I):
            if current is None:
                current = {"model": "", "specs": {}}
                facts.append(current)
            price = _PRICE_RE.search(line)
            if price:
                current.setdefault("price_usd", float(price.group().replace("$", "").replace(",", "")))
                current.setdefault("currency", "USD")
        elif "AUTHORITATIVE SPEC VALUES" in line:
            in_authoritative = True
            if current is None:
                current = {"model": "", "specs": {}}
                facts.append(current)
        elif line.strip().startswith("Additional context"):
            in_authoritative = False
        elif in_authoritative and ":" in line:
            label, value = line.split(":", 1)
            field = _spec_field(label)
            if field is None:
                continue
            label_unit = re.search(r"(?<![a-z])(?:" + _UNIT_PATTERN + r")", label, re.I)
            unit = label_unit.group() if label_unit else ("lb" if field.endswith("_lb") else "in")
            for number, value_unit, _, _ in _measurements(value, unit):
                current.setdefault("_measurements", {}).setdefault(field, []).append((float(number), value_unit))
    for fact in facts:
        measurements = fact.setdefault("_measurements", {})
        for field, value in fact["specs"].items():
            if isinstance(value, (float, int)) and not isinstance(value, bool) and field.endswith(("_lb", "_in")):
                # Structured model facts take precedence over supplemental text.
                measurements[field] = [(float(value), "lb" if field.endswith("_lb") else "in")]
        for field in ("minimum_user_height_in", "maximum_user_height_in"):
            if field in measurements:
                measurements.setdefault("user_height_in", []).extend(measurements[field])
    return [fact for fact in facts if fact["model"] or fact["specs"] or fact.get("_measurements")]


def _measurement_conflicts(text: str, fact: dict, query: str = "", default_unit: str | None = None) -> bool:
    measurements = fact.get("_measurements", {})
    matches = list(_MEASUREMENT_RE.finditer(text))
    ranges = []
    for match in matches:
        if match.group("low"):
            ranges.append((match.group("low"), match.group("unit"), match.group("number"),
                           match.group("unit"), match.start(), match.end()))
    for first, second in zip(matches, matches[1:]):
        if re.fullmatch(r"\s*(?:[-–—]|to)\s*", text[first.end():second.start()], re.I):
            ranges.append((first.group("number"), first.group("unit"), second.group("number"),
                           second.group("unit"), first.start(), second.end()))
    for low, low_unit, high, high_unit, start, end in ranges:
        low_kind, low_factor = _unit_info(low_unit)
        high_kind, high_factor = _unit_info(high_unit)
        if low_kind != high_kind or float(low) * low_factor > float(high) * high_factor:
            return True
        field = _claim_field(text, start, end, query)
        expected = measurements.get(field, [])
        if len(expected) > 1:
            ordered = sorted(expected, key=lambda item: item[0] * _unit_info(item[1])[1])
            if not (_measurement_matches(low, low_unit, ordered[0])
                    and _measurement_matches(high, high_unit, ordered[-1])):
                return True
    claims = _measurements(text, default_unit)
    # Tool-authored units may be implicit in prose ("maximum user weight is
    # 300"). Guard those values too, without interpreting model/price numbers.
    for label in _FIELD_RE.finditer(text):
        tail = re.match(r"[^\d\n$|]{0,30}(\d+(?:\.\d+)?)", text[label.end():])
        if tail:
            start, end = label.end() + tail.start(1), label.end() + tail.end(1)
            if any(existing_start <= start < existing_end for _, _, existing_start, existing_end in claims):
                continue
            field = _claim_field(text, start, end, query)
            if field:
                claims.append((tail.group(1), "lb" if field.endswith("_lb") else "in", start, end))
    for number, unit, start, end in claims:
        field = _claim_field(text, start, end, query)
        if field is None:
            # A bare measurement cannot borrow a number from an unrelated field.
            return True
        expected = measurements.get(field, [])
        if not any(_measurement_matches(number, unit, value) for value in expected):
            return True
    return False


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


def _bound_claim_conflicts(text: str, fact: dict, query: str = "", default_unit: str | None = None) -> bool:
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
    if _measurement_conflicts(text, fact, query, default_unit):
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


def _table_claim_conflicts(label: str, cell: str, fact: dict) -> bool:
    unit = re.search(r"(?<![a-z])(?:" + _UNIT_PATTERN + r")", label, re.I)
    # Units may live in a header. Bare counts/prices are not measurements.
    default_unit = unit.group() if unit and _spec_field(label) else None
    return _bound_claim_conflicts(f"{label}: {cell}", fact, label, default_unit)


def _has_model_fact_conflict(response: str, blob: str, query: str = "") -> bool:
    facts = _model_facts(blob)
    if not facts:
        return False
    # A feature-by-model table carries its model identity in the header, not
    # in each value row. Check the column binding before ordinary prose.
    columns: dict[int, dict] = {}
    field_headers: list[str] = []
    bound_table_rows: set[str] = set()
    for row in response.splitlines():
        if row.count("|") < 2:
            columns = {}
            field_headers = []
            continue
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        header = {index: mentions[0][2] for index, cell in enumerate(cells)
                  if (mentions := _fact_mentions(cell, facts)) and len(mentions) == 1}
        if header:
            # A model in column zero normally identifies a data row. A model
            # elsewhere identifies feature-by-model column headers.
            if not columns and 0 in header and field_headers:
                fact = header[0]
                bound_table_rows.add(row.strip())
                for index, label in enumerate(field_headers[1:], 1):
                    if index < len(cells) and _table_claim_conflicts(label, cells[index], fact):
                        return True
                continue
        if header and 0 not in header:
            columns = header
            field_headers = []
            bound_table_rows.add(row.strip())
            continue
        if columns and cells:
            bound_table_rows.add(row.strip())
            for index, fact in columns.items():
                if index < len(cells) and _table_claim_conflicts(cells[0], cells[index], fact):
                    return True
            continue
        if any(_spec_field(cell) or re.search(r"price|mechanism|foot roller", cell, re.I) for cell in cells[1:]):
            field_headers = cells
            bound_table_rows.add(row.strip())
    query_mentions = _fact_mentions(query, facts)
    current = facts[0] if len(facts) == 1 else (query_mentions[0][2] if len(query_mentions) == 1 else None)
    for line in re.split(r"\n|(?<=[.;])\s+|\bwhile\b", response):
        if line.strip() in bound_table_rows:
            continue
        mentions = _fact_mentions(line, facts)
        if not mentions:
            if current is not None and _bound_claim_conflicts(line, current, query):
                return True
            if current is None and (_measurements(line) or _PRICE_RE.search(line)):
                # Unnamed comparison values must agree for every possible model.
                if any(_bound_claim_conflicts(line, fact, query) for fact in facts):
                    return True
            continue
        for index, (start, _, fact) in enumerate(mentions):
            end = mentions[index + 1][0] if index + 1 < len(mentions) else len(line)
            segment = line[0 if index == 0 else start:end]
            if _bound_claim_conflicts(segment, fact, query):
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

    if _has_model_fact_conflict(text, blob, user_query):
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
    return text.strip()
