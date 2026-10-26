"""Deterministic safety reports and customer copy for English, Korean, Spanish.

These rules identify reports that must stop troubleshooting; they are not a
medical diagnosis or an estimate of a real incident's severity. Negation and
reference filtering are scoped so another reported hazard remains visible.
"""

from __future__ import annotations

import re

_LATIN_ACCENTS = str.maketrans("áéíóúüñ", "aeiouun")
_EN_PHYSICAL = (
    r"(?:too hot|overheat\w*|smok(?:e|es|ed|ing|y)|"
    r"(?:burn(?:ing|t)|scorch(?:ing|ed))\s+(?:plastic\s+)?(?:smell|odou?r)|"
    r"smells?\s+(?:like\s+)?(?:burning|burnt|smoke|melting plastic)|sparks?|arcing|"
    r"electric(?:al)?\s+shocks?|electrocut\w*|"
    r"(?:got|getting|gave|gives|felt|feel)\s+(?:me\s+)?(?:(?:an?|mild|small|electric|electrical)\s+)*shocks?|"
    r"shock(?:ed|ing)\s+(?:me|my|our|his|her)|zapped\s+me|got zapped|short[- ]circuit\w*|has a short|"
    r"(?:caught|catching|on) fire|fire|flames?|burning|burn(?:ed|t)\s+(?:my|our|me|his|her|skin))"
)
_EN = rf"(?:{_EN_PHYSICAL}|pain(?:ful)?|hurts?|discomfort)"
_ES_PHYSICAL = (
    r"(?:humo|hume(?:a|an|ando|o)|incendio|fuego|ard(?:e|en|iendo|io)|llamas|(?:una|la) llama|"
    r"olor(?:es)?\s+(?:a|de)\s+(?:quemad[oa]|plastico quemado)|huele a quemado|"
    r"chispas?|chispe(?:a|an|ando)|descargas? electricas?|"
    r"(?:dio|da|dando)\s+(?:una\s+)?(?:descarga|corriente|calambre)|"
    r"electrocut\w*|cortocircuito|sobrecalent\w*|demasiado caliente|muy caliente)"
)
_ES = rf"(?:{_ES_PHYSICAL}|dolor|duele\w*|molestias?)"
_KO_ROOT = (
    r"(?:연기|불꽃|화재|스파크|감전|전기\s*충격|누전|합선|과열|통증|찌릿|"
    r"타는\s*냄새|탄\s*냄새|불|(?:너무\s*|지나치게\s*)?뜨겁|"
    r"(?:너무\s*|지나치게\s*)?뜨거|아프|아파)"
)
_KO_PHYSICAL = (
    r"(?:연기|불꽃|화재|스파크|감전|전기\s*충격|누전|합선|과열|찌릿|"
    r"타는\s*냄새|탄\s*냄새|불(?:이|은|도)?\s*(?:나|났|난|납|붙|타)|"
    r"(?:너무|지나치게)\s*(?:뜨겁|뜨거)|뜨거워(?:서|졌|요))"
)
_KO = rf"(?:{_KO_PHYSICAL}|통증|아프|아파)"
_HAZARD = re.compile(rf"\b(?:{_EN}|{_ES})\b|{_KO}", re.I)
_PHYSICAL_HAZARD = re.compile(rf"\b(?:{_EN_PHYSICAL}|{_ES_PHYSICAL})\b|{_KO_PHYSICAL}", re.I)
_CHAIR_CAUSED_INJURY = re.compile(
    r"\b(?:chair|seat|roller|footrest|armrest|frame)\s+(?:has\s+)?(?:hurt|hurts|burned)\s+(?:me|my|our|his|her)\b|"
    r"\bsilla\s+(?:me|nos)\s+(?:ha\s+)?(?:causado|causo|provocado|provoco)\s+dolor\b|"
    r"(?:의자|안마기)\s*때문에[^.!?\n]{0,20}(?:통증|아프|아파)", re.I,
)
_EN_NEGATIVE = re.compile(
    rf"\b(?:no|not|without|never|isn't|aren't|wasn't|weren't|hasn't|haven't|doesn't|don't|didn't)\s+"
    rf"(?:(?:any|signs? of|evidence of|experiencing|feel(?:ing)?|caus(?:e|es|ed|ing)|have|had|"
    rf"been|in|a|an|physical|smell(?:ing)?|see|seeing|saw|get|getting|got|"
    rf"emit(?:ting)?|produc(?:e|es|ed|ing)|show(?:ing)?)\s+)*{_EN}"
    rf"(?:\s*,?\s*(?:or|and)\s+(?:any\s+)?{_EN}|"
    rf"\s*,\s*(?:any\s+)?{_EN}(?=\s*(?:,|(?:or|and)\b|[.!?;]|$)))*\b|"
    rf"\b{_EN}\s+(?:is|are|was|were)\s+(?:not\s+(?:present|visible|seen|detected)|absent)\b|"
    rf"\b{_EN}\s+(?:isn't|aren't|wasn't|weren't)\s+(?:present|visible|seen|detected)\b", re.I,
)
_ES_NEGATIVE = re.compile(
    rf"\b(?:no|sin|nunca)\s+(?:(?:hay|veo|vemos|vi|ven|ve|observan|tiene|tengo|siento|sufro|"
    rf"presento|noto|percibo|me|ha|he|han|esta|en|se|produce|sale|salen|saliendo|dado|dio|da|una|"
    rf"echando|echa|echado|emitiendo|emite|emitido|expulsando|soltando|desprendiendo|produciendo|"
    rf"nada de|ni|signos de|ningun(?:a)?)\s+)*{_ES}"
    rf"(?:\s*,?\s*(?:ni|o|y)\s+{_ES})*\b|"
    rf"\b{_ES}\s+no\s+(?:hay|existe|existen|esta|estan|se ve|se ven)\b", re.I,
)
_KO_NEGATIVE = re.compile(
    rf"{_KO_ROOT}(?:\s*(?:와|과|나|및|또는)\s*{_KO_ROOT})*(?:은|는|이|가|도|을|를)?\s*(?:전혀\s*)?"
    r"(?:(?:난|붙은|발생한|된|당한|겪은)\s*(?:적(?:이|은|도)?|일(?:이|은|도)?)\s*)?"
    r"(?:없(?:습니다|어요|었어요|었고|었지만|음|다|고|지만|는데|는)?|아니(?:에요|었어요|고|지만|라|다)?|"
    r"(?:나지|나오지|피어오르지|발생하지|보이지|느껴지지|(?:되|하)?지)\s*않(?:습니다|아요|았어요|고|지만|는데|음|다)?|"
    r"안\s*(?:나요|납니다|났어요|났습니다|붙었어요|붙었습니다|나|보여요|보입니다|보여|느껴져요|뜨거워요|뜨겁습니다|아파요|아픕니다))"
)
_KO_PREFIX_NEGATIVE = re.compile(r"안\s*(?:뜨거워\w*|뜨겁\w*|아파\w*|아프\w*)")
_KO_DENIED_MODIFIER = re.compile(rf"{_KO_ROOT}(?:은|는|이|가|도)?\s*"
    r"(?:되는|된|나는|나오는|난|발생한|발생하는|있는|심한|운|워진)\s*"
    r"(?:것|건|상태|증상)(?:은|는|이|가)?\s*아니(?:에요|고|지만|다|라)?")
_KO_SCHEDULE = r"(?:배송|배달|일정|예약|주문|도착|날짜|시간|설치|방문|수령|약속)"
_KO_POSTPONE_VERB = re.compile(
    rf"(?P<context>{_KO_SCHEDULE}(?:(?!연기|의자|안마기|제품|기기)[^.!?\n]){{0,60}})"
    r"연기(?=하|해|했|합|할|되|돼|됐|된|될)"
)
_KO_POSTPONE_REQUEST = re.compile(r"연기(?=하자|해\s*(?:주|줘)|할까요|할게요|하겠|합시다)")
_KO_POSTPONE_NOUN = re.compile(
    rf"(?P<context>{_KO_SCHEDULE}(?:\s*(?:날짜|일정|시간))?(?:을|를|이|가|은|는|의)?\s*)"
    r"연기(?=(?:은|는|도)?\s*(?:요청|부탁|가능|불가|변경|문의|사유|관련|방법|안내)|\s*(?:[.!?]|$))"
)
_SPANISH_CALL = re.compile(
    r"\b(?:(?:me|te|se|nos|os|le|les)\s+llamas?|llama\s+(?:a|al)\b|llamame|llamanos)\b", re.I,
)
_SPANISH_NAME = re.compile(r"\bse llama\s+(?:\"[^\"\n]+\"|'[^'\n]+'|[a-z0-9_-]+\b)", re.I)
_REFERENCE_START = re.compile(r"^(?:(?:the|a|in the|el|la|en el|en la)\s+)?"
    r"(?:manual|guide|instructions|example|sample|translation|definition|warning label|ejemplo|instrucciones|guia|traduccion)\b|"
    r"^(?:설명서|매뉴얼|예시|예문|문구|번역|안내문)", re.I)
_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\'[^\'\n]+\'(?!\w)|‘[^’\n]*’')
_META_QUOTE_SUFFIX = re.compile(r"^\s*(?:(?:is|was|es|means)\s+)?(?:(?:an?|the|un|una)\s+)?"
    r"(?:example|sample|translation|definition|warning|ejemplo|traduccion)\b|"
    r"^\s*(?:는|은|이|가)?\s*(?:설명서\s*)?(?:예문|예시|문구|번역)", re.I)
_DEFINITION_PREFIX = re.compile(r"(?:what does|meaning of|translate|que significa|como se traduce)\s*$", re.I)
_REFERENCE_REPORT_TAIL = re.compile(
    r"\b(?:and|but|y|pero)\s+(?P<latin>(?:(?:now|ahora)\s+)?"
    r"(?:my|our|the|this|mi|nuestra|la|esta)\s+(?:chair|seat|unit|silla)\b.*)|"
    r"(?:있고|고|지만)\s*(?P<korean>(?:(?:지금|실제로|제|우리)\s*)?"
    r"(?:의자|안마기|제품|기기)(?:에서|가|는|를|에).*)", re.I,
)
_CLAUSE = re.compile(r"[.;\n]+|\b(?:but|however|pero|aunque|sin embargo)\b", re.I)
_HYPOTHETICAL = re.compile(r"\b(?:what|how|should|could|can)\b[^.!?]*\bif\b|"
                          r"\b(?:que|como|debo|deberia)\b[^.!?]*\bsi\b|"
                          r"(?:나면|발생하면|감전되면)[^.!?]*(?:어떻게|해야|하나요|할까요|대처)", re.I)
_ACTUAL = re.compile(r"\b(?:right now|currently|yesterday|i see|i saw|i smell|i felt|i got|ahora|ayer|veo|siento|me dio)\b|"
                    r"지금|어제|현재|실제로", re.I)


def _normalized(text: str) -> str:
    return (text or "").lower().replace("’", "'").translate(_LATIN_ACCENTS)


def _reports_hazard(text: str, *, physical_only: bool = False) -> bool:
    value = _normalized(text)
    if physical_only:
        # Contact local-parts and document URLs are metadata, not incident
        # reports. Keep any real report outside those spans visible.
        value = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|https?://[^\s<>]+", "[contact]", value, flags=re.I)
    value = _KO_POSTPONE_VERB.sub(lambda match: match["context"], value)
    value = _KO_POSTPONE_REQUEST.sub("", value)
    value = _KO_POSTPONE_NOUN.sub(lambda match: match["context"], value)
    value = _SPANISH_NAME.sub("", value)
    value = _SPANISH_CALL.sub("", value)
    value = re.sub(r"\b(?:fire[- ](?:resistant|retardant|proof)|smoke[- ](?:gray|grey|free)|burning calories)\b", "", value)
    # A manual example may be quoted, but quotes around a real report ("my
    # chair is 'smoking'") do not erase it. Actual reports in another clause
    # are evaluated independently from references and hypothetical questions.
    for clause in _CLAUSE.split(value):
        def reference_quote(match):
            before, after = clause[:match.start()], clause[match.end():]
            explicit_reference = (_REFERENCE_START.search(before.strip()) or
                                  _META_QUOTE_SUFFIX.search(after) or _DEFINITION_PREFIX.search(before))
            return "" if explicit_reference else match[0]

        clause = _QUOTED.sub(reference_quote, clause)
        if _REFERENCE_START.search(clause.strip()):
            report_tail = _REFERENCE_REPORT_TAIL.search(clause)
            if report_tail:
                clause = report_tail["latin"] or report_tail["korean"]
            elif not _ACTUAL.search(clause):
                continue
        if _HYPOTHETICAL.search(clause) and not _ACTUAL.search(clause):
            continue
        for negative in (_EN_NEGATIVE, _ES_NEGATIVE, _KO_NEGATIVE, _KO_PREFIX_NEGATIVE, _KO_DENIED_MODIFIER):
            clause = negative.sub("", clause)
        # Keep the original negation/reference processing for both modes. A
        # denied physical marker in one clause must not combine with ordinary
        # shopping pain in another clause to manufacture a physical incident.
        physical_clause = re.sub(r"\bburning\s+(?:(?:back|neck|leg|foot|feet)\s+)?(?:pain|sensations?)\b", "[pain symptom]", clause) if physical_only else clause
        physical = _PHYSICAL_HAZARD.search(physical_clause) or (_CHAIR_CAUSED_INJURY.search(clause) and _HAZARD.search(clause))
        hazard = physical if physical_only else _HAZARD.search(clause)
        if hazard:
            return True
    return False


def reported_safety_hazard(ticket=None, turns=(), text: str = "", *, physical_only: bool = False) -> bool:
    """Keep reported incidents; physical-only mode excludes general pain needs."""
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    values = [text, str(collected.get("safety_report") or ""), str(collected.get("human_support_safety_report") or ""),
              str(collected.get("intake_summary") or ""), str(collected.get("intake_raw_message") or "")]
    values.extend(str(getattr(turn, "customer_answer", "") or "") for turn in turns or ())
    return any(_reports_hazard(value, physical_only=physical_only) for value in values if value)


def detect_customer_language(text: str, default: str = "en") -> str:
    """Short hazard reports such as 'Sale humo' also carry Spanish evidence."""
    value = _normalized(text)
    explicit = re.search(r"(?:in|speak|reply|answer)\s+(english|korean|spanish)\b", value)
    if explicit:
        return {"english": "en", "korean": "ko", "spanish": "es"}[explicit[1]]
    if re.search(r"[가-힣]", value):
        return "ko"
    if re.search(r"\b(?:hola|gracias|silla|pedido|garantia|necesito|quiero|puedes|por favor|humo|humea|"
                 r"fuego|llamas|chispas|descarga|calambre|corriente|cortocircuito|dolor|duele|llamame|llamanos|"
                 r"olor|quemado|no se|de acuerdo)\b|[¿¡]", value):
        return "es"
    if len(re.findall(r"\b[a-z]{2,}\b", value)) >= 4:
        return "en"
    return default if default in {"en", "ko", "es"} else "en"


def _notification_copy(language: str, sent: bool, status: str | None) -> str:
    if sent:
        return {"en": "A notification has been submitted to our support inbox. A callback has not been booked. ",
                "ko": "지원팀 수신함으로 알림을 제출했습니다. 콜백은 아직 예약되지 않았습니다. ",
                "es": "Se ha enviado una notificación al buzón de soporte. Todavía no se ha reservado una llamada de seguimiento. "}[language]
    if language == "en":
        pending = "The support notification is queued; delivery has not been confirmed. " if status in {"pending", "processing"} else ""
        return "I haven't sent a notification or booked a callback. " + pending
    if status in {"pending", "processing"}:
        return {"ko": "지원팀 알림이 발송 대기 중입니다. 전달 완료는 아직 확인되지 않았으며, 콜백도 예약되지 않았습니다. ",
                "es": "La notificación a soporte está en cola; su entrega aún no está confirmada. Todavía no se ha reservado una llamada de seguimiento. "}[language]
    return {"ko": "지원팀 알림 전달은 아직 확인되지 않았습니다. 콜백은 예약되지 않았습니다. ",
            "es": "La entrega de la notificación a soporte aún no está confirmada. No se ha reservado una llamada de seguimiento. "}[language]


def safety_handoff_message(language: str, phone: str, notification_sent: bool = False,
                           *, notification_status: str | None = None) -> str:
    """Localize the safety stop without claiming staff response or callback."""
    language = language if language in {"en", "ko", "es"} else "en"
    notification = _notification_copy(language, notification_sent, notification_status)
    if language == "ko":
        return ("의자 사용을 즉시 중단하세요. 안전하게 할 수 있을 때만 전원을 끄고 플러그를 뽑으세요. "
                "다시 시험하거나 사용을 재개하지 말고, 패널을 열거나 직접 분해하지 마세요. "
                "불이 나고 있거나 즉각적인 위험이 있으면 안전한 곳으로 이동하고 현지 긴급 구조 기관에 연락하세요. "
                "신고하신 안전 문제와 상담 내용을 지원팀이 검토할 수 있도록 이 접수에 저장했습니다. "
                f"{notification}직접 지원을 받으려면 **{phone}**로 전화해 주세요.")
    if language == "es":
        return ("Deje de usar la silla. Apáguela y desenchúfela solo si puede hacerlo de forma segura. "
                "No haga otra prueba, no vuelva a usarla ni abra paneles. "
                "Si hay fuego activo o peligro inmediato, aléjese y contacte con los servicios de emergencia locales. "
                "He guardado el problema de seguridad y el contexto de esta conversación en su caso para que el equipo los revise. "
                f"{notification}Para recibir ayuda directa, llame al **{phone}**.")
    return ("Stop using the chair. Switch it off and unplug it only if you can do so safely. "
            "Do not run another test, resume use, or open panels. "
            "If there is an active fire or immediate danger, move away and contact local emergency services. "
            "I've saved the reported safety concern and conversation context with this case for team review. "
            f"{notification}Please call **{phone}** for direct support.")


def support_handoff_message(language: str, phone: str, reason: str = "customer_request",
                            notification_sent: bool = False, *, notification_status: str | None = None) -> str:
    if reason == "safety_hazard":
        return safety_handoff_message(language, phone, notification_sent, notification_status=notification_status)
    language = language if language in {"en", "ko", "es"} else "en"
    notification = _notification_copy(language, notification_sent, notification_status)
    if language == "ko":
        request = "콜백 요청" if reason == "callback_request" else "담당자 상담 요청"
        return (f"{request}과 상담 내용을 지원팀이 검토할 수 있도록 이 접수에 저장했습니다. "
                f"{notification}직접 도움을 받으려면 **{phone}**로 전화해 주세요. 문제 해결 단계를 다시 반복하실 필요는 없습니다.")
    if language == "es":
        request = "solicitud de llamada" if reason == "callback_request" else "solicitud de hablar con una persona"
        return (f"He guardado su {request} y el contexto de la conversación en su caso para que el equipo los revise. "
                f"{notification}Para recibir ayuda directa, llame al **{phone}**. No necesita repetir aquí las pruebas de solución de problemas.")
    request = "callback request" if reason == "callback_request" else "request for a person"
    return (f"I've saved your {request} with this case and its conversation context for team review. "
            f"{notification}For direct help, please call **{phone}**. You do not need to repeat the troubleshooting steps here.")
