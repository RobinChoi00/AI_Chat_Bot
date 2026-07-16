"""Opening greeting helpers for the main chat agent."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence


def is_conversation_start(chat_history: Sequence[Any] | None) -> bool:
    return not chat_history or len(chat_history) == 0


def is_opening_greeting(user_query: str) -> bool:
    """True when the first message is only a greeting or vague opener."""
    q = (user_query or "").strip()
    if not q:
        return False
    if re.match(
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|"
        r"thanks|thank\s+you|thx)(?:\s+(there|everyone|team))?[!.?\s]*$",
        q,
        re.IGNORECASE,
    ):
        return True
    if len(q) <= 40 and re.match(
        r"^(help|i\s+need\s+help|can\s+you\s+help|support|anyone\s+there)[!.?\s]*$",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.match(r"^(안녕(?:하세요)?|도와주세요|도움이\s*필요해요?)[!.?\s]*$", q):
        return True
    if re.match(
        r"^[¿¡]?(hola|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|ayuda|necesito\s+ayuda)[!.?¿¡\s]*$",
        q,
        re.IGNORECASE,
    ):
        return True
    return False


def build_chat_welcome_message(language: str = "en") -> str:
    from config import CHAT_WELCOME_MESSAGE

    if language == "es":
        return (
            "¡Hola! Bienvenido al soporte de Osaki y Titan. 👋\n\n"
            "¿Qué modelo de silla de masaje tiene? El modelo aparece en la etiqueta "
            "del número de serie (por ejemplo, OS-4000T, Solo Flex o Hypnos 4D).\n\n"
            "Comparta el modelo o pregunte sobre especificaciones, precios, pedidos, "
            "entrega, garantía o solución de problemas.\n\n"
            "---\n\n"
            "Esta conversación puede grabarse, almacenarse y revisarse para mejorar el servicio."
        )
    if language == "ko":
        return (
            "안녕하세요! Osaki & Titan 고객지원입니다. 👋\n\n"
            "사용 중인 마사지 의자 모델을 알려주시겠어요? 모델명은 의자의 "
            "시리얼 번호 스티커에서 확인할 수 있습니다(예: OS-4000T, Solo Flex, Hypnos 4D).\n\n"
            "모델명과 함께 제품 사양, 가격, 주문, 배송, 보증 또는 문제 해결 내용을 말씀해 주세요.\n\n"
            "---\n\n"
            "이 대화는 서비스 개선을 위해 기록·저장 및 검토될 수 있습니다."
        )
    return CHAT_WELCOME_MESSAGE
