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
    return False


def build_chat_welcome_message() -> str:
    from config import CHAT_WELCOME_MESSAGE

    return CHAT_WELCOME_MESSAGE
