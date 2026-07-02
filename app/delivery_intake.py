"""
delivery_intake.py
==================
Validate free-text answers in the warranty delivery workflow before advancing.

When the customer asks a product-spec question (box size, minimum doorway, etc.)
instead of order/tracking info, answer from the spec catalog and re-prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w.-]+\.\w+$")
_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{3,30}$")
_TRACKING_COMPACT_RE = re.compile(r"^[A-Za-z0-9\-]{8,40}$")

_QUESTION_WORDS_RE = re.compile(
    r"\b(what|how|give|tell|show|size|dimension|measure|length|width|height|"
    r"please|help|where|when|why|can you|could you|do you|is there|the box|"
    r"shipping box|carton|crate|package|doorway|door|weight|heavy|fit)\b",
    re.IGNORECASE,
)

_BOX_SIZE_Q_RE = re.compile(
    r"\b(box|carton|crate|package|shipping).{0,40}\b(size|dimension|measure|"
    r"length|width|height|big|large)\b|"
    r"\b(size|dimension).{0,40}\b(box|carton|crate|package)\b|"
    r"\bwhat size\b|\bsize of (the )?(box|carton|package)\b|"
    r"\bhow big\b|\bcarton dimensions\b|\bshipping dimensions\b",
    re.IGNORECASE,
)

_DOORWAY_Q_RE = re.compile(
    r"\b(minimum\s+)?(doorway|doorway clearance|door width|door size|"
    r"door opening|door frame)\b|"
    r"\bfit through (the )?door\b|\b(get|go) through (the )?door\b|"
    r"\bhow wide.{0,20}door\b|\bminimum entrance\b",
    re.IGNORECASE,
)

_WEIGHT_Q_RE = re.compile(
    r"\b(how heavy|how much does it weigh|product weight|chair weight|"
    r"shipping weight|weighs)\b|\bweight\b",
    re.IGNORECASE,
)

_CHAIR_DIMENSIONS_Q_RE = re.compile(
    r"\b(chair|unit|massage chair).{0,30}\b(dimension|dimensions|size|wide|"
    r"width|height|depth|long)\b|"
    r"\b(dimension|dimensions|size).{0,30}\b(chair|unit|massage chair)\b|"
    r"\bhow (wide|tall|deep|long)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeliverySpecQuestion:
    topic_id: str
    title: str
    search_query: str
    line_keywords: tuple[str, ...]


_SPEC_QUESTION_RULES: Sequence[tuple[re.Pattern[str], DeliverySpecQuestion]] = (
    (
        _BOX_SIZE_Q_RE,
        DeliverySpecQuestion(
            topic_id="box_size",
            title="shipping carton dimensions",
            search_query="shipping carton box package dimensions length width height",
            line_keywords=("carton", "box", "package", "shipping", "dimension"),
        ),
    ),
    (
        _DOORWAY_Q_RE,
        DeliverySpecQuestion(
            topic_id="doorway",
            title="minimum doorway clearance",
            search_query="minimum doorway door width entrance clearance",
            line_keywords=("doorway", "door", "entrance", "clearance"),
        ),
    ),
    (
        _WEIGHT_Q_RE,
        DeliverySpecQuestion(
            topic_id="weight",
            title="chair weight",
            search_query="chair weight product weight shipping weight",
            line_keywords=("weight", "lbs", "kg", "pound"),
        ),
    ),
    (
        _CHAIR_DIMENSIONS_Q_RE,
        DeliverySpecQuestion(
            topic_id="chair_dimensions",
            title="chair dimensions",
            search_query="dimensions width height depth standing",
            line_keywords=("dimension", "width", "height", "depth", "length", "standing"),
        ),
    ),
)


def is_plausible_email(text: str) -> bool:
    return bool(_EMAIL_RE.fullmatch((text or "").strip()))


def is_plausible_order_id(text: str) -> bool:
    clean = (text or "").replace("#", "").strip()
    if not clean or " " in clean:
        return False
    if _QUESTION_WORDS_RE.search(clean):
        return False
    return bool(_ORDER_ID_RE.fullmatch(clean))


def is_plausible_tracking_number(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or len(raw) < 8:
        return False
    if "?" in raw or "!" in raw:
        return False
    if _QUESTION_WORDS_RE.search(raw) and not _TRACKING_COMPACT_RE.fullmatch(
        re.sub(r"\s+", "", raw)
    ):
        return False

    words = raw.split()
    if len(words) > 3:
        return False
    if len(words) > 1 and not any(re.search(r"\d{6,}", word) for word in words):
        return False

    compact = re.sub(r"\s+", "", raw)
    return bool(_TRACKING_COMPACT_RE.fullmatch(compact))


def detect_delivery_spec_question(text: str) -> Optional[DeliverySpecQuestion]:
    """Return the best-matching delivery-relevant spec question, if any."""
    for pattern, spec in _SPEC_QUESTION_RULES:
        if pattern.search(text or ""):
            return spec
    return None


def looks_like_box_size_question(text: str) -> bool:
    return bool(_BOX_SIZE_Q_RE.search(text or ""))


def _get_products_retriever():
    try:
        import main  # noqa: WPS433

        return getattr(main, "products_retriever", None)
    except Exception:
        return None


def fetch_delivery_spec_answer(
    model_name: str,
    spec: DeliverySpecQuestion,
) -> str:
    """Best-effort spec answer from the product catalog for one model."""
    if not (model_name or "").strip():
        return ""

    retriever = _get_products_retriever()
    if retriever is None:
        return ""

    try:
        from agent_tools import tool_search_chair_specs  # noqa: WPS433

        raw = tool_search_chair_specs(
            products_retriever=retriever,
            query=spec.search_query,
            model_name=model_name,
        )
    except Exception:
        return ""

    if not raw or raw.startswith("NO_RESULTS"):
        return ""

    hits = [
        line.strip()
        for line in raw.splitlines()
        if line.strip().startswith("- ")
        and any(kw in line.lower() for kw in spec.line_keywords)
    ]
    if hits:
        body = "\n".join(hits[:5])
        return f"For **{model_name}**, here is what we have on {spec.title}:\n{body}"

    snippet = raw.strip().replace("\n", " ")
    if len(snippet) > 420:
        snippet = snippet[:417].rstrip() + "..."
    return (
        f"For **{model_name}**, here is what we found on {spec.title}:\n"
        f"{snippet}"
    )


def fetch_box_size_hint(model_name: str) -> str:
    """Backward-compatible wrapper for box-size lookups."""
    spec = _SPEC_QUESTION_RULES[0][1]
    return fetch_delivery_spec_answer(model_name, spec)


def _build_spec_side_answer(
    *,
    model_name: str,
    spec: DeliverySpecQuestion,
    reprompt: str,
) -> str:
    parts: list[str] = []
    if model_name:
        answer = fetch_delivery_spec_answer(model_name, spec)
        if answer:
            parts.append(answer)
        else:
            parts.append(
                f"I don't have exact **{spec.title}** for **{model_name}** in our "
                "catalog right now. Our warranty team can confirm the numbers when "
                "they review your case."
            )
    else:
        parts.append(
            f"I can look up **{spec.title}** once we know your chair model. "
            "Please confirm your model at the start of this chat if you haven't yet."
        )
    parts.append(reprompt)
    return "\n\n".join(parts)


def _reprompt_order_or_email() -> str:
    return (
        "To look up your delivery, please enter your **order number** "
        "(for example `#12345` or `OSKUS11308`) or the **email address** "
        "used at checkout."
    )


def _reprompt_tracking_number() -> str:
    return (
        "Please enter your carrier **tracking number** "
        "(usually 8–40 letters and numbers, such as `1Z999AA10123456784`)."
    )


def validate_delivery_text_answer(
    node_id: str,
    answer: str,
    *,
    model_name: str = "",
) -> None:
    """
    Raise ValueError with a customer-facing message when input should not advance.
    """
    text = (answer or "").strip()
    if not text:
        raise ValueError("Please enter a response before continuing.")

    spec = detect_delivery_spec_question(text)

    if node_id == "delivery_get_name":
        if spec:
            raise ValueError(
                _build_spec_side_answer(
                    model_name=model_name,
                    spec=spec,
                    reprompt=_reprompt_order_or_email(),
                )
            )

        from warranty_email import extract_email  # noqa: WPS433

        embedded_email = extract_email(text)
        if embedded_email and is_plausible_email(embedded_email):
            return
        if is_plausible_email(text) or is_plausible_order_id(text):
            return

        raise ValueError(_reprompt_order_or_email())

    if node_id == "delivery_get_tracking_number":
        if spec:
            raise ValueError(
                _build_spec_side_answer(
                    model_name=model_name,
                    spec=spec,
                    reprompt=_reprompt_tracking_number(),
                )
            )

        if is_plausible_tracking_number(text):
            return

        raise ValueError(_reprompt_tracking_number())
