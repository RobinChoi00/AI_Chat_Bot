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
    r"\b(box|carton|crate|package|shipping).{0,40}\b(size|sizes|dimension|dimensions|measure|"
    r"length|width|height|big|large)\b|"
    r"\b(size|sizes|dimension|dimensions).{0,40}\b(box|carton|crate|package)\b|"
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
    r"\bhow (wide|tall|deep)\b|"
    r"\bhow long\b(?!.{0,40}\b(take|takes|ship|shipping|deliver|delivery|arrive|wait|transit|days?|weeks?)\b)",
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
        "(for example `#12345` or `OSKUS11308`) and the **email address** "
        "used at checkout."
    )


def _reprompt_tracking_number() -> str:
    return (
        "Please enter your carrier **tracking number** "
        "(usually 8–40 letters and numbers, such as `1Z999AA10123456784`)."
    )


IDENTITY_NODES = frozenset({"delivery_get_name", "delivery_status_get_order_email"})
TRACKING_QUESTION_NODES = frozenset({"delivery_tracking_q", "delivery_status_tracking_q"})

_NEVER_ARRIVED_RE = re.compile(
    r"\b(never|didn'?t|did not)\s+(arrived?|came|come|showed? up|got (?:it|here|delivered)|been delivered)\b|"
    r"\b(not|never) (delivered|received)\b|\bstolen\b|\b(marked|says|shows|showing)\s+(as\s+)?delivered\b|"
    r"\b(lost|missing) (package|shipment|delivery|chair)\b",
    re.I,
)
_STILL_COMING_RE = re.compile(
    r"\b(hasn'?t|has not|haven'?t|have not)\s+(arrived|come|received it|gotten it)\b|\bnot\s+(arrived|here)\s+yet\b|"
    r"\bstill (on (its|the) way|coming)\b",
    re.I,
)
_WRONG_ITEM_RE = re.compile(
    r"\bwrong (item|chair|model|colou?r|product|one|unit)\b|\bnot what i ordered\b|\bdifferent (chair|model|colou?r)\b",
    re.I,
)
_DAMAGE_RE = re.compile(
    r"\b(damag\w*|broke|broken|crack\w*|scratch\w*|dent\w*|torn|tear|ripped|crushed|smashed|bent|punctur\w*|hole)\b",
    re.I,
)
_MISSING_RE = re.compile(
    r"\b(missing|incomplete|not included|didn'?t (come|arrive) with|no remote|only (one|two|1|2) box(es)?)\b",
    re.I,
)
_LATE_RE = re.compile(
    r"\b(late|delay\w*|still waiting|past (the )?(due )?date|overdue|should have (arrived|come|been here))\b",
    re.I,
)
_STATUS_RE = re.compile(
    r"\b(where|when|arriv\w*|eta|track\w*|status|shipped|on (its|the) way|update|estimate\w*)\b",
    re.I,
)
_NO_TRACKING_RE = re.compile(
    r"\b(don'?t|do not|dont)\s+(know|have|see)\b|\bnot sure\b|\bno idea\b|\b(can'?t|cannot|couldn'?t)\s+find\b|"
    r"\b(lost|deleted)\b|\b(haven'?t|have not|didn'?t|did not|never)\s+(got|gotten|received?|get)\b|"
    r"\bno tracking\b|\b(in|on|from) (my|the) (e-?mail|inbox|confirmation)\b|^\s*(no|nope|nah)\b",
    re.I,
)
_SIGNED_CLEARED_RE = re.compile(
    r"\b(didn'?t|did not|forgot to|never)\s+(note|write|mark|mention)\b|"
    r"\bsigned\b.{0,25}\b(clear\w*|fine|ok(ay)?|good|normal|without)\b|\bno damage (was )?(noted|written)\b",
    re.I,
)
_SIGNED_DAMAGED_RE = re.compile(
    r"\b(noted|wrote|written|marked|signed|refused)\b.{0,30}\bdamag\w*|\bdamag\w*\b.{0,30}\b(noted|on the (receipt|paper|slip))\b",
    re.I,
)
_SIGNED_UNKNOWN_RE = re.compile(
    r"\b(didn'?t|did not|never|nobody|no one|wasn'?t)\b.{0,20}\bsign\w*|\bno (signature|receipt)\b|"
    r"\bleft (it )?(at|by|on|outside)\b|\b(don'?t|do not|can'?t|cannot)\s+(remember|recall)\b|\bnot sure\b|"
    r"\bno idea\b|\b(lost|threw away|don'?t have) the (receipt|paper\w*|slip)\b",
    re.I,
)
_UNBOX_NOW_RE = re.compile(
    r"\b(right away|immediately|as soon as|when (i|we) (opened|unboxed|unpacked)|at unboxing|same day|right when)\b",
    re.I,
)
_UNBOX_LATER_RE = re.compile(
    r"\b(later|afterwards?|next day|days? later|weeks? later|after (a|some|few|using|a few)|when (i|we) (sat|used|assembled|set))\b",
    re.I,
)
_FINE_RE = re.compile(r"\b(fine|ok(ay)?|good|perfect|intact|normal|no damage|not damaged|looked new)\b", re.I)

_SALES_TOPICS: Sequence[tuple[str, re.Pattern[str], str]] = (
    ("reschedule", re.compile(r"\b(re-?schedul\w*|change (the )?(delivery )?(date|day|time)|deliver(ed)? (on|another|a different)|hold (the )?delivery)\b", re.I),
     "I can't reschedule a delivery from this chat."),
    ("address", re.compile(r"\b(change|update|new)\b.{0,25}\baddress\b|\baddress\b.{0,25}\b(change|update)\b", re.I),
     "I can't change a delivery address from this chat."),
    ("service", re.compile(r"\bwhite[- ]?glove\b|\b(in-?home|room of choice|threshold) delivery\b|\b(set ?up|assembly|install\w*) service\b", re.I),
     "Delivery service options depend on the order, and this chat can't add or change them."),
    ("cost", re.compile(r"\b(delivery|shipping|freight)\s+(fee|cost|charge|price)s?\b|\bhow much\b.{0,25}\b(ship|deliver)\w*\b|\bfree (shipping|delivery)\b", re.I),
     "I can't quote delivery costs from this chat."),
    ("timing", re.compile(r"\bhow long\b.{0,40}\b(take|takes|ship|shipping|deliver\w*|arrive|transit)\b|\bshipping time\b|\btransit time\b", re.I),
     "I can't promise a delivery time here. Once the order ships, the carrier estimate appears in the tracking status."),
    ("carrier", re.compile(r"\b(which|what|who)\b.{0,20}\b(carrier|courier|shipping company|delivery company)\b", re.I),
     "The carrier shows up in the tracking status once I can look up the order."),
)


def delivery_sales_topic(text: str) -> Optional[str]:
    """The customer-facing lead for an order question that Sales owns, if any."""
    for _key, pattern, lead in _SALES_TOPICS:
        if pattern.search(text or ""):
            return lead
    return None


def tracking_number_explainer(text: str) -> Optional[str]:
    if re.search(r"\bwhat\b.{0,20}\btracking (number|#|code|id)\b|\bwhere\b.{0,25}\b(find|get|see)\b.{0,20}\btracking\b", text or "", re.I):
        return (
            "A tracking number is the carrier's shipment number. It is in the shipping confirmation email "
            "that arrives once the order ships. If you don't have it, choose **No, I don't have a tracking number** "
            "and I'll use your order number and checkout email instead."
        )
    return None


def map_delivery_free_text(node_id: str, text: str) -> Optional[str]:
    """Map a free-text reply on a delivery menu to one answer_key, or None."""
    raw = (text or "").strip()
    if not raw or delivery_sales_topic(raw) or tracking_number_explainer(raw):
        return None
    if node_id == "delivery_intent_q":
        already_came = re.search(r"\b(arrived|came|delivered|showed up|got (it|here))\b", raw, re.I)
        if _STILL_COMING_RE.search(raw) or (_LATE_RE.search(raw) and not already_came and not _DAMAGE_RE.search(raw)):
            return "status_check"
        if any(p.search(raw) for p in (_NEVER_ARRIVED_RE, _WRONG_ITEM_RE, _DAMAGE_RE, _MISSING_RE, _LATE_RE)):
            return "damage_issue"
        if _STATUS_RE.search(raw):
            return "status_check"
        return None
    if node_id in TRACKING_QUESTION_NODES:
        compact = re.sub(r"\s+", "", raw)
        if is_plausible_tracking_number(raw) and re.search(r"\d", compact) and not is_plausible_email(raw):
            return "has_tracking"
        from warranty_email import extract_email  # noqa: WPS433

        if extract_email(raw) or _NO_TRACKING_RE.search(raw):
            return "no_tracking"
        return None
    if node_id == "delivery_problem_type_q":
        for pattern, key in (
            (_NEVER_ARRIVED_RE, "never_arrived"),
            (_STILL_COMING_RE, "never_arrived"),
            (_WRONG_ITEM_RE, "wrong_item"),
            (_DAMAGE_RE, "damaged_in_transit"),
            (_MISSING_RE, "missing_parts"),
            (_LATE_RE, "late_delivery"),
        ):
            if pattern.search(raw):
                return key
        return None
    if node_id == "delivery_visible_damage_q":
        if re.search(r"\bbox\b", raw, re.I) and _DAMAGE_RE.search(raw) and not re.search(r"\b(no|not)\b.{0,10}damag", raw, re.I):
            return "yes_box_damage"
        if _FINE_RE.search(raw):
            return "no_box_damage"
        return None
    if node_id == "delivery_chair_inside_damage_q":
        if re.search(r"\bchair\b", raw, re.I) and _DAMAGE_RE.search(raw) and not re.search(r"\b(no|not)\b.{0,10}damag", raw, re.I):
            return "yes_chair_inside_damage"
        if _FINE_RE.search(raw) or _MISSING_RE.search(raw):
            return "no_chair_inside_damage"
        return None
    if node_id == "delivery_signed_q":
        if _SIGNED_CLEARED_RE.search(raw):
            return "signed_cleared"
        if _SIGNED_DAMAGED_RE.search(raw):
            return "signed_damaged"
        if _SIGNED_UNKNOWN_RE.search(raw):
            return "signed_unknown"
        return None
    if node_id == "delivery_damage_when_opened_q":
        if _UNBOX_LATER_RE.search(raw):
            return "noticed_later"
        if _UNBOX_NOW_RE.search(raw):
            return "visible_at_unboxing"
        return None
    return None


_DELIVERY_CONTEXT_RE = re.compile(r"\b(deliver\w*|arriv\w*|came|shipped|shipment|package|box(es)?|order|tracking|carrier|driver)\b", re.I)
_BRANDED_ORDER_TOKEN_RE = re.compile(r"^(OSTI|OSKUS|OSKMC|OSK)", re.I)


def looks_like_delivery_report(text: str) -> bool:
    """A first message about a delivery that the issue classifier might miss."""
    raw = text or ""
    order, email = delivery_identity_parts(raw)
    if order and email and not _DAMAGE_RE.search(raw):
        return True
    if not _DELIVERY_CONTEXT_RE.search(raw):
        return False
    return any(p.search(raw) for p in (_NEVER_ARRIVED_RE, _STILL_COMING_RE, _MISSING_RE, _WRONG_ITEM_RE))


def tracking_number_in(text: str) -> Optional[str]:
    for token in re.findall(r"[A-Za-z0-9-]{10,40}", text or ""):
        digits = re.sub(r"\D", "", token)
        mixed = bool(re.search(r"[A-Za-z]", token)) and len(digits) >= 6
        if (mixed or (token.isdigit() and len(token) >= 12)) and not _BRANDED_ORDER_TOKEN_RE.match(token) \
                and is_plausible_tracking_number(token):
            return token
    return None


def delivery_identity_parts(text: str) -> tuple[str, str]:
    """(order_number, email) found in one message; either may be empty."""
    from delivery_lookup import parse_order_or_email  # noqa: WPS433

    raw = text or ""
    order, email = parse_order_or_email(raw)
    if order and not re.search(r"\d", order):
        order = ""
    return order, email


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

    if node_id in ("delivery_get_name", "delivery_status_get_order_email"):
        from warranty_email import extract_email  # noqa: WPS433

        embedded_email = extract_email(text)
        if embedded_email and is_plausible_email(embedded_email):
            return
        if is_plausible_email(text) or is_plausible_order_id(text):
            return

        raise ValueError(_reprompt_order_or_email())

    if node_id in ("delivery_get_tracking_number", "delivery_status_get_tracking"):
        if is_plausible_tracking_number(text):
            return

        raise ValueError(_reprompt_tracking_number())
