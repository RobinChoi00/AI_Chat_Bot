"""Deterministic answers for common warranty questions.

These replies use published contacts and policy pages only. They never approve
a replacement, refund, repair quote, technician visit, or timeline.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    from config import (
        SALES_PHONE,
        WARRANTY_BUSINESS_HOURS,
        WARRANTY_PHONE,
        WARRANTY_TEAM_EMAIL,
    )
except ImportError:  # pragma: no cover
    SALES_PHONE = "+1-888-848-2630 ext. 2"
    WARRANTY_PHONE = "+1-888-848-2630 ext. 3"
    WARRANTY_BUSINESS_HOURS = "Mon-Fri, 10:00 AM - 6:00 PM CST"
    WARRANTY_TEAM_EMAIL = "service@osakititan.com"

from warranty_defaults import DEFAULT_WARRANTY_DOMAIN, normalize_warranty_domain

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_SPANISH_RE = re.compile(
    r"\b(?:qué|que|cómo|como|cuánto|cuanto|dónde|donde|por qué|porque|"
    r"hola|gracias|silla|garantía|garantia)\b",
    re.I,
)
_GIBBERISH_RE = re.compile(
    r"^(?:asdfg?h?j?k?l?|qwer+t?y?|zxcv+b?n?m?|[sdfghjkl]{5,})$",
    re.I,
)

_WARRANTY_LENGTH_RE = re.compile(
    r"\b(?:how long|how many years?|what(?:'s| is) the (?:length|term|period) of)\b"
    r".{0,40}\bwarrant|"
    r"\bwarrant\w*.{0,20}\b(?:how long|how many years?|length|term|period|coverage)\b|"
    r"\b(?:warranty (?:length|term|period|coverage)|years? of (?:the )?warranty)\b",
    re.I,
)
_COVERED_RE = re.compile(
    r"\b(?:is (?:this|it|that|my (?:chair|issue|problem)) covered|"
    r"(?:does|do) (?:the )?warrant\w* cover|"
    r"under warrant\w*|warrant\w* cover)\b",
    re.I,
)
_REPLACE_RE = re.compile(
    r"\b(?:will you replace|replace (?:my|the) (?:chair|unit)|"
    r"send me a new chair|new chair instead)\b",
    re.I,
)
_SEND_PART_RE = re.compile(
    r"\b(?:just send|send|ship|mail)\b.{0,30}\b(?:new\s+)?(?:part|remote|controller|fuse|airbag|cover|cord)\b|"
    r"\b(?:new|replacement)\s+(?:part|remote|controller)\b",
    re.I,
)
_REFUND_RE = re.compile(
    r"\b(?:refund|money back|chargeback)\b",
    re.I,
)
_RETURN_RE = re.compile(
    r"\b(?:return (?:it|the chair|my chair|this)|send it back|"
    r"can i return|want to return)\b",
    re.I,
)
_REPAIR_COST_RE = re.compile(
    r"\b(?:how much|what(?:'s| is) the (?:cost|price|fee)|"
    r"repair cost|cost of (?:a |the )?repair|pay for (?:a |the )?repair|"
    r"out of pocket)\b.{0,40}\b(?:repair|fix|service|technician|labor)\b|"
    r"\b(?:repair|service call|technician)\b.{0,25}\b(?:cost|price|fee|charge)\b",
    re.I,
)
_PART_SHIP_COST_RE = re.compile(
    r"\b(?:pay for shipping|shipping (?:for|on) (?:the )?parts?|"
    r"parts? shipping|freight for (?:the )?parts?)\b",
    re.I,
)
_TECH_WHEN_RE = re.compile(
    r"\b(?:when|how soon).{0,40}\b(?:technician|tech(?:nician)? visit|"
    r"service (?:tech|visit|appointment|call)|repair (?:person|tech|visit))|"
    r"\b(?:technician|tech)\b.{0,20}\b(?:come|arrive|visit|scheduled)\b",
    re.I,
)
_HOW_LONG_RE = re.compile(
    r"\bhow long (?:will|does|is) this (?:take|going to take)|"
    r"\bhow long (?:will|does) (?:the )?(?:repair|fix|service|case)\b|"
    r"\b(?:eta|timeline) (?:for|on) (?:the )?(?:repair|case|service)\b",
    re.I,
)
_CASE_STATUS_RE = re.compile(
    r"\b(?:status|update) of my (?:case|ticket|claim|warranty)|"
    r"\b(?:my )?(?:case|ticket|claim) status\b|"
    r"\bwhat(?:'s| is) the status of my (?:case|ticket|claim)\b",
    re.I,
)
_PHONE_RE = re.compile(
    r"\b(?:what(?:'s| is)|whats)\b.{0,20}\b(?:phone|number|email|contact)|"
    r"\b(?:phone number|call you|contact (?:number|info|information|email))\b|"
    r"\bhow do i (?:call|reach|contact)\b",
    re.I,
)
_MODEL_WHERE_RE = re.compile(
    r"\b(?:where|how)\b.{0,40}\b(?:find|locate|look|read)\b.{0,45}"
    r"\b(?:model\s+(?:number|name|label)|serial)|"
    r"\bwhere(?:'s|\s+is|\s+are)\s+(?:the\s+|my\s+)?(?:model|serial)\b|"
    r"\b(?:can't|cannot|couldn't|could not)\s+find\s+(?:the\s+|my\s+)?"
    r"(?:model|serial)\b",
    re.I,
)
_DONT_KNOW_RE = re.compile(
    r"^\s*(?:i\s+)?(?:don'?t|do not)\s+know\b|"
    r"^\s*(?:not sure|no idea|idk|dunno)\s*[.!?]*\s*$|"
    r"\b(?:can't|cannot|couldn'?t)\s+(?:check|tell|see|find out)\b",
    re.I,
)
_CONFUSION_RE = re.compile(
    r"\b(?:i\s+)?(?:don'?t|do not)\s+understand\b|"
    r"\bwhat does this mean\b|\bwhat do you mean\b|"
    r"\b(?:can you )?(?:explain|clarify)\b|"
    r"\bnone of (?:these|those|the) (?:choices?|options?)\b",
    re.I,
)
_ALREADY_TRIED_RE = re.compile(
    r"\b(?:i\s+)?(?:already|just) tried\b|\bthat didn'?t work\b|"
    r"\bstill (?:the same|not working|doesn'?t work)\b|"
    r"\bdid that already\b",
    re.I,
)

_DESCRIPTION_TEXT_NODES = frozenset({
    "delivery_missing_parts_q",
    "delivery_other_problem_q",
    "install_model",
})


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _reprompt(node: dict) -> str:
    prompt = str((node or {}).get("prompt") or "").strip()
    if prompt:
        return prompt
    return "Please choose one of the options above to continue."


def _domain_from(ticket) -> str:
    if ticket is None:
        return DEFAULT_WARRANTY_DOMAIN
    return normalize_warranty_domain(str(getattr(ticket, "domain", "") or ""))


def _is_exact_option_text(node: dict, text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return False
    for opt in (node or {}).get("options") or []:
        label = _normalize(str(opt.get("label") or ""))
        key = _normalize(str(opt.get("answer_key") or ""))
        if norm == label or (key and norm == key):
            return True
    return False


def _join(lead: str, node: dict) -> str:
    return f"{lead.rstrip()}\n\n{_reprompt(node)}"


def _contact_blurb() -> str:
    return (
        f"Warranty support is **{WARRANTY_PHONE}**, {WARRANTY_BUSINESS_HOURS}. "
        f"Email **{WARRANTY_TEAM_EMAIL}**."
    )


def _policy_url(domain: str, path: str) -> str:
    from store_config import get_storefront_base_url

    return f"{get_storefront_base_url(domain).rstrip('/')}/{path.lstrip('/')}"


def _warranty_length_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/warranty")
    return (
        "Coverage depends on the chair model and the original purchase date. "
        "This chat cannot confirm or approve a claim. The warranty team does that "
        "after reviewing your case.\n\n"
        "The published storefront terms describe standard coverage as **three (3) years** "
        "for defects in parts, workmanship, or structural defects, with year 1 including "
        "labor. Extended 4-year and 5-year plans are sold at purchase on some chairs. "
        "Your chair may differ.\n\n"
        f"Full terms: {url}\n\n"
        f"{_contact_blurb()}"
    )


def _returns_lead(domain: str) -> str:
    url = _policy_url(domain, "pages/sales-policy")
    return (
        "This chat cannot start a return, refund, or RMA. "
        f"**Sales** handles that at **{SALES_PHONE}**, and I have not changed your order.\n\n"
        "The published return window is **30 days from delivery**, with both-way freight "
        "paid by the customer and an RMA required first. White Glove fees are not refundable.\n\n"
        f"Full policy: {url}"
    )


def _replacement_answer() -> str:
    return (
        "This chat cannot approve a replacement chair or ship a part. "
        "After we finish the questions here, the warranty team reviews the case and decides "
        "the next step. Nothing has been approved.\n\n"
        f"{_contact_blurb()}"
    )


def _cost_answer() -> str:
    return (
        "I won't quote a repair cost, service-call fee, or part-shipping charge here. "
        "Those depend on the chair, the part, and whether the team finds the issue in warranty. "
        "Nothing has been billed from this chat.\n\n"
        f"{_contact_blurb()}"
    )


def _timing_answer() -> str:
    return (
        "Nothing has been scheduled from this chat, and I can't promise a technician visit, "
        "repair date, or case timeline. A callback has not been booked.\n\n"
        f"{_contact_blurb()}"
    )


def _phone_answer() -> str:
    return (
        f"{_contact_blurb()}\n"
        f"For sales, orders, or returns, call **{SALES_PHONE}**."
    )


def _model_where_answer() -> str:
    return (
        "Look for the model name on the serial-number sticker on the chair base, "
        "but check it only if it is visible and easy to reach. "
        "You can use **Snap a photo of the serial sticker** to help read a visible label. "
        "Do not lift, tip, or open the chair to find it. "
        "If the label is inaccessible or unreadable, ask for a person to help identify your chair."
    )


def looks_like_factual_common_question(text: str) -> bool:
    """True when the message is a common policy/contact question, not a menu pick."""
    raw = (text or "").strip()
    if not raw:
        return False
    return bool(
        _WARRANTY_LENGTH_RE.search(raw)
        or _COVERED_RE.search(raw)
        or _REPLACE_RE.search(raw)
        or _SEND_PART_RE.search(raw)
        or _REFUND_RE.search(raw)
        or _RETURN_RE.search(raw)
        or _REPAIR_COST_RE.search(raw)
        or _PART_SHIP_COST_RE.search(raw)
        or _TECH_WHEN_RE.search(raw)
        or _HOW_LONG_RE.search(raw)
        or _CASE_STATUS_RE.search(raw)
        or _PHONE_RE.search(raw)
        or _MODEL_WHERE_RE.search(raw)
    )


def looks_like_confusion(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if _GIBBERISH_RE.match(raw):
        return True
    return bool(
        _DONT_KNOW_RE.search(raw)
        or _CONFUSION_RE.search(raw)
        or _ALREADY_TRIED_RE.search(raw)
    )


def try_factual_common_question(
    node: dict,
    text: str,
    *,
    ticket=None,
) -> Optional[str]:
    """Answer a policy/contact question without advancing the flowchart."""
    raw = (text or "").strip()
    if not raw or not node or _is_exact_option_text(node, raw):
        return None
    domain = _domain_from(ticket)
    if _WARRANTY_LENGTH_RE.search(raw) or _COVERED_RE.search(raw):
        return _join(_warranty_length_answer(domain), node)
    if _REPLACE_RE.search(raw) or _SEND_PART_RE.search(raw):
        return _join(_replacement_answer(), node)
    if _REFUND_RE.search(raw) or _RETURN_RE.search(raw):
        return _join(_returns_lead(domain), node)
    if _REPAIR_COST_RE.search(raw) or _PART_SHIP_COST_RE.search(raw):
        return _join(_cost_answer(), node)
    if _TECH_WHEN_RE.search(raw) or _HOW_LONG_RE.search(raw) or _CASE_STATUS_RE.search(raw):
        return _join(_timing_answer(), node)
    if _PHONE_RE.search(raw):
        return _join(_phone_answer(), node)
    if _MODEL_WHERE_RE.search(raw):
        return _join(_model_where_answer(), node)
    return None


def try_confusion_common_question(node: dict, text: str) -> Optional[str]:
    """Re-explain the current step when the customer is stuck, without advancing."""
    raw = (text or "").strip()
    if not raw or not node or _is_exact_option_text(node, raw):
        return None
    node_id = str(node.get("node_id") or "")
    hangul = bool(_HANGUL_RE.search(raw))
    spanish = bool(_SPANISH_RE.search(raw))

    if hangul and node_id not in _DESCRIPTION_TEXT_NODES:
        return _join(
            "이 채팅은 아래 영어 선택지로 이어집니다. 상담이 필요하시면 "
            f"**{WARRANTY_PHONE}** ({WARRANTY_BUSINESS_HOURS})로 전화해 주세요.\n\n"
            "This chat continues in English with the current question. "
            f"For a person, call **{WARRANTY_PHONE}** ({WARRANTY_BUSINESS_HOURS}).",
            node,
        )
    if spanish and node_id not in _DESCRIPTION_TEXT_NODES and not re.search(r"[a-z]{4,}", raw, re.I):
        return _join(
            "Este chat continúa en inglés con la pregunta de abajo. "
            f"Para hablar con una persona, llame al **{WARRANTY_PHONE}** "
            f"({WARRANTY_BUSINESS_HOURS}).",
            node,
        )

    if _ALREADY_TRIED_RE.search(raw):
        return _join(
            "Understood — I'll keep that you already tried the last step. "
            "Please answer the current question so we can continue, or ask for a person. "
            f"{_contact_blurb()}",
            node,
        )
    if _CONFUSION_RE.search(raw) or _DONT_KNOW_RE.search(raw):
        extra = (
            "If you can't check this from where you are, choose the closest option "
            "or ask for a person. "
        )
        if node.get("type") == "question_text":
            extra = (
                "If you don't have that detail, say so in a short sentence "
                "(for example *I don't have the order number*) and I'll tell you what we can use instead. "
            )
        return _join(
            "No problem — here is the current question again. "
            + extra
            + _contact_blurb(),
            node,
        )
    if _GIBBERISH_RE.match(raw):
        return _join(
            "I didn't catch that. Please answer the current question, or tap one of the options.",
            node,
        )
    return None


def try_answer_common_question(
    node: dict,
    text: str,
    *,
    ticket=None,
    allow_confusion: bool = True,
) -> Optional[str]:
    factual = try_factual_common_question(node, text, ticket=ticket)
    if factual:
        return factual
    if allow_confusion:
        return try_confusion_common_question(node, text)
    return None
