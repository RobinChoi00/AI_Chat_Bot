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
_EXTENDED_COVER_RE = re.compile(
    r"\bextended warrant|"
    r"\bwarrant\w*.{0,60}\bcover|"
    r"\bwhat does (?:the |it |extended )?warrant|"
    r"\bwhat does it cover\b",
    re.I,
)
_WHITE_GLOVE_RE = re.compile(
    r"\bwhite[- ]?glove\b.{0,90}\b("
    r"include|included|assembly|install|set\s*up|living\s+room|packaging|"
    r"what(?:'s| is)|fee|cost|\$\d"
    r")|"
    r"\b("
    r"include|included|assembly|what(?:'s| is) included"
    r").{0,90}\bwhite[- ]?glove\b",
    re.I,
)
_BLUETOOTH_PAIR_RE = re.compile(
    r"\b(?:how do i|how to|help(?: me)?)\b.{0,40}\b(?:pair|connect)\b.{0,40}\bbluetooth|"
    r"\bpair(?:ing)? (?:my |the )?(?:chair|bluetooth)|"
    r"\bbluetooth.{0,40}\b(?:pair|connect|set up|setup)\b",
    re.I,
)
_BLUETOOTH_FAULT_RE = re.compile(
    r"\b(?:will\s+not|won't|does\s+not|doesn't|cannot|can't)\s+(?:pair|connect)\b|"
    r"\bbluetooth\s+(?:connection\s+)?(?:fault|problem|failure|issue)\b",
    re.I,
)
_FSA_RECEIPT_RE = re.compile(
    r"\b(?:fsa|hsa|flexible spending|health savings)\b|"
    r"\breceipt\b.{0,40}\b(?:fsa|hsa|eligible)|"
    r"\b(?:fsa|hsa|eligible).{0,40}\breceipt\b",
    re.I,
)
_HALLWAY_FIT_RE = re.compile(
    r"\b(?:tight|narrow)\s+(?:hallway|doorway|stair(?:s|case)?|doors?)\b|"
    r"\b(?:hallway|doorway|stair(?:s|case)?)\b.{0,60}\b(?:fit|fits|ok|okay|delivery|deliver|width|wide)|"
    r"\b(?:fit|fits|fitting)\b.{0,60}\b(?:hallway|doorway|stair(?:s|case)?|doors?)\b|"
    r"\bfit\s+(?:through|in)\s+(?:the\s+|my\s+|a\s+)?.{0,24}\b(?:hallway|doors?|doorway|stair)|"
    r"\b(?:door|doorway|hallway)\s+width\b|"
    r"\b\d{2}(?:\.\d+)?\s*(?:\"|in(?:ch(?:es)?)?)\s+(?:wide\s+)?(?:door|doorway|hallway|opening)|"
    r"\b(?:door|doorway|hallway|opening)\s+(?:is\s+|of\s+)?\d{2}(?:\.\d+)?\s*(?:\"|in)",
    re.I,
)
_ASSEMBLY_HOWTO_RE = re.compile(
    r"\b(?:how\s+(?:do|can)\s+(?:i|you|we)|how\s+to|how\s+do\s+you)\b.{0,40}\b"
    r"(?:assembl(?:e|y|ing)|install(?:ation)?|set\s*up)\b|"
    r"\bassembl(?:y|e|ing)\s+(?:instructions?|guide|video|manual|steps?)\b",
    re.I,
)
_AIR_INTENSITY_RE = re.compile(
    r"\b(?:airbags?|air bags?|air)\b.{0,50}\b(?:too\s+(?:intense|strong|hard|much)|over[\s-]?inflat)|"
    r"\b(?:too\s+(?:intense|strong|hard)|over[\s-]?inflat).{0,50}\b(?:airbags?|air bags?|air)|"
    r"\b(?:lower|reduce|turn\s+down)\b.{0,40}\b(?:air\s+)?(?:intensity|pressure|airbags?)\b|"
    r"\bhow\s+(?:do|can)\s+i\b.{0,40}\b(?:lower|reduce|turn\s+down).{0,24}\b(?:intensity|pressure|airbags?)",
    re.I,
)
_AIR_NOT_INFLATING_RE = re.compile(
    r"\b(?:will\s+not|won't|does\s+not|doesn't|cannot|can't|not)\s+"
    r"(?:inflat(?:e|ing)|stay\s+inflated)\b|"
    r"\bno\s+air\b|\bnot\s+inflating\b",
    re.I,
)
_ASSEMBLY_FAULT_RE = re.compile(
    r"\b(?:will\s+not|won't|cannot|can't|does\s+not|doesn't)\s+"
    r"(?:assembl(?:e|y)|attach|install|fit|go\s+together)\b|"
    r"\b(?:missing|broken|damaged)\s+(?:part|piece|bolt|screw|panel|footrest)\b|"
    r"\b(?:won't|will\s+not|can't)\s+(?:attach|go\s+on|fit\s+together)\b",
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


def _join(lead: str, node: dict, *, reprompt: bool = True) -> str:
    text = lead.rstrip()
    if not reprompt:
        return text
    return f"{text}\n\n{_reprompt(node)}"


def _join_without_issue_menu(lead: str, node: dict) -> str:
    """Sales/how-to answers should not reopen setup / delivery / malfunction."""
    node_id = str((node or {}).get("node_id") or "")
    if node_id in ("", "root", "issue_type"):
        return _join(lead, node, reprompt=False)
    return _join(lead, node)


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


def _white_glove_answer(domain: str) -> str:
    url = _policy_url(domain, "pages/shipping-handling")
    return (
        "**White Glove** is in-home delivery plus assembly of the chair. "
        "Standard curbside delivery does not include assembly. "
        "The published **basic White Glove fee is $275**. Extra charges can apply for "
        "stairs (a third person, **$120–$300**), more on-site time, added distance, "
        "or a redelivery attempt. This chat cannot add White Glove to an order or book a crew.\n\n"
        "Published details: the usual team is two people; they do not move your furniture, "
        "remove doors, or haul away an old chair. The White Glove fee is not refundable. "
        "Sales confirms the amount for your address and whether it is on your order.\n\n"
        f"Full details: {url}\n\n"
        f"Call **{SALES_PHONE}** for White Glove, fees, or scheduling."
    )


def _bluetooth_pair_answer() -> str:
    return (
        "This chat does not walk through a model-specific pairing sequence. "
        "On most Osaki and Titan chairs, Bluetooth is turned on from the remote or tablet, "
        "then you pair from your phone's Bluetooth list. Button names differ by model.\n\n"
        "If pairing still fails after that, choose **setup** below if you are assembling the chair, "
        "or **malfunction** if an already-setup chair will not connect.\n\n"
        f"{_contact_blurb()}"
    )


def _fsa_receipt_answer() -> str:
    return (
        "This chat cannot issue an FSA/HSA letter or a copy of your receipt. "
        f"**Sales** handles order documents at **{SALES_PHONE}**. I have not changed your order."
    )


def _hallway_model_name(text: str, ticket=None) -> str:
    try:
        from product_catalog import resolve_model_name  # noqa: WPS433

        named = resolve_model_name(text)
        if named:
            return named
    except Exception:
        pass
    return str(getattr(ticket, "model_name", "") or "").strip()


def _hallway_opening_inches(text: str) -> Optional[float]:
    try:
        from sales_constraints import doorway_width_from_text  # noqa: WPS433

        inches = doorway_width_from_text(text)
        if inches is not None:
            return inches
    except Exception:
        pass
    match = re.search(
        r"(\d{2}(?:\.\d+)?)\s*(?:\"|in(?:ch(?:es)?)?)\s+(?:wide\s+)?"
        r"(?:hallway|door|doorway|opening|stair)|"
        r"(?:hallway|door|doorway|opening|stair).{0,24}?"
        r"(\d{2}(?:\.\d+)?)\s*(?:\"|in(?:ch(?:es)?)?)",
        text or "",
        re.I,
    )
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        width = float(raw)
    except (TypeError, ValueError):
        return None
    if 20 <= width <= 48:
        return width
    return None


def _hallway_fit_answer(text: str, ticket=None) -> str:
    """Compare the shopper's opening to published assembled / disassembled doorway width."""
    model = _hallway_model_name(text, ticket)
    inches = _hallway_opening_inches(text)
    fit = None
    if model:
        try:
            from sales_spec_index import lookup_fit_spec  # noqa: WPS433

            fit = lookup_fit_spec(model)
        except Exception:
            fit = None

    lines = [
        "We check hallway and door fit against the **published doorway width** "
        "for the model — assembled and, when listed, disassembled. "
        "Measure the **narrowest opening** on the path (door, hallway, or stair width).",
    ]
    if model and fit is not None and fit.door_asm_in is not None:
        from sales_spec_index import doorway_ok  # noqa: WPS433

        lines.append(f"**{model}** published doorway:")
        lines.append(f"- Assembled: **{fit.door_asm_in:g} in**")
        if fit.door_dis_in is not None:
            lines.append(f"- Disassembled: **{fit.door_dis_in:g} in**")
        if inches is not None:
            if doorway_ok(model, limit_in=inches, mode="assembled"):
                lines.append(
                    f"A **{inches:g} in** opening is at least as wide as the assembled figure, "
                    "so that published width fits."
                )
            else:
                lines.append(
                    f"The assembled figure is wider than a **{inches:g} in** opening."
                )
                if fit.door_dis_in is not None:
                    if doorway_ok(model, limit_in=inches, mode="disassembled"):
                        lines.append(
                            "The disassembled figure fits that width. "
                            "I won't promise a delivery crew will disassemble on site."
                        )
                    else:
                        lines.append(
                            "Even the disassembled figure is wider than that opening."
                        )
        else:
            lines.append(
                "Tell me the width in inches of the **narrowest door or hallway** on the path "
                "and I will compare it to those numbers."
            )
    elif model:
        lines.append(
            f"I don't have a published doorway number for **{model}**. "
            "Share the **narrowest opening in inches** and Sales can still check "
            f"at **{SALES_PHONE}**."
        )
    elif inches is not None:
        lines.append(
            f"A **{inches:g} in** opening is tight for some chairs. "
            "Which **model** should I compare against the published doorway figures?"
        )
    else:
        lines.append(
            "Tell me the **model** and the **narrowest door or hallway width in inches**, "
            "and I will compare them to the published doorway figures."
        )
    lines.append(
        "This check is **width only**. It does not confirm a stair turn, furniture in the path, "
        "or that White Glove will remove a door."
    )
    return "\n".join(lines)


def _is_air_intensity_question(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or _AIR_NOT_INFLATING_RE.search(raw):
        return False
    return bool(_AIR_INTENSITY_RE.search(raw))


def _air_intensity_answer() -> str:
    return (
        "This chat's **air** path is for bags that **will not inflate**, not for lowering strength. "
        "Published warranty materials do not list a model-specific air-intensity control, "
        "so I will not invent a remote setting.\n\n"
        "Look up your exact model at [https://www.otasupport.com](https://www.otasupport.com) "
        "for the remote buttons that are documented there. "
        "If a bag **stays over-inflated and will not release**, say that — that is a malfunction."
    )


def _assembly_howto_answer() -> str:
    return (
        "This chat does not walk through a model-specific assembly sequence. "
        "Published setup guides are at [https://www.otasupport.com](https://www.otasupport.com) — "
        "look up your exact model there.\n\n"
        "If a part will not attach, a piece is missing, or an already-assembled chair is wrong, "
        "say that and we can continue with **setup** or a **malfunction**."
    )


def _is_assembly_howto(text: str) -> bool:
    """How-to assembly questions only — a failed install is still a setup case."""
    raw = (text or "").strip()
    if not raw or _ASSEMBLY_FAULT_RE.search(raw):
        return False
    return bool(_ASSEMBLY_HOWTO_RE.search(raw))


def is_sales_bound_opening_question(text: str) -> bool:
    """FSA, hallway fit, White Glove policy — answer these without the issue menu."""
    raw = (text or "").strip()
    if not raw:
        return False
    return bool(
        _FSA_RECEIPT_RE.search(raw)
        or _HALLWAY_FIT_RE.search(raw)
        or _WHITE_GLOVE_RE.search(raw)
        or _is_assembly_howto(raw)
    )


def _is_bluetooth_pairing_howto(text: str) -> bool:
    """How-to pairing questions only — a failed pair is a malfunction, not a FAQ."""
    raw = (text or "").strip()
    if not raw or _BLUETOOTH_FAULT_RE.search(raw):
        return False
    return bool(_BLUETOOTH_PAIR_RE.search(raw))


def looks_like_factual_common_question(text: str) -> bool:
    """True when the message is a common policy/contact question, not a menu pick."""
    raw = (text or "").strip()
    if not raw:
        return False
    return bool(
        _WARRANTY_LENGTH_RE.search(raw)
        or _COVERED_RE.search(raw)
        or _EXTENDED_COVER_RE.search(raw)
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
        or _WHITE_GLOVE_RE.search(raw)
        or _is_bluetooth_pairing_howto(raw)
        or _FSA_RECEIPT_RE.search(raw)
        or _HALLWAY_FIT_RE.search(raw)
        or _is_assembly_howto(raw)
        or _is_air_intensity_question(raw)
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
    if (
        _WARRANTY_LENGTH_RE.search(raw)
        or _COVERED_RE.search(raw)
        or (re.search(r"\bwarrant", raw, re.I) and re.search(r"\bcover", raw, re.I))
        or _EXTENDED_COVER_RE.search(raw)
    ):
        return _join(_warranty_length_answer(domain), node)
    if _WHITE_GLOVE_RE.search(raw):
        return _join_without_issue_menu(_white_glove_answer(domain), node)
    if _is_bluetooth_pairing_howto(raw):
        return _join(_bluetooth_pair_answer(), node)
    if _FSA_RECEIPT_RE.search(raw):
        return _join_without_issue_menu(_fsa_receipt_answer(), node)
    if _HALLWAY_FIT_RE.search(raw):
        return _join_without_issue_menu(_hallway_fit_answer(raw, ticket), node)
    if _is_assembly_howto(raw):
        return _join_without_issue_menu(_assembly_howto_answer(), node)
    if _is_air_intensity_question(raw):
        return _join(_air_intensity_answer(), node)
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
