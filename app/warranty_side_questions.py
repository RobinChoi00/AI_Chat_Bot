"""
Answer customer side questions without advancing the warranty workflow.

When a customer asks about specs, dimensions, or general troubleshooting while
the flowchart expects a menu answer or delivery detail, we respond from the
product catalog / knowledge base and re-show the current question.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from delivery_intake import (
    DeliverySpecQuestion,
    detect_delivery_spec_question,
    fetch_delivery_spec_answer,
    is_plausible_email,
    is_plausible_order_id,
    is_plausible_tracking_number,
)
from warranty_common_questions import (
    try_confusion_common_question,
    try_factual_common_question,
    WARRANTY_PHONE,
    WARRANTY_TEAM_EMAIL,
)
from warranty_intake_context import enrich_path_text
from warranty_knowledge import contextual_search_knowledge, customer_step_entries
from warranty_self_help import (
    _collect_fallback_hints,
    build_path_text,
    infer_defect_category_from_turns,
)
from warranty_step_enrichment import _pick_step_tips, format_step_message

_FAQ_START_RE = re.compile(
    r"^\s*(what|how|can|could|does|do|is|are|will|where|when|why|"
    r"give|tell|show|please)\b",
    re.IGNORECASE,
)

_QUESTION_HINT_RE = re.compile(
    r"\b(what|how|give|tell|show|size|dimension|measure|length|width|height|"
    r"please|help|where|when|why|can you|could you|do you|is there|the box|"
    r"shipping box|carton|crate|package|doorway|door|weight|heavy|fit)\b",
    re.IGNORECASE,
)

_SKIP_SIDE_NODE_IDS = frozenset({"root"})

_DELIVERY_TEXT_NODES = frozenset(
    {
        "delivery_get_name",
        "delivery_get_tracking_number",
        "delivery_status_get_order_email",
        "delivery_status_get_tracking",
    }
)


def answer_pre_purchase_product_media(text: str, *, ticket=None) -> Optional[str]:
    """Point shoppers to an exact published product, independent of repair state."""
    from warranty_scope import is_pre_purchase_product_media, support_contact

    if not is_pre_purchase_product_media(text):
        return None

    from catalog_identity import sentence_product
    from sales_catalog import load_product_index

    # Never reuse a previously saved warranty model as the shopping target:
    # someone asking about another chair should not get an unrelated gallery.
    product, _named = sentence_product(text, load_product_index())
    contact = support_contact("sales", str(getattr(ticket, "domain", "") or ""))["phone"]
    if product is None or product.status.lower() != "active" or not re.fullmatch(r"[a-z0-9-]{1,120}", product.handle):
        return (
            "I couldn't verify the exact product for your photo or video request. "
            f"Please share its full model name or the product page, or contact **Sales at {contact}**."
        )
    # The catalog is the official Osaki USA storefront export. Other store
    # handles can differ; use the catalog's source rather than guessing one.
    url = f"https://osakiusa.com/products/{product.handle}"
    message = f"For the **{product.display_name}**, see the [official product photos and details]({url})."
    if re.search(r"\b(?:videos?|demo(?:nstration)?s?)\b", text, re.I):
        message += f" I don't have a verified official video link for this model. **Sales at {contact}** can check whether one is available."
    return message


def _remote_display_observation(node: dict, text: str) -> tuple[bool, bool, str]:
    """Interpret explicit display and control information only at remote checks."""
    from warranty_common_questions import _is_exact_option_text

    if str(node.get("node_id") or "") not in {"defect_remote_power_q", "defect_remote_screen_q"} or _is_exact_option_text(node, text):
        return False, False, ""
    raw = (text or "").strip().replace("’", "'")
    if re.search(r"\b(?:cable|cord)\b[^.!?;\n]{0,20}\b(?:cut|broken|damaged|frayed)\b", raw, re.I):
        return False, False, ""  # Let the separate cable symptom be handled.
    display = r"(?:screen|display|touch\s*screen)"
    states = r"(?:blank|black|dark|off|not\s+(?:showing|displaying)\s+anything)"
    shows_image = bool(re.search(
        rf"\b{display}\s+(?:(?:is|was)\s+not|(?:isn't|wasn't))\s+(?:blank|black|dark|off)\b|"
        rf"\b{display}\s+(?:shows|displays|lights\s+up|comes\s+on)\b",
        raw,
        re.I,
    )) and not bool(re.search(rf"\b{display}\s+(?:shows|displays)\s+nothing\b", raw, re.I))
    blank_or_unspecified = not shows_image and bool(re.search(
        rf"\b{display}\b[^.!?;\n]{{0,24}}\b{states}\b|\b{states}\s+(?:remote\s+)?{display}\b|\bno\s+display\b|"
        rf"^\s*(?:the\s+)?{display}[!.?]*\s*$",
        raw,
        re.I,
    ))
    controls = r"(?:commands?|buttons?|controls?)"
    fails = r"(?:(?:do\s+not|don't|will\s+not|won't|cannot|can't)\s+(?:work|respond|move)|(?:are\s+)?(?:unresponsive|dead)|fail)"
    responds = r"(?:work|respond|move)(?:s|ing)?"
    some = rf"\b(?:some|certain|only\s+(?:some|a\s+few))\s+(?:of\s+the\s+)?{controls}\s+"
    if re.search(rf"{some}(?:still\s+)?{responds}\b", raw, re.I):
        control_status = "partial_working"
    elif re.search(rf"{some}{fails}\b|\bnot\s+all\s+{controls}\s+{responds}\b", raw, re.I):
        control_status = "partial_failure"
    elif re.search(rf"\b(?:no|none\s+of\s+the)\s+{controls}\s+{responds}\b|\b{controls}\s+{fails}\b", raw, re.I):
        control_status = "unresponsive"
    elif re.search(rf"\b{controls}\s+(?:(?:still|all|do)\s+){{0,2}}{responds}\b", raw, re.I):
        control_status = "working"
    else:
        control_status = ""
    return blank_or_unspecified, shows_image, control_status


def explicit_remote_display_answer_key(node: dict, text: str) -> Optional[str]:
    """Resolve a stated result instead of guessing it from the word 'screen'."""
    blank, shows_image, controls = _remote_display_observation(node, text)
    if node.get("node_id") == "defect_remote_power_q":
        if shows_image or controls in {"working", "partial_working"}:
            return "has_power"
        if blank and controls == "unresponsive":
            return "no_power"
    elif node.get("node_id") == "defect_remote_screen_q":
        if controls in {"unresponsive", "partial_working", "partial_failure"}:
            return "commands_not_responding"
        # A bare mention of the screen is not evidence that it is blank.
        if blank and controls == "working" and len(text.split()) > 2:
            return "blank_screen_commands_ok"
    return None


def answer_ambiguous_remote_display(node: dict, text: str) -> Optional[str]:
    """Do not turn a blank display into an assumed power failure or replacement."""
    blank, _shows_image, controls = _remote_display_observation(node, text)
    if not blank or (controls and not (controls == "partial_failure" and node.get("node_id") == "defect_remote_power_q")):
        return None
    return (
        "A blank screen by itself doesn't tell us whether the remote can still send commands. "
        "Do the remote buttons still move the chair, or are the buttons unresponsive too? "
        "Please tell me both what you see on the screen and whether the buttons work."
    )


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _node_prompt(node: dict) -> str:
    return str(node.get("prompt") or "").strip()


def _reprompt_current(node: dict) -> str:
    prompt = _node_prompt(node)
    if prompt:
        return prompt
    return "Please choose one of the options above to continue."


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


_BRANDED_ORDER_RE = re.compile(r"\b((?:OSTI|OSKUS|OSKMC|OSK)[A-Za-z0-9-]{3,})\b", re.I)
_ORDER_LABEL_RE = re.compile(
    r"(?:order\s*(?:number|no\.?|#)?\s*[:#]?\s*|#)\s*([A-Za-z0-9-]{4,})",
    re.I,
)
_CASE_PROGRESS_RE = re.compile(
    r"\b(?:working on (?:it|this|my(?:\s+order)?)|are you working|wrong number|worng number|"
    r"wrong order|is this (?:the )?(?:wrong|right)|did you (?:start|get|receive))\b",
    re.I,
)


def mentioned_order_number(text: str) -> Optional[str]:
    """An order id the customer stated, including store prefixes such as OSTI."""
    raw = text or ""
    branded = _BRANDED_ORDER_RE.search(raw)
    if branded:
        return branded.group(1).upper()
    if re.search(r"\border\b|#", raw, re.I):
        numbered = _ORDER_LABEL_RE.search(raw)
        if numbered and any(char.isdigit() for char in numbered.group(1)):
            return numbered.group(1).upper()
    return None


def _order_progress_reply(node: dict, text: str, ticket) -> Optional[str]:
    """Answer an order number or a 'are you working on it' question without a lookup."""
    if _looks_like_valid_workflow_answer(node, text):
        return None
    order = mentioned_order_number(text)
    saved = ""
    if ticket is not None and hasattr(ticket, "get_collected"):
        saved = str(ticket.get_collected().get("order_number") or "").strip()
    number = order or saved
    progress = bool(_CASE_PROGRESS_RE.search(text or ""))
    if not order and not progress:
        return None
    if order and progress:
        lead = (
            f"I saved **{order}**, but nobody is working on it yet and this chat has not verified that number. "
            "If it is wrong, send the correct order number."
        )
    elif order:
        lead = (
            f"I saved order number **{order}**. Nobody is working on it yet, and this chat has not looked it up. "
            "The number alone does not show whether it is correct; a lookup also needs the checkout email."
        )
    elif number:
        lead = (
            f"Nobody is working on this yet, and **{number}** has not been verified. "
            "If that number is wrong, send the correct order number."
        )
    else:
        lead = "Nobody is working on this yet, and I do not have an order number saved. Send the order number if you have it."
    return f"{lead}\n\n{_reprompt_current(node)}"


def _delivery_side_reply(
    node: dict,
    text: str,
    issue_type: str,
    ticket=None,
) -> Optional[str]:
    """Order questions owned by Sales, and what a tracking number is, during a delivery case."""
    from delivery_intake import delivery_sales_topic, tracking_number_explainer  # noqa: WPS433
    from warranty_scope import SALES_PHONE  # noqa: WPS433

    node_id = str(node.get("node_id") or "")
    if not (node_id.startswith("delivery_") or node_id == "issue_type" or issue_type == "delivery"):
        return None
    if _looks_like_valid_workflow_answer(node, text):
        return None
    explainer = tracking_number_explainer(text)
    if explainer:
        return f"{explainer}\n\n{_reprompt_current(node)}"
    lead = delivery_sales_topic(text)
    if not lead:
        return None
    # Ask before creating a Sales review request (does not change the order).
    if ticket is not None and hasattr(ticket, "set_collected"):
        ticket.set_collected("awaiting_sales_submit_consent", True)
        ticket.set_collected("sales_submit_pending_text", (text or "")[:2000])
        ticket.set_collected("human_support_department", "sales")
    return (
        f"{lead} I can submit this to **Sales** for review — that does **not** "
        f"change your order, address, or delivery schedule.\n\n"
        f"Would you like me to submit it to Sales?\n\n"
        f"(You can also call **Sales** at **{SALES_PHONE}**.)\n\n"
        f"{_reprompt_current(node)}"
    )


def _looks_like_faq_question(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if "?" in raw:
        return True
    if _FAQ_START_RE.search(raw):
        return True
    if _QUESTION_HINT_RE.search(raw) and len(raw.split()) >= 4:
        return True
    return False


def _matches_option_heuristic(node: dict, text: str) -> bool:
    if not (node.get("options") or []):
        return False
    from warranty_nlp import _heuristic_option_match  # noqa: WPS433

    return _heuristic_option_match(list(node.get("options") or []), text) is not None


def _looks_like_valid_workflow_answer(node: dict, text: str) -> bool:
    """Return True when the text should be treated as a normal workflow answer."""
    node_id = str(node.get("node_id") or "")
    raw = (text or "").strip()
    if not raw:
        return False

    options = list(node.get("options") or [])
    if options:
        from delivery_intake import map_delivery_free_text  # noqa: WPS433
        from warranty_error_code_gate import is_gate_node, map_gate_free_text  # noqa: WPS433
        from warranty_common_questions import looks_like_factual_common_question  # noqa: WPS433

        if looks_like_factual_common_question(raw):
            return False
        if map_delivery_free_text(node_id, raw):
            return True
        if is_gate_node(node_id):
            mapped = map_gate_free_text(node_id, raw)
            if mapped:
                return True
        keys = {_normalize(str(o.get("answer_key") or "")) for o in options}
        if _normalize(raw) in keys:
            return True
        if _matches_option_heuristic(node, raw):
            return True
        return False

    if node_id in _DELIVERY_TEXT_NODES:
        from warranty_email import extract_email  # noqa: WPS433

        if node_id in ("delivery_get_name", "delivery_status_get_order_email"):
            embedded = extract_email(raw)
            if embedded and is_plausible_email(embedded):
                return True
            return is_plausible_email(raw) or is_plausible_order_id(raw)

        if node_id in ("delivery_get_tracking_number", "delivery_status_get_tracking"):
            return is_plausible_tracking_number(raw)

    if node.get("type") == "question_text" and node_id == "install_model":
        if _looks_like_faq_question(raw):
            return False
        return len(raw) >= 2 and not raw.endswith("?")

    return False


def _should_handle_side_question(node: dict, text: str) -> bool:
    node_id = str(node.get("node_id") or "")
    if node.get("type") == "terminal" or node_id in _SKIP_SIDE_NODE_IDS:
        return False
    if _looks_like_valid_workflow_answer(node, text):
        return False
    if detect_delivery_spec_question(text):
        return True
    return _looks_like_faq_question(text)


_GUIDE_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "to", "of", "for", "on", "in", "at", "is",
        "are", "do", "does", "how", "can", "i", "my", "me", "you", "your", "with",
        "this", "that", "it", "chair", "please", "help", "about", "from",
    }
)


def _guide_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) >= 3 and tok not in _GUIDE_STOPWORDS
    }


def _tips_match_question(query_text: str, tips: Sequence[str]) -> bool:
    """Require tip text to share meaningful tokens with the asked question."""
    query_tokens = _guide_tokens(query_text)
    if not query_tokens or not tips:
        return False
    tip_tokens = set()
    for tip in tips:
        tip_tokens |= _guide_tokens(tip)
    # At least two overlapping content tokens, or one strong multi-word cue.
    overlap = query_tokens & tip_tokens
    if len(overlap) >= 2:
        return True
    strong = {
        "autocheck", "auto", "check", "bluetooth", "watt", "watts", "diagram",
        "hose", "airbag", "roller", "rollers", "mechanism", "remote", "heater",
    }
    return bool(overlap & strong) and len(overlap) >= 1


def _format_knowledge_side_answer(
    *,
    node: dict,
    query_text: str,
    model_name: str,
    issue_type: str,
    turns: Sequence[Any],
    ticket=None,
) -> Optional[str]:
    if re.search(
        r"too\s+intense|over[\s-]?inflat|not\s+enough\s+(?:pressure|squeeze)|"
        r"(?:not|isn't|aren't|don't|doesn't)\s+squeez|lower\s+the\s+intensity|"
        r"auto[\s-]?check|automatic\s+check|\bwatts?\b|wattage|parts?\s+diagram|"
        r"replacement\s+parts",
        query_text or "",
        re.I,
    ):
        return None

    path_text = enrich_path_text(
        f"{query_text} {build_path_text(turns)}".strip(),
        ticket,
    )
    defect_category = infer_defect_category_from_turns(turns)
    matches = customer_step_entries(
        contextual_search_knowledge(
            path_text=path_text,
            issue_type=issue_type,
            defect_category=defect_category,
            model_name=model_name,
            limit=2,
        ),
        model_name=model_name or None,
    )
    node_id = str(node.get("node_id") or "")
    fallback = _collect_fallback_hints(turns, node_id)
    tips = _pick_step_tips(matches, fallback)

    if not matches and not tips:
        return None
    # Do not paste an unrelated published check (e.g. air hose for auto-check).
    if not _tips_match_question(query_text, tips):
        return None

    model_display = (model_name or "your chair").strip()
    if defect_category:
        summary = (
            f"Here's a published check we can share for the **{defect_category}** "
            f"issue on your **{model_display}** before the next step."
        )
    else:
        summary = (
            f"Here's a published check we can share about your **{model_display}** "
            "before the next step."
        )

    return format_step_message(
        base_prompt=_reprompt_current(node),
        summary=summary,
        tips=tips,
    )


def try_answer_side_question(
    *,
    node: dict,
    answer: str,
    model_name: str = "",
    issue_type: str = "",
    turns: Optional[Sequence[Any]] = None,
    ticket=None,
) -> Optional[str]:
    """
    If ``answer`` is a side question, return a customer-facing reply plus the
    current workflow re-prompt. Returns ``None`` when normal submit should run.
    """
    text = (answer or "").strip()
    if not text or not node:
        return None
    product_media = answer_pre_purchase_product_media(text, ticket=ticket)
    if product_media:
        return product_media
    display_help = answer_ambiguous_remote_display(node, text)
    if display_help:
        return display_help
    factual = try_factual_common_question(node, text, ticket=ticket)
    if factual:
        return factual
    progress = _order_progress_reply(node, text, ticket)
    if progress:
        return progress
    delivery = _delivery_side_reply(node, text, issue_type, ticket=ticket)
    if delivery:
        return delivery
    if not _looks_like_valid_workflow_answer(node, text):
        confusion = try_confusion_common_question(node, text)
        if confusion:
            return confusion
    if not _should_handle_side_question(node, text):
        return None

    node_id = str(node.get("node_id") or "")
    spec = detect_delivery_spec_question(text)
    if spec:
        if node_id in ("delivery_get_name", "delivery_status_get_order_email"):
            reprompt = _reprompt_order_or_email()
        elif node_id in ("delivery_get_tracking_number", "delivery_status_get_tracking"):
            reprompt = _reprompt_tracking_number()
        else:
            reprompt = _reprompt_current(node)
        return _build_spec_side_answer(
            model_name=model_name,
            spec=spec,
            reprompt=reprompt,
        )

    kb_answer = _format_knowledge_side_answer(
        node=node,
        query_text=text,
        model_name=model_name,
        issue_type=issue_type,
        turns=turns or (),
        ticket=ticket,
    )
    if kb_answer:
        return kb_answer

    if _looks_like_faq_question(text):
        return (
            f"I don't have a published answer for that in this chat. "
            f"Warranty support is **{WARRANTY_PHONE}**. Email **{WARRANTY_TEAM_EMAIL}**. "
            "This chat cannot approve a replacement, refund, or repair quote.\n\n"
            f"{_reprompt_current(node)}"
        )

    return None
