"""Shopping follow-ups answered from saved context and published catalog facts.

Each handler returns ``None`` unless its referent is unambiguous: a named
model, the current shortlist, the saved household users, or a general feature
term. Nothing here supplies a capacity, width, height range or setting that
the catalog does not publish.
"""
from __future__ import annotations

import re
from typing import Optional

from sales_catalog import ProductSpecs, list_active_products
from sales_constraints import _read_person_measurements, extract_constraints, product_constraint_issues
from sales_height import (STANDARD_MIN_IN, STANDARD_RANGE, XL_MAX_IN, XL_RANGE, height_guidance, is_xl_line,
                          needs_xl_line)
from sales_spec_index import lookup_fit_spec

WARRANTY_EMAIL = "service@osakititan.com"
WARRANTY_PHONE = "+1-888-848-2630 ext. 3"
SALES_PHONE = "+1-888-848-2630 ext. 2"
WARRANTY_PORTAL = "https://titanchair.freshdesk.com/support/home"

_PERSON_LABELS = {
    'self': 'you', 'user': 'the user', 'buyer': 'you', 'person': 'the user',
    'father': 'your father', 'mother': 'your mother', 'husband': 'your husband', 'wife': 'your wife',
    'spouse': 'your spouse', 'partner': 'your partner', 'son': 'your son', 'daughter': 'your daughter',
    'first_user': 'the first user', 'second_user': 'the second user',
    'heavier_user': 'the heavier user', 'lighter_user': 'the lighter user',
}
_NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8}


def _person(label: str) -> str:
    if label in _PERSON_LABELS:
        return _PERSON_LABELS[label]
    base, _, index = label.rpartition('_')
    if base == 'user' and index.isdigit():
        return f'user {index}'
    return label.replace('_', ' ')


def _feet(inches: float) -> str:
    feet, rest = divmod(round(inches), 12)
    return f"{feet}'{rest}\""


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _saved_users(rec: dict) -> list[tuple[str, Optional[float], Optional[float]]]:
    """(label, height_in, weight_lb) for every household user that has a saved measurement."""
    heights = _read_person_measurements(rec.get('user_heights_in'))
    weights = _read_person_measurements(rec.get('user_weights_lb'))
    if not heights and rec.get('height_in'):
        heights = {'user': float(rec['height_in'])}
    if not weights and rec.get('weight_lb'):
        weights = {'user': float(rec['weight_lb'])}
    if len(heights) == 1 and len(weights) == 1 and set(heights) != set(weights):
        label = next(label for label in (*heights, *weights) if label not in {'user', 'person'})
        heights = {label: next(iter(heights.values()))}
        weights = {label: next(iter(weights.values()))}
    labels = list(dict.fromkeys([*weights, *heights]))
    return [(label, heights.get(label), weights.get(label)) for label in labels]


def _relabel_single_user(users, raw: str):
    """"My father's requirements" names the only saved user when it was unlabeled."""
    named = re.search(r"\bmy\s+(father|dad|mother|mom|husband|wife|spouse|partner|son|daughter)(?:['’]s)?\b", raw or '', re.I)
    if not named or len(users) != 1 or users[0][0] not in {'user', 'self', 'person', 'buyer'}:
        return users
    label = {'dad': 'father', 'mom': 'mother'}.get(named.group(1).lower(), named.group(1).lower())
    return [(label, users[0][1], users[0][2])]


def _owner_phrase(users) -> str:
    named = [_person(label) for label, _, _ in users if label not in {'user', 'person', 'self', 'buyer'}]
    if len(named) == 1:
        return named[0] + "'s"
    if len(named) > 1:
        return "both users'"
    return 'your'


def _shortlist(data: dict) -> list[ProductSpecs]:
    from sales_agent import _product_from_tier_pick
    out: list[ProductSpecs] = []
    for pick in data.get('pending_tier_picks') or []:
        product = _product_from_tier_pick(pick) if isinstance(pick, dict) else None
        if product is not None and product.handle not in {item.handle for item in out}:
            out.append(product)
    return out


def _height_in_line_range(height: float, product: ProductSpecs) -> bool:
    if height < STANDARD_MIN_IN or height > XL_MAX_IN:
        return False
    return is_xl_line(product) == needs_xl_line(height)


def _height_confirmation(products: list[ProductSpecs], users) -> Optional[str]:
    heights = [(label, height) for label, height, _ in users if height]
    if not heights or not products:
        return None
    if len(heights) == 1 and len(products) == 1:
        label, height = heights[0]
        return f"Height for {_person(label)}: {height_guidance(height, products[0])}"
    lines = []
    for product in products:
        xl = is_xl_line(product)
        fits = [f"{_person(label)} ({_feet(height)})" for label, height in heights if _height_in_line_range(height, product)]
        misses = [(label, height) for label, height in heights if not _height_in_line_range(height, product)]
        parts = [f"✅ {', '.join(fits)}"] if fits else []
        for label, height in misses:
            if STANDARD_MIN_IN <= height <= XL_MAX_IN:
                parts.append(f"{_person(label)} ({_feet(height)}) fits our {'standard line' if xl else 'XL line'} better")
            else:
                parts.append(f"{_person(label)} ({_feet(height)}) is outside our published ranges; a specialist should confirm")
        lines.append(f"- **{product.display_name}** ({'XL line, ' + XL_RANGE if xl else 'standard line, ' + STANDARD_RANGE}): "
                     + '; '.join(parts))
    return '\n'.join(lines)


def _weights_from_message(raw: str) -> list[float]:
    """Person weights the customer asks about; a box or chair weight is not one."""
    values = []
    for match in re.finditer(r'(?<!\d)(\d{2,3})\s*-?\s*(?:lbs?|pounds?)\b', raw or '', re.I):
        start = max(raw.rfind(mark, 0, match.start()) for mark in '.!?') + 1
        ends = [index for index in (raw.find(mark, match.end()) for mark in '.!?') if index >= 0]
        window = raw[max(start, match.start() - 45):min([*ends, match.end() + 20])]
        if re.search(r"\b(?:box|package|packag\w*|shipping|carton|crate|chair\s+(?:itself|weighs)|product\s+weighs|weighs\s+only|net\s+weight|gross)\b", window, re.I):
            continue
        values.append(float(match.group(1)))
    return values


def _reply(text, *, intent, tools, products=(), patch=None, human=True, extra_buttons=()):
    from sales_agent import SalesReply, QuickReply
    buttons = [QuickReply(label=label, payload=payload) for label, payload in extra_buttons]
    if human:
        buttons.append(QuickReply(label='Talk to a human', payload='human'))
    return SalesReply(reply=text, intent=intent, tools_used=list(tools),
                      products=[product.as_public_dict() for product in products],
                      quick_replies=buttons, prefs_patch=dict(patch or {}))


# --- Repair and new purchase in one conversation -------------------------

_REPAIR = re.compile(
    r"\b(?:broken|repairs?|service\s+(?:case|request)|technical\s+support|faults?|defect(?:ive)?|"
    r"stopped\s+working|not\s+working|(?:to|for|with)\s+(?:the\s+)?warranty)\b", re.I,
)
_PURCHASE = re.compile(
    r"\b(?:new[\s-]+purchase|new[\s-]+chair|second\s+chair|another\s+chair|buy(?:ing)?|(?:to|for|with)\s+(?:the\s+)?sales)\b", re.I,
)
_SEPARATE = re.compile(r"\b(?:separate(?:ly)?|route|split|two\s+(?:different\s+)?(?:requests|issues|things|tasks))\b", re.I)
_OLD_CHAIR = re.compile(
    r"\b(?:(?:the|my|our)\s+)?(?:old|current|owned|existing|broken)\s+(?:chair|model|one)\b|"
    r"\b(?:the\s+)?one\s+that(?:'s|\s+is)\s+broken\b|"
    r"\b(?:the\s+)?(?:prior|old|original|earlier|existing)\s+(?:repair|fault|service\s+(?:case|request))\b", re.I,
)
_NEW_CHAIR = re.compile(
    r"\b(?:the|my|our|a)\s+(?:(?:separate|second|another|new|office)\s+(?:chair|model|one)|new[\s-]+purchase)\b|"
    r"\b(?:the\s+)?one\s+i(?:'m|\s+am)\s+(?:buying|considering|shopping\s+for)\b", re.I,
)
_CONTACT_OR_STATUS = re.compile(
    r"\b(?:who|whom|where|what\s+(?:number|email)|how\s+(?:can|do|should)\s+i)\b.{0,65}"
    r"\b(?:contact|call|email|reach|support|service|help|send|submit|open)\b|"
    r"\b(?:right\s+team|contact\s+(?:number|details)|technical\s+support|repair\s+status|service\s+status|"
    r"(?:repair|service|technician)\s+(?:cost|fee|pricing)|"
    r"(?:scheduled|booked|approved)|status\s+of\s+(?:the\s+)?(?:repair|service\s+request))\b|"
    r"\bwhat\s+(?:number|email)\b", re.I,
)
_MODEL_DETAIL = re.compile(
    r"\b(?:price|pricing|how\s+much|costs?|stock|availability|available|colou?rs?|door(?:way)?|width|wide|"
    r"dimensions?|clearance|spec(?:s|ifications)?|capacity|weight|heat(?:ing|er)?|rollers?|zero\s+gravity|"
    r"airbags?|track|intensity|bluetooth|programs?)\b", re.I,
)
_TASK_BREAK = re.compile(
    r"(?<=[.!?;])\s+|\s*,?\s+\b(?:and|but|while|whereas)\s+"
    r"(?=(?:separately\b|(?:sales|warranty)\s+for\b|(?:i|we)\s+(?:also\s+)?(?:want|need|would\s+like|plan|own|have|am\s+buying)\b|"
    r"(?:i|we)'d\s+(?:also\s+)?like\b|"
    r"(?:buy|purchase|recommend|compare|fix|repair)\b|(?:the|my|our)\s+(?:old|new|broken|current|second)\b))", re.I,
)
_NEGATED_REPAIR = re.compile(
    r"\b(?:not|never|isn't|isnt|wasn't|wasnt)\s+(?:actually\s+|really\s+)?(?:broken|defective|damaged|faulty)\b|"
    r"\b(?:not|don't|don't\s+want\s+to)\s+(?:asking|ask|talk|talking|need\s+help|discuss)\s+(?:about|with)\s+"
    r"(?:(?:the|my|our)\s+)?(?:repair|old\s+chair|service\s+(?:case|request))\b", re.I,
)
_NEGATED_PURCHASE = re.compile(
    r"\b(?:not|no|don't|do\s+not|never|isn't)\s+(?:(?:asking|looking)\s+for\s+|"
    r"(?:want|need|plan)\s+)?(?:a\s+|the\s+|to\s+)?$", re.I,
)
_CORRECT_MODEL = re.compile(
    r"\b(?:correction|i\s+meant|actually|correct\s+model|not\s+the)\b", re.I,
)
_OLDER_STATED_CODE = re.compile(r"\b(OS|TP)[\s-]?(\d{3,5}[A-Za-z]{0,3})\b", re.I)
_FUTURE_CHAIR = re.compile(r"\b(?:next|future|prospective|new)\s+(?:chair|model|one)\b", re.I)
_CONDITIONAL_FAULT = re.compile(
    r"\b(?:if|in\s+case)\b[^.!?;]{0,90}\b(?:breaks?|fails?|stops?\s+working|needs?\s+(?:a\s+)?repairs?|"
    r"(?:is|becomes?)\s+(?:broken|defective|faulty))\b", re.I,
)
_ISSUE = re.compile(
    r"\b(?:its|the|my|our)\s+((?:[\w'’-]+\s+){0,4}?[\w'’-]+)\s+(stopped\s+working|(?:is|are)\s+(?:broken|not\s+working)|"
    r"(?:doesn't|don't|won't)\s+work)\b", re.I,
)


def _explicit_context_model(text: str) -> Optional[ProductSpecs]:
    """A fuzzy catalog hit alone cannot replace a saved conversation model."""
    from catalog_identity import generation_normalized
    from sales_agent import _guess_model_from_text
    from sales_intent_fallback import named_model_in_text

    # A live catalog may contain a model called One (or another ordinary
    # word). Remove complete referential phrases, not the name inside an
    # explicit statement like "the broken chair is actually Osaki One 4D".
    source = _OLD_CHAIR.sub(' ', _NEW_CHAIR.sub(' ', (text or '').replace('’', "'")))
    product = _guess_model_from_text(source)
    if product is None:
        return None

    def words(value: str) -> list[str]:
        return re.findall(r'[a-z0-9]+(?:\.\d+)?', generation_normalized(value).casefold())

    haystack = ' ' + ' '.join(words(source)) + ' '
    for name in (product.display_name, product.title, product.handle):
        terms = words(re.sub(r'\([^)]*\)', ' ', name))
        if len(terms) >= 2 and ' ' + ' '.join(terms) + ' ' in haystack:
            return product
        if terms and terms[0] in {'osaki', 'titan', 'amamedic', 'kanji'}:
            terms = terms[1:]
        # A brandless single common name is not sufficient, but the exact
        # designation "One 4D" (or "TP Epic") is deliberate.
        if len(terms) >= 2 and ' ' + ' '.join(terms) + ' ' in haystack:
            return product
    nickname = named_model_in_text(source)
    if nickname:
        named = _guess_model_from_text(nickname)
        if named is not None and named.handle == product.handle:
            return product
    return None


def _purchase_in(text: str) -> bool:
    return any(not _NEGATED_PURCHASE.search(text[:match.start()]) and not re.match(
        r"\s+(?:(?:an?|the)\s+)?(?:replacement\s+)?(?:remotes?|upholstery|controllers?|"
        r"replacement\s+parts?|parts?|accessor(?:y|ies)|extended\s+warranty|warranty\s+(?:plan|coverage))\b",
        text[match.end():], re.I,
    ) for match in _PURCHASE.finditer(text))


def prospective_repair_question(message: str) -> bool:
    """A hypothetical problem with an unpurchased chair is a policy question."""
    raw = (message or '').replace('’', "'")
    if not (_purchase_in(raw) or _FUTURE_CHAIR.search(raw)) or _OLD_CHAIR.search(raw):
        return False
    pieces = _TASK_BREAK.split(raw)
    return any(_CONDITIONAL_FAULT.search(piece) for piece in pieces) and not any(
        _REPAIR.search(piece) and not _NEGATED_REPAIR.search(piece) and not _CONDITIONAL_FAULT.search(piece)
        and not _purchase_in(piece) for piece in pieces
    )


def owner_task_clauses(message: str, data: dict) -> tuple[list[str], list[str]]:
    """Split genuinely independent tasks before reading models or fit limits."""
    from sales_policy import reports_owned_chair_problem

    raw = (message or '').replace('’', "'")
    pieces = [piece.strip(' ,;') for piece in _TASK_BREAK.split(raw) if piece.strip(' ,;')]
    repair = [piece for piece in pieces if (_REPAIR.search(piece) or _OLD_CHAIR.search(piece))
              and not _NEGATED_REPAIR.search(piece) and not _purchase_in(piece)
              and (not _CONDITIONAL_FAULT.search(piece) or _OLD_CHAIR.search(piece))]
    shopping = [piece for piece in pieces if _purchase_in(piece)
                and not _NEGATED_REPAIR.search(piece) and not _OLD_CHAIR.search(piece)]
    owned_model_fault = any(
        re.search(r"\b(?:my|our)\b", piece, re.I) and not re.search(r"\b(?:if|would|might|could)\b", piece, re.I)
        and _explicit_context_model(piece) is not None for piece in repair
    )
    owner_evidence = data.get('owner_service_case') or owned_model_fault or any(reports_owned_chair_problem(piece) for piece in repair) or _OLD_CHAIR.search(raw) or re.search(
        r"\b(?:already\s+own|my\s+(?:chair|model)|(?:stopped|is\s+not)\s+working|needs?\s+(?:a\s+)?repair)\b", raw, re.I)
    if repair and shopping and (owner_evidence or _SEPARATE.search(raw)):
        return (repair, shopping)
    distinct_repair = _OLD_CHAIR.search(raw) or re.search(r"\b(?:old|prior|earlier|original|broken)\s+[\w-]+\s+(?:repair|is\s+a\s+service\s+case)\b", raw, re.I)
    if _SEPARATE.search(raw) and data.get('owner_service_case') and distinct_repair and _REPAIR.search(raw) and _purchase_in(raw):
        # E.g. "route my repair to Warranty and my new-purchase question to
        # Sales" names neither product; use the distinct saved referents.
        return ([piece for piece in pieces if _OLD_CHAIR.search(piece)] or [raw], shopping or [raw])
    return ([], [])


def _repair_clauses(message: str) -> list[str]:
    raw = (message or '').replace('’', "'")
    pieces = _TASK_BREAK.split(raw)
    relevant = [piece for piece in pieces if (_REPAIR.search(piece) or _OLD_CHAIR.search(piece))
                and not _NEGATED_REPAIR.search(piece) and not _purchase_in(piece)]
    return relevant or [piece for piece in pieces if not _NEGATED_REPAIR.search(piece) and not _purchase_in(piece)]


def owner_case_patch(message: str, data: dict, *, clauses: Optional[list[str]] = None) -> dict:
    """Remember only the existing chair; a model switch never inherits its symptom."""
    from sales_policy import reports_owned_chair_problem
    case = dict(data.get('owner_service_case') or {})
    pieces = _repair_clauses(message) if clauses is None else clauses
    product = next((found for piece in pieces if not _purchase_in(piece)
                    and (found := _explicit_context_model(piece)) is not None), None)
    # Service still accepts retired models that no longer appear in the
    # storefront. Retain a single explicit model code as stated, not a fuzzy
    # match to another generation (e.g. OS-4000T is not OS-4000XT).
    codes = {(m[1].upper(), m[2].upper()) for piece in pieces if not _purchase_in(piece)
             for m in _OLDER_STATED_CODE.finditer(piece)}
    model = product.display_name if product is not None else None
    if model is None and len(codes) == 1:
        prefix, code = next(iter(codes))
        model = f"{'Osaki' if prefix == 'OS' else 'Titan'} {prefix}-{code}"
    if model is not None:
        if case.get('model') and case['model'] != model:
            case['issue'] = None  # Session dictionaries are shallow-merged.
        case['model'] = model
    for piece in pieces:
        if _purchase_in(piece):
            continue
        issue = _ISSUE.search(piece)
        if issue:
            part = re.sub(r"^.*['’]s\s+", '', issue.group(1))
            description = f"{part} {issue.group(2)}".strip()
            # A generic restatement or model correction must not replace a
            # known component with "chair is broken" or its full model name.
            if case.get('issue') and re.fullmatch(r"(?:current\s+|old\s+|broken\s+|new\s+)?(?:chair|model|one)", part, re.I):
                continue
            if product is not None and _explicit_context_model(part) is not None:
                continue
            case['issue'] = description[:80]
            break
    if not case and any(reports_owned_chair_problem(piece) for piece in pieces):
        case = {'model': None}  # A real repair, but the model is still unknown.
    return {'owner_service_case': case} if case else {}


def is_owner_repair_followup(message: str, data: dict) -> bool:
    """Recognize a return to a known repair, not a new recommendation or order."""
    raw = (message or '').replace('’', "'")
    case = data.get('owner_service_case') or {}
    if not case or _purchase_in(raw) or prospective_repair_question(raw):
        return False
    explicit = bool(_OLD_CHAIR.search(raw) or re.search(r"\b(?:the|my|our)\s+(?:repair|service\s+(?:case|request))\b", raw, re.I))
    correction = bool(_CORRECT_MODEL.search(raw) and _OLD_CHAIR.search(raw))
    repair_confirmation = bool(re.search(r"\b(?:back|return)\s+to\b|\b(?:still|again)\b", raw, re.I)
                               and (_REPAIR.search(raw) or explicit))
    return correction or (explicit and (repair_confirmation or bool(_CONTACT_OR_STATUS.search(raw)))) or (
        data.get('technical_support_topic') == 'warranty_redirect' and bool(_CONTACT_OR_STATUS.search(raw))
        and not _MODEL_DETAIL.search(raw)
    )


def owner_purchase_question(message: str, data: dict) -> Optional[str]:
    """Bind an explicit new-chair reference to the saved purchase model only."""
    from sales_agent import _guess_model_from_text
    raw = (message or '').replace('’', "'")
    if not data.get('owner_service_case') or not _NEW_CHAIR.search(raw) or not _MODEL_DETAIL.search(raw):
        return None
    if _NEGATED_REPAIR.search(raw):
        # A customer correcting the route explicitly is opting out of the
        # old task, not asking us to restate or reopen it.
        raw = ' '.join(piece for piece in _TASK_BREAK.split(raw) if not _NEGATED_REPAIR.search(piece))
    if _OLD_CHAIR.search(raw) or _REPAIR.search(raw) or _explicit_context_model(raw) is not None:
        return None
    # Do not reinterpret delivery terms, service fees or refunds as a
    # product-price follow-up merely because a new chair was mentioned.
    if re.search(r"\b(?:warranty|coverage|shipping|delivery|refunds?|return|installation|technician|"
                 r"financ\w*|order\s+(?:status|number)|price\s+match)\b", raw, re.I):
        return None
    product = _guess_model_from_text(str(data.get('owner_purchase_model') or ''))
    if product is None:
        fallback = _guess_model_from_text(str(data.get('pending_primary') or ''))
        if fallback and fallback.display_name != (data.get('owner_service_case') or {}).get('model'):
            product = fallback
    return _NEW_CHAIR.sub(product.display_name, raw, count=1) if product else None


def ambiguous_owner_cost_reply(message: str, data: dict):
    """Both repair costs and new-chair price are plausible after a lane switch."""
    from sales_agent import _guess_model_from_text
    old = (data.get('owner_service_case') or {}).get('model')
    product = _guess_model_from_text(str(data.get('owner_purchase_model') or ''))
    if (not old or product is None or data.get('technical_support_topic') != 'warranty_redirect'
            or not re.fullmatch(r"\s*(?:and\s+)?(?:how\s+much(?:\s+is\s+it|\s+does\s+it\s+cost)?|"
                                r"what\s+(?:does\s+it\s+cost|is\s+the\s+price))[?.!\s]*", message, re.I)):
        return None
    return _reply(
        f"Do you mean the **repair cost for {old}**, or the **purchase price for {product.display_name}**? "
        "Warranty needs to confirm repair pricing; I can look up the catalog price for the new chair.",
        intent='unclear', tools=['conversation.owner_cost_clarify'], human=False,
    )


def owner_purchase_model_patch(message: str, data: dict, result) -> dict:
    """Record an unambiguous shopping model without consuming a repair mention."""
    from sales_agent import _guess_model_from_text
    if (not data.get('owner_service_case') or 'conversation.dual_task' in result.tools_used
            or result.intent not in {'specs', 'price', 'stock', 'recommend'} or _OLD_CHAIR.search(message or '')):
        return {}
    named = _explicit_context_model(message)
    if named is None and not data.get('owner_purchase_model') and result.intent == 'recommend' and result.products:
        named = _guess_model_from_text(str((result.prefs_patch or {}).get('pending_primary') or ''))
    if named is None:
        return {}
    if named.display_name == (data.get('owner_service_case') or {}).get('model') and not _purchase_in(message):
        return {}
    return {'owner_purchase_model': named.handle}


def dual_task_reply(message: str, data: dict):
    raw = message or ''
    repair, shopping = owner_task_clauses(raw, data)
    if not repair or not shopping:
        return None
    from sales_agent import _guess_model_from_text
    patch = owner_case_patch(raw, data, clauses=repair)
    case = patch.get('owner_service_case') or {}
    repair_model = case.get('model')
    explicit_purchase = next((found for piece in shopping if not _OLD_CHAIR.search(piece)
                              and not (_REPAIR.search(piece) and _SEPARATE.search(piece))
                              and (found := _explicit_context_model(piece)) is not None), None)
    saved_purchase = _guess_model_from_text(str(data.get('owner_purchase_model') or ''))
    fallback = _guess_model_from_text(str(data.get('pending_primary') or ''))
    if re.search(r"\b(?:buy|purchase|another)\b.{0,50}\b(?:same|identical)\s+(?:chair|model|one)\b", raw, re.I):
        explicit_purchase = explicit_purchase or _guess_model_from_text(str(repair_model or ''))
    product = explicit_purchase or saved_purchase or (fallback if fallback and fallback.display_name != repair_model else None)
    purchase_model = product.display_name if product else None
    if product is not None:
        patch['owner_purchase_model'] = product.handle
    rec = dict(data.get('recommend_prefs') or {})
    repair_label = repair_model or 'your current chair'
    issue = f" ({case['issue']})" if case.get('issue') else ''
    lines = [
        "I'll keep these as two separate requests:",
        "",
        f"**1. Repair — {repair_label}{issue} → Warranty Department**",
        f"Email **{WARRANTY_EMAIL}** with the model, a description of the issue, and your order number or proof of purchase, "
        f"or call **{WARRANTY_PHONE}**. You can also open a service request at {WARRANTY_PORTAL}. "
        "I have not submitted a service request, booked a callback, or approved a repair or replacement.",
        "",
        f"**2. New purchase — {purchase_model or 'your new chair'} → Sales**",
    ]
    saved = []
    if rec.get('budget_max_usd'):
        saved.append(f"under {_money(float(rec['budget_max_usd']))}")
    if rec.get('doorway_in') and rec['doorway_in'] != 'skip':
        saved.append(f"{rec['doorway_in']}\" doorway ({rec.get('doorway_fit') or 'assembled'})")
    if rec.get('weight_lb'):
        saved.append(f"{rec['weight_lb']} lb user")
    keep = f" with your saved requirements ({', '.join(saved)})" if saved else ''
    lines.append(f"I can keep helping with the new chair right here{keep}. Sales is also at **{SALES_PHONE}**. "
                 "The repair stays with Warranty and won't change the new-purchase recommendation.")
    buttons = [('Keep shopping', 'recommend')]
    return _reply('\n'.join(lines), intent='recommend', tools=['conversation.dual_task', 'routing.warranty', 'routing.sales'],
                  patch=patch,
                  extra_buttons=buttons)


def new_purchase_note(message: str, data: dict) -> Optional[str]:
    """One-time reminder that an open repair is not mixed into a new purchase."""
    case = data.get('owner_service_case') or {}
    if not case or data.get('owner_case_acknowledged'):
        return None
    if not _purchase_in((message or '').replace('’', "'")) or not re.search(r"\b(?:separately|second\s+chair|another\s+chair|new[\s-]+purchase|also\s+(?:want|like)\s+to\s+buy)\b", message or '', re.I):
        return None
    model = f"**{case['model']}**" if case.get('model') else 'current chair'
    return (f"Your {model} repair stays with Warranty (**{WARRANTY_EMAIL}**, {WARRANTY_PHONE}). "
            "This is a separate new purchase, so I'll help with it here.")


# --- Named-model capacity / fit -------------------------------------------

_CAPACITY_ASK = re.compile(
    r"\b(?:supports?|hold|holds|handle|handles|accommodate|rated\s+for|capacity|weight\s+limit|"
    r"max(?:imum)?\s+(?:user\s+)?weight|work\s+for\s+(?:both|them|my|a\s+\d))\b", re.I,
)
_KEEP_MODEL = re.compile(
    r"\b(?:keep|stay|continue|stick)\s+(?:discussing|talking\s+about|on|with|focused\s+on)\b|\bsame\s+(?:model|chair)\b", re.I,
)


def _capacity_lines(product: ProductSpecs, rec: dict, explicit: list[float]) -> list[str]:
    fit = lookup_fit_spec(product.display_name)
    cap = getattr(fit, 'max_user_lb', None) if fit else None
    lines = []
    if cap is None:
        lines.append(f"**{product.display_name}** has no published maximum user weight in our verified specs, so I can't confirm "
                     "it supports that weight. Sales needs to confirm it.")
        return lines
    lines.append(f"**{product.display_name}** lists a **{cap:g} lb maximum user weight**.")
    checks = [(f"a {weight:g} lb user", weight) for weight in explicit]
    if not checks:
        checks = [(f"{_person(label)} ({weight:g} lb)", weight) for label, _, weight in _saved_users(rec) if weight]
    for label, weight in checks:
        if weight <= cap:
            lines.append(f"- {label[0].upper() + label[1:]}: ✅ within the published limit ({cap - weight:g} lb to spare).")
        else:
            lines.append(f"- {label[0].upper() + label[1:]}: ❌ above the published limit by {weight - cap:g} lb.")
    return lines


def _doorway_budget_lines(product: ProductSpecs, rec: dict) -> list[str]:
    lines = []
    door = rec.get('doorway_in')
    if door and door != 'skip':
        mode = rec.get('doorway_fit') or 'assembled'
        fit = lookup_fit_spec(product.display_name)
        width = fit.doorway_for_mode(mode) if fit else None
        label = 'with disassembly' if mode == 'disassembled' else 'assembled'
        if width is None:
            lines.append(f"- Your {door}\" doorway: the {label} width is not published, so it needs confirmation.")
        elif width <= float(door):
            lines.append(f"- Your {door}\" doorway: ✅ it lists {width:g}\" {label}.")
        else:
            lines.append(f"- Your {door}\" doorway: ❌ it lists {width:g}\" {label}, wider than your doorway.")
    cap = rec.get('budget_max_usd')
    if cap and product.price_usd is not None:
        ok = product.price_usd <= float(cap)
        lines.append(f"- Budget: {'✅' if ok else '❌'} {_money(product.price_usd)} vs your {_money(float(cap))} maximum.")
    return lines


def named_model_fit_reply(message: str, data: dict):
    from sales_agent import _guess_model_from_text, _focus_product_from_prefs
    raw = message or ''
    keep = bool(_KEEP_MODEL.search(raw))
    product = _guess_model_from_text(raw)
    if product is None and keep and re.search(r'\bsame\s+(?:model|chair)\b', raw, re.I):
        product = _focus_product_from_prefs(data)
    if product is None:
        return None
    explicit = _weights_from_message(raw)
    capacity_ask = bool(_CAPACITY_ASK.search(raw)) and not re.search(r'\b(?:door(?:way)?|opening)\b', raw, re.I)
    if not (capacity_ask or keep):
        return None
    if keep and not capacity_ask and not (explicit or re.search(r"\b\d\s*(?:'|foot|feet|ft)\b", raw, re.I)):
        return None
    rec = dict(data.get('recommend_prefs') or {})
    if keep or re.search(r'\b(?:i|i\s+am|i\'m|my)\b', raw, re.I) and not capacity_ask:
        rec = extract_constraints(raw, rec)
        explicit = []
    lines = []
    if keep:
        lines.append(f"I'll stay on **{product.display_name}** — no new recommendation.")
    lines.extend(_capacity_lines(product, rec, explicit))
    lines.extend(_doorway_budget_lines(product, rec))
    height = _height_confirmation([product], _saved_users(rec))
    if height:
        lines.append(height)
    patch = {'pending_primary': product.display_name}
    if keep:
        patch['recommend_prefs'] = rec
    return _reply('\n'.join(lines), intent='specs', tools=['catalog.weight_capacity', 'cases.fit_gates'], products=[product],
                  patch=patch, extra_buttons=[('Check the price', f'price:{product.handle}')])


_WIDTH_COMPARE = re.compile(
    r"\b(narrower|wider|smaller|bigger|larger|less|more)\s+than\s+(?:a\s+|my\s+|the\s+)?(\d{2}(?:\.\d+)?)\s*(?:\"|in\b|inch(?:es)?\b)", re.I,
)


def named_width_reply(message: str, data: dict):
    """Answer "is <model> narrower than N inches" from the published width only."""
    from sales_agent import _guess_model_from_text
    raw = message or ''
    match = _WIDTH_COMPARE.search(raw)
    product = _guess_model_from_text(raw) if match else None
    if product is None:
        return None
    fit = lookup_fit_spec(product.display_name)
    mode = 'disassembled' if re.search(r'\bdisassembl', raw, re.I) else 'assembled'
    width = fit.doorway_for_mode(mode) if fit else None
    limit = float(match.group(2))
    if width is None:
        text = f"The {mode} doorway width for **{product.display_name}** is not published, so I can't compare it with {limit:g}\"."
    else:
        narrower = width < limit
        if width == limit:
            verdict = f"it is exactly {limit:g}\", with no margin"
        elif narrower:
            verdict = f"yes, it is {limit - width:g}\" narrower than {limit:g}\""
        else:
            verdict = f"no, it is {width - limit:g}\" wider than {limit:g}\""
        text = f"**{product.display_name}** lists **{width:g}\" {mode}**, so {verdict}."
        other = fit.doorway_for_mode('disassembled' if mode == 'assembled' else 'assembled')
        if other is not None and other != width:
            text += f" Its {'disassembled' if mode == 'assembled' else 'assembled'} width is listed at {other:g}\"."
    text += "\n\nA published width is only a width check; turns, stairs and packaging still need confirmation before delivery."
    return _reply(text, intent='specs', tools=['catalog.doorway_fit'], products=[product],
                  patch={'pending_primary': product.display_name})


_CAPACITY_LIST = re.compile(
    r"\b(?:which|what|list|show)\b.{0,40}\b(?:chairs|models)\b.{0,60}"
    r"\b(?:capacity|support|supports|hold|holds|rated|weight\s+limit)\b", re.I,
)


def capacity_list_reply(message: str, data: dict):
    from sales_agent import _guess_model_from_text
    raw = message or ''
    if not _CAPACITY_LIST.search(raw) or _guess_model_from_text(raw) is not None:
        return None
    if re.search(r'\b(?:these|those|the)\s+two\b|\bof\s+(?:these|those|your|the)\s+(?:options|picks)\b', raw, re.I):
        return None
    rec = dict(data.get('recommend_prefs') or {})
    explicit = _weights_from_message(raw)
    need = max(explicit) if explicit else float(rec['weight_lb']) if rec.get('weight_lb') else None
    if need is None:
        return None
    rows = []
    for product in list_active_products():
        fit = lookup_fit_spec(product.display_name)
        cap = getattr(fit, 'max_user_lb', None) if fit else None
        if cap is not None and cap >= need and product.price_usd is not None:
            rows.append((product.price_usd, product.display_name, product, cap))
    rows.sort(key=lambda row: row[:2])
    if not rows:
        return _reply(f"No active chair publishes a maximum user weight of at least **{need:g} lb**. Sales can check heavier-duty options.",
                      intent='specs', tools=['catalog.weight_capacity'])
    limits = {key: value for key, value in rec.items() if key in {'budget_max_usd', 'doorway_in', 'doorway_fit'}}
    lines = [f"Active chairs with a published maximum user weight of **at least {need:g} lb** ({len(rows)} total):"]
    passing = []
    for _, _, product, cap in rows[:10]:
        fit = lookup_fit_spec(product.display_name)
        door = fit.door_asm_in if fit else None
        issues = product_constraint_issues(product, limits) if limits else []
        status = ''
        if limits:
            status = ' · ✅ fits your saved budget and doorway' if not issues else ' · ❌ misses your saved ' + ' and '.join(
                {'budget': 'budget', 'doorway': 'doorway'}.get(issue, issue) for issue in issues)
        if limits and not issues:
            passing.append(product)
        door_bit = f" · {door:g}\" assembled" if door is not None else ' · assembled width not published'
        lines.append(f"- **{product.display_name}** — {_money(product.price_usd)} · {cap:g} lb{door_bit}{status}")
    if len(rows) > 10:
        lines.append(f"…and {len(rows) - 10} more. Tell me a price range to narrow it down.")
    lines.append("Chairs without a published capacity are left out rather than guessed.")
    return _reply('\n'.join(lines), intent='specs', tools=['catalog.weight_capacity', 'cases.fit_gates'],
                  products=passing or [row[2] for row in rows[:3]])


# --- Current shortlist ------------------------------------------------------

_SHORTLIST_REF = re.compile(
    r"\b(?:which|any|do\s+any|does\s+any|how\s+many)\b.{0,25}\b(?:of\s+)?(?:your|these|those|the|my)\s+"
    r"(?:options|picks|recommendations|suggestions|choices|chairs|ones|shortlist)\b", re.I,
)


def _roller_count(value: str) -> Optional[int]:
    match = re.search(r'\b(\d+)\b', value or '')
    if match:
        return int(match.group(1))
    word = re.search(r'\b(' + '|'.join(_NUMBER_WORDS) + r')\b', (value or '').lower())
    return _NUMBER_WORDS[word.group(1)] if word else None


def shortlist_feature_reply(message: str, data: dict):
    from sales_agent import _SPEC_QUESTION_PATTERNS, _spec_customer_value, _spec_is_published
    raw = message or ''
    picks = _shortlist(data)
    if len(picks) < 2 or not _SHORTLIST_REF.search(raw):
        return None
    facets = [(key, label) for pattern, key, label in _SPEC_QUESTION_PATTERNS if pattern.search(raw)]
    if not facets:
        return None
    key, label = facets[0]
    label = 'Foot rollers' if key == 'foot_roller' else label
    lines = [f"{label} across your current shortlist:"]
    having = []
    for product in picks:
        value = str(product.as_public_dict()['specs'].get(key) or '')
        if _spec_is_published(value):
            having.append(product)
            lines.append(f"- **{product.display_name}**: ✅ {_spec_customer_value(value) or value}")
        else:
            lines.append(f"- **{product.display_name}**: not listed in the published specs")
    if len(having) == len(picks):
        lines.append(f"\n{'Both' if len(picks) == 2 else f'All {len(picks)}'} options list {label.lower()}.")
    elif having:
        lines.append(f"\nOnly {' and '.join(f'**{p.display_name}**' for p in having)} list{'s' if len(having) == 1 else ''} {label.lower()}.")
    else:
        lines.append(f"\nNone of these options lists {label.lower()}; I won't assume it.")
    if key == 'foot_roller' and re.search(r'\bcalf\b', raw, re.I):
        lines.append('A separate calf roller is not established by a foot-roller listing.')
    return _reply('\n'.join(lines), intent='specs', tools=['catalog.specs', 'conversation.shortlist'], products=having or picks)


_ONE_PICK = re.compile(
    r"\b(?:show|give|pick|recommend|suggest|choose)\b.{0,12}\b(?:me\s+)?(?:one|a\s+single|the\s+best|your\s+top|the\s+most)\b.{0,30}"
    r"\b(?:option|chair|pick|model|choice)\b|\bwhich\s+fit\s+(?:information|details|facts|checks?)\b", re.I,
)


def single_pick_reply(message: str, data: dict, *, domain: str):
    raw = message or ''
    rec = dict(data.get('recommend_prefs') or {})
    if not _ONE_PICK.search(raw) or not rec:
        return None
    from sales_agent import _guess_model_from_text, _verified_catalog_picks, _fmt_price
    from sales_cta import product_page_url
    from sales_compare import request_from_prefs
    if _guess_model_from_text(raw) is not None:
        return None
    candidates = [product for product in _shortlist(data) if not product_constraint_issues(product, rec)]
    if not candidates:
        candidates = [product for product, _ in _verified_catalog_picks(rec, request=request_from_prefs(rec), domain=domain)]
    users = _relabel_single_user(_saved_users(rec), raw)
    owner = _owner_phrase(users)
    if not candidates:
        return _reply(f"I can't show a chair that passes every published limit in {owner} saved requirements. "
                      "Sales can check options outside the published specs.", intent='recommend', tools=['cases.fit_gates'])
    product = candidates[0]
    fit = lookup_fit_spec(product.display_name)
    url = product_page_url(domain, product.handle)
    lines = [f"One option using {owner} requirements: **{product.display_name}** — {_fmt_price(product.price_usd)}", url, "",
             "Checked against published specs:"]
    if rec.get('budget_max_usd'):
        lines.append(f"- ✅ Price {_fmt_price(product.price_usd)} is within the {_money(float(rec['budget_max_usd']))} budget")
    for label, _, weight in users:
        if weight and fit and fit.max_user_lb:
            lines.append(f"- ✅ {fit.max_user_lb:g} lb maximum user weight covers {_person(label)} ({weight:g} lb)")
    door = rec.get('doorway_in')
    if door and door != 'skip' and fit:
        mode = rec.get('doorway_fit') or 'assembled'
        width = fit.doorway_for_mode(mode)
        lines.append(f"- ✅ {width:g}\" {'with disassembly' if mode == 'disassembled' else 'assembled'} fits the {door}\" doorway")
    heights = [(label, height) for label, height, _ in users if height]
    unconfirmed = []
    for label, height in heights:
        if _height_in_line_range(height, product):
            line_range = XL_RANGE if is_xl_line(product) else STANDARD_RANGE
            lines.append(f"- ✅ {_person(label).capitalize()} ({_feet(height)}) is within the {line_range} design range of this chair's line")
        else:
            unconfirmed.append(f"- Height for {_person(label)}: {height_guidance(height, product)}")
    lines.extend(["", "Still needs confirmation:", *unconfirmed])
    if rec.get('intensity') == 'Gentle' or re.search(r'\bgentle\b', str(rec.get('goal') or '') + raw, re.I):
        lines.append("- Gentle feel: pressure range and intensity levels are not verified; a showroom trial or specialist can confirm")
    lines.append("- Delivery path: turns, stairs and packaging beyond the doorway width")
    patch = {'pending_primary': product.display_name, 'pending_product_url': url}
    return _reply('\n'.join(lines), intent='recommend', tools=['cases.fit_gates', 'catalog.weight_capacity', 'cta.product_url'],
                  products=[product], patch=patch, extra_buttons=[('Email me this pick', 'lead:save_pick')])


_HEIGHT_FIT = re.compile(
    r"\bheights?\b.{0,40}\b(?:fit|range|limit|confirm\w*|published|verified|support\w*)\b|"
    r"\b(?:fit|range|limit|confirm\w*)\b.{0,30}\bheights?\b", re.I,
)


def saved_users_height_reply(message: str, data: dict):
    from sales_agent import _guess_model_from_text, _focus_product_from_prefs
    from sales_context import _pair_from_prefs
    raw = message or ''
    if not _HEIGHT_FIT.search(raw) or _guess_model_from_text(raw) is not None:
        return None
    rec = dict(data.get('recommend_prefs') or {})
    users = [user for user in _saved_users(rec) if user[1]]
    if len(users) < 2 and not re.search(r'\b(?:both|each|all|every|users?)\b', raw, re.I):
        return None
    products = _shortlist(data)
    if not products:
        pair = _pair_from_prefs(data)
        products = list(pair) if pair else [p for p in [_focus_product_from_prefs(data)] if p]
    if not products or not users:
        return None
    line = _height_confirmation(products, users)
    text = ("A height range for each exact model is not published, so here is how each user compares with our line-level "
            f"design ranges ({STANDARD_RANGE} standard, {XL_RANGE} XL):\n" + line +
            "\n\nThat is a design-range check, not a confirmed fitting: Sales can confirm the fit for both users before you order, "
            "and the showroom is the best place to check the feel together.")
    return _reply(text, intent='specs', tools=['catalog.height_fit', 'conversation.saved_requirements'], products=products)


# --- General feature concepts ----------------------------------------------

_CONCEPTS: tuple[tuple[str, str, re.Pattern[str], str], ...] = (
    ('zero_gravity', 'Zero gravity', re.compile(r"\bzero[\s-]*gravity\b|\brecline(?:d)?\s+position\b|\breclin(?:e|ing)\b", re.I),
     "**Zero gravity** is a reclined position: the chair tilts back and raises your legs to about heart level, so your weight "
     "spreads across the backrest and seat instead of pressing down on your spine and hips. People use it to relax, and it lets "
     "the back rollers press into you more evenly. It is about posture, not the rollers. The number of stages varies by model."),
    ('track', 'Massage track (S / L / SL)', re.compile(r"\b(?:sl|s|l)[\s-]?tracks?\b|\btrack\s+type\b|\broller\s+track\b", re.I),
     "The **track** is the rail the back rollers travel along, so it decides *where on your body the rollers reach*. "
     "An **S-Track** follows the S-curve of the spine from neck to lower back. An **L-Track** continues under the seat to the "
     "glutes and upper hamstrings. An **SL-Track** combines both: the spine curve plus the extension under the seat."),
    ('mechanism', '2D / 3D / 4D mechanism', re.compile(r"\b[2345]\s*d\b(?!\s*(?:chair|model))|\bmechanisms?\b", re.I),
     "The **2D / 3D / 4D** label describes how the rollers move. **2D** rollers move up-down and side-to-side. **3D** adds "
     "depth: how far the rollers push out toward your body. **4D** adds control of speed and rhythm, so strokes vary like hands. "
     "Labels beyond 4D are manufacturer terms, so check that model's description. A higher number does not by itself mean stronger pressure."),
    ('airbag', 'Airbags', re.compile(r"\bair\s*bags?\b|\bair\s+(?:compression|massage)\b", re.I),
     "**Airbags** inflate and deflate to squeeze and release areas like the shoulders, arms, hips, calves and feet "
     "(compression massage). Their count and placement vary by model. More airbags does not by itself mean stronger pressure."),
    ('heating', 'Heating', re.compile(r"\bheat(?:ing|ed)?\b", re.I),
     "**Heating** gently warms specific areas, most often the lower back and sometimes the calves or seat, to help muscles relax. "
     "The heated areas vary by model, and it is comfort warmth, not a medical heat treatment."),
    ('rollers', 'Massage rollers', re.compile(r"(?<!foot\s)(?<!calf\s)\brollers?\b(?!\s+track)", re.I),
     "The **massage rollers** are the motorized heads inside the backrest that knead, tap and roll along your back as they travel the track. "
     "How they move is described by the 2D/3D/4D mechanism. Roller count and shape vary by model."),
    ('foot_roller', 'Foot rollers', re.compile(r"\bfoot\s*rollers?\b|\breflexology\b", re.I),
     "**Foot rollers** are spinning rollers under the soles that knead the bottoms of your feet, reflexology style. "
     "Some chairs pair them with calf airbags; the number of rollers varies by model."),
    ('body_scan', 'Body scanning', re.compile(r"\bbody[\s-]*scann?(?:ing|er)?\b", re.I),
     "**Body scanning** uses the chair's sensors to estimate landmarks like your shoulder height so the program can position the rollers. "
     "The method and adjustment range vary by model; it is not a medical examination."),
    ('auto_programs', 'Auto programs', re.compile(r"\bauto(?:matic)?\s+programs?\b|\bpreset\s+programs?\b", re.I),
     "**Auto programs** are preset routines that combine roller strokes, airbags and recline for a goal like relaxing or stretching, "
     "so you don't have to set each part yourself. The number and names vary by model."),
)
_CONCEPT_CUE = re.compile(
    r"\b(?:mean|meaning|means|explain|concept|general\s+term|in\s+general|generally|what\s+(?:is|are|does|do)|"
    r"how\s+(?:does|do)\b.{0,40}\bwork|what\s+is\s+.{0,40}\bfor|purpose|same\s+(?:thing|as)|difference|different|vs\.?|versus)\b", re.I,
)
_DIFFERENCE_FOLLOWUP = re.compile(r"\b(?:explain|tell\s+me|what(?:'s|\s+is))\b.{0,15}\b(?:the|that|this)\s+difference\b|\bhow\s+(?:are|do)\s+they\s+differ\b", re.I)
_PAIR_CONTRAST = {
    frozenset({'zero_gravity', 'track'}):
        "**Zero gravity** is how the chair *positions you* (recline and leg lift). The **track** is where the rollers *travel* "
        "along your back. They are separate features, and each model lists them separately.",
    frozenset({'mechanism', 'track'}):
        "The **track** decides *where* the rollers travel. The **2D/3D/4D mechanism** decides *how* the rollers move along it.",
    frozenset({'airbag', 'rollers'}):
        "No: **rollers** press and knead mechanically, while **airbags** squeeze with air pressure. Most chairs combine both.",
    frozenset({'airbag', 'mechanism'}):
        "The **mechanism** describes the rollers' motion; **airbags** squeeze with air pressure. Many chairs combine both.",
}


def concept_reply(message: str, data: dict):
    from sales_agent import _guess_model_from_text
    raw = message or ''
    found = [(key, title, copy) for key, title, pattern, copy in _CONCEPTS if pattern.search(raw)]
    generic = bool(re.search(r"\b(?:general\s+term|in\s+general|not\s+(?:a|any)\s+(?:particular|specific)\s+(?:model|chair)|the\s+concept)\b", raw, re.I))
    if not found and _DIFFERENCE_FOLLOWUP.search(raw) and len(data.get('last_concepts') or []) >= 2:
        keys = list(data['last_concepts'])[:2]
        found = [(key, title, copy) for key, title, _, copy in _CONCEPTS if key in keys]
    if not found:
        return None
    if not (_CONCEPT_CUE.search(raw) or generic or _DIFFERENCE_FOLLOWUP.search(raw)):
        return None
    if not generic and _guess_model_from_text(raw) is not None:
        return None
    if re.search(r"\b(?:does|do|has|have|is\s+there)\b.{0,40}\b(?:it|this|that|the\s+chair)\b.{0,15}\b(?:have|has|come\s+with)\b", raw, re.I):
        return None
    if len(found) == 1 and found[0][0] == 'mechanism' and not re.search(r"\bmechanism|\b[2345]\s*d\b.{0,30}\b(?:mean|stand|difference)", raw, re.I):
        return None
    keys = [key for key, _, _ in found][:3]
    lines = []
    if len(found) >= 2 and re.search(r"\bsame\b", raw, re.I):
        lines.append("No, they are different features.")
    elif len(found) >= 2:
        lines.append("Here is the difference:")
    lines.extend(copy for _, _, copy in found[:3])
    contrast = _PAIR_CONTRAST.get(frozenset(keys[:2])) if len(keys) >= 2 else None
    if contrast:
        lines.append(contrast)
    lines.append("This is the general meaning. Name a model and I'll show its published settings.")
    return _reply('\n\n'.join(lines), intent='specs', tools=['catalog.feature_definition'],
                  patch={'last_concepts': keys}, human=True, extra_buttons=[('See all models', 'list')])


_LABEL_LOCATION = re.compile(
    r"\b(?:where|how)\b.{0,40}\b(?:find|look|locate|see|check|read|is|are)\b.{0,30}"
    r"\b(?:serial\b|model\s*(?:number|no\b|#|label|sticker|tag|plate)\b)", re.I,
)


def label_location_reply(message, data):
    """Owner help for reading the model/serial label; this copy matches the Warranty chat."""
    if not _LABEL_LOCATION.search(message or ''):
        return None
    return _reply(
        "Look for the model name on the **serial-number sticker on the chair base**, but check it only if it is visible and easy to reach. "
        "Do not lift, tip, or open the chair to find it. If the label is inaccessible or unreadable, our Warranty Department can help identify "
        f"your chair: email **{WARRANTY_EMAIL}**, call **{WARRANTY_PHONE}**, or open a service request at {WARRANTY_PORTAL}.",
        intent='specs', tools=['owner_help.label_location'],
    )


_USER_HEIGHT_LIMIT = re.compile(
    r"\b(?:height\s+(?:max(?:imum)?|limit|range)|"
    r"max(?:imum)?\s+(?:user\s+)?height|"
    r"good\s+(?:up\s+)?(?:to|till|until)|"
    r"up\s+to\s+what\s+height|height\s+maximum|"
    r"how\s+tall\s+(?:can|should|of\s+(?:a\s+)?(?:person|user|someone))|"
    r"what\s+height\s+(?:can|should|is\s+(?:this|it|the\s+chair)\s+good))\b",
    re.I,
)
_SOURCE_ASK = re.compile(
    r"\bwhere\b.{0,40}\b(?:say|said|listed|written|published|find|come\s+from)\b|"
    r"\bwhere\s+does\s+(?:it|that|this)\s+say\b",
    re.I,
)


def focused_height_limit_reply(message: str, data: dict):
    """Answer the user-height question for the chair already in focus."""
    raw = message or ""
    if not _USER_HEIGHT_LIMIT.search(raw):
        return None
    if re.search(r"\bhow\s+tall\s+is\s+the\s+chair\b|\b(?:door|doorway|product|chair)\s+height\b", raw, re.I):
        return None
    from sales_agent import _focus_product_from_prefs, _guess_model_from_text
    from sales_height import STANDARD_RANGE, XL_RANGE, is_xl_line

    product = _guess_model_from_text(raw) or _focus_product_from_prefs(data)
    if product is None:
        return None
    xl = is_xl_line(product)
    line = XL_RANGE if xl else STANDARD_RANGE
    kind = "XL" if xl else "standard"
    return _reply(
        f"A user-height range for **{product.display_name}** itself is not published. "
        f"It is a **{kind}** chair, designed for **{line}**. "
        "Outside that range, a specialist should confirm the fit.",
        intent="specs",
        tools=["catalog.height_fit"],
        products=[product],
        patch={"pending_primary": product.display_name},
    )


def published_source_reply(message: str, data: dict):
    """Point back at the catalog figure instead of restarting the menu."""
    raw = message or ""
    if not _SOURCE_ASK.search(raw):
        return None
    from sales_agent import _focus_product_from_prefs, _guess_model_from_text

    product = _guess_model_from_text(raw) or _focus_product_from_prefs(data)
    if product is None:
        return None
    return _reply(
        f"That figure for **{product.display_name}** comes from the **published catalog specs** this chat uses. "
        "I can't open the product-page sentence it was copied from. "
        "A specialist can point you to the listing if you need the page itself.",
        intent="specs",
        tools=["catalog.source"],
        products=[product],
        patch={"pending_primary": product.display_name},
    )


def followup_reply(message: str, prefs: Optional[dict], *, domain: str):
    """Context-grounded shopping follow-up, or ``None`` to continue normal routing."""
    data = prefs or {}
    for handler in (published_source_reply, focused_height_limit_reply, label_location_reply, concept_reply,
                    named_width_reply, named_model_fit_reply, capacity_list_reply,
                    shortlist_feature_reply, saved_users_height_reply):
        reply = handler(message, data)
        if reply is not None:
            return reply
    return single_pick_reply(message, data, domain=domain)
