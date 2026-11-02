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


def _title_height_claim(product: ProductSpecs) -> Optional[str]:
    match = re.search(r"\b(\d)'\s?(\d{1,2})\"?", product.display_name or '')
    return f"{match.group(1)}'{match.group(2)}\"" if match else None


def _height_confirmation(products: list[ProductSpecs], users) -> Optional[str]:
    heights = [(label, height) for label, height, _ in users if height]
    if not heights or not products:
        return None
    who = ' and '.join(f"{_person(label)} ({_feet(height)})" for label, height in heights)
    names = ' and '.join(f"**{product.display_name}**" for product in products)
    line = (f"Height fit for {who}: the supported user-height range for {names} is not published in our verified specs, "
            "so Sales needs to confirm height fit before you order.")
    claims = [(product, _title_height_claim(product)) for product in products]
    claims = [(product, claim) for product, claim in claims if claim]
    if claims:
        line += ' ' + ' '.join(
            f"(The {product.display_name.split('(')[0].strip()} product title mentions up to {claim}; that is a listing claim, not a verified fit check.)"
            for product, claim in claims
        )
    return line


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
_ISSUE = re.compile(
    r"\b(?:its|the|my|our)\s+((?:[\w'’-]+\s+){0,4}?[\w'’-]+)\s+(stopped\s+working|(?:is|are)\s+(?:broken|not\s+working)|"
    r"(?:doesn't|don't|won't)\s+work)\b", re.I,
)


def owner_case_patch(message: str, data: dict) -> dict:
    """Durable repair context saved when Sales routes a fault to Warranty."""
    from sales_agent import _guess_model_from_text
    case = dict(data.get('owner_service_case') or {})
    product = _guess_model_from_text(message or '')
    if product is not None:
        case['model'] = product.display_name
    issue = _ISSUE.search(message or '')
    if issue:
        part = re.sub(r"^.*['’]s\s+", '', issue.group(1)) if product is not None else issue.group(1)
        case['issue'] = f"{part} {issue.group(2)}".strip()[:80]
    return {'owner_service_case': case} if case else {}


def dual_task_reply(message: str, data: dict):
    raw = message or ''
    case = data.get('owner_service_case') or {}
    purchase = [match for match in _PURCHASE.finditer(raw)
                if not re.search(r"\b(?:not|no|don't|do\s+not|never)\s+(?:want\s+|need\s+)?(?:a\s+|to\s+)?$", raw[:match.start()], re.I)]
    if not (_REPAIR.search(raw) and purchase and (_SEPARATE.search(raw) or case)):
        return None
    from sales_agent import _guess_model_from_text
    sentences = [part for part in re.split(r'(?<=[.!?])\s+', raw) if part.strip()]
    repair_model = case.get('model')
    purchase_model = None
    for sentence in sentences:
        product = _guess_model_from_text(sentence)
        if product is None:
            continue
        if _REPAIR.search(sentence) and not _PURCHASE.search(sentence):
            repair_model = repair_model or product.display_name
        elif product.display_name != repair_model:
            purchase_model = product.display_name
    purchase_model = purchase_model or str(data.get('pending_primary') or '') or None
    if purchase_model == repair_model:
        purchase_model = None
    rec = dict(data.get('recommend_prefs') or {})
    repair_label = repair_model or 'your current chair'
    issue = f" ({case['issue']})" if case.get('issue') else ''
    lines = [
        "I'll keep these as two separate requests:",
        "",
        f"**1. Repair — {repair_label}{issue} → Warranty Department**",
        f"Email **{WARRANTY_EMAIL}** with the model, a description of the issue, and your order number or proof of purchase, "
        f"or call **{WARRANTY_PHONE}**. You can also open a service request at {WARRANTY_PORTAL}. "
        "Warranty reviews the issue and coverage; this chat has not approved a repair or replacement.",
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
                  patch={'owner_service_case': case or ({'model': repair_model} if repair_model else {})},
                  extra_buttons=buttons)


def new_purchase_note(message: str, data: dict) -> Optional[str]:
    """One-time reminder that an open repair is not mixed into a new purchase."""
    case = data.get('owner_service_case') or {}
    if not case or data.get('owner_case_acknowledged'):
        return None
    if not re.search(r"\b(?:separately|second\s+chair|another\s+chair|new[\s-]+purchase|also\s+(?:want|like)\s+to\s+buy)\b", message or '', re.I):
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
    elif _title_height_claim(product):
        lines.append(f"The product title mentions up to {_title_height_claim(product)}; the supported height range is not in our verified specs.")
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
    lines.extend(["", "Still needs confirmation:"])
    heights = [(label, height) for label, height, _ in users if height]
    for label, height in heights:
        lines.append(f"- Height fit for {_person(label)} ({_feet(height)}): the supported height range is not published")
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
    text = line + "\n\nI won't treat height fit as confirmed. A Sales specialist can check both users before you order."
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


def followup_reply(message: str, prefs: Optional[dict], *, domain: str):
    """Context-grounded shopping follow-up, or ``None`` to continue normal routing."""
    data = prefs or {}
    for handler in (label_location_reply, concept_reply, named_width_reply, named_model_fit_reply, capacity_list_reply,
                    shortlist_feature_reply, saved_users_height_reply):
        reply = handler(message, data)
        if reply is not None:
            return reply
    return single_pick_reply(message, data, domain=domain)
