"""Resolve catalog follow-ups from saved conversation context.

Objective questions need the relevant product and published facts, rather than
another body-fit interview. This module never supplies inventory or fit facts
that are absent from the catalog.
"""
from __future__ import annotations

import re

from sales_catalog import color_is_listed, parse_asked_color
from sales_compare import lookup_shop_models, product_by_handle, split_compare_terms
from sales_constraints import constraint_verification_notes, extract_constraints, product_constraint_issues
from sales_spec_index import lookup_fit_spec


_RECAP = re.compile(
    r"\b(?:confirm|recap|summarize|repeat|list|show|what|which)\b.{0,65}"
    r"\b(?:constraints?|requirements?|limits?|preferences?|measurements?)\b.{0,45}"
    r"\b(?:applying|using|saved|remember|have|recorded|so\s+far)\b|"
    r"\b(?:recap|summarize|repeat|list)\s+(?:my|our|the\s+saved)\s+(?:constraints?|requirements?|limits?|preferences?|measurements?)\b",
    re.I,
)
_WAREHOUSE = re.compile(
    r"\b(?:warehouse|physically\s+(?:there|available|in\s+stock)|"
    r"(?:actual|real|live)\s+(?:stock|inventory)|(?:quoting|just|only)\s+(?:the\s+)?catalog)\b", re.I,
)
_PAIR_REF = re.compile(
    r"\b(?:which|either|both|these(?:\s+two)?|those(?:\s+two)?|between\s+(?:them|the\s+two)|"
    r"their|them|each\s+(?:one|model|chair)|the\s+(?:same\s+)?two(?:\s+(?:models|chairs))?|"
    r"first\s+(?:one|chair|model)|second\s+(?:one|chair|model))\b", re.I,
)
_KEEP_PAIR = re.compile(
    r"\b(?:keep|stay\s+with|stick\s+(?:with|to)|continue\s+with)\b.{0,25}\b(?:the\s+)?(?:same\s+(?:two|2|pair)|(?:these|those)\s+two|both)\b|"
    r"\bonly\s+(?:compare\s+)?(?:these|those|the)\s+two\b|\b(?:don't|do\s+not)\s+(?:choose|pick|add|suggest)\s+(?:another|a\s+different|other)\b", re.I,
)
_SIDE_BY_SIDE = re.compile(r"\b(?:together|side[\s-]by[\s-]side|summar(?:y|i[sz]e)|table|line\s+(?:them\s+)?up|in\s+one\s+place)\b", re.I)


def _pair_referenced(raw, data):
    """Loose pronouns (their/them) only refer to the pair right after a comparison."""
    refs = [match.group(0).lower() for match in _PAIR_REF.finditer(raw or '')]
    if any(ref not in {'their', 'them'} for ref in refs):
        return True
    return bool(refs) and data.get('last_rateable_intent') == 'compare'


def _count_in(value):
    match = re.search(r'\b(\d+)\b', value or '')
    if match:
        return int(match.group(1))
    words = {'one': 1, 'single': 1, 'two': 2, 'dual': 2, 'three': 3, 'tri': 3, 'four': 4, 'five': 5, 'six': 6}
    word = re.search(r'\b(one|single|two|dual|three|tri|four|five|six)\b|\b(tri|dual)-', (value or '').lower())
    return words[word.group(1) or word.group(2)] if word else None


def _pair_verdicts(message, pair, rec):
    """Direct one-line answers for objective "which of the two" questions."""
    lines = []
    fits = [lookup_fit_spec(product.display_name) for product in pair]
    names = [product.display_name for product in pair]
    if re.search(r'\b(?:more|most|fewer|less)\b.{0,20}\b(?:foot\s*rollers?|rollers?)\b', message, re.I):
        counts = [_count_in(product.foot_roller) for product in pair]
        if None in counts:
            lines.append('A foot-roller count is not published for both models, so I cannot say which has more.')
        elif counts[0] == counts[1]:
            lines.append(f'Both list the same number of foot rollers ({counts[0]}).')
        else:
            more = 0 if counts[0] > counts[1] else 1
            fewer = 1 - more
            lines.append(f"**{names[more]}** has more foot rollers: {counts[more]} vs {counts[fewer]} on {names[fewer]}.")
    door_ask = re.search(r'\b(?:smaller|narrower|smallest|narrowest|less|least|tighter)\b.{0,30}\b(?:door(?:way)?|opening|width)\b|'
                         r'\b(?:door(?:way)?|opening)\b.{0,30}\b(?:smaller|narrower|less)\b', message, re.I)
    if door_ask:
        mode = 'disassembled' if re.search(r'\bdisassembl', message, re.I) else 'assembled'
        widths = [fit.doorway_for_mode(mode) if fit else None for fit in fits]
        if None in widths:
            lines.append(f'A {mode} doorway width is not published for both models, so I cannot rank them.')
        elif widths[0] == widths[1]:
            lines.append(f'Both list the same {mode} doorway width ({widths[0]:g} in).')
        else:
            small = 0 if widths[0] < widths[1] else 1
            lines.append(f"**{names[small]}** needs the smaller {mode} doorway: {widths[small]:g} in vs "
                         f"{widths[1 - small]:g} in for {names[1 - small]}.")
    weight_ask = re.search(r'\b(?:support|supports|hold|holds|handle|capacity|rated)\b', message, re.I)
    weight = None
    stated = re.search(r'(?<!\d)(\d{2,3})\s*(?:lbs?|pounds?)\b', message, re.I)
    if weight_ask and stated:
        weight = float(stated.group(1))
    elif weight_ask and rec.get('weight_lb'):
        weight = float(rec['weight_lb'])
    if weight is not None:
        caps = [fit.max_user_lb if fit else None for fit in fits]
        ok = [name for name, cap in zip(names, caps) if cap is not None and cap >= weight]
        detail = ', '.join(f"{name} {cap:g} lb" if cap is not None else f"{name} not published" for name, cap in zip(names, caps))
        if len(ok) == 2:
            lines.append(f"Both publish a capacity that supports **{weight:g} lb** ({detail}).")
        elif ok:
            lines.append(f"Only **{ok[0]}** publishes a capacity that supports **{weight:g} lb** ({detail}).")
        else:
            lines.append(f"Neither publishes a capacity that supports **{weight:g} lb** ({detail}).")
    return lines


def _pair_side_by_side_reply(pair, rec, *, intro):
    from sales_agent import SalesReply, QuickReply, _fmt_price, _spec_customer_value
    left, right = pair
    fits = [lookup_fit_spec(product.display_name) for product in pair]

    def fit_value(fit, attr, unit):
        value = getattr(fit, attr, None) if fit else None
        return f'{value:g}{unit}' if value is not None else 'not published'

    rows = [
        ('Price', _fmt_price(left.price_usd), _fmt_price(right.price_usd)),
        ('Mechanism / track', f"{left.massage_mechanism or 'not listed'} / {left.track_type or 'not listed'}",
         f"{right.massage_mechanism or 'not listed'} / {right.track_type or 'not listed'}"),
        ('Foot rollers', _spec_customer_value(left.foot_roller) or 'not listed', _spec_customer_value(right.foot_roller) or 'not listed'),
        ('Max user weight', fit_value(fits[0], 'max_user_lb', ' lb'), fit_value(fits[1], 'max_user_lb', ' lb')),
        ('Assembled doorway', fit_value(fits[0], 'door_asm_in', ' in'), fit_value(fits[1], 'door_asm_in', ' in')),
        ('Disassembled doorway', fit_value(fits[0], 'door_dis_in', ' in'), fit_value(fits[1], 'door_dis_in', ' in')),
    ]
    lines = [intro, '', f"**{left.display_name}** vs **{right.display_name}**"]
    lines.extend(f"- **{label}**: {a} vs {b}" for label, a, b in rows)
    issue_labels = {'budget': 'budget', 'mechanism': 'required mechanism', 'weight_capacity': 'user-weight capacity', 'doorway': 'doorway width'}
    if any(rec.get(key) for key in ('budget_max_usd', 'weight_lb', 'doorway_in')):
        lines.append('')
        for product in pair:
            issues = product_constraint_issues(product, rec)
            lines.append(f"- {product.display_name}: " + ('✅ passes your saved limits' if not issues else
                         '❌ misses your saved ' + ', '.join(issue_labels.get(key, key) for key in issues)))
    lines.append('\nPublished doorway width is only a width check; turns, stairs and packaging still need confirmation before delivery.')
    return SalesReply(reply='\n'.join(lines), intent='compare', products=[product.as_public_dict() for product in pair],
                      tools_used=['catalog.compare', 'conversation.product_context'],
                      quick_replies=[QuickReply(label=product.display_name, payload=f'specs:{product.handle}') for product in pair]
                      + [QuickReply(label='Talk to a human', payload='human')],
                      prefs_patch={'pending_compare_pair': [product.handle for product in pair], 'awaiting_compare_recommend': False})
_OBJECTIVE = re.compile(
    r"\b(?:less\s+money|cheaper|cheap(?:est)?|(?:less|more|least|most)\s+expensive|"
    r"costs?|prices?|budget|door(?:way)?|opening|width|wide|clearance|"
    r"capacity|weigh(?:t|s)?|heavy|heavier|mass|supports?|holds?|mechanism|[2345]d|track|(?:foot|calf)\s+rollers?|features?|specs?|"
    r"airbags?|heat(?:ing)?|zero\s+gravity|auto\s+programs?|massage\s+styles?|"
    r"bluetooth|body\s+scann?(?:ing)?|colou?rs?)\b", re.I,
)
_FIT_QUESTION = re.compile(
    r"\b(?:will|would|can|does|could)\b.{0,40}\b(?:fit|go\s+through|pass\s+through)\b|"
    r"\b(?:doorway|door|opening)\s+fit\b|"
    r"\b(?:fits?|pass(?:es)?(?:\s+through)?|go(?:es)?\s+through)\b.{0,40}\b(?:door(?:way)?|opening|entrance)\b", re.I,
)


def _pair_from_prefs(prefs):
    handles = prefs.get('pending_compare_pair') or []
    if not isinstance(handles, list) or len(handles) != 2:
        return None
    pair = tuple(product_by_handle(str(handle)) for handle in handles)
    return pair if all(pair) and pair[0].handle != pair[1].handle else None


def _unrecognized_fit_subject(raw):
    subject = re.search(r'\b(?:will|would|can|could|does)\s+(?:the\s+)?(.+?)\s+(?:fit|go\s+through|pass\s+through)\b', raw, re.I)
    if not subject:
        return False
    name = subject.group(1).strip()
    if re.fullmatch(r'(?:it|this|that|this\s+chair|that\s+chair|chair|one|that\s+one|this\s+one|one\s+you\s+(?:just\s+)?recommended)', name, re.I):
        return False
    return not lookup_shop_models(re.sub(r'^(?:this|that)\s+', '', name, flags=re.I)).matches


def _replace_compared_model(raw, pair, data, domain):
    from sales_agent import SalesReply, QuickReply, _compare_reply
    if re.search(r"\b(?:do\s+not|don't)\s+(?:replace|swap|switch|change)\b", raw, re.I):
        if _OBJECTIVE.search(raw):
            return _objective_pair_reply(raw, pair, data, domain)
        return _compare_reply(f'{pair[0].display_name} vs {pair[1].display_name}', prefs=data, domain=domain)
    change = re.search(
        r'\b(?:replace|swap|switch|change)\s+(?:the\s+)?(.+?)\s+(?:with|for|to)\s+(.+?)'
        r'(?:\s+in\s+(?:that|the|our|this)\s+comparison)?[.!?]*$', raw, re.I,
    )
    if not change or re.search(r"\b(?:don't|do\s+not|not)\s*$", raw[:change.start()], re.I):
        return None
    old_lookup = lookup_shop_models(change.group(1))
    old = [product for product in pair if any(product.handle == match.handle for match in old_lookup.matches)]
    if len(old) != 1:
        return SalesReply(reply='Which of the two compared models should I replace? I will keep the other one.', intent='compare',
                          tools_used=['conversation.product_context'],
                          quick_replies=[QuickReply(label=product.display_name, payload=f'specs:{product.handle}') for product in pair])
    other = next(product for product in pair if product.handle != old[0].handle)
    return _compare_reply(f'{other.display_name} vs {change.group(2)}', prefs=data, domain=domain)


def _recap_reply(prefs):
    from sales_agent import SalesReply, QuickReply
    rec = dict(prefs.get('recommend_prefs') or {})
    lines = ["Here are the requirements saved for this conversation:"]
    fields = (
        ('height_in', 'User height', ' in'), ('weight_lb', 'User weight', ' lb'),
        ('budget_max_usd', 'Maximum budget', ''), ('doorway_in', 'Doorway width', ' in'),
        ('doorway_fit', 'Doorway check', ''), ('mechanism', 'Required mechanism', ''),
        ('goal', 'Main focus', ''), ('intensity', 'Pressure preference', ''),
        ('space', 'Space', ''),
    )
    shown = False
    for key, label, unit in fields:
        value = rec.get(key)
        if not value:
            continue
        shown = True
        if key == 'budget_max_usd':
            value = f"${float(value):,.0f}"
        elif key == 'doorway_in' and value == 'skip':
            value, unit = 'No doorway limit supplied', ''
        lines.append(f"- **{label}**: {value}{unit}")
    if not shown:
        lines.append("No buyer measurements or limits are saved yet. A model's published specifications are not your personal measurements.")
    lines.extend(constraint_verification_notes(rec))
    lines.append("Tell me what to correct; I will keep the other saved requirements.")
    return SalesReply(reply='\n'.join(lines), intent='specs', tools_used=['conversation.saved_requirements'],
                      quick_replies=[QuickReply(label='Talk to a human', payload='human')])


def _objective_pair_reply(message, pair, prefs, domain):
    from sales_agent import SalesReply, QuickReply, _fmt_price, _SPEC_QUESTION_PATTERNS, _spec_customer_value
    rec = dict(prefs.get('recommend_prefs') or {})
    # A width supplied in a fit question is a doorway requirement. Product
    # capacities/mechanism labels elsewhere in the sentence remain facts.
    if _FIT_QUESTION.search(message) or re.search(r'\b(?:fits?|for)\s+(?:a\s+|my\s+)?\d', message, re.I):
        rec.update({key: value for key, value in extract_constraints(message).items()
                    if key in {'doorway_in', 'doorway_fit', 'space'}})
    own_mass = bool(re.search(r"\bweight\s+of\s+(?:the\s+)?chair\b|\bchair(?:['’]s)?\s+(?:own\s+)?weight\b|\bheavier\s+to\s+(?:move|carry)\b|\bchair\s+(?:itself\s+)?weigh", message, re.I)
                    or re.search(r'\b(?:weighs?|weight|heavy|heavier|mass)\b', message, re.I) and re.search(r'\b(?:chair|product)\s+(?:itself|mass)\b', message, re.I))
    if re.search(r"\bI(?:\s+(?:weigh|am)|['’]m)\b.{0,12}\d{2,3}\s*(?:lbs?|pounds?)\b", message, re.I):
        rec.update({key: value for key, value in extract_constraints(message, rec).items()
                    if key in {'weight_lb', 'user_weights_lb'}})
    verdicts = [] if own_mass else _pair_verdicts(message, pair, rec)
    lines = (verdicts + [""] if verdicts else []) + ["For the two models we are comparing:"]
    if own_mass:
        lines.append("The chair's own weight is not verified in this catalog, so I cannot determine which chair is heavier to move. Maximum user-weight capacity measures the person sitting in it; it is not the chair's mass.")
    eligible = []
    issue_labels = {'budget': 'budget', 'mechanism': 'required mechanism',
                    'weight_capacity': 'user-weight capacity', 'doorway': 'doorway width'}
    for product in pair:
        lines.append(f"\n**{product.display_name}** — {_fmt_price(product.price_usd)}")
        if own_mass:
            lines.append('- Chair weight: not verified.')
            if not product_constraint_issues(product, rec):
                eligible.append(product)
            continue
        lines.append(f"- Published mechanism / track: {product.massage_mechanism or 'not listed'} / {product.track_type or 'not listed'}.")
        specs = product.as_public_dict()['specs']
        generic_specs = bool(re.search(r'\b(?:features?|specs?)\b', message, re.I))
        facets = {key: label for pattern, key, label in _SPEC_QUESTION_PATTERNS if pattern.search(message)}
        if generic_specs:
            facets.update({'zero_gravity': 'Zero gravity', 'heating': 'Heating', 'airbag': 'Airbags',
                           'foot_roller': 'Foot roller', 'auto_programs': 'Auto programs', 'massage_styles': 'Massage styles'})
        if re.search(r'\bauto\s+programs?\b', message, re.I):
            facets['auto_programs'] = 'Auto programs'
        if re.search(r'\bmassage\s+styles?\b', message, re.I):
            facets['massage_styles'] = 'Massage styles'
        for key, label in facets.items():
            if key in {'massage_mechanism', 'track_type'}:
                continue
            if key == 'foot_roller':
                label = 'Foot roller'
            lines.append(f"- {label}: {_spec_customer_value(specs.get(key)) or 'not verified'}.")
        fit = lookup_fit_spec(product.display_name)
        if fit:
            for label, value, unit in (
                ('Maximum user weight', fit.max_user_lb, ' lb'),
                ('Assembled doorway', fit.door_asm_in, ' in'),
                ('Disassembled doorway', fit.door_dis_in, ' in'),
            ):
                lines.append(f"- {label}: {f'{value:g}{unit}' if value is not None else 'not verified'}.")
        else:
            lines.append('- Doorway and user-weight limits are not verified for this model.')
        issues = product_constraint_issues(product, rec)
        if issues:
            lines.append('- Does not pass every saved verified requirement: ' + ', '.join(issue_labels.get(key, key) for key in issues) + '.')
        else:
            eligible.append(product)
    if rec.get('doorway_in') and rec['doorway_in'] != 'skip':
        lines.append(f"\nI am checking your **{rec['doorway_in']} in doorway** with the chair **{rec.get('doorway_fit') or 'assembled'}**.")
    priced = [product for product in pair if product.price_usd is not None]
    if priced and not own_mass:
        # Respect an explicit request for the higher-priced model. A locally
        # negated cheaper preference must not choose the cheaper chair.
        from sales_intent import _NEGATED_REQUEST_PREFIX
        price_directions = [match for match in re.finditer(r'\b(?:(?:more|most|less|least)\s+expensive|cheap(?:er|est)?|prici(?:er|est)|costlier)\b', message, re.I)
                            if not _NEGATED_REQUEST_PREFIX.search(message[max(0, match.start() - 70):match.start()])]
        higher = bool(price_directions and re.match(r'(?:more|most|prici|costlier)', price_directions[-1].group(0), re.I))
        selected = (max if higher else min)(priced, key=lambda product: product.price_usd)
        if len(priced) == 2 and priced[0].price_usd == priced[1].price_usd:
            lines.append('\nTheir published catalog prices are equal.')
        elif re.search(r'\b(?:money|cheap\w*|cost\w*|price\w*|expensive|budget)\b', message, re.I):
            scope = 'Among these models that pass the saved verified limits' if len(eligible) == len(pair) else 'Of the two compared models'
            lines.append(f"\n{scope}, **{selected.display_name}** has the {'higher' if higher else 'lower'} published price at **{_fmt_price(selected.price_usd)}**.")
        if len(priced) == 2 and re.search(r'\bprice\s+(?:gap|difference)\b', message, re.I):
            lines.append(f"The published price gap is **{_fmt_price(abs(priced[0].price_usd - priced[1].price_usd))}**.")
    if not eligible:
        lines.append('\nNeither of these two passes every saved verified requirement. I will not substitute another model without asking.')
    lines.extend(constraint_verification_notes(rec))
    if re.search(r'\bcalf\s+rollers?\b', message, re.I):
        lines.append('A separate calf-roller specification is not verified here; a listed foot roller does not establish a calf roller.')
    lines.append('Published doorway width is only a width check; turns, packaging, access and any disassembly still need confirmation before delivery.')
    return SalesReply(reply='\n'.join(lines), intent='compare', products=[product.as_public_dict() for product in eligible],
                      tools_used=['catalog.compare', 'cases.fit_gates'],
                      quick_replies=[QuickReply(label='Talk to a human', payload='human')],
                      prefs_patch={'pending_compare_pair': [product.handle for product in pair],
                                   'recommend_prefs': rec,
                                   'awaiting_compare_recommend': False})


def relative_price_followup(message, prefs, *, domain):
    """Recover a catalog price comparison, retaining actual negotiation routes."""
    from sales_intent import _DISCOUNT_RE, _requested_matches
    data = prefs or {}
    pair = _pair_from_prefs(data)
    if not pair or _requested_matches(_DISCOUNT_RE, message or ''):
        return None
    if re.search(r'\b(?:costco|walmart|amazon)\b|\bcompetitor\b', message or '', re.I):
        return None
    if not re.search(r'\b(?:price\s+(?:gap|difference)|(?:more|most|less|least)\s+expensive|cheap(?:er|est)?)\b', message or '', re.I):
        return None
    replacement = _replace_compared_model(message, pair, data, domain)
    return replacement or _objective_pair_reply(message, pair, data, domain)


def _saved_capacity_reply(raw, data):
    from sales_agent import SalesReply, QuickReply, _focus_product_from_prefs, _product_from_tier_pick
    rec = dict(data.get('recommend_prefs') or {})
    if not rec.get('weight_lb') or not re.search(
        r'\b(?:which|what)\b.{0,45}\b(?:support|hold|handle)\s+(?:me|my\s+weight)\b', raw, re.I,
    ):
        return None
    candidates = []
    for pick in data.get('pending_tier_picks') or []:
        product = _product_from_tier_pick(pick) if isinstance(pick, dict) else None
        if product is not None and product.handle not in {item.handle for item in candidates}:
            candidates.append(product)
    if not candidates:
        focus = _focus_product_from_prefs(data)
        if focus is not None:
            candidates = [focus]
    eligible = [product for product in candidates if not product_constraint_issues(product, rec)]
    lines = [f"I am using your saved **{rec['weight_lb']} lb user weight**, not the chair or shipping-box weight."]
    for product in eligible:
        fit = lookup_fit_spec(product.display_name)
        lines.append(f"- **{product.display_name}** lists a **{fit.max_user_lb:g} lb maximum user capacity** and passes the saved verified numerical limits.")
    if not eligible:
        lines.append('None of the currently discussed models has a verified capacity and other published limits that meet every saved requirement. A specialist should confirm an appropriate model.')
    lines.extend(constraint_verification_notes(rec))
    return SalesReply(reply='\n'.join(lines), intent='recommend', tools_used=['catalog.weight_capacity', 'cases.fit_gates'],
                      products=[product.as_public_dict() for product in eligible],
                      quick_replies=[QuickReply(label='Talk to a human', payload='human')])


def _personal_height_fit_reply(raw, data):
    from sales_agent import SalesReply, QuickReply, _focus_product_from_prefs, _guess_model_from_text
    personal_question = re.search(
        r'\b(?:fits?|suit(?:s)?|work(?:s)?\s+for)\s+(?:me|my\s+(?:body|height))\b|'
        r'\b(?:am\s+i|i\s+am|i.m)\s+too\s+(?:tall|short)\b', raw, re.I,
    )
    measurement_caution = re.search(r"\b(?:don't|do\s+not)\s+guess\b", raw, re.I) and re.search(r'\b(?:height|fit|measurements?|dimensions?)\b', raw, re.I)
    if not personal_question and not measurement_caution:
        return None
    if _unrecognized_fit_subject(raw):
        return SalesReply(reply='Which exact model should I check? I cannot establish fit for an unrecognized model or variant.', intent='specs',
                          tools_used=['conversation.product_context'], quick_replies=[QuickReply(label='Talk to a human', payload='human')])
    named = _guess_model_from_text(raw)
    pair = _pair_from_prefs(data)
    product = named or _focus_product_from_prefs(data) or (pair[0] if pair and data.get('last_rateable_intent') == 'compare' else None)
    if product is None:
        return None
    rec = data.get('recommend_prefs') or {}
    height = f"Your saved height is **{rec['height_in']} in**. " if rec.get('height_in') else ''
    names = f'**{product.display_name}**'
    if pair and named is None and data.get('last_rateable_intent') == 'compare':
        names = ' and '.join(f'**{item.display_name}**' for item in pair)
    return SalesReply(
        reply=height + f"The supported user-height range for {names} is not verified in this catalog, so I cannot confirm personal height fit. "
        "A specialist should confirm the minimum and maximum supported height before purchase; I will not infer it from the model name or track type.",
        intent='specs', tools_used=['catalog.height_fit'],
        quick_replies=[QuickReply(label='Talk to a human', payload='human')],
    )


def reply_for_context(message: str, prefs: dict | None, *, domain: str):
    """Return an answer only when the requested referent is unambiguous."""
    from sales_agent import (SalesReply, QuickReply, _compare_reply, _compare_recommend_reply,
                             _doorway_fit_reply, _focus_product_from_prefs,
                             _guess_model_from_text, _price_reply, _specs_reply, _stock_reply)
    data = prefs or {}
    raw = message or ''
    if _RECAP.search(raw):
        return _recap_reply(data)

    height_fit = _personal_height_fit_reply(raw, data)
    if height_fit is not None:
        return height_fit

    capacity = _saved_capacity_reply(raw, data)
    if capacity is not None:
        return capacity

    pair = _pair_from_prefs(data)
    if pair:
        replacement = _replace_compared_model(raw, pair, data, domain)
        if replacement is not None:
            return replacement
        terms = split_compare_terms(raw)
        named_pair = bool(terms) and all(lookup_shop_models(term).matches for term in terms)
        if _guess_model_from_text(raw) is None and not named_pair:
            rec = dict(data.get('recommend_prefs') or {})
            if _KEEP_PAIR.search(raw):
                return _pair_side_by_side_reply(
                    pair, rec, intro=f"I'll keep comparing only **{pair[0].display_name}** and **{pair[1].display_name}**; "
                                     "I won't swap in another chair.")
            if _SIDE_BY_SIDE.search(raw) and _pair_referenced(raw, data):
                return _pair_side_by_side_reply(pair, rec, intro="Here are both models side by side:")

    terms = split_compare_terms(raw)
    # A valid named pair takes priority over a single-name spec fallback. Use
    # the comparison resolver for family ambiguity and unavailable models.
    if terms:
        lookups = [lookup_shop_models(term) for term in terms]
        if all(item.matches for item in lookups):
            return _compare_reply(f'{terms[0]} vs {terms[1]}', prefs=data, domain=domain)

    named = _guess_model_from_text(raw)
    if _WAREHOUSE.search(raw) and not re.search(r"\b(?:not\s+asking\s+about|don'?t\s+(?:ask|check|want))\b.{0,35}\b(?:warehouse|stock|inventory)\b", raw, re.I):
        if pair and named is None and data.get('last_rateable_intent') == 'compare':
            return SalesReply(reply='Which of the two models should I check? Purchase availability does not confirm a physical warehouse location or allocated color.',
                              intent='stock', tools_used=['conversation.product_context'],
                              quick_replies=[QuickReply(label=product.display_name, payload=f'stock:{product.handle}') for product in pair])
        reply = _stock_reply(raw, prefs=data, domain=domain)
        reply.reply = ('I cannot verify a physical warehouse location or a color allocation from catalog status or purchase availability.\n\n' + reply.reply)
        color = parse_asked_color(str(data.get('last_shopper_question') or ''))
        product = named or _focus_product_from_prefs(data)
        if color and product:
            listed = color_is_listed(product, color)
            reply.reply += (f"\n\nFor the previously requested **{color}**, " +
                            ('a listed color still needs checkout confirmation; I cannot verify a warehouse allocation.' if listed else
                             f"that color is not among the published options: {', '.join(product.colors) or 'not listed'}."))
        return reply

    if named is not None and re.match(r'^(?:and\s+)?(?:what|how)\s+about\b|^and\s+(?:the\s+)?', raw, re.I):
        previous = _focus_product_from_prefs(data)
        if previous is not None and previous.handle != named.handle:
            from sales_intent import classify
            label = classify(raw).label
            if label not in {'price', 'stock', 'specs'}:
                label = data.get('last_rateable_intent')
            builder = {'price': _price_reply, 'stock': _stock_reply}.get(label, _specs_reply)
            reply = builder(raw, prefs={**data, 'pending_primary': named.display_name}, domain=domain)
            reply.prefs_patch = {**(reply.prefs_patch or {}),
                                 'pending_compare_pair': [previous.handle, named.handle],
                                 'pending_compare_left': previous.handle,
                                 'pending_compare_right': named.handle,
                                 'awaiting_compare_recommend': False}
            return reply

    # Do not turn a new search or a single named product question into the old
    # comparison. The retained pair is for referential follow-ups only.
    question = re.split(r'[.!?]\s+', raw)[-1]
    named_only_in_background = question != raw and _PAIR_REF.search(question) and _guess_model_from_text(question) is None
    if pair and (named is None or named_only_in_background) and _pair_referenced(raw, data) and not re.search(r'\b(?:new\s+search|start\s+over|other\s+models?|different\s+models?)\b', raw, re.I):
        if _OBJECTIVE.search(raw):
            return _objective_pair_reply(raw, pair, data, domain)
        if re.search(r'\b(?:better|choose|pick|recommend|right\s+for|should\s+i)\b', raw, re.I):
            return _compare_recommend_reply(raw, prefs=data, domain=domain)

    saved_doorway = (data.get('recommend_prefs') or {}).get('doorway_in')
    if _FIT_QUESTION.search(raw) and (re.search(r'\b(?:door(?:way)?|opening|entrance)\b', raw, re.I)
                                    or saved_doorway and re.search(r'\b(?:width|wide|through)\b', raw, re.I)):
        if _unrecognized_fit_subject(raw):
            return SalesReply(reply='Which exact model should I check against your doorway? I cannot use the previously discussed chair for an unrecognized model name.',
                              intent='specs', tools_used=['conversation.product_context'],
                              quick_replies=[QuickReply(label='Talk to a human', payload='human')])
        if named is None and pair and data.get('last_rateable_intent') == 'compare':
            return SalesReply(reply='Which of the two compared models should I check against your saved doorway?',
                              intent='specs', tools_used=['conversation.product_context'],
                              quick_replies=[QuickReply(label=product.display_name, payload=f'specs:{product.handle}') for product in pair])
        product = named or _focus_product_from_prefs(data)
        if product is not None:
            return _doorway_fit_reply(raw, prefs={**data, 'pending_primary': product.display_name}, domain=domain)
    return None
