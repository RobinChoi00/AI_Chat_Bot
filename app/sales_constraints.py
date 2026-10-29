"""Exact shopper constraints shared by deterministic Sales recommendation paths.

Case-book buckets rank candidates; they must never replace a stated numerical
limit. Missing published fit information is not evidence of a safe fit.
"""
from __future__ import annotations

import json
import math
import re
from typing import Mapping


_NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
}
_HEIGHT_MEASUREMENT = re.compile(
    r"(?<![\w.])(?P<feet>[3-7]|three|four|five|six|seven)[\s-]*"
    r"(?:['’′]|ft\b|feet\b|foot\b)[\s-]*"
    r"(?:(?:and\s+)?(?P<inches>\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)"
    r"(?!\w)(?:[\s-]*(?:[\"″]|inches\b|inch\b|in\b))?)?",
    re.I,
)


def _negated_measurement(text: str, start: int) -> bool:
    prefix = text[max(0, start - 60):start].replace('’', "'")
    return bool(re.search(
        r"\b(?:not|instead of|rather than|used to be|previously|formerly|"
        r"(?:do\s+not|don't|never)\s+(?:use|save|assume|set|count))\s*(?:a\s+)?$",
        prefix, re.I,
    ))


def _negated_preference(text: str, start: int) -> bool:
    prefix = text[max(0, start - 70):start].replace('’', "'")
    # A single rejection governs a coordinated list ("not 4D or 5D"),
    # but ends before a new affirmative clause ("but I want 3D").
    modifier = r'(?:(?:a|an|the|any|too|very)\s+)*'
    option = r'(?:[2345]\s*d|deep[ -]?tissue|strong|firm|deep|intense|hard|gentle|soft|light(?:\s+pressure)?|balanced|medium|moderate|adjustable|customi[sz]e|variable)'
    previous_options = r'(?:' + modifier + option + r'(?:\s+(?:massage|pressure|chair|model|one))?\s*(?:,?\s*(?:or|and)\s+|,\s*))*'
    return bool(re.search(
        r"\b(?:not|no|avoid|don't\s+(?:want|need|recommend)|do\s+not\s+(?:want|need|recommend)|instead of|rather than)\s*"
        + previous_options + modifier + r'$', prefix, re.I,
    ))


def intensity_preference(text: str) -> str | None:
    """Use affirmative corrections, never the intensity a shopper rejected."""
    positive = []
    rejected_strong = False
    for match in re.finditer(
        r'\b(?:deep[ -]?tissue|strong|firm|deep|intense|hard|gentle|soft|light(?:\s+pressure)?|balanced|medium|moderate|adjustable|customi[sz]e|variable)\b',
        text or '', re.I,
    ):
        word = match.group().lower()
        value = ('Strong' if re.match(r'deep|strong|firm|intense|hard', word) else
                 'Gentle' if re.match(r'gentle|soft|light', word) else
                 'Balanced' if word in {'balanced', 'medium', 'moderate'} else 'Highly Adjustable')
        if _negated_preference(text, match.start()):
            rejected_strong |= value == 'Strong'
        else:
            positive.append(value)
    if positive:
        return positive[-1]
    return 'Gentle' if rejected_strong else None


def _latest_subject_position(pattern: str, prefix: str) -> int:
    return max((match.end() for match in re.finditer(pattern, prefix, re.I)), default=-1)


def _is_person_height(text: str, start: int, end: int) -> bool:
    prefix = re.split(r'[.;!?]', text[:start])[-1][-120:]
    suffix = text[end:]
    if re.match(r'\s*(?:user\s+)?height\s+(?:range|limit|rating)\b', suffix, re.I):
        return False
    if re.match(r'\s*(?:tall\s+)?(?:user|person|buyer|man|woman)\b', suffix, re.I):
        return True
    if re.match(r'\s*(?:(?:tall|high)\s+(?:and\s+)?)?(?:of\s+)?(?:clearance|space|room|wide|deep|long|behind|door(?:way)?|opening|ceiling)\b', suffix, re.I):
        return False
    person = _latest_subject_position(r"\b(?:i\s+am|i'm|my\s+(?:body\s+)?height(?:\s+is)?|(?:buyer|user|person|husband|wife|father|mother|dad|mom|spouse)(?:\s+(?:is|who\s+is))?)\b", prefix)
    dimension = _latest_subject_position(r'\b(?:door(?:way)?|opening|entrance|ceiling|hall(?:way)?|chair|package|product|shipment|box|unit|device|room|clearance|space|height\s+(?:range|limit)|supported\s+height)\b', prefix)
    return dimension < 0 or person > dimension


def height_measurements(text: str, *, include_negated: bool = False, include_dimensions: bool = False) -> list[tuple[int, int, int]]:
    """Return measured body heights and their text spans, including spoken units."""
    values = []
    for match in _HEIGHT_MEASUREMENT.finditer(text or ''):
        if not include_negated and _negated_measurement(text, match.start()):
            continue
        if not include_dimensions and not _is_person_height(text, match.start(), match.end()):
            continue
        feet = match.group('feet').lower()
        inches = (match.group('inches') or '0').lower()
        ft = int(feet) if feet.isdigit() else _NUMBER_WORDS[feet]
        inch = int(inches) if inches.isdigit() else _NUMBER_WORDS[inches]
        # Counts of foot rollers are product features, not the shopper's height.
        if not match.group('inches') and re.match(r'\s*(?:rollers?|massagers?|massage)\b', (text or '')[match.end():], re.I):
            continue
        if inch < 12:
            values.append((match.start(), match.end(), ft * 12 + inch))
    for match in re.finditer(r'(?<![\w.])(\d{1,3}(?:\.\d+)?)\s*(cm|centimet(?:er|re)s?|m|met(?:er|re)s?)(?![a-z])', text or '', re.I):
        centimeters = float(match.group(1)) * (100 if match.group(2).lower() in {'m', 'meter', 'meters', 'metre', 'metres'} else 1)
        if 100 <= centimeters <= 229 and (include_negated or not _negated_measurement(text, match.start())) and (include_dimensions or _is_person_height(text, match.start(), match.end())):
            values.append((match.start(), match.end(), round(centimeters / 2.54)))
    for match in re.finditer(r'(?<![\w.])(\d{2}(?:\.\d+)?)\s*(?:inches\b|inch\b|in\b|["″])', text or '', re.I):
        if any(start <= match.start() < end for start, end, _ in values):
            continue
        if (include_negated or not _negated_measurement(text, match.start())) and (include_dimensions or _is_person_height(text, match.start(), match.end())):
            inches = float(match.group(1))
            if 38 <= inches <= 90:
                values.append((match.start(), match.end(), round(inches)))
    return sorted(values)


def without_height_measurements(text: str) -> str:
    """Keep body-part preferences while excluding foot/feet used as height units."""
    raw = text or ''
    for start, end, _ in reversed(height_measurements(raw, include_negated=True, include_dimensions=True)):
        raw = raw[:start] + ' ' * (end - start) + raw[end:]
    return raw


def _is_person_weight(text: str, start: int, end: int) -> bool:
    prefix = re.split(r'[.;!?]', text[:start])[-1][-150:]
    suffix = text[end:]
    if re.match(r'\s*(?:heavy\s+)?(?:chair|package|product|shipment|box|unit|device)\b', suffix, re.I):
        return False
    requirement = re.search(
        r'\b(?:recommend(?:\s+(?:me|us))?|need|require|want|looking\s+for|shopping\s+for)\s+'
        r'(?:(?:a|an|the)\s+)?(?:massage\s+)?(?:chair|model|seat)\s+'
        r'(?:for\s+(?:a\s+)?|(?:that\s+can|to)\s+(?:support|handle|hold)\s+|'
        r'(?:with|that\s+has)\s+(?:a\s+)?(?:(?:published|verified|maximum)\s+)*)'
        r'(?:(?:up\s+to|at\s+least)\s*)?$', prefix, re.I,
    )
    if requirement and not re.search(r"\b(?:don't|do\s+not|not)\s*$", prefix[:requirement.start()], re.I):
        return True
    if re.match(r'\s*(?:capacity|(?:maximum\s+)?(?:user\s+)?weight\s+(?:capacity|limit|rating)|(?:user\s+)?(?:capacity|limit|rating))\b', suffix, re.I):
        return False
    person = _latest_subject_position(r"\b(?:i\s+(?:am|weigh)|i'm|my\s+(?:body\s+)?weight(?:\s+is)?|(?:buyer|user|person|husband|wife|father|mother|dad|mom|spouse)(?:\s+(?:is|weighs?|who\s+weighs?))?)\b", prefix)
    product = _latest_subject_position(r"\b(?:capacity|rated(?:\s+for)?|rating|limits?|max(?:imum)?|supports?|holds?|can\s+handle|(?:chair|product|shipping)(?:'s)?\s+weight|costs?)\b", prefix)
    if product > person:
        return False
    if re.match(r'\s*(?:user|person|buyer|man|woman)\b', suffix, re.I):
        return True
    object_subject = _latest_subject_position(r'\b(?:chair|package|product|shipment|box|unit|device)\b', prefix)
    return object_subject < 0 or person > object_subject


def _person_label(text: str, start: int) -> str:
    """Keep named household users separate without storing customer names."""
    # A household user is often introduced in the preceding sentence:
    # "My mother will also use it. She weighs 150 pounds."
    prefix = text[:start][-220:]
    labels = list(re.finditer(
        r"\b(?:father|dad|mother|mom|husband|wife|spouse|partner|son|daughter|"
        r"first\s+user|second\s+user|heavier\s+user|lighter\s+user|buyer|user|person|"
        r"i|i'm|me|myself|my\s+(?:body\s+)?(?:height|weight))\b",
        prefix, re.I,
    ))
    label = labels[-1].group().lower() if labels else 'user'
    return {'dad': 'father', 'mom': 'mother', "i'm": 'self', 'i': 'self', 'me': 'self', 'myself': 'self',
            'my weight': 'self', 'my body weight': 'self', 'my height': 'self',
            'my body height': 'self'}.get(label, label.replace(' ', '_'))


def _read_person_measurements(value: str | None) -> dict[str, float]:
    try:
        parsed = json.loads(value or '{}')
        if not isinstance(parsed, dict):
            return {}
        return {key: float(amount) for key, amount in parsed.items()
                if isinstance(key, str) and isinstance(amount, (int, float))
                and not isinstance(amount, bool) and math.isfinite(amount) and amount > 0}
    except (TypeError, ValueError):
        return {}


def _retain_person_measurements(text: str, measurements: list[tuple[int, float]], previous: str | None) -> dict[str, float]:
    current = {}
    corrected = bool(re.search(r'\b(?:actually|correction|correcting|instead|now|not)\b', text, re.I))
    for start, value in measurements:
        label = _person_label(text, start)
        if label in {'user', 'buyer', 'person'} and not corrected:
            # Unnamed users still have separate requirements. Do not let the
            # final lighter measurement silently replace the first one.
            if label in current:
                current[label + '_1'] = current.pop(label)
            if label + '_1' in current:
                index = 2
                while label + '_' + str(index) in current:
                    index += 1
                label += '_' + str(index)
        current[label] = value
    saved = _read_person_measurements(previous)
    # A single-user correction replaces the prior value. Once two actual users
    # are known, updating one must not discard the other person's requirements.
    if len(saved) > 1:
        for label, value in current.items():
            if label in {'heavier_user', 'lighter_user'}:
                target = (max if label == 'heavier_user' else min)(saved, key=saved.get)
                saved[target] = value
            elif label in {'user', 'buyer', 'person'}:
                # An unnamed requirement can raise a shared minimum; it cannot
                # silently erase an identified user's larger body measurement.
                if value > max(saved.values()):
                    saved['user'] = value
            else:
                saved[label] = value
        return saved
    if len(current) > 1:
        return current
    label = next(iter(current), 'user')
    sharing = re.search(r'\b(?:also|too|both|together|share|sharing|second\s+user)\b', text, re.I)
    named = {'father', 'mother', 'husband', 'wife', 'spouse', 'partner', 'son', 'daughter', 'self'}
    if sharing and saved and label in named and label not in saved:
        return {**saved, **current}
    return current


def _doorway_measurements(text: str) -> list[tuple[int, float]]:
    values = []
    unit_pattern = r'(?:inches\b|inch\b|in\b|["″]|mm\b|millimet(?:er|re)s?\b|cm\b|centimet(?:er|re)s?\b|m\b|met(?:er|re)s?\b)'
    for match in re.finditer(r'(?<![\w.])(\d{1,4}(?:\.\d+)?)\s*-?\s*(' + unit_pattern + ')', text, re.I):
        if _negated_measurement(text, match.start()) or not _is_customer_doorway(text, match.start(), match.end()):
            continue
        prefix = re.split(r'[;!?]|\.(?=\s|$)', text[:match.start()])[-1][-140:]
        suffix = text[match.end():]
        before = re.search(r'\b(?:door(?:way)?|entrance|opening)\b', prefix, re.I)
        after = re.match(r'\s*(?:wide\s+)?(?:door(?:way)?|entrance|opening)\b', suffix, re.I)
        if not (before or after) or re.match(r'\s*(?:tall|high)\b', suffix, re.I):
            continue
        # A later body measurement is not another width of the earlier door.
        if before and not after and _latest_subject_position(r"\b(?:i\s+am|i'm|my\s+height|(?:father|mother|husband|wife)\s+is)\b", prefix) > _latest_subject_position(r'\b(?:door(?:way)?|entrance|opening)\b', prefix):
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith(('mm', 'millimet')):
            value /= 25.4
        elif unit.startswith(('cm', 'centimet')):
            value /= 2.54
        elif unit in {'m', 'meter', 'meters', 'metre', 'metres'}:
            value *= 100 / 2.54
        if 0 < value <= 1000:
            values.append((match.start(), value))
    return values


_SPOKEN_NUMBERS = {**_NUMBER_WORDS, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                   'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
                   'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}


def _spoken_amount(text: str) -> int:
    total = current = 0
    for token in re.split(r'[\s-]+', text.lower().strip()):
        if token == 'hundred':
            current = (current or 1) * 100
        elif token == 'thousand':
            total += (current or 1) * 1000
            current = 0
        elif token != 'and':
            current += _SPOKEN_NUMBERS.get(token, 0)
    return total + current


def _is_customer_doorway(text: str, start: int, end: int) -> bool:
    """A model's published doorway requirement is not a home measurement."""
    prefix = re.split(r'[;!?]|\.(?=\s|$)', text[:start])[-1][-160:]
    scope = prefix + text[start:end]
    customer = _latest_subject_position(r'\b(?:my|our|your)\s+(?:(?:own|home|house)\s+)?(?:door(?:way)?|opening|entrance)\b', scope)
    product = _latest_subject_position(
        r"\b(?:published|listed|catalog|specifications?|specs?|required|requires|minimum|assembled|disassembled)\b|"
        r"\b(?:chair|model|product|unit)(?:['’]s|\s+(?:has|needs|requires|lists))\b|\b(?!(?:my|our|your)\b)[a-z][\w-]*['’]s\b", scope,
    )
    access_request = _latest_subject_position(r'\b(?:fit|fits|pass|passes|go|goes)\s+through\b|\b(?:must|needs?\s+to|has\s+to)\s+fit\b', scope)
    return product < 0 or customer > product or access_request > product


def extract_constraints(text: str, previous: Mapping[str, str] | None = None) -> dict[str, str]:
    out = dict(previous or {})
    raw = text or ""
    heights = height_measurements(raw)
    if heights:
        users = _retain_person_measurements(raw, [(start, value) for start, _, value in heights], out.get('user_heights_in'))
        out['user_heights_in'] = json.dumps(users, sort_keys=True, separators=(',', ':'))
        out['height_in'] = f'{max(users.values()):g}'
    weights = []
    for match in re.finditer(r"(?<!\d)(\d{1,4}(?:\.\d+)?)\s*(lbs?|pounds?|kgs?|kilograms?|stone|st)(?![a-z])", raw, re.I):
        if _negated_measurement(raw, match.start()):
            continue
        if not _is_person_weight(raw, match.start(), match.end()):
            continue
        unit = match.group(2).lower()
        value = float(match.group(1)) * (2.20462 if unit.startswith(('kg', 'kilogram')) else 14 if unit in {'stone', 'st'} else 1)
        if 40 <= value <= 2200:
            weights.append((match.start(), value))
    if weights:
        users = _retain_person_measurements(raw, weights, out.get('user_weights_lb'))
        out['user_weights_lb'] = json.dumps(users, sort_keys=True, separators=(',', ':'))
        out['weight_lb'] = f'{max(users.values()):g}'
    door_values = _doorway_measurements(raw)
    if door_values:
        # Different doors along one delivery route must all be cleared. An
        # explicit correction supersedes the older measurement in this turn.
        correction = bool(re.search(r'\b(?:actually|correction|correcting|remeasured|instead|not|now)\b', raw, re.I))
        value = max(door_values)[1] if correction else min(width for _, width in door_values)
        if 0 < value <= 1000:
            out['doorway_in'] = f'{value:g}'
            if out.get('space') != 'Small Room':
                out['space'] = 'Narrow Doorway'
    if re.search(r"\b(?:assembled[ -]only|fully assembled|no disassembly|without\s+(?:any\s+)?disassembl\w*|without\s+(?:taking|pulling)\s+(?:it|the chair|anything)\s+apart|(?:can(?:not|'t)|do not|don't)\s+(?:(?:safely|be)\s+)?(?:disassembl\w*|remove\s+(?:the\s+|my\s+)?(?:chair['’]s\s+)?side\s+panels?))\b", raw, re.I):
        out['doorway_fit'] = 'assembled'
    elif re.search(r"\b(?:ok(?:ay)?\s+to\s+disassembl\w*|allow\s+disassembly|disassembly\s+(?:is\s+)?(?:ok(?:ay)?|fine|allowed)|(?:i|we)\s+can\s+take\s+(?:it|the chair)\s+apart|(?:i|we)(?:'m|\s+am|\s+are)?\s+(?:ok(?:ay)?|fine)\s+with\s+disassembl\w*)\b", raw, re.I):
        out['doorway_fit'] = 'disassembled'
    if re.search(r'\b(?:no\s+(?:doorway|door)\s+(?:limit|constraint)|doorway\s+(?:is\s+)?(?:not|no longer)\s+a\s+constraint)\b', raw, re.I):
        out['doorway_in'] = 'skip'
        if out.get('space') != 'Small Room':
            out['space'] = 'No Space Constraint'
    budgets = []
    contextual_budgets = []
    for match in re.finditer(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?|(?<![\w.])(\d+(?:\.\d+)?)\s*k\b|\b(?:budget|under|up to|maximum|max|no more than)\s*(?:(?:is|of|now|USD)\s*)*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?|\bUSD\s+(\d[\d,]*(?:\.\d+)?)|(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(?:USD|dollars?)\b", raw, re.I):
        if _negated_measurement(raw, match.start()):
            continue
        if re.match(r'\s*(?:lbs?|pounds?|kgs?|kilograms?|stone|cm|centimet(?:er|re)s?|mm|millimet(?:er|re)s?|inches|inch|feet|ft|[2345]?d)\b', raw[match.end():], re.I):
            continue
        prefix = raw[max(0, match.start()-5):match.start()]
        if re.search(r'\b(?:CAD|AUD|CA|AU)\s*$', prefix, re.I) or re.search(r'\b(?:C|A)$', prefix) or re.match(r'\s*(?:CAD|AUD|EUR|GBP|KRW|JPY)\b', raw[match.end():], re.I):
            continue
        value = float((match.group(1) or match.group(3) or match.group(4) or match.group(6) or match.group(7)).replace(',', ''))
        if match.group(2) or match.group(3) or match.group(5):
            value *= 1000
        if math.isfinite(value) and 0 <= value:
            budgets.append((match.start(), value))
            if match.group(4):
                contextual_budgets.append((match.start(), value))
    number_words = '|'.join((*_SPOKEN_NUMBERS, 'hundred', 'thousand', 'and'))
    for match in re.finditer(r'\b(?:budget|under|up to|maximum|max|no more than)\s*(?:(?:is|of|now|USD)\s+)*(?P<amount>(?:' + number_words + r')\b(?:[\s-]+(?:' + number_words + r')\b){0,13})(?:(?:US\s+)?dollars?\b|USD\b)?', raw, re.I):
        words = match.group('amount').strip().rstrip('-')
        if words and not _negated_measurement(raw, match.start()) and any(word in _SPOKEN_NUMBERS for word in re.split(r'[\s-]+', words.lower())):
            value = _spoken_amount(words)
            budgets.append((match.start(), value))
            contextual_budgets.append((match.start(), value))
    if budgets:
        out['budget_max_usd'] = f'{max(contextual_budgets or budgets)[1]:g}'
    if re.search(r"\b(?:no budget (?:limit|cap)|unlimited budget|budget (?:is )?unlimited|budget\s+(?:is\s+)?(?:no longer\s+(?:a\s+)?|not\s+a\s+)(?:restriction|constraint|limit|concern|issue))\b", raw, re.I):
        out['budget_max_usd'] = ''
        out['budget'] = ''
    excluded = set(filter(None, out.get('excluded_mechanisms', '').split(',')))
    positive_mechanisms = []
    for match in re.finditer(r"\b([2345])\s*D\b", raw, re.I):
        mechanism = match.group(1) + 'D'
        if _negated_preference(raw, match.start()):
            excluded.add(mechanism)
            if out.get('mechanism') == mechanism:
                out.pop('mechanism')
        else:
            out['mechanism'] = mechanism
            positive_mechanisms.append(mechanism)
            excluded.discard(mechanism)
    alternatives = re.search(r'\b([2345])\s*d\s+or\s+([2345])\s*d\b', raw, re.I)
    if alternatives and not _negated_preference(raw, alternatives.start()):
        allowed = {alternatives.group(1) + 'D', alternatives.group(2) + 'D'}
        out['allowed_mechanisms'] = ','.join(sorted(allowed))
        out['mechanism'] = ''
        excluded.difference_update(allowed)
    elif positive_mechanisms:
        out['allowed_mechanisms'] = ''
    if excluded or 'excluded_mechanisms' in out:
        out['excluded_mechanisms'] = ','.join(sorted(excluded))
    intensity = intensity_preference(raw)
    if intensity:
        out['intensity'] = intensity
    return out


def doorway_width_from_text(text: str) -> float | None:
    """A grounded home width, bare width answer, or explicit access question."""
    exact = extract_constraints(text).get('doorway_in')
    if exact and exact != 'skip':
        return float(exact)
    match = re.fullmatch(r'\s*(\d{2}(?:\.\d+)?)\s*(?:"|in(?:ch(?:es)?)?)\s*(?:wide)?[.!?\s]*', text or '', re.I)
    if match is None:
        match = re.search(r'\b(?:fit|pass|go)\s+through\s+(?:a\s+|my\s+)?(\d{2}(?:\.\d+)?)\s*(?:"|in(?:ch(?:es)?)?)\b', text or '', re.I)
    if match:
        width = float(match.group(1))
        if 20 <= width <= 48:
            return width
    return None


def product_constraint_issues(product, constraints: Mapping[str, str], *, price_usd=None) -> list[str]:
    """Return failed or unverified hard gates; an empty list permits ranking."""
    from sales_spec_index import lookup_fit_spec

    if product is None:
        return ['product_unverified']
    issues = []
    cap = constraints.get('budget_max_usd')
    price = price_usd if price_usd is not None else getattr(product, 'price_usd', None)
    if cap and (price is None or float(price) > float(cap)):
        issues.append('budget')
    allowed = set(filter(None, constraints.get('allowed_mechanisms', '').lower().split(',')))
    mechanism = constraints.get('mechanism') if not allowed else None
    if mechanism and mechanism.lower() not in str(getattr(product, 'massage_mechanism', '')).lower():
        issues.append('mechanism')
    if allowed and not any(value in str(getattr(product, 'massage_mechanism', '')).lower() for value in allowed):
        issues.append('mechanism')
    excluded = set(filter(None, constraints.get('excluded_mechanisms', '').lower().split(',')))
    if any(value in str(getattr(product, 'massage_mechanism', '')).lower() for value in excluded):
        issues.append('excluded_mechanism')
    weight = constraints.get('weight_lb')
    if weight:
        fit = lookup_fit_spec(getattr(product, 'display_name', '') or getattr(product, 'title', ''))
        if fit is None or fit.max_user_lb is None or float(weight) > fit.max_user_lb:
            issues.append('weight_capacity')
    if constraints.get('doorway_in') and constraints.get('doorway_in') != 'skip':
        fit = lookup_fit_spec(getattr(product, 'display_name', '') or getattr(product, 'title', ''))
        mode = constraints.get('doorway_fit') or 'assembled'
        width = fit.doorway_for_mode(mode) if fit else None
        if width is None or width > float(constraints['doorway_in']):
            issues.append('doorway')
    return issues


def constraint_verification_notes(constraints: Mapping[str, str]) -> list[str]:
    notes = []
    if constraints.get('height_in'):
        notes.append('Your exact height is saved, but supported user-height limits are not published in this catalog. A specialist should confirm height fit before purchase.')
    if constraints.get('intensity') == 'Strong':
        notes.append('Deep-tissue feel varies by model and user; the mechanism label alone does not verify pressure strength. A showroom trial or specialist confirmation is recommended.')
    return notes
