"""Exact shopper constraints shared by deterministic Sales recommendation paths.

Case-book buckets rank candidates; they must never replace a stated numerical
limit. Missing published fit information is not evidence of a safe fit.
"""
from __future__ import annotations

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
    return bool(re.search(r'\b(?:not|instead of|rather than|used to be|previously|formerly)\s*$', text[max(0, start - 32):start], re.I))


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
    dimension = _latest_subject_position(r'\b(?:door(?:way)?|opening|entrance|ceiling|hall(?:way)?|chair|room|clearance|space|height\s+(?:range|limit)|supported\s+height)\b', prefix)
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
    for match in re.finditer(r'(?<!\d)(1\d{2}|2[0-2]\d)\s*cm(?![a-z])', text or '', re.I):
        if (include_negated or not _negated_measurement(text, match.start())) and (include_dimensions or _is_person_height(text, match.start(), match.end())):
            values.append((match.start(), match.end(), round(int(match.group(1)) / 2.54)))
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
        r'(?:for\s+(?:a\s+)?|(?:that\s+can|to)\s+(?:support|handle|hold)\s+)'
        r'(?:(?:up\s+to|at\s+least)\s*)?$', prefix, re.I,
    )
    if requirement and not re.search(r"\b(?:don't|do\s+not|not)\s*$", prefix[:requirement.start()], re.I):
        return True
    person = _latest_subject_position(r"\b(?:i\s+(?:am|weigh)|i'm|my\s+(?:body\s+)?weight(?:\s+is)?|(?:buyer|user|person|husband|wife|father|mother|dad|mom|spouse)(?:\s+(?:is|weighs?|who\s+weighs?))?)\b", prefix)
    product = _latest_subject_position(r"\b(?:capacity|rated(?:\s+for)?|rating|limits?|max(?:imum)?|supports?|holds?|can\s+handle|(?:chair|product|shipping)(?:'s)?\s+weight|weighs|costs?)\b", prefix)
    if product > person:
        return False
    if re.match(r'\s*(?:user|person|buyer|man|woman)\b', suffix, re.I):
        return True
    object_subject = _latest_subject_position(r'\b(?:chair|package|product|shipment|box|unit|device)\b', prefix)
    return object_subject < 0 or person > object_subject


def extract_constraints(text: str, previous: Mapping[str, str] | None = None) -> dict[str, str]:
    out = dict(previous or {})
    raw = text or ""
    heights = height_measurements(raw)
    if heights:
        out['height_in'] = str(heights[-1][2])
    weights = []
    for match in re.finditer(r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*(lbs?|pounds?|kg)(?![a-z])", raw, re.I):
        if _negated_measurement(raw, match.start()):
            continue
        if not _is_person_weight(raw, match.start(), match.end()):
            continue
        value = float(match.group(1)) * (2.20462 if match.group(2).lower() == 'kg' else 1)
        if 40 <= value <= 2200:
            weights.append((match.start(), value))
    if weights:
        out['weight_lb'] = f'{max(weights)[1]:g}'
    doors = [match for match in re.finditer(
        r'\b(\d{2}(?:\.\d+)?)\s*-?\s*(?:"|inch(?:es)?|in\b)\s*(?:wide\s+)?(?:door(?:way)?|opening|entrance)\b|'
        r'\b(?:door(?:way)?|entrance)\b(?:\s+(?:opening|width|clearance|is|measures|at|only|just|about|exactly)|\s*[:=]){0,6}\s*'
        r'(\d{2}(?:\.\d+)?)\s*-?\s*(?:"|inch(?:es)?|in\b)|'
        r'\b(?:door(?:way)?|entrance)\b[^.;!?]{0,60}?\b(\d{2}(?:\.\d+)?)\s*-?\s*(?:"|inch(?:es)?|in\b)\s+wide\b', raw, re.I)
        if not _negated_measurement(raw, match.start())]
    if doors:
        value = float(next(group for group in doors[-1].groups() if group is not None))
        if 20 <= value <= 48:
            out['doorway_in'] = f'{value:g}'
            if out.get('space') != 'Small Room':
                out['space'] = 'Narrow Doorway'
    if re.search(r"\b(?:assembled[ -]only|fully assembled|no disassembly|without\s+(?:any\s+)?disassembl\w*|without\s+(?:taking|pulling)\s+(?:it|the chair|anything)\s+apart|(?:can(?:not|'t)|do not|don't)\s+(?:be\s+)?disassembl\w*)\b", raw, re.I):
        out['doorway_fit'] = 'assembled'
    elif re.search(r"\b(?:(?:ok(?:ay)?|able)\s+to\s+disassembl\w*|allow\s+disassembly|disassembly\s+(?:is\s+)?(?:ok(?:ay)?|fine|allowed)|can\s+(?:be\s+disassembled|take\s+(?:it|the chair)\s+apart))\b", raw, re.I):
        out['doorway_fit'] = 'disassembled'
    if re.search(r'\b(?:no\s+(?:doorway|door)\s+(?:limit|constraint)|doorway\s+(?:is\s+)?(?:not|no longer)\s+a\s+constraint)\b', raw, re.I):
        out['doorway_in'] = 'skip'
        if out.get('space') != 'Small Room':
            out['space'] = 'No Space Constraint'
    budgets = []
    contextual_budgets = []
    for match in re.finditer(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?|(?<![\w.])(\d+(?:\.\d+)?)\s*k\b|\b(?:budget|under|up to|maximum|max|no more than)\s*(?:is\s*|of\s*|now\s*)?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", raw, re.I):
        if _negated_measurement(raw, match.start()):
            continue
        value = float((match.group(1) or match.group(3) or match.group(4)).replace(',', ''))
        if match.group(2) or match.group(3) or match.group(5):
            value *= 1000
        if 500 <= value <= 50000:
            budgets.append((match.start(), value))
            if match.group(4):
                contextual_budgets.append((match.start(), value))
    if budgets:
        out['budget_max_usd'] = f'{max(contextual_budgets or budgets)[1]:g}'
    if re.search(r"\b(?:no budget (?:limit|cap)|unlimited budget|budget (?:is )?unlimited)\b", raw, re.I):
        out['budget_max_usd'] = ''
    mechanisms = list(re.finditer(r"\b([2345])\s*D\b", raw, re.I))
    positive = [m for m in mechanisms if not re.search(r"\b(?:not|no|avoid|don't want|don't need|instead of|rather than)\s*$", raw[max(0, m.start()-20):m.start()], re.I)]
    if positive:
        out['mechanism'] = positive[-1].group(1) + 'D'
    if re.search(r"\b(?:not\s+(?:too\s+)?(?:strong|firm)|gentle|soft|light pressure)\b", raw, re.I):
        out['intensity'] = 'Gentle'
    elif re.search(r"\b(?:deep[ -]?tissue|strong|firm)\b", raw, re.I):
        out['intensity'] = 'Strong'
    return out


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
    mechanism = constraints.get('mechanism')
    if mechanism and mechanism.lower() not in str(getattr(product, 'massage_mechanism', '')).lower():
        issues.append('mechanism')
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
