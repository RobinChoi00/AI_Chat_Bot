"""Exact shopper constraints shared by deterministic Sales recommendation paths.

Case-book buckets rank candidates; they must never replace a stated numerical
limit. Missing published fit information is not evidence of a safe fit.
"""
from __future__ import annotations

import re
from typing import Mapping


def extract_constraints(text: str, previous: Mapping[str, str] | None = None) -> dict[str, str]:
    out = dict(previous or {})
    raw = text or ""
    heights = []
    for match in re.finditer(r"(?<!\d)([3-7])\s*(?:'|ft\b|feet\b)\s*(\d{1,2})?\s*(?:\"|in\b|inches\b)?", raw, re.I):
        inches = int(match.group(2) or 0)
        if inches < 12:
            heights.append((match.start(), int(match.group(1)) * 12 + inches))
    for match in re.finditer(r"(?<!\d)(1\d{2}|2[0-2]\d)\s*cm(?![a-z])", raw, re.I):
        heights.append((match.start(), round(int(match.group(1)) / 2.54)))
    if heights:
        out['height_in'] = str(max(heights)[1])
    weights = []
    for match in re.finditer(r"(?<!\d)(\d{2,3}(?:\.\d+)?)\s*(lbs?|pounds?|kg)(?![a-z])", raw, re.I):
        value = float(match.group(1)) * (2.20462 if match.group(2).lower() == 'kg' else 1)
        if 40 <= value <= 700:
            weights.append((match.start(), value))
    if weights:
        out['weight_lb'] = f'{max(weights)[1]:g}'
    doors = list(re.finditer(r"\b(\d{2}(?:\.\d+)?)\s*(?:\"|inch(?:es)?|in\b)\s*(?:wide\s+)?(?:door|doorway)\b|\b(?:door|doorway)\s*(?:is\s*)?(\d{2}(?:\.\d+)?)\s*(?:\"|inch(?:es)?|in\b)", raw, re.I))
    if doors:
        value = float(doors[-1].group(1) or doors[-1].group(2))
        if 20 <= value <= 48:
            out['doorway_in'] = f'{value:g}'
    budgets = []
    contextual_budgets = []
    for match in re.finditer(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?|(?<![\w.])(\d+(?:\.\d+)?)\s*k\b|\b(?:budget|under|up to|maximum|max|no more than)\s*(?:is\s*|of\s*|now\s*)?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", raw, re.I):
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
