"""Company height guidance for chair fit, approved by Sales (Nov 2026).

Standard-line chairs are designed for users 5'0"–6'2"; the XL line covers
6'3"–6'5". These are line-level design ranges, not per-model specifications,
so heights outside both ranges still go to a showroom trial or a specialist.
"""

from __future__ import annotations

import json
import re
from typing import Optional

STANDARD_MIN_IN = 60
STANDARD_MAX_IN = 74
XL_MAX_IN = 77

STANDARD_RANGE = "5'0\"–6'2\""
XL_RANGE = "6'3\"–6'5\""

_XL_TITLE = re.compile(r"\bXL\b|\bbig\s*(?:&|and)\s*tall\b", re.I)


def feet(inches: float) -> str:
    whole_feet, rest = divmod(round(inches), 12)
    return f"{whole_feet}'{rest}\""


def is_xl_line(product) -> bool:
    title = f"{getattr(product, 'display_name', '') or ''} {getattr(product, 'title', '') or ''}"
    return bool(_XL_TITLE.search(title))


def needs_xl_line(height_in) -> bool:
    try:
        return float(height_in) > STANDARD_MAX_IN
    except (TypeError, ValueError):
        return False


def saved_heights(constraints) -> dict[str, float]:
    """Every saved user height (household users first, else the single height)."""
    try:
        users = json.loads((constraints or {}).get('user_heights_in') or '{}')
    except (TypeError, ValueError):
        users = {}
    heights = {str(label): float(value) for label, value in (users or {}).items() if value}
    if not heights and (constraints or {}).get('height_in'):
        try:
            heights = {'user': float(constraints['height_in'])}
        except (TypeError, ValueError):
            heights = {}
    return heights


def height_guidance(height_in: float, product=None) -> str:
    """One sentence on how a height fits the standard and XL lines, or one model."""
    height = feet(height_in)
    name = f"**{product.display_name}**" if product is not None else None
    if height_in < STANDARD_MIN_IN:
        return (f"{height} is below the {STANDARD_RANGE} design range of our standard chairs, "
                "so please try the chair in the showroom or have a specialist confirm the fit before you order.")
    if height_in > XL_MAX_IN:
        return (f"{height} is above the {XL_RANGE} design range of our XL line, the tallest range we publish, "
                "so a specialist should confirm the fit before you order.")
    if product is None:
        if height_in <= STANDARD_MAX_IN:
            return f"{height} is within the {STANDARD_RANGE} design range of our standard chairs."
        return (f"{height} is above the {STANDARD_RANGE} range of our standard chairs, "
                f"so our XL line, designed for {XL_RANGE}, is the right fit.")
    if is_xl_line(product):
        if height_in > STANDARD_MAX_IN:
            return f"{height} is within the {XL_RANGE} design range of our XL line, which includes {name}."
        return (f"{name} is an XL-line chair designed for {XL_RANGE}; for {height}, a standard chair ({STANDARD_RANGE}) "
                "is the closer fit, or try the XL in the showroom.")
    if height_in <= STANDARD_MAX_IN:
        return f"{height} is within the {STANDARD_RANGE} design range of our standard chairs, including {name}."
    return (f"{name} is a standard-line chair designed for up to 6'2\"; for {height}, "
            f"our XL line ({XL_RANGE}) is the recommended fit.")


def height_note(constraints) -> Optional[str]:
    """Line-level height guidance for the saved users of a recommendation."""
    heights = saved_heights(constraints)
    if not heights:
        return None
    if len(heights) == 1:
        return "Height: " + height_guidance(next(iter(heights.values())))
    listed = ', '.join(feet(value) for value in heights.values())
    tallest, shortest = max(heights.values()), min(heights.values())
    if tallest > XL_MAX_IN or shortest < STANDARD_MIN_IN:
        return (f"Height: your users ({listed}) include a height outside our published design ranges "
                f"({STANDARD_RANGE} standard, {XL_RANGE} XL), so a specialist should confirm the fit for both.")
    if needs_xl_line(tallest):
        return (f"Height: the taller user ({feet(tallest)}) is above the {STANDARD_RANGE} standard range, so I'm keeping to "
                f"our XL line ({XL_RANGE}); try it together in the showroom to check the fit at {feet(shortest)} too.")
    return f"Height: both users ({listed}) are within the {STANDARD_RANGE} design range of our standard chairs."
