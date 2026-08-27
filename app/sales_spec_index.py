"""
sales_spec_index.py
===================
Runtime fit specs keyed by sales / case-book model name.

Authoritative doorway (assembled / disassembled), max user weight, and wall
clearance come from ``data/sales/spec_index.json`` (exported from the Massage
Chair specification workbook). Recommendation filtering must use this index
per *model*, not the shared case-row reason string.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sales"
_SPEC_INDEX_PATH = _DATA_DIR / "spec_index.json"

# Required capacity = top of the shopper weight band (same policy as refill).
WEIGHT_HARD_LB: dict[str, float] = {
    "≤180 lb": 180,
    "181–220 lb": 220,
    "221–260 lb": 260,
    "261–300 lb": 297,
    "301+ lb": 330,
}

SMALL_ROOM_MAX_WALL_IN = 6.0


@dataclass(frozen=True)
class ModelFitSpec:
    name: str
    brand: str = ""
    max_user_lb: Optional[float] = None
    door_asm_in: Optional[float] = None
    door_dis_in: Optional[float] = None
    wall_clearance_in: Optional[float] = None

    def doorway_for_mode(self, mode: str) -> Optional[float]:
        """Return the clearance inches to compare against the shopper's door.

        * ``assembled`` (default / conservative): assembled only
        * ``disassembled``: prefer disassembled when listed, else assembled
        * ``either``: best (smallest) of the two available numbers
        """
        mode = (mode or "assembled").strip().lower()
        asm = self.door_asm_in
        dis = self.door_dis_in
        if mode == "disassembled":
            return dis if dis is not None else asm
        if mode == "either":
            vals = [v for v in (asm, dis) if v is not None]
            return min(vals) if vals else None
        return asm if asm is not None else dis


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


@lru_cache(maxsize=1)
def load_spec_index() -> dict[str, ModelFitSpec]:
    """Map normalized model name → ModelFitSpec."""
    path = _SPEC_INDEX_PATH
    if not path.is_file():
        logger.warning("sales_spec_index: missing %s", path)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("sales_spec_index: failed to read %s", path)
        return {}
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, list):
        logger.warning("sales_spec_index: unexpected shape in %s", path)
        return {}

    out: dict[str, ModelFitSpec] = {}
    for row in models:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        spec = ModelFitSpec(
            name=name,
            brand=str(row.get("brand") or "").strip(),
            max_user_lb=_as_float(row.get("max_user_lb")),
            door_asm_in=_as_float(row.get("door_asm_in")),
            door_dis_in=_as_float(row.get("door_dis_in")),
            wall_clearance_in=_as_float(row.get("wall_clearance_in")),
        )
        keys = {_normalize_key(name)}
        for alias in row.get("aliases") or []:
            alias_s = str(alias or "").strip()
            if alias_s:
                keys.add(_normalize_key(alias_s))
        for key in keys:
            if key:
                out[key] = spec
    return out


def lookup_fit_spec(model_name: str) -> Optional[ModelFitSpec]:
    key = _normalize_key(model_name)
    if not key:
        return None
    index = load_spec_index()
    hit = index.get(key)
    if hit is not None:
        return hit
    # Soft contains match against longer index keys / names.
    for cand_key, spec in index.items():
        if key in cand_key or cand_key in key:
            return spec
    return None


def doorway_inches_for_model(
    model_name: str,
    *,
    mode: str = "assembled",
) -> Optional[float]:
    spec = lookup_fit_spec(model_name)
    if spec is None:
        return None
    return spec.doorway_for_mode(mode)


def weight_ok(model_name: str, weight_bucket: str) -> bool:
    """True when unknown or capacity covers the shopper band."""
    need = WEIGHT_HARD_LB.get((weight_bucket or "").strip())
    if need is None:
        return True
    spec = lookup_fit_spec(model_name)
    if spec is None or spec.max_user_lb is None:
        return True
    return spec.max_user_lb >= need


def wall_ok(model_name: str, space: str) -> bool:
    if (space or "").strip() != "Small Room":
        return True
    spec = lookup_fit_spec(model_name)
    if spec is None or spec.wall_clearance_in is None:
        return True
    return spec.wall_clearance_in <= SMALL_ROOM_MAX_WALL_IN


def doorway_ok(
    model_name: str,
    *,
    limit_in: Optional[float],
    mode: str = "assembled",
) -> bool:
    """True when no limit, unknown spec, or chair clears the doorway."""
    if limit_in is None:
        return True
    door = doorway_inches_for_model(model_name, mode=mode)
    if door is None:
        return True
    return door <= limit_in


def _as_float(value) -> Optional[float]:
    if value is None or value == "" or value == "-":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else None


def clear_spec_index_cache() -> None:
    load_spec_index.cache_clear()
