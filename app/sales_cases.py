"""
sales_cases.py
==============
Lookup layer for the Sales practical-case workbooks.

Source of truth (Excel on Desktop / raw_data/sales):
  - All_Practical_Cases — 28,800 height×weight×budget×goal×… combinations
  - Active_Models — sales priority list

Runtime uses compact gzipped CSVs under ``data/sales/`` so we do not parse
multi-MB xlsx on every chat turn.

Brand selection (storefront → workbook):
  - titanchair.com + osakiusa.com → practical_cases_titan.csv.gz  (Titan & USA.xlsx)
  - osakimassagechair.com       → practical_cases_osaki.csv.gz  (Osakimassagechair.xlsx)
"""

from __future__ import annotations

import csv
import gzip
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "sales"

# Exact labels from the workbook (must match CSV).
HEIGHTS = (
    'Petite (<5\'4")',
    'Average (5\'4"–5\'11")',
    'Tall (6\'0"–6\'2")',
    'Extra Tall (6\'3"+)',
)
WEIGHTS = (
    "≤180 lb",
    "181–220 lb",
    "221–260 lb",
    "261–300 lb",
    "301+ lb",
)
BUDGETS = (
    "Under $3,000",
    "$3,000–$4,999",
    "$5,000–$6,999",
    "$7,000–$9,999",
    "$10,000+",
)
GOALS = (
    "Neck & Shoulders",
    "Upper Back",
    "Lower Back",
    "Hips & Seat",
    "Arms & Hands",
    "Foot & Calf",
    "Full-Body Relaxation",
    "Stretching & Mobility",
)
INTENSITIES = ("Gentle", "Balanced", "Strong", "Highly Adjustable")
FOOT_PRIORITIES = ("Not Important", "Important", "Top Priority")
SPACES = ("No Space Constraint", "Small Room", "Narrow Doorway")

_BUDGET_CODE = {
    "under_3000": "Under $3,000",
    "3000_4999": "$3,000–$4,999",
    "5000_6999": "$5,000–$6,999",
    "7000_9999": "$7,000–$9,999",
    "10000_plus": "$10,000+",
    # legacy button amounts from sales_agent
    "2000": "Under $3,000",
    "3000": "Under $3,000",
    "5000": "$5,000–$6,999",
    "6000": "$5,000–$6,999",
    "8000": "$7,000–$9,999",
}
_HEIGHT_CODE = {
    "petite": 'Petite (<5\'4")',
    "average": 'Average (5\'4"–5\'11")',
    "tall": 'Tall (6\'0"–6\'2")',
    "extra_tall": 'Extra Tall (6\'3"+)',
}
_WEIGHT_CODE = {
    "le180": "≤180 lb",
    "181_220": "181–220 lb",
    "221_260": "221–260 lb",
    "261_300": "261–300 lb",
    "301_plus": "301+ lb",
}
_GOAL_CODE = {
    "neck": "Neck & Shoulders",
    "upper_back": "Upper Back",
    "lower_back": "Lower Back",
    "hips": "Hips & Seat",
    "arms": "Arms & Hands",
    "feet": "Foot & Calf",
    "full_body": "Full-Body Relaxation",
    "stretch": "Stretching & Mobility",
    "back": "Lower Back",  # free-text alias
}
_INTENSITY_CODE = {
    "gentle": "Gentle",
    "balanced": "Balanced",
    "strong": "Strong",
    "adjustable": "Highly Adjustable",
}
_FOOT_CODE = {
    "not_important": "Not Important",
    "important": "Important",
    "top": "Top Priority",
}
_SPACE_CODE = {
    "none": "No Space Constraint",
    "small_room": "Small Room",
    "narrow_door": "Narrow Doorway",
}


@dataclass(frozen=True)
class CaseMatch:
    scenario_id: str
    primary_model: str
    alternative_1: str
    alternative_2: str
    reason: str
    trade_off: str
    do_not_recommend_when: str
    buckets: dict[str, str]
    brand: str


def brand_for_domain(domain: Optional[str]) -> str:
    """Map storefront domain to practical-case workbook key.

    - ``osakimassagechair.com`` → ``osaki`` (Osakimassagechair.xlsx)
    - ``titanchair.com`` / ``osakiusa.com`` (and default) → ``titan`` (Titan & USA.xlsx)
    """
    d = (domain or "").strip().lower()
    # Strip scheme / path / www.
    d = d.replace("https://", "").replace("http://", "")
    if "/" in d:
        d = d.split("/", 1)[0]
    if d.startswith("www."):
        d = d[4:]

    if "osakimassagechair" in d:
        return "osaki"
    # Titan Chair + OsakiUSA share the Titan & USA recommendation book.
    if "titanchair" in d or "osakiusa" in d or "titan" in d:
        return "titan"
    # Safe default for sales chat when domain is missing: OsakiUSA lineup.
    return "titan"


def _cases_path(brand: str) -> Path:
    return _DATA_DIR / f"practical_cases_{brand}.csv.gz"


@lru_cache(maxsize=2)
def _load_case_index(brand: str) -> dict[tuple[str, ...], dict[str, str]]:
    path = _cases_path(brand)
    if not path.is_file():
        logger.warning("sales_cases: missing %s", path)
        return {}

    index: dict[tuple[str, ...], dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                (row.get("Height") or "").strip(),
                (row.get("Weight") or "").strip(),
                (row.get("Budget") or "").strip(),
                (row.get("Primary Goal") or "").strip(),
                (row.get("Intensity") or "").strip(),
                (row.get("Foot & Calf Priority") or "").strip(),
                (row.get("Space Constraint") or "").strip(),
            )
            if not all(key):
                continue
            index[key] = {
                "scenario_id": (row.get("Scenario ID") or "").strip(),
                "primary": (row.get("Primary Model") or "").strip(),
                "alt1": (row.get("Alternative Model 1") or "").strip(),
                "alt2": (row.get("Alternative Model 2") or "").strip(),
                "reason": (row.get("Recommendation Reason") or "").strip(),
                "trade_off": (row.get("Main Trade-Off") or "").strip(),
                "do_not": (row.get("Do Not Recommend When") or "").strip(),
            }
    logger.info("sales_cases: loaded %d cases for brand=%s", len(index), brand)
    return index


def cases_available(brand: str = "osaki") -> bool:
    return bool(_load_case_index(brand))


def height_bucket(height_in: Optional[int]) -> Optional[str]:
    if height_in is None:
        return None
    if height_in < 64:
        return HEIGHTS[0]
    if height_in <= 71:
        return HEIGHTS[1]
    if height_in <= 74:
        return HEIGHTS[2]
    return HEIGHTS[3]


def weight_bucket(weight_lb: Optional[int]) -> Optional[str]:
    if weight_lb is None:
        return None
    if weight_lb <= 180:
        return WEIGHTS[0]
    if weight_lb <= 220:
        return WEIGHTS[1]
    if weight_lb <= 260:
        return WEIGHTS[2]
    if weight_lb <= 300:
        return WEIGHTS[3]
    return WEIGHTS[4]


def budget_bucket(budget_usd: Optional[int]) -> Optional[str]:
    if budget_usd is None:
        return None
    if budget_usd < 3000:
        return BUDGETS[0]
    if budget_usd < 5000:
        return BUDGETS[1]
    if budget_usd < 7000:
        return BUDGETS[2]
    if budget_usd < 10000:
        return BUDGETS[3]
    return BUDGETS[4]


def goal_bucket(focus_areas: Optional[list[str]], text: str = "") -> Optional[str]:
    areas = {a.lower() for a in (focus_areas or [])}
    raw = (text or "").lower()
    if "feet" in areas or "foot" in raw or "calf" in raw:
        return "Foot & Calf"
    if "neck" in areas or "shoulder" in raw:
        return "Neck & Shoulders"
    if "hip" in raw or "glute" in raw:
        return "Hips & Seat"
    if "stretch" in raw or "mobility" in raw:
        return "Stretching & Mobility"
    if "arm" in raw or "hand" in raw:
        return "Arms & Hands"
    if "upper back" in raw:
        return "Upper Back"
    if "lower back" in raw or "lumbar" in raw or "sciatica" in raw:
        return "Lower Back"
    if "back" in areas or "back" in raw:
        return "Lower Back"
    if "relax" in raw or "full body" in raw or "full-body" in raw:
        return "Full-Body Relaxation"
    return None


def intensity_bucket(text: str = "") -> Optional[str]:
    raw = (text or "").lower()
    if re.search(r"\b(gentle|soft|light|elderly|senior)\b", raw):
        return "Gentle"
    if re.search(r"\b(strong|deep|firm|intense|hard)\b", raw):
        return "Strong"
    if re.search(r"\b(adjustable|customi[sz]e|variable)\b", raw):
        return "Highly Adjustable"
    if re.search(r"\b(balanced|medium|moderate)\b", raw):
        return "Balanced"
    return None


def foot_bucket(text: str = "", focus_areas: Optional[list[str]] = None) -> Optional[str]:
    raw = (text or "").lower()
    areas = {a.lower() for a in (focus_areas or [])}
    if re.search(r"\b(top\s+priority|must\s+have.*foot|need.*foot)\b", raw):
        return "Top Priority"
    if "feet" in areas or re.search(r"\b(foot|feet|calf|calves)\b", raw):
        return "Important"
    if re.search(r"\b(no\s+foot|foot\s+not|don'?t\s+care.*foot)\b", raw):
        return "Not Important"
    return None


def space_bucket(text: str = "") -> Optional[str]:
    raw = (text or "").lower()
    if re.search(r"\b(narrow\s+door|doorway|tight\s+door)\b", raw):
        return "Narrow Doorway"
    if re.search(r"\b(small\s+room|apartment|tight\s+space|wall\s+clearance)\b", raw):
        return "Small Room"
    if re.search(r"\b(plenty\s+of\s+space|large\s+room|no\s+space\s+issue)\b", raw):
        return "No Space Constraint"
    return None


def apply_payload_codes(prefs: dict[str, str], payload: str) -> dict[str, str]:
    """Merge ``recommend:…`` button codes into a prefs dict."""
    out = dict(prefs or {})
    parts = [p.strip() for p in (payload or "").split(":") if p.strip()]
    if not parts or parts[0].lower() != "recommend":
        return out

    # Legacy single-token goals: recommend:back | recommend:neck | recommend:feet
    if len(parts) == 2 and parts[1].lower() in _GOAL_CODE:
        out["goal"] = _GOAL_CODE[parts[1].lower()]
        return out

    # recommend:budget:under_3000 | recommend:height:petite | …
    i = 1
    while i + 1 < len(parts):
        key = parts[i].lower()
        code = parts[i + 1].lower().rstrip("+")
        i += 2
        if key == "budget" and code in _BUDGET_CODE:
            out["budget"] = _BUDGET_CODE[code]
        elif key == "height" and code in _HEIGHT_CODE:
            out["height"] = _HEIGHT_CODE[code]
        elif key == "weight" and code in _WEIGHT_CODE:
            out["weight"] = _WEIGHT_CODE[code]
        elif key in {"goal", "focus"} and code in _GOAL_CODE:
            out["goal"] = _GOAL_CODE[code]
        elif key == "intensity" and code in _INTENSITY_CODE:
            out["intensity"] = _INTENSITY_CODE[code]
        elif key == "foot" and code in _FOOT_CODE:
            out["foot"] = _FOOT_CODE[code]
        elif key == "space" and code in _SPACE_CODE:
            out["space"] = _SPACE_CODE[code]
        elif key == "doorway":
            # recommend:doorway:30 | recommend:doorway:skip
            if code == "skip":
                out["doorway_in"] = "skip"
                # Skipping inches also skips the disassemble follow-up.
                out.setdefault("doorway_fit", "assembled")
            else:
                try:
                    inches = float(code)
                except ValueError:
                    inches = None
                if inches is not None and 20 <= inches <= 48:
                    out["doorway_in"] = (
                        str(int(inches)) if inches == int(inches) else str(inches)
                    )
                    if "space" not in out:
                        out["space"] = "Narrow Doorway"
        elif key == "doorway_fit":
            # recommend:doorway_fit:assembled|disassembled|either
            if code in {"assembled", "disassembled", "either"}:
                out["doorway_fit"] = code
    return out


def merge_prefs_from_hints(
    prefs: dict[str, str],
    *,
    height_in: Optional[int] = None,
    weight_lb: Optional[int] = None,
    budget_usd: Optional[int] = None,
    focus_areas: Optional[list[str]] = None,
    free_text: str = "",
) -> dict[str, str]:
    out = dict(prefs or {})
    if "height" not in out:
        hb = height_bucket(height_in)
        if hb:
            out["height"] = hb
    if "weight" not in out:
        wb = weight_bucket(weight_lb)
        if wb:
            out["weight"] = wb
    if "budget" not in out:
        bb = budget_bucket(budget_usd)
        if bb:
            out["budget"] = bb
    if "goal" not in out:
        gb = goal_bucket(focus_areas, free_text)
        if gb:
            out["goal"] = gb
    # Secondary axes: free-text / refine answers may override earlier defaults.
    ib = intensity_bucket(free_text)
    if ib:
        out["intensity"] = ib
    fb = foot_bucket(free_text, focus_areas)
    if fb:
        out["foot"] = fb
    sb = space_bucket(free_text)
    if sb:
        out["space"] = sb
    # Doorway inches from free text ("30 inch door", '32"').
    if "doorway_in" not in out:
        dm = re.search(
            r"(\d{2}(?:\.\d)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\s*(?:door|doorway)?",
            free_text or "",
            re.I,
        )
        if dm:
            try:
                inches = float(dm.group(1))
            except ValueError:
                inches = None
            if inches is not None and 20 <= inches <= 48:
                out["doorway_in"] = (
                    str(int(inches)) if inches == int(inches) else str(inches)
                )
                if "space" not in out:
                    out["space"] = "Narrow Doorway"
    return out


# Must ask these first. Weight and space default so we can show three chairs
# after two answers; the shopper can correct them if fit is tight.
ASK_PREF_KEYS = (
    "height",
    "goal",
)

CORE_PREF_KEYS = (
    "height",
    "weight",
    "space",
    "goal",
)

# Filled automatically once the core four are known so free-text / short
# button paths do not force extra intensity/foot turns.
SECONDARY_PREF_KEYS = (
    "intensity",
    "foot",
)

REQUIRED_PREF_KEYS = CORE_PREF_KEYS + SECONDARY_PREF_KEYS

# Safe mid-case defaults when the customer didn't mention these axes.
_SECONDARY_DEFAULTS = {
    "intensity": "Balanced",
    "foot": "Not Important",
}

_CORE_DEFAULTS = {
    "weight": "181–220 lb",
    "space": "No Space Constraint",
}

DEFAULTABLE_PREF_KEYS = ("weight", "space", "intensity", "foot")

# Soft intensity from massage goal when the shopper didn't say gentle/strong.
_GOAL_INTENSITY = {
    "Neck & Shoulders": "Balanced",
    "Upper Back": "Balanced",
    "Lower Back": "Strong",
    "Hips & Seat": "Balanced",
    "Arms & Hands": "Gentle",
    "Foot & Calf": "Balanced",
    "Full-Body Relaxation": "Gentle",
    "Stretching & Mobility": "Highly Adjustable",
}

# Value / Mid / Premium shelves mapped onto the case-book budget bands.
# Mid covers the full mid stack ($3k–$7k) so the $3–5k band is not orphaned.
TIER_BUDGETS = (
    ("Value (under ~$3k)", ("Under $3,000",)),
    ("Mid-range (~$3–7k)", ("$3,000–$4,999", "$5,000–$6,999")),
    ("Premium ($7k+)", ("$7,000–$9,999", "$10,000+")),
)


def missing_required(prefs: dict[str, str]) -> list[str]:
    return [k for k in REQUIRED_PREF_KEYS if not (prefs or {}).get(k)]


def missing_ask(prefs: dict[str, str]) -> list[str]:
    """Questions we actually ask. Weight/space are defaulted, not quizzed."""
    return [k for k in ASK_PREF_KEYS if not (prefs or {}).get(k)]


def missing_core(prefs: dict[str, str]) -> list[str]:
    return [k for k in CORE_PREF_KEYS if not (prefs or {}).get(k)]


def enrich_implied_prefs(prefs: dict[str, str]) -> dict[str, str]:
    """Fill axes implied by other answers, then defaults after height + goal."""
    out = dict(prefs or {})
    goal = (out.get("goal") or "").strip()
    if goal == "Foot & Calf" and "foot" not in out:
        # Asking foot again after they chose Foot & Calf as the main goal is noise.
        out["foot"] = "Important"
    # Two-question path: height + goal is enough to pick three chairs.
    if out.get("height") and out.get("goal"):
        if not out.get("weight"):
            out["weight"] = _CORE_DEFAULTS["weight"]
        if not out.get("space"):
            out["space"] = _CORE_DEFAULTS["space"]
    # Once height/weight/space/goal are known, skip intensity/foot questions.
    if all(out.get(k) for k in CORE_PREF_KEYS):
        if not out.get("intensity"):
            out["intensity"] = _GOAL_INTENSITY.get(goal) or _SECONDARY_DEFAULTS["intensity"]
        if not out.get("foot"):
            out["foot"] = _SECONDARY_DEFAULTS["foot"]
    return out


def secondary_defaults_applied(
    before: dict[str, str], after: dict[str, str]
) -> list[str]:
    """Return keys that were filled by enrich_implied_prefs defaults."""
    applied: list[str] = []
    for key in DEFAULTABLE_PREF_KEYS:
        if not (before or {}).get(key) and (after or {}).get(key):
            applied.append(key)
    return applied


def lookup_case(
    prefs: dict[str, str],
    *,
    domain: Optional[str] = None,
    brand: Optional[str] = None,
) -> Optional[CaseMatch]:
    brand_key = (brand or brand_for_domain(domain)).strip().lower()
    filled = enrich_implied_prefs(prefs)
    # Budget is optional in the chat flow (we recommend across tiers), but
    # each case-book row is keyed by a specific budget band.
    if missing_required(filled) or not filled.get("budget"):
        return None

    key = (
        filled["height"],
        filled["weight"],
        filled["budget"],
        filled["goal"],
        filled["intensity"],
        filled["foot"],
        filled["space"],
    )
    row = _load_case_index(brand_key).get(key)
    if not row:
        logger.info("sales_cases: no row for %s brand=%s", key, brand_key)
        return None
    return CaseMatch(
        scenario_id=row["scenario_id"],
        primary_model=row["primary"],
        alternative_1=row["alt1"],
        alternative_2=row["alt2"],
        reason=row["reason"],
        trade_off=row["trade_off"],
        do_not_recommend_when=row["do_not"],
        buckets=filled,
        brand=brand_key,
    )


@dataclass(frozen=True)
class ActiveModel:
    name: str
    priority: int  # 1 = primary focus … 5 = de-emphasize; 99 = unknown
    list_price: Optional[float]
    notes: str


@lru_cache(maxsize=2)
def load_active_models(brand: str) -> tuple[ActiveModel, ...]:
    path = _DATA_DIR / f"active_models_{brand}.csv"
    if not path.is_file():
        return ()
    rows: list[ActiveModel] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            raw_pri = (row.get("priority") or "").strip()
            try:
                priority = int(float(raw_pri)) if raw_pri else 99
            except ValueError:
                priority = 99
            if priority < 1 or priority > 10:
                priority = 99
            raw_price = (row.get("list_price") or "").strip()
            try:
                list_price = float(raw_price) if raw_price else None
            except ValueError:
                list_price = None
            rows.append(
                ActiveModel(
                    name=name,
                    priority=priority,
                    list_price=list_price,
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return tuple(rows)


def _norm_model(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def active_priority(model_name: str, brand: str) -> int:
    needle = _norm_model(model_name)
    if not needle:
        return 99
    best = 99
    for row in load_active_models(brand):
        key = _norm_model(row.name)
        if not key:
            continue
        if needle == key or needle in key or key in needle:
            best = min(best, row.priority)
    return best


def rank_case_models(
    primary: str,
    alt1: str,
    alt2: str,
    *,
    brand: str,
) -> tuple[str, list[str], Optional[str]]:
    """Reorder case picks so stronger Active_Models priority leads.

    Returns (lead_model, other_models, optional_note).
    """
    names = [n for n in (primary, alt1, alt2) if (n or "").strip()]
    if not names:
        return "", [], None

    scored = sorted(
        ((active_priority(n, brand), i, n) for i, n in enumerate(names)),
        key=lambda t: (t[0], t[1]),
    )
    lead = scored[0][2]
    others = [n for _, _, n in scored[1:]]
    note = None
    if lead != primary and scored[0][0] < active_priority(primary, brand):
        note = (
            f"Leading with **{lead}** (higher sales priority than "
            f"{primary} for this storefront)."
        )
    return lead, others, note


def short_do_not_recommend(text: str, *, max_len: int = 220) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return ""
    # Keep the first sentence-ish chunk so replies stay scannable.
    for sep in ("; or ", ". ", "; "):
        if sep in raw:
            raw = raw.split(sep, 1)[0].rstrip(".;") + "."
            break
    if len(raw) > max_len:
        raw = raw[: max_len - 1].rstrip() + "…"
    return raw


def clear_case_cache() -> None:
    _load_case_index.cache_clear()
    load_active_models.cache_clear()