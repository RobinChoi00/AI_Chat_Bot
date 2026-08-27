"""
sales_agent.py
==============
Sales AI orchestrator — takes a raw customer message, runs the intent
classifier, applies guardrails, calls deterministic catalog tools, and
returns a structured response (text + quick-reply buttons + optional
handoff signal).

No LLM calls happen here. Everything the customer sees is either:
  1. Hard-coded copy tied to a specific intent, or
  2. Deterministic facts from ``sales_catalog`` (price, specs) / Shopify stock, or
  3. A row from the practical-case workbook (``sales_cases``) for recommendations.

That's what lets us honestly aim for 100% customer satisfaction: the AI
never invents a price, promises a delivery date, or diagnoses a defect.
Anything it isn't sure about drops cleanly to a human handoff.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from sales_catalog import (
    ProductSpecs,
    RecommendationRequest,
    compare,
    list_active_products,
    parse_recommendation_hints,
    price_tier_label,
    recommend,
    resolve_product,
)
from sales_cases import (
    TIER_BUDGETS,
    apply_payload_codes,
    brand_for_domain,
    cases_available,
    enrich_implied_prefs,
    lookup_case,
    merge_prefs_from_hints,
    missing_required,
    rank_case_models,
    secondary_defaults_applied,
    short_do_not_recommend,
)
from sales_cta import (
    after_hours_blurb,
    extract_email,
    financing_page_url,
    format_defaults_note,
    format_fit_guide_summary,
    is_sales_after_hours,
    is_strong_buy_path,
    product_page_url,
    showroom_blurb,
)
from sales_shopify_stock import LiveStockSnapshot, fetch_live_stock, stock_badge
from sales_intent import (
    INTENT_COMPARE,
    INTENT_GREETING,
    INTENT_INTENSITY,
    INTENT_ORDER_STATUS,
    INTENT_PRICE,
    INTENT_RECOMMEND,
    INTENT_SPECS,
    INTENT_STOCK,
    INTENT_UNCLEAR,
    HANDOFF_INTENTS,
    SalesIntent,
    classify,
    handoff_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response DTO
# ---------------------------------------------------------------------------


@dataclass
class QuickReply:
    label: str
    payload: str


@dataclass
class SalesReply:
    reply: str
    intent: str
    handoff: bool = False
    handoff_reason: Optional[str] = None
    quick_replies: list[QuickReply] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    products: list[dict] = field(default_factory=list)  # public dicts, for UI cards
    # Merged into sales_sessions.collected_data by the router (recommend prefs, etc.).
    prefs_patch: Optional[dict] = None
    # Router creates a SalesLead when email is present (chat-native "Email me this pick").
    lead_capture: Optional[dict] = None
    # Tidio Flow stage for static Decision (quick reply) branching — free plan.
    # menu | ask_height | ask_weight | ask_space | ask_goal | recommend | lead | handoff | warranty
    flow_stage: str = "menu"

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "intent": self.intent,
            "handoff": self.handoff,
            "handoff_reason": self.handoff_reason,
            "quick_replies": [{"label": q.label, "payload": q.payload} for q in self.quick_replies],
            "tools_used": self.tools_used,
            "products": self.products,
            "flow_stage": self.flow_stage,
        }


# ---------------------------------------------------------------------------
# Copy helpers
# ---------------------------------------------------------------------------


def _fmt_price(price_usd: Optional[float]) -> str:
    if price_usd is None:
        return "price not published — a rep can confirm"
    return f"${price_usd:,.0f}"


def _menu_quick_replies() -> list[QuickReply]:
    """Default set of quick-reply buttons — always present as an escape hatch."""
    return [
        QuickReply(label="Recommend a chair", payload="recommend"),
        QuickReply(label="Check a price", payload="price"),
        QuickReply(label="Availability / stock", payload="stock"),
        QuickReply(label="Compare two models", payload="compare"),
        QuickReply(label="Talk to a human", payload="human"),
    ]


_MENU_INTRO = (
    "Hi! I'm the Osaki shopping assistant. I can help with **model "
    "recommendations, pricing, specs, and availability**.\n\n"
    "• Already ordered — **shipping / tracking / parts / repair**: "
    "I'll share Warranty Department contact info\n"
    "• **Cancel / refund / discount**: I can connect you with an agent\n\n"
    "What would you like to do?"
)


# ---------------------------------------------------------------------------
# Extract "topic" (model name) from free text
# ---------------------------------------------------------------------------


def _guess_model_from_text(text: str) -> Optional[ProductSpecs]:
    return resolve_product(text or "")


# ---------------------------------------------------------------------------
# Response builders per intent
# ---------------------------------------------------------------------------


def _handoff_reply(intent: SalesIntent) -> SalesReply:
    reply = handoff_message(intent) or (
        "Let me connect you with a human. Share your **email** and someone "
        "from our team will reach out within one business day."
    )
    return SalesReply(
        reply=reply,
        intent=intent.label,
        handoff=True,
        handoff_reason=intent.label,
        quick_replies=[
            QuickReply(label="Share my email", payload="lead:email"),
            QuickReply(label="Back to menu", payload="menu"),
        ],
    )


def _price_candidates(limit: int = 3) -> list[ProductSpecs]:
    """A few mid-catalog anchors when the shopper didn't name a model."""
    active = [p for p in list_active_products() if p.price_usd]
    if not active:
        return []
    active.sort(key=lambda p: p.price_usd or 0)
    # Pick around the 25th / 50th / 75th percentile so we don't dump only cheap chairs.
    n = len(active)
    idxs = sorted({max(0, min(n - 1, int(n * frac))) for frac in (0.25, 0.5, 0.75)})
    picks: list[ProductSpecs] = []
    for i in idxs[:limit]:
        picks.append(active[i])
    return picks


def _price_reply(message: str) -> SalesReply:
    product = _guess_model_from_text(message)
    if product is None:
        samples = _price_candidates()
        lines = [
            "Sure — which model are you asking about? Type the model name "
            "(e.g. *Osaki OS-Pro Maestro LE*), or tap **Recommend a chair** "
            "and I'll match fit first."
        ]
        if samples:
            lines.append("\nPopular price points right now:")
            for p in samples:
                lines.append(f"- **{p.display_name}** — {_fmt_price(p.price_usd)}")
        return SalesReply(
            reply="\n".join(lines),
            intent=INTENT_PRICE,
            quick_replies=[
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.resolve_product"],
            products=[p.as_public_dict() for p in samples],
        )

    price_txt = _fmt_price(product.price_usd)
    availability = (
        "✅ Currently in our active catalog."
        if product.status.lower() == "active"
        else "⚠️ This model is not in our active catalog right now — a rep can confirm availability."
    )
    reply = (
        f"**{product.display_name}** — {price_txt}.\n\n"
        f"{availability}\n\n"
        "This is the published base price. For any other offer, I can connect "
        "you with our sales team."
    )
    return SalesReply(
        reply=reply,
        intent=INTENT_PRICE,
        quick_replies=[
            QuickReply(label=f"Specs for {product.display_name}", payload=f"specs:{product.handle}"),
            QuickReply(label="Compare with another model", payload="compare"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.resolve_product"],
        products=[product.as_public_dict()],
    )


def _stock_reply(message: str, *, domain: str = "osakiusa.com") -> SalesReply:
    product = _guess_model_from_text(message)
    if product is None:
        return SalesReply(
            reply=(
                "Happy to check live inventory — which model? Type the name or "
                "tap **See all models**."
            ),
            intent=INTENT_STOCK,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.resolve_product"],
        )

    live = fetch_live_stock(
        product.handle,
        domain=domain,
        title=product.title or product.display_name,
    )
    tools = ["catalog.resolve_product"]
    if live is not None:
        tools.append("shopify.inventory")
        if live.in_stock:
            qty = (
                f" About **{live.total_inventory}** unit(s) show in inventory right now."
                if live.total_inventory is not None
                else ""
            )
            low = " (low stock)" if live.is_low else ""
            reply = (
                f"**{product.display_name}** is **available to buy** right now{low}.{qty}\n\n"
                "Exact configuration (color/options) is confirmed at checkout."
            )
        else:
            reply = (
                f"**{product.display_name}** is **not available to buy right now** "
                "(out of stock or not listed for sale). A rep can check restock — "
                "I won't invent a date."
            )
    elif product.status.lower() == "active":
        reply = (
            f"**{product.display_name}** is **active in our catalog**, but I "
            "couldn't reach live inventory just now. Checkout will show the "
            "final availability — or I can hand you to a rep."
        )
    else:
        reply = (
            f"**{product.display_name}** is **not in our active catalog** right now. "
            "A rep can confirm restock timing — I won't guess a date."
        )
    return SalesReply(
        reply=reply,
        intent=INTENT_STOCK,
        quick_replies=[
            QuickReply(label="Check the price", payload=f"price:{product.handle}"),
            QuickReply(label="See similar models", payload="recommend"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=tools,
        products=[product.as_public_dict()],
    )


_SPEC_QUESTION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bzero[\s-]?grav", re.I), "zero_gravity", "Zero gravity"),
    (re.compile(r"\bheat(?:ing|er)?\b", re.I), "heating", "Heating"),
    (re.compile(r"\bairbags?\b", re.I), "airbag", "Airbags"),
    (re.compile(r"\bfoot\s*rollers?\b|\bcalf\s*rollers?\b", re.I), "foot_roller", "Foot/calf roller"),
    (re.compile(r"\b(?:sl|l|s)[\s-]?track\b|\btrack\s*type\b", re.I), "track_type", "Track"),
    (re.compile(r"\b(?:2|3|4)\s*d\b|\bmechanism\b", re.I), "massage_mechanism", "Mechanism"),
)


def _yes_no_spec(value: str) -> str:
    v = (value or "").strip().lower()
    if not v or v in {"n/a", "na", "none", "-", "no", "false", "0"}:
        return "No"
    if v in {"yes", "true", "1", "y"}:
        return "Yes"
    return value.strip()


def _specs_reply(message: str) -> SalesReply:
    product = _guess_model_from_text(message)
    if product is None:
        return SalesReply(
            reply=(
                "Which model would you like specs for? Type the model name or "
                "tap **See all models**."
            ),
            intent=INTENT_SPECS,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.resolve_product"],
        )

    specs = product.as_public_dict()["specs"]
    lines = [f"**{product.display_name}** — {_fmt_price(product.price_usd)}"]

    asked = [
        (label, key)
        for pattern, key, label in _SPEC_QUESTION_PATTERNS
        if pattern.search(message or "")
    ]
    if asked:
        lines.append("")
        for label, key in asked:
            val = str(specs.get(key) or "").strip()
            if key in {"zero_gravity", "heating", "airbag", "foot_roller"}:
                lines.append(f"- **{label}**: {_yes_no_spec(val)}")
            else:
                lines.append(f"- **{label}**: {val or '—'}")
        lines.append("\nFull quick specs:")

    for label, key in (
        ("Mechanism", "massage_mechanism"),
        ("Track", "track_type"),
        ("Zero gravity", "zero_gravity"),
        ("Heating", "heating"),
        ("Airbag", "airbag"),
        ("Foot roller", "foot_roller"),
        ("Auto programs", "auto_programs"),
        ("Massage styles", "massage_styles"),
    ):
        val = str(specs.get(key) or "").strip()
        if val:
            lines.append(f"- **{label}**: {val}")
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_SPECS,
        quick_replies=[
            QuickReply(label="Check the price", payload=f"price:{product.handle}"),
            QuickReply(label="Compare with another model", payload="compare"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.resolve_product"],
        products=[product.as_public_dict()],
    )


_HEIGHT_REPLIES = (
    QuickReply(label='Under 5\'4"', payload="recommend:height:petite"),
    QuickReply(label='5\'4"–5\'11"', payload="recommend:height:average"),
    QuickReply(label='6\'0"–6\'2"', payload="recommend:height:tall"),
    QuickReply(label='6\'3"+', payload="recommend:height:extra_tall"),
)

_WEIGHT_REPLIES = (
    QuickReply(label="≤180 lb", payload="recommend:weight:le180"),
    QuickReply(label="181–220 lb", payload="recommend:weight:181_220"),
    QuickReply(label="221–260 lb", payload="recommend:weight:221_260"),
    QuickReply(label="261–300 lb", payload="recommend:weight:261_300"),
    QuickReply(label="301+ lb", payload="recommend:weight:301_plus"),
)

_GOAL_REPLIES = (
    QuickReply(label="Neck & shoulders", payload="recommend:goal:neck"),
    QuickReply(label="Lower back", payload="recommend:goal:lower_back"),
    QuickReply(label="Upper back", payload="recommend:goal:upper_back"),
    QuickReply(label="Foot & calf", payload="recommend:goal:feet"),
    QuickReply(label="Full-body relax", payload="recommend:goal:full_body"),
    QuickReply(label="Stretch / mobility", payload="recommend:goal:stretch"),
)

_INTENSITY_REPLIES = (
    QuickReply(label="Gentle", payload="recommend:intensity:gentle"),
    QuickReply(label="Balanced", payload="recommend:intensity:balanced"),
    QuickReply(label="Strong / deep", payload="recommend:intensity:strong"),
    QuickReply(label="Highly adjustable", payload="recommend:intensity:adjustable"),
)

_FOOT_REPLIES = (
    QuickReply(label="Foot not important", payload="recommend:foot:not_important"),
    QuickReply(label="Foot important", payload="recommend:foot:important"),
    QuickReply(label="Foot is top priority", payload="recommend:foot:top"),
)

_SPACE_REPLIES = (
    QuickReply(label="No space issue", payload="recommend:space:none"),
    QuickReply(label="Small room", payload="recommend:space:small_room"),
    QuickReply(label="Narrow doorway", payload="recommend:space:narrow_door"),
)

_DOORWAY_INCH_REPLIES = (
    QuickReply(label='28"', payload="recommend:doorway:28"),
    QuickReply(label='30"', payload="recommend:doorway:30"),
    QuickReply(label='32"', payload="recommend:doorway:32"),
    QuickReply(label='36"+', payload="recommend:doorway:36"),
    QuickReply(label="Not sure", payload="recommend:doorway:skip"),
)

_COMPACT_SPACES = frozenset({"Narrow Doorway", "Small Room"})

_DOORWAY_MSG_RE = re.compile(
    r"(\d{2}(?:\.\d)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
    re.I,
)


def _why_pick(product: ProductSpecs, request: RecommendationRequest) -> str:
    """One short reason so recommendations don't feel like a random list."""
    bits: list[str] = []
    if request.budget_usd and product.price_usd is not None:
        bits.append(f"near your ${request.budget_usd:,.0f} budget")
    if request.height_in and request.height_in >= 74 and product.track_type in {
        "L-Track",
        "SL-Track",
    }:
        bits.append(f"{product.track_type} suits taller users")
    if "back" in request.focus_areas and product.track_type in {"L-Track", "SL-Track"}:
        bits.append("strong back / full-body track coverage")
    if "neck" in request.focus_areas and product.massage_mechanism in {"3D", "4D"}:
        bits.append(f"{product.massage_mechanism} depth for neck/shoulders")
    if "feet" in request.focus_areas and "yes" in (product.foot_roller or "").lower():
        bits.append("includes foot/calf rollers")
    if product.massage_mechanism and not bits:
        bits.append(f"{product.massage_mechanism} mechanism")
    elif product.massage_mechanism and product.massage_mechanism not in " ".join(bits):
        bits.append(product.massage_mechanism)
    return "; ".join(bits[:2])


def _parse_doorway_inches_message(text: str) -> Optional[float]:
    """Pull a doorway width like 30\" / 32 in from free text."""
    match = _DOORWAY_MSG_RE.search(text or "")
    if not match:
        return None
    try:
        inches = float(match.group(1))
    except ValueError:
        return None
    if 20 <= inches <= 48:
        return inches
    return None


def _needs_doorway_inches(prefs: dict[str, str]) -> bool:
    space = (prefs.get("space") or "").strip()
    if space not in _COMPACT_SPACES:
        return False
    return not (prefs.get("doorway_in") or "").strip()


def _doorway_limit_in(prefs: dict[str, str]) -> Optional[float]:
    raw = (prefs.get("doorway_in") or "").strip().lower()
    if not raw or raw == "skip":
        return None
    try:
        return float(raw.rstrip("+"))
    except ValueError:
        return None


def _clarify_doorway_inches(prefs: dict[str, str]) -> SalesReply:
    return SalesReply(
        reply=(
            "What's the **narrowest doorway** on the delivery path "
            "(inches)?\n\n"
            "I'll only keep chairs that can fit — or tap **Not sure** to skip."
        ),
        intent=INTENT_RECOMMEND,
        quick_replies=[
            *_DOORWAY_INCH_REPLIES,
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cases.clarify"],
        prefs_patch={"recommend_prefs": prefs},
        flow_stage="ask_doorway",
    )



def _clarify_recommend(missing: str, prefs: dict[str, str]) -> SalesReply:
    patch = {"recommend_prefs": prefs}
    stage = f"ask_{missing}" if missing in {
        "height", "weight", "goal", "intensity", "foot", "space", "doorway_in"
    } else "ask_height"
    if missing == "height":
        return SalesReply(
            reply=(
                "Happy to recommend a chair. What's the **user height**?\n\n"
                "I'll match fit first, then show **Value / Mid / Premium** options."
            ),
            intent=INTENT_RECOMMEND,
            quick_replies=[*_HEIGHT_REPLIES, QuickReply(label="Talk to a human", payload="human")],
            tools_used=["cases.clarify"],
            prefs_patch=patch,
            flow_stage=stage,
        )
    if missing == "weight":
        return SalesReply(
            reply="Thanks. Roughly what **weight range**?",
            intent=INTENT_RECOMMEND,
            quick_replies=[*_WEIGHT_REPLIES, QuickReply(label="Talk to a human", payload="human")],
            tools_used=["cases.clarify"],
            prefs_patch=patch,
            flow_stage=stage,
        )
    if missing == "space":
        return SalesReply(
            reply=(
                "Any **doorway / space** constraint?\n\n"
                "Narrow doorways and tight rooms matter for delivery and placement."
            ),
            intent=INTENT_RECOMMEND,
            quick_replies=[*_SPACE_REPLIES, QuickReply(label="Talk to a human", payload="human")],
            tools_used=["cases.clarify"],
            prefs_patch=patch,
            flow_stage=stage,
        )
    if missing == "doorway_in":
        return _clarify_doorway_inches(prefs)
    if missing == "goal":
        return SalesReply(
            reply="What's the **main focus** for the massage?",
            intent=INTENT_RECOMMEND,
            quick_replies=[*_GOAL_REPLIES, QuickReply(label="Talk to a human", payload="human")],
            tools_used=["cases.clarify"],
            prefs_patch=patch,
            flow_stage=stage,
        )
    if missing == "intensity":
        return SalesReply(
            reply="Preferred **massage intensity**?",
            intent=INTENT_RECOMMEND,
            quick_replies=[
                *_INTENSITY_REPLIES,
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["cases.clarify"],
            prefs_patch=patch,
            flow_stage=stage,
        )
    if missing == "foot":
        return SalesReply(
            reply="How important are **foot & calf** rollers / kneading?",
            intent=INTENT_RECOMMEND,
            quick_replies=[*_FOOT_REPLIES, QuickReply(label="Talk to a human", payload="human")],
            tools_used=["cases.clarify"],
            prefs_patch=patch,
            flow_stage=stage,
        )
    return SalesReply(
        reply="Got it. What's the **user height**?",
        intent=INTENT_RECOMMEND,
        quick_replies=[*_HEIGHT_REPLIES, QuickReply(label="Talk to a human", payload="human")],
        tools_used=["cases.clarify"],
        prefs_patch=patch,
        flow_stage="ask_height",
    )


def _why_case_pick_lines(
    *,
    prefs: dict[str, str],
    reason: str,
    product: Optional[ProductSpecs],
    priority_note: Optional[str] = None,
) -> list[str]:
    """Customer-facing 'why this chair' bullets from fit guide + catalog facts."""
    lines = ["**Why we recommend it:**"]
    fit_bits: list[str] = []
    if prefs.get("goal"):
        fit_bits.append(f"targets **{prefs['goal']}**")
    if prefs.get("height"):
        fit_bits.append(f"sized for **{prefs['height']}**")
    if prefs.get("weight"):
        fit_bits.append(f"rated for **{prefs['weight']}**")
    if prefs.get("intensity"):
        fit_bits.append(f"**{prefs['intensity']}** intensity")
    if fit_bits:
        lines.append("• Built around your fit: " + "; ".join(fit_bits) + ".")
    cleaned_reason = (reason or "").strip()
    if cleaned_reason:
        lines.append(f"• {cleaned_reason}")
    if product is not None:
        bits = [
            bit
            for bit in (product.massage_mechanism, product.track_type)
            if bit
        ]
        if bits:
            lines.append(
                "• This model’s hardware: **"
                + " + ".join(bits)
                + "**."
            )
        if product.price_usd is not None:
            lines.append(f"• Listed around {_fmt_price(product.price_usd)} on the storefront.")
    if priority_note:
        # Strip markdown stars for nested reuse; caller may already bold names.
        note = re.sub(r"\*\*([^*]+)\*\*", r"\1", priority_note).strip()
        if note:
            lines.append(f"• {note}")
    if len(lines) == 1:
        lines.append("• Best match from our sales fit guide for your answers.")
    return lines


def _short_tier_blurb(
    product: Optional[ProductSpecs],
    *,
    prefs: dict[str, str],
    reason: str = "",
    doorway_in: Optional[float] = None,
) -> Optional[str]:
    """One short line under each tier pick — chat-scannable, no case-book dump."""
    bits: list[str] = []
    if product is not None:
        for bit in (product.massage_mechanism, product.track_type):
            if bit and bit not in bits:
                bits.append(bit)
        if doorway_in is not None and prefs.get("space") in {
            "Narrow Doorway",
            "Small Room",
        }:
            bits.append(f'~{doorway_in:g}" doorway')
        if "yes" in (product.foot_roller or "").lower() and prefs.get("goal") == "Foot & Calf":
            bits.append("foot/calf rollers")
        elif (
            prefs.get("goal")
            and product.track_type in {"L-Track", "SL-Track"}
            and len(bits) < 3
        ):
            if prefs["goal"] in {
                "Lower Back",
                "Upper Back",
                "Neck & Shoulders",
                "Full-Body Relaxation",
            }:
                bits.append(f"fits {prefs['goal'].lower()}")
    if bits:
        return " · ".join(bits[:3])
    cleaned = re.sub(
        r"^Lead with this model in [^:]+:\s*",
        "",
        (reason or "").strip(),
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > 90:
        cleaned = cleaned[:87].rstrip(" ,;") + "…"
    return cleaned


_DOORWAY_IN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*in(?:ch(?:es)?)?\s*min(?:imum)?\s*doorway",
    re.I,
)


def _parse_min_doorway_in(reason: str) -> Optional[float]:
    match = _DOORWAY_IN_RE.search(reason or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _compact_space(prefs: dict[str, str]) -> bool:
    return (prefs.get("space") or "") in {"Narrow Doorway", "Small Room"}


def _stock_rank(snap: Optional[LiveStockSnapshot]) -> int:
    if snap is None:
        return 1
    if snap.in_stock:
        return 0
    return 2


def _collect_tier_candidates(
    prefs: dict[str, str],
    *,
    budgets: tuple[str, ...],
    brand: str,
    used_models: set[str],
) -> list[tuple[str, str, float]]:
    """Return (model_name, reason, doorway_in_or_999) candidates for a tier."""
    seen: set[str] = set()
    out: list[tuple[str, str, float]] = []
    limit = _doorway_limit_in(prefs)
    for budget in budgets:
        match = lookup_case(
            enrich_implied_prefs({**prefs, "budget": budget}),
            brand=brand,
        )
        if match is None:
            continue
        lead, others, _ = rank_case_models(
            match.primary_model,
            match.alternative_1,
            match.alternative_2,
            brand=brand,
        )
        reason = match.reason or ""
        doorway = _parse_min_doorway_in(reason)
        door_key = doorway if doorway is not None else 999.0
        for name in [lead, *others]:
            if not name or name in used_models or name in seen:
                continue
            # Hard filter when shopper gave a doorway width.
            if limit is not None and door_key < 999 and door_key > limit:
                continue
            seen.add(name)
            out.append((name, reason, door_key))
    # If the hard filter wiped the tier, fall back to unfiltered candidates.
    if not out and limit is not None:
        for budget in budgets:
            match = lookup_case(
                enrich_implied_prefs({**prefs, "budget": budget}),
                brand=brand,
            )
            if match is None:
                continue
            lead, others, _ = rank_case_models(
                match.primary_model,
                match.alternative_1,
                match.alternative_2,
                brand=brand,
            )
            reason = match.reason or ""
            doorway = _parse_min_doorway_in(reason)
            door_key = doorway if doorway is not None else 999.0
            for name in [lead, *others]:
                if not name or name in used_models or name in seen:
                    continue
                seen.add(name)
                out.append((name, reason, door_key))
    return out


def _choose_tier_pick(
    candidates: list[tuple[str, str, float]],
    *,
    domain: str,
    prefer_compact: bool,
) -> Optional[tuple[str, Optional[ProductSpecs], Optional[LiveStockSnapshot], str, Optional[float]]]:
    """Pick best candidate by stock, then doorway when space is tight, then list order."""
    if not candidates:
        return None
    scored: list[tuple] = []
    for idx, (name, reason, door_key) in enumerate(candidates):
        product = resolve_product(name)
        snap: Optional[LiveStockSnapshot] = None
        if product is not None:
            snap = fetch_live_stock(
                product.handle,
                domain=domain,
                title=product.title or product.display_name or name,
            )
        door_sort = door_key if prefer_compact else 0.0
        scored.append(
            (
                _stock_rank(snap),
                door_sort,
                idx,
                name,
                product,
                snap,
                reason,
                door_key if door_key < 999 else None,
            )
        )
    scored.sort(key=lambda row: (row[0], row[1], row[2]))
    best = scored[0]
    return best[3], best[4], best[5], best[6], best[7]


def _tiered_case_recommend_reply(
    prefs: dict[str, str],
    *,
    domain: str,
    request: RecommendationRequest,
    defaults_applied: Optional[list[str]] = None,
) -> Optional[SalesReply]:
    """Fit-first recommend: one primary pick per Value / Mid / Premium budget band."""
    brand = brand_for_domain(domain)
    if not cases_available(brand):
        return None

    fit_bits = [
        prefs.get("height"),
        prefs.get("weight"),
        prefs.get("space"),
        prefs.get("goal"),
    ]
    fit_label = ", ".join(b for b in fit_bits if b) or "your answers"
    lines = [
        f"Based on **{fit_label}**, here are three options:",
        "",
    ]

    public_products: list[dict] = []
    used_models: set[str] = set()
    tier_leads: list[dict] = []
    stock_checked = False
    prefer_compact = _compact_space(prefs)
    n = 0

    for tier_label, budgets in TIER_BUDGETS:
        candidates = _collect_tier_candidates(
            prefs,
            budgets=budgets,
            brand=brand,
            used_models=used_models,
        )
        chosen = _choose_tier_pick(
            candidates,
            domain=domain,
            prefer_compact=prefer_compact,
        )
        if chosen is None:
            continue
        pick_name, product, snap, reason, doorway_in = chosen
        if snap is not None:
            stock_checked = True
        # Swap to an in-stock alt when the lead is confirmed OOS.
        if snap is not None and not snap.in_stock:
            for alt_name, alt_reason, alt_door in candidates:
                if alt_name == pick_name:
                    continue
                alt_product = resolve_product(alt_name)
                if alt_product is None:
                    continue
                alt_snap = fetch_live_stock(
                    alt_product.handle,
                    domain=domain,
                    title=alt_product.title or alt_product.display_name or alt_name,
                )
                if alt_snap is not None and alt_snap.in_stock:
                    pick_name, product, snap = alt_name, alt_product, alt_snap
                    reason = alt_reason
                    doorway_in = alt_door if alt_door < 999 else None
                    break

        used_models.add(pick_name)
        badge = stock_badge(snap) if stock_checked else None
        store_handle = (snap.handle if snap and snap.handle else None) or (
            product.handle if product else None
        )
        url = product_page_url(domain, store_handle or "") if store_handle else None
        n += 1
        display = (product.display_name if product else None) or pick_name
        price = _fmt_price(product.price_usd) if product else "price on request"
        stock_bit = f" · *{badge}*" if badge else ""

        lines.append(f"**{n}. {tier_label}** — **{display}** · {price}{stock_bit}")
        blurb = _short_tier_blurb(
            product,
            prefs=prefs,
            reason=reason,
            doorway_in=doorway_in,
        )
        if blurb:
            lines.append(blurb)
        if url:
            lines.append(url)
        if product is not None:
            card = product.as_public_dict()
            if url:
                card["product_url"] = url
            if badge:
                card["stock"] = badge
            public_products.append(card)
        lines.append("")
        tier_leads.append(
            {
                "tier": tier_label,
                "model": pick_name,
                "display": display,
                "handle": product.handle if product else None,
                "url": url,
                "stock": badge,
            }
        )

    if not tier_leads:
        return None

    defaults_note = format_defaults_note(defaults_applied or [], prefs)
    if defaults_note:
        lines.append(defaults_note)
        lines.append("")

    if is_sales_after_hours():
        lines.append(after_hours_blurb())
        lines.append("")

    lines.append(
        "Reply **1 / 2 / 3** for that chair, email these picks, "
        "or ask to visit the Carrollton showroom."
    )

    primary = tier_leads[0]
    primary_url = primary.get("url")
    pick_summary = format_fit_guide_summary(
        domain=domain,
        prefs=prefs,
        primary=primary["model"],
        alternatives=[t["model"] for t in tier_leads[1:]],
        product_url=primary_url,
        stock_label=primary.get("stock"),
    )

    # Order matters: numbered menu 1–3 must match Value / Mid / Premium.
    quick: list[QuickReply] = []
    for i, tier in enumerate(tier_leads[:3], start=1):
        short_tier = tier["tier"].split("(")[0].strip()
        label = f"{short_tier}: {tier['display']}"
        quick.append(QuickReply(label=label, payload=f"tier:{i}"))
    if len(tier_leads) >= 2:
        quick.append(
            QuickReply(
                label="Compare Value vs Mid",
                payload="compare:tiers:1:2",
            )
        )
    quick.append(QuickReply(label="Email me these picks", payload="lead:save_pick"))
    quick.append(QuickReply(label="Talk to a human", payload="human"))
    # Keep 1–3 as the tier picks after Tidio button ranking/cap.

    tools = ["cases.lookup", "cases.tiered", "catalog.resolve_product", "cta.product_url"]
    if stock_checked:
        tools.append("shopify.inventory")

    return SalesReply(
        reply="\n".join(lines).rstrip(),
        intent=INTENT_RECOMMEND,
        quick_replies=quick,
        tools_used=tools,
        products=public_products,
        flow_stage="recommend",
        prefs_patch={
            "recommend_prefs": prefs,
            "pending_pick_summary": pick_summary,
            "pending_primary": primary["model"],
            "pending_product_url": primary_url,
            "pending_tier_picks": tier_leads,
        },
    )


def _case_recommend_reply(
    prefs: dict[str, str],
    *,
    domain: str,
    request: RecommendationRequest,
    defaults_applied: Optional[list[str]] = None,
) -> Optional[SalesReply]:
    brand = brand_for_domain(domain)
    if not cases_available(brand):
        return None
    match = lookup_case(prefs, brand=brand)
    if match is None:
        return None

    lead, others, priority_note = rank_case_models(
        match.primary_model,
        match.alternative_1,
        match.alternative_2,
        brand=brand,
    )
    model_names = [n for n in [lead, *others] if n]

    resolved: list[tuple[str, Optional[ProductSpecs], Optional[LiveStockSnapshot]]] = []
    stock_checked = False
    for name in model_names:
        product = resolve_product(name)
        snap: Optional[LiveStockSnapshot] = None
        if product is not None:
            snap = fetch_live_stock(
                product.handle,
                domain=domain,
                title=product.title or product.display_name or name,
            )
            if snap is not None:
                stock_checked = True
        resolved.append((name, product, snap))

    def _stock_rank(
        item: tuple[str, Optional[ProductSpecs], Optional[LiveStockSnapshot]],
    ) -> tuple:
        _name, _product, snap = item
        if snap is None:
            return (1, 0)  # unknown — keep mid priority
        if snap.in_stock:
            return (0, 1 if snap.is_low else 0)
        return (2, 0)  # OOS last

    ordered = sorted(enumerate(resolved), key=lambda pair: (_stock_rank(pair[1]), pair[0]))
    resolved = [item for _, item in ordered]

    if resolved and resolved[0][0] != lead:
        # Stock demotion changed the lead.
        lead = resolved[0][0]
        others = [n for n, _, _ in resolved[1:]]
        priority_note = (
            (priority_note + " ") if priority_note else ""
        ) + f"Showing **{lead}** first because live inventory looks better right now."

    bucket_bits = [
        match.buckets.get("budget"),
        match.buckets.get("height"),
        match.buckets.get("weight"),
        match.buckets.get("goal"),
        match.buckets.get("intensity"),
        match.buckets.get("foot"),
        match.buckets.get("space"),
    ]
    lead_product = resolved[0][1] if resolved else None
    lines = [
        "Based on our **sales fit guide** "
        f"({', '.join(b for b in bucket_bits if b)}):",
        "",
        f"**Primary pick: {lead}**",
        "",
    ]
    lines.extend(
        _why_case_pick_lines(
            prefs=match.buckets,
            reason=match.reason,
            product=lead_product,
            priority_note=priority_note,
        )
    )
    defaults_note = format_defaults_note(defaults_applied or [], match.buckets)
    if defaults_note:
        lines.append(f"\n_{defaults_note}_")
    if others:
        lines.append(f"\n**Also consider:** {' / '.join(others)}")
    if match.trade_off and "no major" not in match.trade_off.lower():
        lines.append(f"\nTrade-off: {match.trade_off}")
    caveat = short_do_not_recommend(match.do_not_recommend_when)
    if caveat:
        lines.append(f"\n**Skip this pick if:** {caveat}")

    products: list[ProductSpecs] = []
    seen_handles: set[str] = set()
    priced_lines: list[str] = []
    public_products: list[dict] = []
    product_links: list[tuple[str, str]] = []
    primary_url: Optional[str] = None
    primary_stock: Optional[str] = None
    for name, product, snap in resolved:
        badge = stock_badge(snap) if stock_checked else None
        store_handle = (snap.handle if snap and snap.handle else None) or (
            product.handle if product else None
        )
        url = product_page_url(domain, store_handle or "") if store_handle else None
        if product is None:
            if badge:
                priced_lines.append(f"- **{name}** — catalog match pending ({badge})")
            if url:
                product_links.append((name, url))
            continue
        if product.handle in seen_handles:
            continue
        seen_handles.add(product.handle)
        products.append(product)
        detail = ", ".join(
            bit for bit in (product.massage_mechanism, product.track_type) if bit
        )
        display = product.display_name or name
        line = (
            f"- **{display}** — {_fmt_price(product.price_usd)}"
            + (f" ({detail})" if detail else "")
        )
        if badge:
            line += f" — *{badge}*"
        if url:
            # Plain URL so Tidio / SMS-style clients stay clickable.
            line += f"\n  → {url}"
            product_links.append((display, url))
        priced_lines.append(line)
        card = product.as_public_dict()
        if url:
            card["product_url"] = url
        if badge:
            card["stock"] = badge
        public_products.append(card)
        if primary_url is None and url:
            primary_url = url
            primary_stock = badge

    if priced_lines:
        lines.append("\nLive catalog + stock:")
        lines.extend(priced_lines)
    elif model_names:
        lines.append(
            "\n(Catalog price lookup didn't match every model name — "
            "a rep can confirm live pricing.)"
        )

    if product_links:
        lines.append("\n**Open these links to shop:**")
        for label, url in product_links:
            lines.append(f"• {label}: {url}")

    if is_sales_after_hours():
        lines.append(f"\n{after_hours_blurb()}")

    buy_path = is_strong_buy_path(product_url=primary_url, stock_label=primary_stock)
    finance_url = financing_page_url(domain, product_url=primary_url) if buy_path else None
    if buy_path:
        lines.append(
            "\nReady when you are: tap a product link above (Affirm / financing "
            "options appear at checkout — I won't invent rates), visit the "
            "Carrollton showroom, or email this pick to sales. "
            "Discounts and delivery dates still need a specialist."
        )
    else:
        lines.append(
            "\nNext step: open a product link above, email this pick to sales, "
            "or talk to a specialist (stock / discounts / delivery dates)."
        )

    pick_summary = format_fit_guide_summary(
        domain=domain,
        prefs=match.buckets,
        primary=lead,
        alternatives=others,
        product_url=primary_url,
        stock_label=primary_stock,
    )

    quick: list[QuickReply] = []
    if buy_path and primary_url:
        quick.append(QuickReply(label="Shop this chair", payload=f"open:{primary_url}"))
        if finance_url:
            quick.append(
                QuickReply(label="Financing at checkout", payload=f"cta:financing:{finance_url}")
            )
        quick.append(QuickReply(label="Visit showroom", payload="cta:showroom"))
    elif primary_url:
        quick.append(QuickReply(label="View this chair", payload=f"open:{primary_url}"))
    quick.append(QuickReply(label="Email me this pick", payload="lead:save_pick"))
    # After secondary defaults, offer a short refine path → re-runs case lookup.
    if defaults_applied:
        quick.extend(
            [
                QuickReply(label="Prefer stronger", payload="recommend:intensity:strong"),
                QuickReply(label="Prefer gentler", payload="recommend:intensity:gentle"),
                QuickReply(label="Foot rollers matter", payload="recommend:foot:important"),
                QuickReply(label="Tight space / doorway", payload="recommend:space:small_room"),
            ]
        )
    for product in products[:2]:
        quick.append(
            QuickReply(label=f"Specs for {product.display_name}", payload=f"specs:{product.handle}")
        )
    quick.append(QuickReply(label="Talk to a human", payload="human"))

    tools = ["cases.lookup", "cases.priority", "catalog.resolve_product", "cta.product_url"]
    if stock_checked:
        tools.append("shopify.inventory")
    if buy_path:
        tools.append("cta.conversion")

    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_RECOMMEND,
        quick_replies=quick,
        tools_used=tools,
        products=public_products,
        flow_stage="recommend",
        prefs_patch={
            "recommend_prefs": match.buckets,
            "pending_pick_summary": pick_summary,
            "pending_primary": lead,
            "pending_product_url": primary_url,
        },
    )


def _catalog_tier_recommend_reply(message: str, request: RecommendationRequest) -> SalesReply:
    """Fallback when practical-case file is missing or incomplete."""
    has_hints = any(
        [request.height_in, request.weight_lb, request.budget_usd, request.focus_areas]
    )
    picks = recommend(request, limit=3)
    if not picks:
        return SalesReply(
            reply=(
                "Happy to recommend a chair. What's the **user height**?\n\n"
                "I'll match fit first, then show **Value / Mid / Premium** options."
            ),
            intent=INTENT_RECOMMEND,
            quick_replies=[
                *_HEIGHT_REPLIES,
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.parse_hints"],
            flow_stage="ask_height",
        )

    header_bits: list[str] = []
    if request.height_in:
        header_bits.append(f"height ~{request.height_in}\"")
    if request.weight_lb:
        header_bits.append(f"weight ~{request.weight_lb} lb")
    if request.focus_areas:
        header_bits.append("focus: " + ", ".join(request.focus_areas))

    if header_bits:
        header = (
            f"Based on {'; '.join(header_bits)}, here are options "
            "across **Value / Mid / Premium**:"
        )
    else:
        header = (
            "Here are three strong options across price tiers — "
            "**Value (under ~$3k)**, **Mid-range (~$5–7k)**, and **Premium ($7k+)**:"
        )

    lines = [header]
    for i, product in enumerate(picks, start=1):
        detail = ", ".join(
            [
                bit
                for bit in (
                    product.massage_mechanism,
                    product.track_type,
                    "Zero-G" if "yes" in product.zero_gravity.lower() else "",
                    "Heating" if "yes" in product.heating.lower() else "",
                )
                if bit
            ]
        )
        tier = price_tier_label(product.price_usd)
        why = _why_pick(product, request)
        tier_bit = f"{tier}" if tier else f"Option {i}"
        line = (
            f"**{i}. {tier_bit}**\n"
            f"- **{product.display_name}** — {_fmt_price(product.price_usd)}"
            + (f"  ({detail})" if detail else "")
        )
        if why:
            line += f"\n  → {why}"
        lines.append(line)
    lines.append(
        "\nReply with a number, ask for specs, or connect with a sales specialist."
    )

    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_RECOMMEND,
        quick_replies=[
            *[
                QuickReply(label=f"Specs for {p.display_name}", payload=f"specs:{p.handle}")
                for p in picks[:3]
            ],
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.recommend"]
        + (["catalog.parse_hints"] if has_hints else []),
        products=[p.as_public_dict() for p in picks],
        flow_stage="recommend",
    )


def _recommend_reply(
    message: str,
    *,
    payload: Optional[str] = None,
    domain: str = "osakiusa.com",
    prefs: Optional[dict] = None,
) -> SalesReply:
    request = parse_recommendation_hints(message)
    merged = dict((prefs or {}).get("recommend_prefs") or {})
    if payload:
        merged = apply_payload_codes(merged, payload)
    merged = merge_prefs_from_hints(
        merged,
        height_in=request.height_in,
        weight_lb=request.weight_lb,
        budget_usd=request.budget_usd,
        focus_areas=request.focus_areas,
        free_text=request.free_text or message,
    )
    # Bare "30 inch" / '32"' answers after ask_doorway.
    door_msg = _parse_doorway_inches_message(message or "")
    if door_msg is not None and not (merged.get("doorway_in") or "").strip():
        merged["doorway_in"] = (
            str(int(door_msg)) if door_msg == int(door_msg) else str(door_msg)
        )
        if "space" not in merged:
            merged["space"] = "Narrow Doorway"

    before_defaults = dict(merged)
    # Budget is never asked and never gates the reply — Value/Mid/Premium
    # tiers inject case-book budget bands internally only.
    merged.pop("budget", None)
    merged = enrich_implied_prefs(merged)
    defaults_applied = secondary_defaults_applied(before_defaults, merged)

    # After compact space, ask doorway inches before goal / recommend.
    if (
        merged.get("height")
        and merged.get("weight")
        and (merged.get("space") or "") in _COMPACT_SPACES
        and _needs_doorway_inches(merged)
    ):
        return _clarify_doorway_inches(merged)

    missing = missing_required(merged)
    if missing:
        return _clarify_recommend(missing[0], merged)

    tiered = _tiered_case_recommend_reply(
        merged,
        domain=domain,
        request=request,
        defaults_applied=defaults_applied,
    )
    if tiered is not None:
        return tiered

    return _catalog_tier_recommend_reply(message, request)



_VS_RE = re.compile(r"\b(vs\.?|versus|compared\s+to|difference\s+between)\b", re.IGNORECASE)


def _split_compare_terms(message: str) -> Optional[tuple[str, str]]:
    if not message:
        return None
    parts = _VS_RE.split(message, maxsplit=1)
    if len(parts) >= 3:
        left, _sep, right = parts[0], parts[1], parts[2]
        left = left.strip(" ?,.")
        right = right.strip(" ?,.")
        if left and right:
            return left, right
    return None


def _compare_reply(message: str) -> SalesReply:
    pair = _split_compare_terms(message)
    if pair is None:
        return SalesReply(
            reply=(
                "Sure — which two models? Try *\"OS-Pro Maestro LE vs Titan Pro "
                "Jupiter LE\"* and I'll line up the specs."
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
        )

    result = compare(*pair)
    if result is None:
        return SalesReply(
            reply=(
                "I couldn't confidently match one of those model names to our "
                "catalog. Could you retype the exact model names, or tap "
                "**See all models**?"
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
        )

    left = result["left"]
    right = result["right"]
    diff = result["diff"]
    lines = [
        f"**{left['model']}** — {_fmt_price(left['price_usd'])}",
        f"**{right['model']}** — {_fmt_price(right['price_usd'])}",
        "",
        f"- **Mechanism**: {diff['mechanism'][0] or '—'} vs {diff['mechanism'][1] or '—'}",
        f"- **Track**: {diff['track'][0] or '—'} vs {diff['track'][1] or '—'}",
        f"- **Zero gravity**: {diff['zero_gravity'][0] or '—'} vs {diff['zero_gravity'][1] or '—'}",
        f"- **Heating**: {diff['heating'][0] or '—'} vs {diff['heating'][1] or '—'}",
        f"- **Foot roller**: {diff['foot_roller'][0] or '—'} vs {diff['foot_roller'][1] or '—'}",
    ]
    if diff["price_delta_usd"] is not None:
        delta = diff["price_delta_usd"]
        if abs(delta) < 1:
            lines.append("- **Price gap**: same published price.")
        else:
            direction = "more" if delta > 0 else "less"
            lines.append(
                f"- **Price gap**: the second is about ${abs(delta):,.0f} {direction}."
            )

    differing = [
        name
        for name, key in (
            ("mechanism", "mechanism"),
            ("track", "track"),
            ("zero gravity", "zero_gravity"),
            ("heating", "heating"),
            ("foot roller", "foot_roller"),
        )
        if (diff.get(key) or ("", ""))[0] != (diff.get(key) or ("", ""))[1]
    ]
    if not differing and abs(diff.get("price_delta_usd") or 0) < 1:
        lines.append(
            "\n**Bottom line:** These two sit in the same tier on published "
            "specs/price — a rep can help you choose by feel/fit."
        )
    elif differing:
        lines.append(
            "\n**Biggest differences:** " + ", ".join(differing) + "."
        )
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_COMPARE,
        quick_replies=[
            QuickReply(label="Recommend which one", payload="recommend"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.compare"],
        products=[left, right],
    )


def _intensity_reply(message: str) -> SalesReply:
    product = _guess_model_from_text(message)
    if product is not None:
        mech = product.massage_mechanism or "adjustable"
        return SalesReply(
            reply=(
                f"**{product.display_name}** uses a **{mech} mechanism** with "
                "multiple intensity levels — most people find level 3 of 5 firm "
                "but comfortable. If you want it stronger, 4D models push deeper "
                "into muscles than 2D/3D. A rep can walk you through settings if "
                "you'd like."
            ),
            intent=INTENT_INTENSITY,
            quick_replies=[
                QuickReply(label="Check the price", payload=f"price:{product.handle}"),
                QuickReply(label="Compare with another model", payload="compare"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.resolve_product"],
            products=[product.as_public_dict()],
        )
    return SalesReply(
        reply=(
            "Osaki chairs offer **multiple intensity levels** (typically 5 steps) "
            "and, on 3D/4D models, an adjustable *depth* separately from speed. "
            "Which model did you have in mind? I can pull the exact spec."
        ),
        intent=INTENT_INTENSITY,
        quick_replies=_menu_quick_replies(),
    )


def _order_status_reply(_message: str) -> SalesReply:
    from sales_intent import SalesIntent, INTENT_ORDER_STATUS, handoff_message

    intent = SalesIntent(label=INTENT_ORDER_STATUS, confidence="high", handoff=True)
    return SalesReply(
        reply=handoff_message(intent) or "",
        intent=INTENT_ORDER_STATUS,
        handoff=True,
        handoff_reason=INTENT_ORDER_STATUS,
        quick_replies=[
            QuickReply(label="Back to menu", payload="menu"),
        ],
    )


def _greeting_reply() -> SalesReply:
    return SalesReply(
        reply=_MENU_INTRO,
        intent=INTENT_GREETING,
        quick_replies=_menu_quick_replies(),
    )


def _unclear_reply() -> SalesReply:
    return SalesReply(
        reply=(
            "I want to make sure I answer you correctly instead of guessing. "
            "Pick what you'd like to do, or type your question in a bit more "
            "detail (e.g. *\"price of the OS-Pro Maestro\"*)."
        ),
        intent=INTENT_UNCLEAR,
        quick_replies=_menu_quick_replies(),
    )


def _list_reply() -> SalesReply:
    picks = list_active_products()[:8]
    if not picks:
        return _unclear_reply()
    lines = ["Here's a snapshot of the current catalog:"]
    for product in picks:
        lines.append(
            f"- **{product.display_name}** — {_fmt_price(product.price_usd)}"
            + (f"  ({product.massage_mechanism} / {product.track_type})"
               if product.massage_mechanism or product.track_type else "")
        )
    lines.append("\nTap a model's name in chat or ask for specs, price, or a comparison.")
    return SalesReply(
        reply="\n".join(lines),
        intent="list",
        quick_replies=[
            QuickReply(label="Recommend a chair", payload="recommend"),
            QuickReply(label="Compare two models", payload="compare"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.list_active"],
        products=[p.as_public_dict() for p in picks],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_PAYLOAD_ROUTES = {
    "menu": lambda _msg: _greeting_reply(),
    "list": lambda _msg: _list_reply(),
    "compare": lambda msg: _compare_reply(msg),
    "price": lambda msg: _price_reply(msg),
    "stock": lambda msg: _stock_reply(msg),
    "specs": lambda msg: _specs_reply(msg),
    "intensity": lambda msg: _intensity_reply(msg),
    "human": lambda _msg: _handoff_reply(
        SalesIntent(label="human", confidence="high", handoff=True)
    ),
}


def _ask_email_for_pick(prefs: Optional[dict]) -> SalesReply:
    summary = ((prefs or {}).get("pending_pick_summary") or "").strip()
    primary = ((prefs or {}).get("pending_primary") or "your recommended chair").strip()
    blurb = (
        f"Gladly — share your **email** and I'll pass **{primary}** "
        "(plus your fit-guide notes) to our sales team for follow-up.\n\n"
        "Type your email address in the chat."
    )
    if is_sales_after_hours():
        blurb += f"\n\n{after_hours_blurb()}"
    return SalesReply(
        reply=blurb,
        intent="lead_capture",
        handoff=False,
        quick_replies=[
            QuickReply(label="Talk to a human instead", payload="human"),
            QuickReply(label="Back to menu", payload="menu"),
        ],
        tools_used=["cta.email_pick"],
        prefs_patch={
            "awaiting_email_for_pick": True,
            "pending_pick_summary": summary,
            "pending_primary": primary,
            "pending_product_url": (prefs or {}).get("pending_product_url"),
        },
    )


def _capture_pick_lead(email: str, prefs: Optional[dict], *, domain: str) -> SalesReply:
    summary = ((prefs or {}).get("pending_pick_summary") or "").strip()
    if not summary:
        summary = format_fit_guide_summary(
            domain=domain,
            prefs=(prefs or {}).get("recommend_prefs") or {},
            primary=str((prefs or {}).get("pending_primary") or "Sales AI pick"),
            alternatives=[],
            product_url=(prefs or {}).get("pending_product_url"),
        )
    primary = ((prefs or {}).get("pending_primary") or "your pick").strip()
    url = ((prefs or {}).get("pending_product_url") or "").strip()
    lines = [
        f"Thanks — I saved **{email}** with your pick (**{primary}**) for the sales team.",
        "Someone will follow up (usually next business day for pricing / delivery / discounts).",
    ]
    if url:
        lines.append(f"\nMeanwhile you can review it here: {url}")
    if is_sales_after_hours():
        lines.append(f"\n{after_hours_blurb()}")
    return SalesReply(
        reply="\n".join(lines),
        intent="lead_capture",
        handoff=True,
        handoff_reason="save_pick",
        quick_replies=[
            QuickReply(label="Back to menu", payload="menu"),
            QuickReply(label="Recommend another chair", payload="recommend"),
        ],
        tools_used=["cta.email_pick", "lead.capture"],
        prefs_patch={
            "awaiting_email_for_pick": False,
            "pending_pick_summary": summary,
        },
        lead_capture={
            "email": email,
            "interest_summary": summary,
            "reason": "save_pick",
        },
    )


def _open_product_reply(url: str, prefs: Optional[dict]) -> SalesReply:
    primary = ((prefs or {}).get("pending_primary") or "this chair").strip()
    return SalesReply(
        reply=(
            f"Here's the product page for **{primary}**:\n{url}\n\n"
            "Financing (Affirm) is offered **at checkout** on that page — "
            "I won't invent rates or terms. "
            "Want me to email this pick to sales, or see other options?"
        ),
        intent=INTENT_RECOMMEND,
        quick_replies=[
            QuickReply(label="Email me this pick", payload="lead:save_pick"),
            QuickReply(label="Visit showroom", payload="cta:showroom"),
            QuickReply(label="Recommend again", payload="recommend"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cta.product_url"],
    )


def _financing_cta_reply(url: str, prefs: Optional[dict]) -> SalesReply:
    primary = ((prefs or {}).get("pending_primary") or "this chair").strip()
    return SalesReply(
        reply=(
            f"For **{primary}**, open the product page and choose **Affirm / "
            f"financing at checkout**:\n{url}\n\n"
            "Eligibility and monthly amounts are shown by Affirm there — "
            "I won't invent rates. A specialist can still help with promotions "
            "or delivery timing."
        ),
        intent=INTENT_RECOMMEND,
        quick_replies=[
            QuickReply(label="Shop this chair", payload=f"open:{url}"),
            QuickReply(label="Email me this pick", payload="lead:save_pick"),
            QuickReply(label="Visit showroom", payload="cta:showroom"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cta.financing"],
    )


def _showroom_cta_reply(prefs: Optional[dict]) -> SalesReply:
    primary = ((prefs or {}).get("pending_primary") or "").strip()
    extra = f"\n\nYour current pick: **{primary}**." if primary else ""
    url = ((prefs or {}).get("pending_product_url") or "").strip()
    quick = [
        QuickReply(label="Email me this pick", payload="lead:save_pick"),
        QuickReply(label="Talk to a human", payload="human"),
    ]
    if url.startswith("https://"):
        quick.insert(0, QuickReply(label="Shop this chair", payload=f"open:{url}"))
    return SalesReply(
        reply=showroom_blurb() + extra,
        intent=INTENT_RECOMMEND,
        quick_replies=quick,
        tools_used=["cta.showroom"],
    )


def _payload_reply(
    payload: str,
    message: str,
    *,
    domain: str,
    prefs: Optional[dict],
) -> Optional[SalesReply]:
    parts = (payload or "").split(":")
    root = parts[0].strip().lower() if parts else ""
    if root == "tier" and len(parts) >= 2:
        try:
            n = int(re.sub(r"\D", "", parts[1]) or "0")
        except ValueError:
            n = 0
        if 1 <= n <= 3:
            return _tier_followup_reply(n - 1, prefs, domain=domain)
        return None
    if root == "compare" and len(parts) >= 4 and parts[1].strip().lower() == "tiers":
        try:
            left_n = int(parts[2])
            right_n = int(parts[3])
        except ValueError:
            return None
        return _compare_pending_tiers_reply(prefs, left_n, right_n)
    if root == "recommend":
        if len(parts) >= 2 and parts[1].strip().lower() == "again":
            return _recommend_reply(
                "recommend",
                domain=domain,
                prefs=prefs,
            )
        return _recommend_reply(
            message or "recommend",
            payload=payload,
            domain=domain,
            prefs=prefs,
        )
    if root == "stock":
        return _stock_reply(message or payload, domain=domain)
    if root == "lead":
        action = parts[1].strip().lower() if len(parts) > 1 else "save_pick"
        if action in {"save_pick", "email", "email_pick"}:
            # If they already typed an email with the button tap, capture immediately.
            email = extract_email(message or "")
            if email:
                return _capture_pick_lead(email, prefs, domain=domain)
            return _ask_email_for_pick(prefs)
    if root == "open" and len(parts) >= 2:
        url = payload.split(":", 1)[1].strip()
        if url.startswith("https://"):
            return _open_product_reply(url, prefs)
    if root == "cta":
        action = parts[1].strip().lower() if len(parts) > 1 else ""
        if action == "showroom":
            return _showroom_cta_reply(prefs)
        if action == "financing":
            url = payload.split(":", 2)[2].strip() if len(parts) >= 3 else ""
            if not url.startswith("https://"):
                url = ((prefs or {}).get("pending_product_url") or "").strip()
            if url.startswith("https://"):
                return _financing_cta_reply(url, prefs)
            return _showroom_cta_reply(prefs)
    factory = _PAYLOAD_ROUTES.get(root)
    if factory is None:
        return None
    return factory(message or payload)


def _finalize_flow_stage(result: SalesReply) -> SalesReply:
    """Fill flow_stage for Tidio static Decision branching when callers omit it."""
    if result.flow_stage and result.flow_stage not in {"", "menu"}:
        # Explicit ask_* / recommend / etc. already set.
        if result.flow_stage.startswith("ask_") or result.flow_stage in {
            "recommend",
            "lead",
            "handoff",
            "warranty",
            "shop",
            "ask_doorway",
        }:
            return result
    from sales_intent import WARRANTY_ROUTE_INTENTS

    if result.intent in WARRANTY_ROUTE_INTENTS:
        result.flow_stage = "warranty"
    elif result.lead_capture:
        result.flow_stage = "lead"
    elif result.handoff:
        result.flow_stage = "handoff"
    elif result.intent == INTENT_GREETING or result.intent == INTENT_UNCLEAR:
        result.flow_stage = "menu"
    elif "cases.clarify" in (result.tools_used or []):
        # Prefer already-set ask_* from clarify; otherwise height (fit-first).
        if not (result.flow_stage or "").startswith("ask_"):
            result.flow_stage = "ask_height"
    elif "cases.lookup" in (result.tools_used or []) or "cta.conversion" in (
        result.tools_used or []
    ):
        result.flow_stage = "recommend"
    elif "cta.product_url" in (result.tools_used or []) or "cta.showroom" in (
        result.tools_used or []
    ):
        result.flow_stage = "shop"
    elif result.intent == INTENT_RECOMMEND:
        result.flow_stage = "ask_height"
    else:
        result.flow_stage = "menu"
    return result


def _tier_followup_reply(
    idx: int,
    prefs: Optional[dict],
    *,
    domain: str = "osakiusa.com",
) -> Optional[SalesReply]:
    """After 1/2/3: why-this-chair card + shop / specs / email / compare."""
    picks = (prefs or {}).get("pending_tier_picks") or []
    if not isinstance(picks, list) or not (0 <= idx < len(picks)):
        return None
    pick = picks[idx]
    if not isinstance(pick, dict):
        return None

    display = (pick.get("display") or pick.get("model") or "this chair").strip()
    tier = (pick.get("tier") or f"Option {idx + 1}").strip()
    url = (pick.get("url") or "").strip()
    handle = (pick.get("handle") or "").strip()
    stock = (pick.get("stock") or "").strip()
    product = resolve_product(handle or display)
    rec_prefs = ((prefs or {}).get("recommend_prefs") or {})

    why_bits: list[str] = []
    if rec_prefs.get("goal"):
        why_bits.append(f"targets **{rec_prefs['goal']}**")
    if rec_prefs.get("height"):
        why_bits.append(f"sized for **{rec_prefs['height']}**")
    if product is not None:
        for bit in (product.massage_mechanism, product.track_type):
            if bit:
                why_bits.append(bit)
    if stock:
        why_bits.append(stock)
    if rec_prefs.get("doorway_in") and rec_prefs.get("doorway_in") != "skip":
        why_bits.append(f"checked against your **{rec_prefs['doorway_in']}\"** doorway")

    lines = [
        f"**{tier}** — **{display}**"
        + (f" · {_fmt_price(product.price_usd)}" if product and product.price_usd else ""),
        "",
        "**Why this pick:**",
    ]
    if why_bits:
        lines.append("• " + " · ".join(why_bits[:4]))
    else:
        lines.append("• Best match in this price tier for your fit answers.")
    if url:
        lines.append(f"\nShop: {url}")
    lines.append(
        "\nFinancing (Affirm) shows at checkout — I won't invent rates. "
        "Want specs, a compare, or email this to sales?"
    )

    patched = dict(prefs or {})
    patched["pending_primary"] = display
    if url:
        patched["pending_product_url"] = url

    quick: list[QuickReply] = []
    if url.startswith("https://"):
        quick.append(QuickReply(label="Shop this chair", payload=f"open:{url}"))
    if handle:
        quick.append(QuickReply(label="Full specs", payload=f"specs:{handle}"))
    quick.append(QuickReply(label="Email me this pick", payload="lead:save_pick"))
    # Compare this tier to the other of Value/Mid when possible.
    if idx == 0 and len(picks) >= 2:
        quick.append(QuickReply(label="Compare vs Mid", payload="compare:tiers:1:2"))
    elif idx == 1 and len(picks) >= 1:
        quick.append(QuickReply(label="Compare vs Value", payload="compare:tiers:1:2"))
    elif idx == 2 and len(picks) >= 2:
        quick.append(QuickReply(label="Compare vs Mid", payload="compare:tiers:2:3"))
    quick.append(QuickReply(label="Back to list", payload="recommend:again"))
    quick.append(QuickReply(label="Talk to a human", payload="human"))

    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_RECOMMEND,
        quick_replies=quick,
        tools_used=["cases.tiered", "cta.product_url"],
        products=[product.as_public_dict()] if product else [],
        flow_stage="recommend",
        prefs_patch=patched,
    )


def _compare_pending_tiers_reply(
    prefs: Optional[dict],
    left_n: int,
    right_n: int,
) -> Optional[SalesReply]:
    """Compare two pending tier picks by 1-based indexes."""
    picks = (prefs or {}).get("pending_tier_picks") or []
    if not isinstance(picks, list):
        return None
    li, ri = left_n - 1, right_n - 1
    if not (0 <= li < len(picks) and 0 <= ri < len(picks)):
        return None
    left_pick = picks[li]
    right_pick = picks[ri]
    if not isinstance(left_pick, dict) or not isinstance(right_pick, dict):
        return None
    left_name = left_pick.get("model") or left_pick.get("display") or ""
    right_name = right_pick.get("model") or right_pick.get("display") or ""
    result = compare(left_name, right_name)
    if result is None:
        # Fall back to handles.
        result = compare(
            left_pick.get("handle") or left_name,
            right_pick.get("handle") or right_name,
        )
    if result is None:
        return SalesReply(
            reply=(
                f"I couldn't line up **{left_pick.get('display') or left_name}** vs "
                f"**{right_pick.get('display') or right_name}** in the catalog. "
                "A specialist can compare them side by side."
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="Back to list", payload="recommend:again"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
            flow_stage="recommend",
        )

    left = result["left"]
    right = result["right"]
    diff = result["diff"]
    lt = (left_pick.get("tier") or "Option A").split("(")[0].strip()
    rt = (right_pick.get("tier") or "Option B").split("(")[0].strip()
    lines = [
        f"**{lt}: {left['model']}** — {_fmt_price(left['price_usd'])}",
        f"**{rt}: {right['model']}** — {_fmt_price(right['price_usd'])}",
        "",
        f"- **Mechanism**: {diff['mechanism'][0] or '—'} vs {diff['mechanism'][1] or '—'}",
        f"- **Track**: {diff['track'][0] or '—'} vs {diff['track'][1] or '—'}",
        f"- **Zero gravity**: {diff['zero_gravity'][0] or '—'} vs {diff['zero_gravity'][1] or '—'}",
        f"- **Heating**: {diff['heating'][0] or '—'} vs {diff['heating'][1] or '—'}",
        f"- **Foot roller**: {diff['foot_roller'][0] or '—'} vs {diff['foot_roller'][1] or '—'}",
    ]
    if diff["price_delta_usd"] is not None:
        delta = diff["price_delta_usd"]
        if abs(delta) >= 1:
            direction = "more" if delta > 0 else "less"
            lines.append(
                f"- **Price gap**: {rt} is about ${abs(delta):,.0f} {direction}."
            )
    lines.append(
        "\nReply **1** for Value, **2** for Mid (or the tier you want), "
        "or ask a specialist to help you choose."
    )
    quick = [
        QuickReply(label=f"Choose {lt}", payload=f"tier:{left_n}"),
        QuickReply(label=f"Choose {rt}", payload=f"tier:{right_n}"),
        QuickReply(label="Back to list", payload="recommend:again"),
        QuickReply(label="Talk to a human", payload="human"),
    ]
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_COMPARE,
        quick_replies=quick,
        tools_used=["catalog.compare", "cases.tiered"],
        products=[left, right],
        flow_stage="recommend",
        prefs_patch={
            "pending_tier_picks": picks,
            "recommend_prefs": (prefs or {}).get("recommend_prefs") or {},
        },
    )


def _tier_digit_reply(message: str, prefs: Optional[dict]) -> Optional[SalesReply]:
    """Map bare 1/2/3 to the matching Value/Mid/Premium follow-up card."""
    digit = re.fullmatch(r"([1-3])[).:\s]*", (message or "").strip())
    if not digit:
        return None
    picks = (prefs or {}).get("pending_tier_picks") or []
    if not isinstance(picks, list) or not picks:
        return None
    return _tier_followup_reply(int(digit.group(1)) - 1, prefs)


def respond(
    message: str,
    *,
    payload: Optional[str] = None,
    domain: str = "osakiusa.com",
    prefs: Optional[dict] = None,
) -> SalesReply:
    """Return a SalesReply for one customer message (+ optional button payload).

    ``payload`` is the ``QuickReply.payload`` value emitted by a previous
    turn. When set, it overrides intent classification so button taps behave
    predictably — critical for hitting 100% satisfaction on menu paths.

    ``prefs`` is ``sales_sessions.collected_data`` (recommend answers accumulate
    across turns via ``prefs_patch`` on the reply).
    """
    # Awaiting email after "Email me this pick" — capture before other intents.
    if (prefs or {}).get("awaiting_email_for_pick"):
        email = extract_email(message or "")
        if email:
            return _finalize_flow_stage(_capture_pick_lead(email, prefs, domain=domain))

    # After a tier list, bare "1"/"2"/"3" opens that chair (chat + Tidio).
    if not payload:
        tier_reply = _tier_digit_reply(message, prefs)
        if tier_reply is not None:
            return _finalize_flow_stage(tier_reply)

    if payload:
        forced = _payload_reply(payload, message, domain=domain, prefs=prefs)
        if forced is not None:
            return _finalize_flow_stage(forced)

    intent = classify(message or "")

    if intent.label in HANDOFF_INTENTS:
        return _finalize_flow_stage(_handoff_reply(intent))

    if intent.label == INTENT_GREETING:
        return _finalize_flow_stage(_greeting_reply())

    if intent.label == INTENT_ORDER_STATUS:
        return _finalize_flow_stage(_order_status_reply(message))

    if intent.label == INTENT_PRICE:
        return _finalize_flow_stage(_price_reply(message))
    if intent.label == INTENT_STOCK:
        return _finalize_flow_stage(_stock_reply(message, domain=domain))
    if intent.label == INTENT_SPECS:
        return _finalize_flow_stage(_specs_reply(message))
    if intent.label == INTENT_RECOMMEND:
        return _finalize_flow_stage(_recommend_reply(message, domain=domain, prefs=prefs))
    if intent.label == INTENT_COMPARE:
        return _finalize_flow_stage(_compare_reply(message))
    if intent.label == INTENT_INTENSITY:
        return _finalize_flow_stage(_intensity_reply(message))

    return _finalize_flow_stage(_unclear_reply())
