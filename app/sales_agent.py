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
    color_is_listed,
    compare_products,
    is_public_browse_pick,
    list_active_products,
    looks_like_color_question,
    parse_asked_color,
    parse_recommendation_hints,
    price_tier_label,
    recommend,
    resolve_product,
)
from sales_compare import (
    is_which_of_pair,
    lookup_shop_models,
    looks_like_compare_pair,
    pair_fit_reasons,
    pair_fit_scores,
    product_by_handle,
    short_model_label,
    split_compare_terms,
)
from sales_cases import (
    TIER_BUDGETS,
    apply_payload_codes,
    brand_for_domain,
    cases_available,
    enrich_implied_prefs,
    height_from_shopper_answer,
    lookup_case,
    merge_prefs_from_hints,
    missing_ask,
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
    showroom_address,
    showroom_blurb,
    showroom_hours,
    showroom_window_label,
)
from sales_shopify_stock import LiveStockSnapshot, fetch_live_stock, stock_badge
from sales_spec_index import (
    doorway_inches_for_model,
    doorway_ok,
    lookup_fit_spec,
    wall_ok,
    weight_ok,
)
from sales_visitor_memory import has_resume_memory
from sales_intent import (
    INTENT_COMPARE,
    INTENT_DISCOUNT,
    INTENT_GREETING,
    INTENT_HUMAN,
    INTENT_INTENSITY,
    INTENT_ORDER_STATUS,
    INTENT_PREPURCHASE_POLICY,
    INTENT_PRICE,
    INTENT_RECOMMEND,
    INTENT_SPECS,
    INTENT_STOCK,
    INTENT_UNCLEAR,
    HANDOFF_INTENTS,
    SalesIntent,
    classify,
    handoff_message,
    looks_like_price_gap,
)
from sales_intent_fallback import named_model_in_text, resolve_unclear, revise_recommend
from sales_policy import (
    TOPIC_REMOTE_SHIPPING,
    TOPIC_SHIPPING,
    TOPIC_SHOWROOM,
    TOPIC_WHITE_GLOVE,
    detect_topic as detect_policy_topic,
    policy_answer,
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
    # Router writes a SalesFeedback row when the shopper rates an answer.
    feedback: Optional[dict] = None
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


def _fmt_live_price(
    snap: Optional[LiveStockSnapshot],
    fallback_price: Optional[float],
) -> str:
    """Format Shopify's current variant price or an honest catalog fallback."""
    live_low = getattr(snap, "price_usd", None) if snap is not None else None
    if live_low is None:
        return _fmt_price(fallback_price)
    low = live_low
    high = getattr(snap, "price_max_usd", None)
    if high is not None and high > low:
        return f"{_fmt_price(low)}–{_fmt_price(high)} depending on options"
    return _fmt_price(low)


_UNPUBLISHED_SPEC = frozenset(
    {"", "-", "—", "–", ".", "n/a", "na", "none", "null", "unknown", "no data"}
)
_UNPUBLISHED_FRAGMENTS = (
    "dont know",
    "don't know",
    "do not know",
    "unknown",
    "no data",
    "not listed",
    "tbd",
)
_TIER_ROLE_BITS = (
    "under ~$3k for this fit",
    "mid-range step-up",
    "top of the published range",
)


def _spec_is_published(value: object) -> bool:
    return _spec_customer_value(value) is not None


def _spec_customer_value(value: object) -> Optional[str]:
    """Catalog text the shopper can see — hide unpublished 'dont know' junk."""
    text = str(value or "").strip()
    if not text:
        return None
    low = text.lower()
    if low in _UNPUBLISHED_SPEC:
        return None
    messy = any(fragment in low for fragment in _UNPUBLISHED_FRAGMENTS)
    if low.startswith("yes"):
        yes_no = "Yes"
    elif re.match(r"^no\b", low) and not low.startswith("none"):
        yes_no = "No"
    else:
        yes_no = None
    if messy:
        return yes_no
    return text


def _auto_programs_bit(raw: str) -> Optional[str]:
    if not _spec_is_published(raw):
        return None
    match = re.match(r"^(\d+)", str(raw).strip())
    if not match:
        return None
    return f"{match.group(1)} auto programs"


def _airbag_bit(raw: str) -> Optional[str]:
    if not _spec_is_published(raw):
        return None
    match = re.match(r"^(\d+)", str(raw).strip())
    if not match:
        return None
    return f"{match.group(1)} airbags"


def _chair_highlight_bits(product: ProductSpecs) -> list[str]:
    """Catalog facts only — skip empty / '-' fields."""
    bits: list[str] = []
    if _spec_is_published(product.massage_mechanism):
        bits.append(product.massage_mechanism.strip())
    if _spec_is_published(product.track_type):
        bits.append(product.track_type.strip())
    fit = lookup_fit_spec(product.display_name)
    if fit is not None:
        if fit.max_user_lb is not None:
            bits.append(f"{fit.max_user_lb:g} lb max")
        if fit.door_asm_in is not None:
            bits.append(f'{fit.door_asm_in:g}" doorway')
    return bits[:4]


def _chair_highlight_line(product: ProductSpecs) -> str:
    return " · ".join(_chair_highlight_bits(product))


def _product_attr_tags(product: Optional[ProductSpecs]) -> dict[str, str]:
    if product is None:
        return {}
    tags: dict[str, str] = {}
    if _spec_is_published(product.massage_mechanism):
        tags["mech"] = product.massage_mechanism.strip()
    if _spec_is_published(product.track_type):
        tags["track"] = product.track_type.strip()
    auto = _auto_programs_bit(product.auto_programs)
    if auto:
        tags["auto"] = auto
    air = _airbag_bit(product.airbag)
    if air:
        tags["air"] = air
    zg = (product.zero_gravity or "").strip()
    if _spec_is_published(zg) and zg.lower() not in {"no", "false", "0"}:
        tags["zg"] = zg if "zero" in zg.lower() else f"{zg} zero gravity"
    heat = (product.heating or "").strip()
    if _spec_is_published(heat) and heat.lower() not in {"no", "false", "0"}:
        tags["heat"] = f"heat ({heat})" if "," in heat or heat.lower() != "yes" else "heat"
    foot = (product.foot_roller or "").strip()
    if _spec_is_published(foot) and foot.lower() not in {"no", "false", "0"}:
        tags["foot"] = "foot/calf rollers"
    return tags


def _contrast_tier_blurbs(
    products: list[Optional[ProductSpecs]],
    *,
    prefs: dict[str, str],
    doorway_ins: list[Optional[float]],
) -> list[str]:
    """One distinct fact-line per tier — do not repeat the same goal phrase."""
    tag_rows = [_product_attr_tags(p) for p in products]
    counts: dict[tuple[str, str], int] = {}
    for tags in tag_rows:
        for key, value in tags.items():
            counts[(key, value)] = counts.get((key, value), 0) + 1

    key_order = ("mech", "track", "auto", "zg", "heat", "air", "foot")
    compact = (prefs.get("space") or "") in {"Narrow Doorway", "Small Room"}
    blurbs: list[str] = []
    for idx, tags in enumerate(tag_rows):
        bits: list[str] = []
        unique = [
            (key, tags[key])
            for key in key_order
            if key in tags and counts.get((key, tags[key]), 0) == 1
        ]
        for _key, value in unique:
            if value not in bits:
                bits.append(value)
            if len(bits) >= 2:
                break
        if len(bits) < 2:
            for key in ("mech", "track"):
                value = tags.get(key)
                if value and value not in bits:
                    bits.append(value)
                if len(bits) >= 2:
                    break
        door = doorway_ins[idx] if idx < len(doorway_ins) else None
        if compact and door is not None:
            door_bit = f'~{door:g}" doorway'
            if door_bit not in bits:
                bits.append(door_bit)
        if prefs.get("goal") == "Foot & Calf" and tags.get("foot") and tags["foot"] not in bits:
            bits.append(tags["foot"])
        if len(bits) < 2 and idx < len(_TIER_ROLE_BITS):
            bits.append(_TIER_ROLE_BITS[idx])
        blurbs.append(" · ".join(bits[:3]))
    return blurbs


def _product_closeout(
    product: ProductSpecs,
    *,
    domain: str,
) -> tuple[str, list[QuickReply], dict]:
    """Why this chair, the product link, and an email-me CTA."""
    why = _chair_highlight_line(product)
    url = product_page_url(domain, product.handle or "")
    lines: list[str] = []
    if why:
        lines.append(why + ".")
    if url:
        lines.append(f"Shop: {url}")
    lines.append("Want this emailed to you? Tap **Email me this pick**.")
    extra = "\n\n" + "\n".join(lines)
    quick: list[QuickReply] = []
    if url and url.startswith("https://"):
        quick.append(QuickReply(label="Shop this chair", payload=f"open:{url}"))
    quick.append(QuickReply(label="Email me this pick", payload="lead:save_pick"))
    summary = f"{product.display_name} — {_fmt_price(product.price_usd)}"
    if url:
        summary += f"\n{url}"
    patch = {
        "pending_primary": product.display_name,
        "pending_product_url": url,
        "pending_pick_summary": summary,
    }
    return extra, quick, patch


def _menu_quick_replies() -> list[QuickReply]:
    """Default set of quick-reply buttons — always present as an escape hatch."""
    return [
        QuickReply(label="Recommend a chair", payload="recommend"),
        QuickReply(label="Check a price", payload="price"),
        QuickReply(label="Availability / stock", payload="stock"),
        QuickReply(label="Compare two models", payload="compare"),
        QuickReply(label="Talk to a human", payload="human"),
    ]


_EMAIL_PICK_RE = re.compile(
    r"email\s+me\s+(?:this|these|the)\s+picks?|"
    r"send\s+(?:this|it|the\s+pick)\s+to\s+my\s+email",
    re.I,
)

_MENU_INTRO = (
    "Hi! I'm the Osaki shopping assistant. Tell me **height** and what the "
    "chair should help with — I'll pick three models.\n\n"
    "I can also check **price**, **specs**, **stock**, **shipping**, and "
    "**returns** here."
)


# ---------------------------------------------------------------------------
# Extract "topic" (model name) from free text
# ---------------------------------------------------------------------------


def _guess_model_from_text(text: str) -> Optional[ProductSpecs]:
    product = resolve_product(text or "")
    if product is not None:
        return product
    # `resolve_product` matches names, not sentences, so "tell me about the
    # Maestro" slips past it. The fallback index knows which catalog token was
    # in the message, so retry with just that name.
    named = named_model_in_text(text or "")
    return resolve_product(named) if named else None


# ---------------------------------------------------------------------------
# Response builders per intent
# ---------------------------------------------------------------------------


def _ask_question_before_human() -> SalesReply:
    return SalesReply(
        reply=(
            "Before I connect you, tell me **the one thing** you need answered. "
            "I can finish price, specs, stock, shipping, returns, or a "
            "recommendation right here.\n\n"
            "A specialist is only needed after that for **discounts**, a "
            "**custom freight quote**, or an **order you already placed**."
        ),
        intent="human_offer",
        handoff=False,
        quick_replies=[
            QuickReply(label="Recommend a chair", payload="recommend"),
            QuickReply(label="Check a price", payload="price"),
            QuickReply(label="Shipping times", payload="ask:shipping"),
            QuickReply(label="Return policy", payload="ask:returns"),
            QuickReply(label="Still connect me to sales", payload="human:confirm"),
        ],
        tools_used=["human.offer"],
        prefs_patch={"human_offered": True, "awaiting_pre_human_question": True},
        flow_stage="menu",
    )


def _shopping_facts_lines() -> list[str]:
    return [
        "- **Shipping:** up to **2 weeks** curbside, **3 weeks** White Glove.",
        "- **Returns:** 30 days from delivery; you pay shipping both ways; "
        "White Glove fee is not refundable.",
        "- **Financing:** Affirm at checkout (I won't quote a rate).",
        "- **Hawaii / Alaska:** we ship, you pay freight (quoted per model and zip).",
        "- **Guam:** we don't ship.",
    ]


def _discount_facts_reply(message: str, prefs: Optional[dict], *, domain: str) -> SalesReply:
    """Published facts only — never a % — then a specialist can negotiate."""
    product = _guess_model_from_text(message or "")
    if product is None:
        named = ((prefs or {}).get("pending_primary") or "").strip()
        if named:
            product = resolve_product(named)
    lines = [
        "I can't quote a **discount, coupon, or sale price** here — a "
        "specialist has to do that.\n",
    ]
    if looks_like_price_gap(message or ""):
        lines.append(
            "That listing is often a **different chair model** than the one on "
            "this site, so the prices will not line up. I can confirm our "
            "published storefront price.\n"
        )
    lines.append("Here is everything I *can* confirm now:")
    if product is not None:
        snap = fetch_live_stock(product.handle, domain=domain)
        lines.append(
            f"\n**{product.display_name}** — {_fmt_live_price(snap, product.price_usd)} "
            f"({stock_badge(snap)})"
        )
    lines.append("")
    lines.extend(_shopping_facts_lines())
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_DISCOUNT,
        handoff=False,
        quick_replies=[
            QuickReply(label="Recommend a chair", payload="recommend"),
            QuickReply(label="Check a price", payload="price"),
            QuickReply(label="Still connect me to sales", payload="human:confirm"),
        ],
        tools_used=["human.complete", "policy.discount_facts"],
        products=[product.as_public_dict()] if product is not None else [],
        flow_stage="menu",
    )


def _attach_human_footer(inner: SalesReply) -> SalesReply:
    """Keep the real answer, then let the shopper dismiss or still connect."""
    footer = (
        "\n\n---\n"
        "If that covers it, you're set. Tap **Still connect me to sales** only "
        "for a discount, a freight quote, or an order you already placed."
    )
    inner.reply = (inner.reply or "").rstrip() + footer
    inner.handoff = False
    inner.handoff_reason = None
    kept = [
        q
        for q in inner.quick_replies
        if q.payload not in {"human", "human:confirm"}
    ]
    kept.append(QuickReply(label="That's all I needed", payload="menu"))
    kept.append(QuickReply(label="Still connect me to sales", payload="human:confirm"))
    inner.quick_replies = kept
    inner.tools_used = list(inner.tools_used or []) + ["human.complete"]
    patch = dict(inner.prefs_patch or {})
    patch["human_offered"] = True
    patch["awaiting_pre_human_question"] = False
    inner.prefs_patch = patch
    if not inner.flow_stage:
        inner.flow_stage = "menu"
    return inner


def _prefs_for_inner(prefs: Optional[dict]) -> dict:
    out = dict(prefs or {})
    out.pop("awaiting_pre_human_question", None)
    out.pop("human_offered", None)
    return out


_BARE_HUMAN_REQUEST_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"talk\s+to\s+(?:a\s+)?(?:human|person|representative|rep|agent|sales|someone)|"
    r"speak\s+(?:with|to)\s+(?:a\s+)?(?:human|person|rep|agent|sales|someone)|"
    r"human(?:\s+please)?|agent|representative|real\s+person"
    r")[!.?\s]*$",
    re.I,
)
_RESTATE_QUESTION_RE = re.compile(
    r"question\s+above|answer\s+my\s+question",
    re.I,
)


def _complete_then_offer_human(
    *,
    prefs: Optional[dict],
    domain: str,
    reason: str = "human",
    message: str = "",
) -> SalesReply:
    """Answer the shopper's actual question before any sales-agent transfer."""
    data = prefs or {}
    question = str(data.get("last_shopper_question") or "").strip()
    asked = (message or "").strip() or question
    summary = str(data.get("pending_pick_summary") or "").strip()

    if reason == "human" and asked:
        if _RESTATE_QUESTION_RE.search(asked):
            return _ask_question_before_human()
        if (
            classify(asked).label == INTENT_HUMAN
            and not _BARE_HUMAN_REQUEST_RE.match(asked)
        ):
            return _handoff_reply(
                SalesIntent(label=INTENT_HUMAN, confidence="high", handoff=True)
            )

    if reason == "discount" or (
        asked and classify(asked).label == INTENT_DISCOUNT
    ):
        return _attach_human_footer(
            _discount_facts_reply(asked, data, domain=domain)
        )

    if question:
        inner_intent = classify(question)
        recovered = (
            resolve_unclear(question)
            if inner_intent.label == INTENT_UNCLEAR
            else None
        )
        label = recovered.label if recovered is not None else inner_intent.label
        if label not in HANDOFF_INTENTS and label != INTENT_HUMAN:
            inner = respond(
                question,
                domain=domain,
                prefs=_prefs_for_inner(data),
                before_handoff=True,
            )
            if inner.intent not in {INTENT_UNCLEAR, INTENT_GREETING, "human_offer"}:
                return _attach_human_footer(inner)

    if summary:
        primary = str(data.get("pending_primary") or "your pick").strip()
        return _attach_human_footer(
            SalesReply(
                reply=f"Here's the pick I already have for you:\n\n{summary}",
                intent=INTENT_RECOMMEND,
                quick_replies=[
                    QuickReply(label=f"See {primary[:28]}", payload="tier:1"),
                    QuickReply(label="Email me these picks", payload="lead:save_pick"),
                ],
                tools_used=["human.complete"],
                flow_stage="recommend",
            )
        )

    return _ask_question_before_human()


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
    active = [
        p
        for p in list_active_products()
        if p.price_usd and is_public_browse_pick(p)
    ]
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


def _price_extreme_product(message: str) -> Optional[ProductSpecs]:
    """Lowest / highest published storefront price — catalog fact, not a guess."""
    raw = message or ""
    active = [
        p
        for p in list_active_products()
        if p.price_usd and is_public_browse_pick(p)
    ]
    if not active:
        return None
    if re.search(r"\b(cheapest|least\s+expensive|lowest\s+price)\b", raw, re.I):
        return min(active, key=lambda p: p.price_usd or 1e9)
    if re.search(
        r"\b(most\s+expensive|highest\s+price|top[\s-]?of[\s-]?the[\s-]?line)\b",
        raw,
        re.I,
    ):
        return max(active, key=lambda p: p.price_usd or 0)
    return None


def _price_reply(message: str, *, domain: str = "osakiusa.com") -> SalesReply:
    product = _guess_model_from_text(message)
    extreme_note = ""
    if product is None:
        extreme = _price_extreme_product(message)
        if extreme is not None:
            product = extreme
            if re.search(
                r"cheapest|least\s+expensive|lowest\s+price", message or "", re.I
            ):
                extreme_note = (
                    "Lowest **published catalog price** I can quote right now "
                    "(not a sale, and Costco-only listings are left out):\n\n"
                )
            else:
                extreme_note = (
                    "Highest **published catalog price** I can quote right now:\n\n"
                )
    if product is None:
        samples = _price_candidates()
        lines = [
            "Sure — which model are you asking about? Type the model name "
            "(e.g. *Osaki OS-Pro Maestro LE*), or tap **Recommend a chair** "
            "and I'll match fit first."
        ]
        if samples:
            lines.append("\nPopular catalog price points:")
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

    live = fetch_live_stock(
        product.handle,
        domain=domain,
        title=product.title or product.display_name,
    )
    price_txt = _fmt_live_price(live, product.price_usd)
    if live is not None:
        availability = (
            "✅ Available to buy now."
            if live.in_stock
            else "⚠️ Not currently available to buy."
        )
    else:
        availability = (
            "✅ Currently in our active catalog; checkout confirms final availability."
            if product.status.lower() == "active"
            else "⚠️ This model is not in our active catalog right now — a rep can confirm availability."
        )
    price_source = (
        "This price was checked live on Shopify."
        if live is not None and live.price_usd is not None
        else "Live pricing was unavailable, so this is the latest catalog price."
    )
    reply = (
        f"{extreme_note}**{product.display_name}** — {price_txt}.\n\n"
        f"{availability}\n\n"
        f"{price_source}"
    )
    extra, close_quick, close_patch = _product_closeout(product, domain=domain)
    public = product.as_public_dict()
    if live is not None and live.price_usd is not None:
        public["price_usd"] = live.price_usd
        public["price_max_usd"] = live.price_max_usd
        public["stock"] = stock_badge(live)
    return SalesReply(
        reply=reply + extra,
        intent=INTENT_PRICE,
        quick_replies=[
            *close_quick,
            QuickReply(label=f"Specs for {product.display_name}", payload=f"specs:{product.handle}"),
            QuickReply(label="Compare with another model", payload="compare"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=[
            "catalog.resolve_product",
            *(["shopify.price_inventory"] if live is not None else []),
        ],
        products=[public],
        prefs_patch=close_patch,
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
            low = " (low stock)" if live.is_low else ""
            reply = (
                f"**{product.display_name}** is **available to buy** right now{low}.\n\n"
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
    extra, close_quick, close_patch = _product_closeout(product, domain=domain)
    return SalesReply(
        reply=reply + extra,
        intent=INTENT_STOCK,
        quick_replies=[
            *close_quick,
            QuickReply(label="Check the price", payload=f"price:{product.handle}"),
            QuickReply(label="See similar models", payload="recommend"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=tools,
        products=[product.as_public_dict()],
        prefs_patch=close_patch,
    )


def _fmt_inches(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _in_recommend_wait(prefs: Optional[dict]) -> bool:
    data = prefs or {}
    if data.get("awaiting_recommend"):
        return True
    rec = data.get("recommend_prefs")
    if not isinstance(rec, dict) or not rec:
        return False
    if missing_ask(rec):
        return True
    space = (rec.get("space") or "").strip()
    if space in _COMPACT_SPACES and (
        _needs_doorway_inches(rec) or _needs_doorway_fit(rec)
    ):
        return True
    return False


def _resume_recommend_after_side(
    side: SalesReply,
    prefs: Optional[dict],
    *,
    domain: str,
) -> SalesReply:
    rec = dict((prefs or {}).get("recommend_prefs") or {})
    side_patch = dict(side.prefs_patch or {})
    extra_rec = side_patch.get("recommend_prefs")
    if isinstance(extra_rec, dict):
        rec.update(extra_rec)
    merged = dict(prefs or {})
    merged["recommend_prefs"] = rec
    resume = _recommend_reply("recommend", domain=domain, prefs=merged)
    side.reply = side.reply.rstrip() + "\n\n" + resume.reply
    patch = dict(side.prefs_patch or {})
    patch.update(resume.prefs_patch or {})
    side.prefs_patch = patch
    side.quick_replies = resume.quick_replies
    side.intent = resume.intent
    side.flow_stage = resume.flow_stage
    side.handoff = resume.handoff
    side.tools_used = list(side.tools_used or []) + list(resume.tools_used or [])
    if resume.products and not side.products:
        side.products = resume.products
    return side


def _color_reply(
    message: str,
    *,
    domain: str = "osakiusa.com",
    prefs: Optional[dict] = None,
) -> SalesReply:
    color = parse_asked_color(message)
    product = _guess_model_from_text(message)
    if product is None:
        pending = str((prefs or {}).get("pending_primary") or "").strip()
        if pending:
            product = resolve_product(pending) or _guess_model_from_text(pending)
    checkout = "Exact color is confirmed at checkout."
    if product is None:
        asked = f"**{color}**" if color else "that color"
        return SalesReply(
            reply=(
                f"I can check {asked} once we know the chair. {checkout}\n\n"
                "Which **model**, or tell me the **user height** and I'll recommend?"
            ),
            intent=INTENT_STOCK,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.color"],
        )

    live = fetch_live_stock(
        product.handle,
        domain=domain,
        title=product.title or product.display_name,
    )
    tools = ["catalog.resolve_product", "catalog.color"]
    if live is not None:
        tools.append("shopify.inventory")
        if live.in_stock:
            low = " (low stock)" if live.is_low else ""
            stock_line = f"The chair is **available to buy** right now{low}."
        else:
            stock_line = (
                "The chair is **not available to buy right now** "
                "(out of stock or not listed for sale)."
            )
    elif product.status.lower() == "active":
        stock_line = (
            "It's **active in our catalog**; checkout confirms final availability."
        )
    else:
        stock_line = "This model is **not in our active catalog** right now."

    if color and product.colors:
        matched = color_is_listed(product, color)
        if matched:
            color_line = (
                f"**{product.display_name}** is listed in **{', '.join(matched)}**."
            )
        else:
            color_line = (
                f"I don't see **{color}** on the published options for "
                f"**{product.display_name}**. Listed: {', '.join(product.colors)}."
            )
    elif product.colors:
        color_line = (
            f"**{product.display_name}** listed colors: {', '.join(product.colors)}."
        )
    elif color:
        color_line = (
            f"**{product.display_name}** — I don't have published color options "
            f"to confirm **{color}** from the catalog."
        )
    else:
        color_line = (
            f"**{product.display_name}** — color options aren't listed in the "
            "catalog I can quote."
        )

    extra, close_quick, close_patch = _product_closeout(product, domain=domain)
    return SalesReply(
        reply=f"{color_line}\n\n{stock_line}\n\n{checkout}{extra}",
        intent=INTENT_STOCK,
        quick_replies=[
            *close_quick,
            QuickReply(label="Check the price", payload=f"price:{product.handle}"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=tools,
        products=[product.as_public_dict()],
        prefs_patch=close_patch,
    )


def _doorway_fit_reply(
    message: str,
    *,
    prefs: Optional[dict] = None,
    domain: str = "osakiusa.com",
) -> SalesReply:
    inches = _parse_doorway_inches_message(message or "")
    product = _guess_model_from_text(message)
    rec = dict((prefs or {}).get("recommend_prefs") or {})
    if inches is not None:
        rec["doorway_in"] = _fmt_inches(inches)
        if "space" not in rec:
            rec["space"] = "Narrow Doorway"

    if product is not None and inches is not None:
        fit = lookup_fit_spec(product.display_name)
        lines = [
            f"For a **{inches:g}\" doorway** and **{product.display_name}**:"
        ]
        if fit is not None and fit.door_asm_in is not None:
            lines.append(
                f"- Assembled doorway listed at **{fit.door_asm_in:g} in**."
            )
            if doorway_ok(product.display_name, limit_in=inches, mode="assembled"):
                lines.append("That assembled figure fits the width you gave.")
            else:
                lines.append(
                    "That assembled figure is wider than the doorway you gave."
                )
                if fit.door_dis_in is not None:
                    lines.append(
                        f"- Disassembled is listed at **{fit.door_dis_in:g} in**."
                    )
                    if doorway_ok(
                        product.display_name, limit_in=inches, mode="disassembled"
                    ):
                        lines.append(
                            "Disassembly may be needed — I won't promise the "
                            "crew will do that."
                        )
                    else:
                        lines.append(
                            "Even the disassembled figure is wider than that doorway."
                        )
        else:
            lines.append(
                "I don't have a published doorway number for that chair. "
                "A specialist can confirm."
            )
        extra, close_quick, close_patch = _product_closeout(product, domain=domain)
        patch = dict(close_patch)
        patch["recommend_prefs"] = rec
        return SalesReply(
            reply="\n".join(lines) + extra,
            intent=INTENT_SPECS,
            quick_replies=[
                *close_quick,
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.doorway"],
            products=[product.as_public_dict()],
            prefs_patch=patch,
        )

    inch_bit = f"**{inches:g}\"** " if inches is not None else ""
    return SalesReply(
        reply=(
            f"A {inch_bit}doorway is tight for some chairs. Which **model**, "
            "or tell me the **user height** and I'll only keep chairs that can fit?"
        ),
        intent=INTENT_RECOMMEND,
        quick_replies=[
            *_HEIGHT_REPLIES,
            QuickReply(label="See all models", payload="list"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.doorway"],
        prefs_patch={"recommend_prefs": rec, "awaiting_recommend": True},
        flow_stage="ask_height",
    )


_SPEC_QUESTION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bzero[\s-]?grav", re.I), "zero_gravity", "Zero gravity"),
    (re.compile(r"\bheat(?:ing|er)?\b", re.I), "heating", "Heating"),
    (re.compile(r"\bairbags?\b", re.I), "airbag", "Airbags"),
    (re.compile(r"\bfoot\s*rollers?\b|\bcalf\s*rollers?\b", re.I), "foot_roller", "Foot/calf roller"),
    (re.compile(r"\b(?:sl|l|s)[\s-]?track\b|\btrack\s*type\b", re.I), "track_type", "Track"),
    (
        re.compile(
            r"\bmechanism\b|\b(?:2|3|4)\s*d\s+(?:roller|massage|mechanism)",
            re.I,
        ),
        "massage_mechanism",
        "Mechanism",
    ),
)


def _yes_no_spec(value: str) -> str:
    v = (value or "").strip().lower()
    if not v or v in {"n/a", "na", "none", "-", "no", "false", "0"}:
        return "No"
    if v in {"yes", "true", "1", "y"}:
        return "Yes"
    return value.strip()


def _specs_reply(message: str, *, domain: str = "osakiusa.com") -> SalesReply:
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

    public = product.as_public_dict()
    specs = public["specs"]
    fit = lookup_fit_spec(product.display_name)
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
            if not _spec_is_published(val):
                lines.append(f"- **{label}**: not listed")
            elif key in {"zero_gravity", "heating", "airbag", "foot_roller"}:
                shown = _spec_customer_value(val) or "not listed"
                lines.append(f"- **{label}**: {_yes_no_spec(shown) if shown != 'not listed' else shown}")
            else:
                shown = _spec_customer_value(val) or "not listed"
                lines.append(f"- **{label}**: {shown}")
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
        shown = _spec_customer_value(val)
        if shown:
            lines.append(f"- **{label}**: {shown}")
    if fit is not None:
        fit_public = {
            "max_user_lb": fit.max_user_lb,
            "doorway_assembled_in": fit.door_asm_in,
            "doorway_disassembled_in": fit.door_dis_in,
            "wall_clearance_in": fit.wall_clearance_in,
        }
        public["fit_specs"] = fit_public
        if fit.max_user_lb is not None:
            lines.append(f"- **Maximum user weight**: {fit.max_user_lb:g} lb")
        if fit.door_asm_in is not None:
            lines.append(f"- **Doorway (assembled)**: {fit.door_asm_in:g} in")
        if fit.door_dis_in is not None:
            lines.append(f"- **Doorway (disassembled)**: {fit.door_dis_in:g} in")
        if fit.wall_clearance_in is not None:
            lines.append(f"- **Wall clearance**: {fit.wall_clearance_in:g} in")
    extra, close_quick, close_patch = _product_closeout(product, domain=domain)
    return SalesReply(
        reply="\n".join(lines) + extra,
        intent=INTENT_SPECS,
        quick_replies=[
            *close_quick,
            QuickReply(label="Check the price", payload=f"price:{product.handle}"),
            QuickReply(label="Compare with another model", payload="compare"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.resolve_product"],
        products=[public],
        prefs_patch=close_patch,
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
    QuickReply(label="Foot & calf", payload="recommend:goal:feet"),
    QuickReply(label="Full-body relax", payload="recommend:goal:full_body"),
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
    QuickReply(label="Not sure", payload="recommend:doorway:skip"),
)

_DOORWAY_FIT_REPLIES = (
    QuickReply(label="Assembled only", payload="recommend:doorway_fit:assembled"),
    QuickReply(
        label="OK to disassemble",
        payload="recommend:doorway_fit:disassembled",
    ),
    QuickReply(label="Not sure", payload="recommend:doorway_fit:assembled"),
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


def _needs_doorway_fit(prefs: dict[str, str]) -> bool:
    """After inches (not skip), ask assembled vs disassemble once."""
    raw = (prefs.get("doorway_in") or "").strip().lower()
    if not raw or raw == "skip":
        return False
    return not (prefs.get("doorway_fit") or "").strip()


def _doorway_limit_in(prefs: dict[str, str]) -> Optional[float]:
    raw = (prefs.get("doorway_in") or "").strip().lower()
    if not raw or raw == "skip":
        return None
    try:
        return float(raw.rstrip("+"))
    except ValueError:
        return None


def _doorway_fit_mode(prefs: dict[str, str]) -> str:
    mode = (prefs.get("doorway_fit") or "assembled").strip().lower()
    if mode in {"assembled", "disassembled", "either"}:
        return mode
    return "assembled"


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
        prefs_patch={"recommend_prefs": prefs, "awaiting_recommend": True},
        flow_stage="ask_doorway",
    )


def _clarify_doorway_fit(prefs: dict[str, str]) -> SalesReply:
    inches = (prefs.get("doorway_in") or "").strip()
    inch_bit = f' ({inches}")' if inches and inches != "skip" else ""
    return SalesReply(
        reply=(
            f"Got it{inch_bit}. Can the delivery team **fully disassemble** "
            "the chair if needed?\n\n"
            "**Assembled only** uses the stricter clearance. "
            "**OK to disassemble** may unlock a few more models."
        ),
        intent=INTENT_RECOMMEND,
        quick_replies=[
            *_DOORWAY_FIT_REPLIES,
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cases.clarify"],
        prefs_patch={"recommend_prefs": prefs, "awaiting_recommend": True},
        flow_stage="ask_doorway_fit",
    )



def _clarify_recommend(missing: str, prefs: dict[str, str]) -> SalesReply:
    patch = {"recommend_prefs": prefs, "awaiting_recommend": True}
    stage = f"ask_{missing}" if missing in {
        "height", "weight", "goal", "intensity", "foot", "space", "doorway_in",
        "doorway_fit",
    } else "ask_height"
    if missing == "height":
        return SalesReply(
            reply=(
                "What's the **user height**? Tap a range, or type it "
                "(like *5'10*). Then I'll ask the main focus and show "
                "**Value / Mid / Premium**."
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
    if missing == "doorway_fit":
        return _clarify_doorway_fit(prefs)
    if missing == "goal":
        return SalesReply(
            reply=(
                "What's the **main focus**? Tap one, or type *back*, *neck*, "
                "or *feet*."
            ),
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


def _short_height_label(height: str) -> str:
    text = (height or "").strip()
    inner = re.search(r"\(([^)]+)\)", text)
    return inner.group(1).strip() if inner else text


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


def _model_doorway_key(
    model_name: str,
    *,
    reason: str,
    mode: str,
) -> float:
    """Per-model doorway inches for sort/filter; 999 = unknown."""
    door = doorway_inches_for_model(model_name, mode=mode)
    if door is not None:
        return door
    parsed = _parse_min_doorway_in(reason)
    return parsed if parsed is not None else 999.0


def _passes_fit_gates(
    model_name: str,
    prefs: dict[str, str],
    *,
    limit: Optional[float],
    mode: str,
) -> bool:
    if not weight_ok(model_name, prefs.get("weight") or ""):
        return False
    if not wall_ok(model_name, prefs.get("space") or ""):
        return False
    if not doorway_ok(model_name, limit_in=limit, mode=mode):
        return False
    return True


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
    mode = _doorway_fit_mode(prefs)

    def _gather(*, apply_hard_filter: bool) -> list[tuple[str, str, float]]:
        local: list[tuple[str, str, float]] = []
        local_seen: set[str] = set()
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
            for name in [lead, *others]:
                if not name or name in used_models or name in local_seen:
                    continue
                if apply_hard_filter and not _passes_fit_gates(
                    name, prefs, limit=limit, mode=mode
                ):
                    continue
                local_seen.add(name)
                door_key = _model_doorway_key(name, reason=reason, mode=mode)
                local.append((name, reason, door_key))
        return local

    return _gather(apply_hard_filter=True)


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


def _no_fit_recommend_reply(
    prefs: dict[str, str],
    *,
    defaults_applied: Optional[list[str]] = None,
) -> SalesReply:
    """Honest empty result when hard fit gates leave no tier picks."""
    bits: list[str] = []
    door = (prefs.get("doorway_in") or "").strip()
    if door and door != "skip":
        mode = _doorway_fit_mode(prefs)
        mode_bit = (
            " (assembled)"
            if mode == "assembled"
            else " (with disassembly)"
            if mode == "disassembled"
            else ""
        )
        bits.append(f'a **{door}"** doorway{mode_bit}')
    if prefs.get("weight"):
        bits.append(f"**{prefs['weight']}** capacity")
    if prefs.get("space") == "Small Room":
        bits.append("**small-room** wall clearance")
    constraint = ", ".join(bits) if bits else "your fit answers"
    lines = [
        f"I couldn't find a chair that safely clears **{constraint}** "
        "in our current Value / Mid / Premium set.",
        "",
        "You can:",
        "• Widen the doorway estimate or allow **disassembly**",
        "• Relax the space constraint",
        "• Talk to a specialist for custom delivery options",
        "",
    ]
    defaults_note = format_defaults_note(defaults_applied or [], prefs)
    if defaults_note:
        lines.append(defaults_note)
        lines.append("")
    lines.append("Or ask to visit the Carrollton showroom.")
    replies = [
        QuickReply(label="Widen doorway / skip", payload="recommend:doorway:skip"),
        QuickReply(
            label="OK to disassemble",
            payload="recommend:doorway_fit:disassembled",
        ),
        QuickReply(label="No space issue", payload="recommend:space:none"),
        QuickReply(label="Talk to a human", payload="human"),
        QuickReply(label="Visit showroom", payload="cta:showroom"),
    ]
    return SalesReply(
        reply="\n".join(lines).rstrip(),
        intent=INTENT_RECOMMEND,
        quick_replies=replies,
        tools_used=["cases.nofit"],
        prefs_patch={"recommend_prefs": prefs, "awaiting_recommend": False},
        flow_stage="recommend_nofit",
    )


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
    header = f"Based on **{fit_label}**, here are three options:"

    public_products: list[dict] = []
    used_models: set[str] = set()
    picked: list[dict] = []
    stock_checked = False
    prefer_compact = _compact_space(prefs)

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
        display = (product.display_name if product else None) or pick_name
        live_price = getattr(snap, "price_usd", None) if snap is not None else None
        catalog_price = product.price_usd if product else None
        show_price = live_price if live_price is not None else catalog_price
        picked.append(
            {
                "tier": tier_label,
                "pick_name": pick_name,
                "product": product,
                "snap": snap,
                "doorway_in": doorway_in,
                "badge": badge,
                "url": url,
                "display": display,
                "show_price": show_price,
                "price": _fmt_live_price(snap, catalog_price),
                "reason": reason,
            }
        )

    if not picked:
        return _no_fit_recommend_reply(prefs, defaults_applied=defaults_applied)

    blurbs = _contrast_tier_blurbs(
        [row["product"] for row in picked],
        prefs=prefs,
        doorway_ins=[row["doorway_in"] for row in picked],
    )
    lines = [header, ""]
    tier_leads: list[dict] = []
    for n, row in enumerate(picked, start=1):
        stock_bit = f" · *{row['badge']}*" if row["badge"] else ""
        lines.append(
            f"**{n}. {row['tier']}** — **{row['display']}** · {row['price']}{stock_bit}"
        )
        if n <= len(blurbs) and blurbs[n - 1]:
            lines.append(blurbs[n - 1])
        if row["url"]:
            lines.append(row["url"])
        product = row["product"]
        if product is not None:
            card = product.as_public_dict()
            if row["show_price"] is not None:
                card["price_usd"] = row["show_price"]
            snap = row["snap"]
            if snap is not None and getattr(snap, "price_max_usd", None) is not None:
                card["price_max_usd"] = snap.price_max_usd
            if row["url"]:
                card["product_url"] = row["url"]
            if row["badge"]:
                card["stock"] = row["badge"]
            public_products.append(card)
        lines.append("")
        tier_leads.append(
            {
                "tier": row["tier"],
                "model": row["pick_name"],
                "display": row["display"],
                "handle": product.handle if product else None,
                "url": row["url"],
                "stock": row["badge"],
            }
        )

    defaults_note = format_defaults_note(defaults_applied or [], prefs)
    if defaults_note:
        lines.append(defaults_note)
        lines.append("")

    if is_sales_after_hours():
        lines.append(after_hours_blurb())
        lines.append("")

    lines.append(
        "Reply **1 / 2 / 3** for that chair, or email these picks. "
        "If a doorway is tight or you weigh over 220 lb, tell me and I'll re-check fit."
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
            "awaiting_recommend": False,
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
            "awaiting_recommend": False,
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
                "What's the **user height**? Tap a range, or type it "
                "(like *5'10*). Then I'll ask the main focus and show "
                "**Value / Mid / Premium**."
            ),
            intent=INTENT_RECOMMEND,
            quick_replies=[
                *_HEIGHT_REPLIES,
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.parse_hints"],
            flow_stage="ask_height",
            prefs_patch={"awaiting_recommend": True},
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
        prefs_patch={"awaiting_recommend": False},
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
    before_defaults = dict(merged)
    awaiting = bool((prefs or {}).get("awaiting_recommend"))
    height_just_set = False
    if awaiting and not merged.get("height"):
        answered_height = height_from_shopper_answer(message or "")
        if answered_height:
            merged["height"] = answered_height
            height_just_set = True
    if (
        awaiting
        and merged.get("height")
        and not merged.get("goal")
        and not height_just_set
    ):
        if re.search(
            r"\b(not\s+sure|idk|i\s+don'?t\s+know|dunno|no\s+idea|everything|general)\b",
            message or "",
            re.I,
        ):
            merged["goal"] = "Full-Body Relaxation"
    # Bare "30 inch" / '32"' answers after ask_doorway.
    door_msg = _parse_doorway_inches_message(message or "")
    if door_msg is not None and not (merged.get("doorway_in") or "").strip():
        merged["doorway_in"] = (
            str(int(door_msg)) if door_msg == int(door_msg) else str(door_msg)
        )
        if "space" not in merged:
            merged["space"] = "Narrow Doorway"

    # Budget is never asked and never gates the reply — Value/Mid/Premium
    # tiers inject case-book budget bands internally only.
    merged.pop("budget", None)
    merged = enrich_implied_prefs(merged)
    defaults_applied = secondary_defaults_applied(before_defaults, merged)

    # After compact space, ask doorway inches, then assembled vs disassemble.
    if (
        merged.get("height")
        and merged.get("weight")
        and (merged.get("space") or "") in _COMPACT_SPACES
        and _needs_doorway_inches(merged)
    ):
        return _clarify_doorway_inches(merged)
    if (
        merged.get("height")
        and merged.get("weight")
        and (merged.get("space") or "") in _COMPACT_SPACES
        and _needs_doorway_fit(merged)
    ):
        return _clarify_doorway_fit(merged)

    missing = missing_ask(merged)
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



_COMPARE_SPEC_FIELDS = (
    ("Mechanism", "mechanism"),
    ("Track", "track"),
    ("Zero gravity", "zero_gravity"),
    ("Heating", "heating"),
    ("Foot roller", "foot_roller"),
)


def _clear_compare_state() -> dict:
    return {
        "awaiting_compare_pick": "",
        "pending_compare_left": "",
        "pending_compare_right": "",
        "pending_compare_left_query": "",
        "pending_compare_right_query": "",
        "pending_compare_candidates": [],
        "pending_compare_pair": [],
        "awaiting_compare_recommend": False,
    }


def _compare_example_prompt() -> str:
    left = lookup_shop_models("Maestro 4D").unique
    right = lookup_shop_models("Paragon").unique
    if left is not None and right is not None:
        return (
            f'Try *"{short_model_label(left)} vs {short_model_label(right)}"* '
            "or *\"Maestro LE vs Champ II\"*."
        )
    return 'Try *"Maestro 4D vs Paragon"* — short names are fine.'


def _compare_spec_lines(diff: dict) -> list[str]:
    lines: list[str] = []
    for label, key in _COMPARE_SPEC_FIELDS:
        pair = diff.get(key) or ("", "")
        left_shown = _spec_customer_value(pair[0]) or "not listed"
        right_shown = _spec_customer_value(pair[1]) or "not listed"
        if left_shown == "not listed" and right_shown == "not listed":
            continue
        lines.append(
            f"- **{label}**: {left_shown} vs {right_shown}"
        )
    delta = diff.get("price_delta_usd")
    if delta is not None:
        if abs(delta) < 1:
            lines.append("- **Price gap**: same published price.")
        else:
            direction = "more" if delta > 0 else "less"
            lines.append(
                f"- **Price gap**: the second is about ${abs(delta):,.0f} {direction}."
            )
    differing = [
        label.lower()
        for label, key in _COMPARE_SPEC_FIELDS
        if (_spec_customer_value((diff.get(key) or ("", ""))[0]) or "not listed")
        != (_spec_customer_value((diff.get(key) or ("", ""))[1]) or "not listed")
        and (
            _spec_customer_value((diff.get(key) or ("", ""))[0]) is not None
            or _spec_customer_value((diff.get(key) or ("", ""))[1]) is not None
        )
    ]
    if not differing and abs(delta or 0) < 1:
        lines.append(
            "\n**Bottom line:** These two sit in the same tier on published "
            "specs/price — a rep can help you choose by feel/fit."
        )
    elif differing:
        lines.append("\n**Biggest differences:** " + ", ".join(differing) + ".")
    return lines


def _compare_resolved_reply(
    left: ProductSpecs,
    right: ProductSpecs,
    *,
    domain: str,
) -> SalesReply:
    result = compare_products(left, right)
    diff = result["diff"]
    left_url = product_page_url(domain, left.handle)
    right_url = product_page_url(domain, right.handle)
    lines = [
        f"**{left.display_name}** — {_fmt_price(left.price_usd)}",
        f"**{right.display_name}** — {_fmt_price(right.price_usd)}",
        "",
        *_compare_spec_lines(diff),
    ]
    quick: list[QuickReply] = []
    if left_url:
        quick.append(
            QuickReply(
                label=f"Shop {short_model_label(left, limit=18)}",
                payload=f"open:{left_url}",
            )
        )
    if right_url:
        quick.append(
            QuickReply(
                label=f"Shop {short_model_label(right, limit=18)}",
                payload=f"open:{right_url}",
            )
        )
    quick.append(
        QuickReply(label="Recommend which one", payload="compare:recommend")
    )
    quick.append(QuickReply(label="Talk to a human", payload="human"))
    patch = _clear_compare_state()
    patch["pending_compare_pair"] = [left.handle, right.handle]
    patch["pending_compare_left"] = left.handle
    patch["pending_compare_right"] = right.handle
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_COMPARE,
        quick_replies=quick,
        tools_used=["catalog.compare"],
        products=[result["left"], result["right"]],
        prefs_patch=patch,
    )


def _compare_missing_reply(
    *,
    found: Optional[ProductSpecs],
    missing_query: str,
    found_slot: str,
    other_query: str,
    domain: str,
) -> SalesReply:
    if found is None:
        return SalesReply(
            reply=(
                f"**{missing_query}** is not on the current store catalog — "
                "it may be an older or warranty-only name. "
                "Which two chairs we sell now should I compare?"
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
            prefs_patch=_clear_compare_state(),
        )
    other_slot = "right" if found_slot == "left" else "left"
    patch = _clear_compare_state()
    patch["awaiting_compare_pick"] = other_slot
    patch[f"pending_compare_{found_slot}"] = found.handle
    patch[f"pending_compare_{found_slot}_query"] = found.display_name
    patch[f"pending_compare_{other_slot}_query"] = other_query
    return SalesReply(
        reply=(
            f"I have **{found.display_name}**. **{missing_query}** is not on "
            "the current store catalog — it may be an older or warranty-only "
            "name.\n\nType another model we sell now, or tap **See all models**."
        ),
        intent=INTENT_COMPARE,
        quick_replies=[
            QuickReply(label="See all models", payload="list"),
            QuickReply(label="Recommend a chair", payload="recommend"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["catalog.compare"],
        products=[found.as_public_dict()],
        prefs_patch=patch,
    )


def _ask_compare_which(
    slot: str,
    query: str,
    matches: list[ProductSpecs],
    *,
    prefs: Optional[dict],
    left_q: str,
    right_q: str,
    left_handle: str = "",
    right_handle: str = "",
    missing_note: str = "",
) -> SalesReply:
    listed = "\n".join(f"- **{item.display_name}**" for item in matches)
    quick: list[QuickReply] = []
    for item in matches[:3]:
        quick.append(
            QuickReply(
                label=short_model_label(item),
                payload=f"compare:pick:{item.handle}",
            )
        )
    if len(matches) > 3:
        extra = "\n".join(f"- **{item.display_name}**" for item in matches[3:])
        listed = listed + "\n" + extra
    quick.append(QuickReply(label="Talk to a human", payload="human"))
    patch = {
        "awaiting_compare_pick": slot,
        "pending_compare_left_query": left_q,
        "pending_compare_right_query": right_q,
        "pending_compare_left": left_handle,
        "pending_compare_right": right_handle,
        "pending_compare_candidates": [
            {"handle": item.handle, "display": item.display_name}
            for item in matches
        ],
        "awaiting_compare_recommend": False,
    }
    head = f"{missing_note}\n\n" if missing_note else ""
    return SalesReply(
        reply=(
            f"{head}Which **{query}** do you mean?\n\n{listed}\n\n"
            "I won't guess the family member."
        ),
        intent=INTENT_COMPARE,
        quick_replies=quick,
        tools_used=["catalog.compare"],
        products=[item.as_public_dict() for item in matches[:3]],
        prefs_patch=patch,
    )


def _finish_compare_pair(
    left: Optional[ProductSpecs],
    right: Optional[ProductSpecs],
    *,
    left_q: str,
    right_q: str,
    prefs: Optional[dict],
    domain: str,
) -> SalesReply:
    if left is not None and right is not None:
        if left.handle == right.handle:
            return SalesReply(
                reply=(
                    f"Those both resolve to **{left.display_name}**. "
                    "Name a second chair to compare it with."
                ),
                intent=INTENT_COMPARE,
                quick_replies=[
                    QuickReply(label="See all models", payload="list"),
                    QuickReply(label="Talk to a human", payload="human"),
                ],
                tools_used=["catalog.compare"],
                prefs_patch={
                    "awaiting_compare_pick": "right",
                    "pending_compare_left": left.handle,
                    "pending_compare_left_query": left.display_name,
                    "pending_compare_right": "",
                    "pending_compare_right_query": "",
                    "pending_compare_candidates": [],
                    "awaiting_compare_recommend": False,
                },
            )
        return _compare_resolved_reply(left, right, domain=domain)
    if left is None and right is None:
        return SalesReply(
            reply=(
                f"I couldn't match **{left_q}** or **{right_q}** to the current "
                "store catalog. Short names are fine — they just have to be "
                "chairs we sell now."
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
            prefs_patch=_clear_compare_state(),
        )
    if left is None:
        return _compare_missing_reply(
            found=right,
            missing_query=left_q,
            found_slot="right",
            other_query=left_q,
            domain=domain,
        )
    return _compare_missing_reply(
        found=left,
        missing_query=right_q,
        found_slot="left",
        other_query=right_q,
        domain=domain,
    )


def _resolve_compare_side(
    query: str,
    *,
    slot: str,
    other: Optional[ProductSpecs],
    other_query: str,
    left_q: str,
    right_q: str,
    prefs: Optional[dict],
    domain: str,
) -> Optional[SalesReply]:
    """Return a reply when this side is not a unique match; None if unique."""
    looked = lookup_shop_models(query)
    if looked.vague:
        return SalesReply(
            reply=(
                f"**{query or 'that'}** is too vague — I need a model nickname "
                f"like Maestro or Paragon. {_compare_example_prompt()}"
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
        )
    if len(looked.matches) > 1:
        left_handle = (
            other.handle if other is not None and slot == "right" else ""
        )
        right_handle = (
            other.handle if other is not None and slot == "left" else ""
        )
        return _ask_compare_which(
            slot,
            query,
            list(looked.matches),
            prefs=prefs,
            left_q=left_q,
            right_q=right_q,
            left_handle=left_handle,
            right_handle=right_handle,
        )
    return None


def _compare_reply(
    message: str,
    *,
    prefs: Optional[dict] = None,
    domain: str = "osakiusa.com",
) -> SalesReply:
    pair = split_compare_terms(message)
    if pair is None:
        return SalesReply(
            reply=(
                "Sure — which two models? Short names are fine. "
                f"{_compare_example_prompt()}"
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
            prefs_patch=_clear_compare_state(),
        )

    left_q, right_q = pair
    left_lookup = lookup_shop_models(left_q)
    right_lookup = lookup_shop_models(right_q)
    left = left_lookup.unique
    right = right_lookup.unique
    left_missing = not left_lookup.vague and not left_lookup.matches
    right_missing = not right_lookup.vague and not right_lookup.matches

    def _not_on_catalog(name: str) -> str:
        return (
            f"**{name}** is not on the current store catalog — "
            "it may be an older or warranty-only name."
        )

    if left_missing or right_missing:
        if left_missing and right_missing:
            return _finish_compare_pair(
                None,
                None,
                left_q=left_q,
                right_q=right_q,
                prefs=prefs,
                domain=domain,
            )
        if right_missing and len(left_lookup.matches) > 1:
            return _ask_compare_which(
                "left",
                left_q,
                list(left_lookup.matches),
                prefs=prefs,
                left_q=left_q,
                right_q=right_q,
                missing_note=_not_on_catalog(right_q),
            )
        if left_missing and len(right_lookup.matches) > 1:
            return _ask_compare_which(
                "right",
                right_q,
                list(right_lookup.matches),
                prefs=prefs,
                left_q=left_q,
                right_q=right_q,
                missing_note=_not_on_catalog(left_q),
            )
        if left_missing:
            return _compare_missing_reply(
                found=right,
                missing_query=left_q,
                found_slot="right",
                other_query=left_q,
                domain=domain,
            )
        return _compare_missing_reply(
            found=left,
            missing_query=right_q,
            found_slot="left",
            other_query=right_q,
            domain=domain,
        )

    blocked = _resolve_compare_side(
        left_q,
        slot="left",
        other=right,
        other_query=right_q,
        left_q=left_q,
        right_q=right_q,
        prefs=prefs,
        domain=domain,
    )
    if blocked is not None:
        return blocked
    blocked = _resolve_compare_side(
        right_q,
        slot="right",
        other=left,
        other_query=left_q,
        left_q=left_q,
        right_q=right_q,
        prefs=prefs,
        domain=domain,
    )
    if blocked is not None:
        return blocked
    return _finish_compare_pair(
        left,
        right,
        left_q=left_q,
        right_q=right_q,
        prefs=prefs,
        domain=domain,
    )


def _apply_compare_pick(
    handle: str,
    prefs: Optional[dict],
    *,
    domain: str,
) -> SalesReply:
    picked = product_by_handle(handle) or resolve_product(handle)
    data = prefs or {}
    slot = str(data.get("awaiting_compare_pick") or "left").strip().lower()
    if slot not in {"left", "right"}:
        slot = "left"
    if picked is None:
        return SalesReply(
            reply=(
                "I couldn't match that tap to a current store chair. "
                f"{_compare_example_prompt()}"
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
        )

    left_q = str(data.get("pending_compare_left_query") or "").strip()
    right_q = str(data.get("pending_compare_right_query") or "").strip()
    left_handle = str(data.get("pending_compare_left") or "").strip()
    right_handle = str(data.get("pending_compare_right") or "").strip()
    if slot == "left":
        left_handle = picked.handle
        left_q = picked.display_name
    else:
        right_handle = picked.handle
        right_q = picked.display_name

    left = product_by_handle(left_handle) if left_handle else None
    right = product_by_handle(right_handle) if right_handle else None
    if left is None and right is None:
        left, right = (picked, None) if slot == "left" else (None, picked)

    if left is None and right_q:
        looked = lookup_shop_models(right_q)
        if looked.unique:
            right = looked.unique
        elif len(looked.matches) > 1:
            return _ask_compare_which(
                "right",
                right_q,
                list(looked.matches),
                prefs=data,
                left_q=left_q or picked.display_name,
                right_q=right_q,
                left_handle=picked.handle,
            )
    if right is None and left_q:
        looked = lookup_shop_models(left_q)
        if looked.unique:
            left = looked.unique
        elif len(looked.matches) > 1:
            return _ask_compare_which(
                "left",
                left_q,
                list(looked.matches),
                prefs=data,
                left_q=left_q,
                right_q=right_q or picked.display_name,
                right_handle=picked.handle,
            )

    return _finish_compare_pair(
        left,
        right,
        left_q=left_q or (left.display_name if left else ""),
        right_q=right_q or (right.display_name if right else ""),
        prefs=data,
        domain=domain,
    )


def _compare_digit_reply(message: str, prefs: Optional[dict], *, domain: str) -> Optional[SalesReply]:
    if not (prefs or {}).get("awaiting_compare_pick"):
        return None
    digit = re.fullmatch(r"([1-9])[).:\s]*", (message or "").strip())
    if not digit:
        return None
    cands = (prefs or {}).get("pending_compare_candidates") or []
    if not isinstance(cands, list):
        return None
    idx = int(digit.group(1)) - 1
    if not (0 <= idx < len(cands)):
        return None
    handle = str((cands[idx] or {}).get("handle") or "").strip()
    if not handle:
        return None
    return _apply_compare_pick(handle, prefs, domain=domain)


def _clarify_compare_recommend(
    missing: str,
    rec: dict[str, str],
    left: ProductSpecs,
    right: ProductSpecs,
) -> SalesReply:
    result = _clarify_recommend(missing, rec)
    names = f"**{left.display_name}** and **{right.display_name}**"
    if missing == "height":
        result.reply = (
            f"To choose between {names}, what's the **user height**?\n\n"
            "I'll pick from these two using published fit specs — "
            "not which one feels better."
        )
    elif missing == "goal":
        result.reply = (
            f"What's the **main focus** so I can choose between {names}?"
        )
    patch = dict(result.prefs_patch or {})
    patch["recommend_prefs"] = rec
    patch["awaiting_compare_recommend"] = True
    patch["pending_compare_pair"] = [left.handle, right.handle]
    result.prefs_patch = patch
    result.tools_used = (result.tools_used or []) + ["catalog.compare"]
    return result


def _compare_recommend_reply(
    message: str,
    *,
    payload: Optional[str] = None,
    prefs: Optional[dict] = None,
    domain: str = "osakiusa.com",
) -> SalesReply:
    data = prefs or {}
    handles = data.get("pending_compare_pair") or []
    if not isinstance(handles, list) or len(handles) < 2:
        left = product_by_handle(str(data.get("pending_compare_left") or ""))
        right = product_by_handle(str(data.get("pending_compare_right") or ""))
    else:
        left = product_by_handle(str(handles[0] or ""))
        right = product_by_handle(str(handles[1] or ""))
    if left is None or right is None:
        return SalesReply(
            reply=(
                "I need two current store chairs first. "
                f"{_compare_example_prompt()}"
            ),
            intent=INTENT_COMPARE,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.compare"],
            prefs_patch={"awaiting_compare_recommend": False},
        )

    rec = dict(data.get("recommend_prefs") or {})
    request = parse_recommendation_hints(message or "")
    if payload:
        rec = apply_payload_codes(rec, payload)
    rec = merge_prefs_from_hints(
        rec,
        height_in=request.height_in,
        weight_lb=request.weight_lb,
        budget_usd=request.budget_usd,
        focus_areas=request.focus_areas,
        free_text=request.free_text or message,
    )
    rec = enrich_implied_prefs(rec)
    missing = missing_ask(rec)
    if missing:
        return _clarify_compare_recommend(missing[0], rec, left, right)

    left_score, right_score = pair_fit_scores(left, right, {"recommend_prefs": rec})
    if abs(left_score - right_score) < 0.75:
        result = compare_products(left, right)
        lines = [
            f"Published fit is too close to call between **{left.display_name}** "
            f"and **{right.display_name}**.",
            "",
            *_compare_spec_lines(result["diff"]),
            "",
            "I won't invent which one feels better — a specialist can sit you "
            "in both at the showroom.",
        ]
        left_url = product_page_url(domain, left.handle)
        right_url = product_page_url(domain, right.handle)
        quick: list[QuickReply] = []
        if left_url:
            quick.append(
                QuickReply(
                    label=f"Shop {short_model_label(left, limit=18)}",
                    payload=f"open:{left_url}",
                )
            )
        if right_url:
            quick.append(
                QuickReply(
                    label=f"Shop {short_model_label(right, limit=18)}",
                    payload=f"open:{right_url}",
                )
            )
        quick.append(QuickReply(label="Talk to a human", payload="human"))
        return SalesReply(
            reply="\n".join(lines),
            intent=INTENT_COMPARE,
            quick_replies=quick,
            tools_used=["catalog.compare", "cases.tiered"],
            products=[result["left"], result["right"]],
            prefs_patch={
                "recommend_prefs": rec,
                "pending_compare_pair": [left.handle, right.handle],
                "awaiting_compare_recommend": False,
            },
        )

    winner, other = (left, right) if left_score >= right_score else (right, left)
    reasons = pair_fit_reasons(winner, other, {"recommend_prefs": rec})
    url = product_page_url(domain, winner.handle)
    other_url = product_page_url(domain, other.handle)
    lines = [
        f"Between these two, **{winner.display_name}** is the better "
        f"**published fit** for **{rec.get('height') or 'your height'}**"
        + (f" and **{rec.get('goal')}**" if rec.get("goal") else "")
        + ".",
        "",
    ]
    if reasons:
        lines.extend(f"- {bit}" for bit in reasons)
    else:
        lines.append(
            "- Catalog track, mechanism, and listed capacity line up better "
            "on this one."
        )
    lines.append(
        "\nThat is catalog fit, not which chair feels stronger in person."
    )
    if url:
        lines.append(f"\nShop: {url}")
    quick = []
    if url:
        quick.append(QuickReply(label="Shop this chair", payload=f"open:{url}"))
    if other_url:
        quick.append(
            QuickReply(
                label=f"Shop {short_model_label(other, limit=18)}",
                payload=f"open:{other_url}",
            )
        )
    quick.append(QuickReply(label="Talk to a human", payload="human"))
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_COMPARE,
        quick_replies=quick,
        tools_used=["catalog.compare", "cases.tiered"],
        products=[winner.as_public_dict(), other.as_public_dict()],
        prefs_patch={
            "recommend_prefs": rec,
            "pending_primary": winner.display_name,
            "pending_product_url": url,
            "pending_compare_pair": [left.handle, right.handle],
            "awaiting_compare_recommend": False,
        },
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


def _short_chair_label(name: str) -> str:
    text = (name or "").strip()
    for prefix in ("Osaki ", "Titan "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
            break
    if len(text) <= 22:
        return text
    return text[:22].rstrip()


def _greeting_reply(prefs: Optional[dict] = None) -> SalesReply:
    data = prefs or {}
    if data.get("visitor_resume_declined") or data.get("visitor_resume_accepted"):
        return SalesReply(
            reply=_MENU_INTRO,
            intent=INTENT_GREETING,
            quick_replies=_menu_quick_replies(),
        )
    if not has_resume_memory(data):
        return SalesReply(
            reply=_MENU_INTRO,
            intent=INTENT_GREETING,
            quick_replies=_menu_quick_replies(),
        )

    primary = str(data.get("pending_primary") or "").strip()
    rec = data.get("recommend_prefs") or {}
    if not isinstance(rec, dict):
        rec = {}
    height = str(rec.get("height") or "").strip()
    goal = str(rec.get("goal") or "").strip()

    if primary:
        lines = [f"Still looking at the **{primary}**?"]
        if height and goal:
            lines.append(
                f"Last fit was {_short_height_label(height)} for **{goal}**."
            )
        lines.append("Pick up there, see three price tiers, or start over.")
    else:
        lines = [
            f"Welcome back. Last fit was {_short_height_label(height)} for **{goal}**.",
            "Want those three picks again, or start over?",
        ]

    quick: list[QuickReply] = []
    if primary:
        quick.append(
            QuickReply(
                label=f"Continue: {_short_chair_label(primary)}",
                payload="resume:continue",
            )
        )
    if height and goal:
        quick.append(
            QuickReply(label="Show my three picks", payload="resume:picks")
        )
    quick.append(QuickReply(label="Start over", payload="resume:reset"))
    quick.append(QuickReply(label="Talk to a human", payload="human"))
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_GREETING,
        quick_replies=quick,
        tools_used=["visitor.resume_offer"],
        prefs_patch={"visitor_resume_offered": True},
        flow_stage="menu",
    )


def _resume_continue_reply(prefs: Optional[dict], *, domain: str) -> SalesReply:
    primary = str((prefs or {}).get("pending_primary") or "").strip()
    product = resolve_product(primary) if primary else None
    if product is None and primary:
        product = _guess_model_from_text(primary)
    if product is None:
        return _resume_picks_reply(prefs, domain=domain)

    _, close_quick, close_patch = _product_closeout(product, domain=domain)
    url = str(close_patch.get("pending_product_url") or "").strip()
    rec = (prefs or {}).get("recommend_prefs") or {}
    has_fit = isinstance(rec, dict) and bool(
        str(rec.get("height") or "").strip() and str(rec.get("goal") or "").strip()
    )
    highlight = _chair_highlight_line(product)
    lines = [
        f"Still looking at the **{product.display_name}** — "
        f"{_fmt_price(product.price_usd)}."
    ]
    if highlight:
        lines.append(highlight + ".")
    if url:
        lines.append(f"Shop: {url}")
    lines.append(
        "Shop this one, see other price tiers, or open the full spec sheet."
    )

    quick: list[QuickReply] = []
    if url.startswith("https://"):
        quick.append(QuickReply(label="Shop this chair", payload=f"open:{url}"))
    if has_fit:
        quick.append(
            QuickReply(label="Other price tiers", payload="resume:picks")
        )
    else:
        quick.append(
            QuickReply(label="Recommend other chairs", payload="recommend")
        )
    if product.handle:
        quick.append(
            QuickReply(label="Full specs", payload=f"specs:{product.handle}")
        )
    quick.extend(close_quick)
    deduped: list[QuickReply] = []
    seen: set[str] = set()
    for item in quick:
        key = item.payload.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.append(QuickReply(label="Talk to a human", payload="human"))

    patch = dict(close_patch)
    patch["visitor_resume_accepted"] = True
    return SalesReply(
        reply="\n".join(lines),
        intent=INTENT_SPECS,
        quick_replies=deduped,
        tools_used=["visitor.resume_continue", "catalog.resolve_product"],
        products=[product.as_public_dict()],
        prefs_patch=patch,
        flow_stage="shop",
    )


def _resume_picks_reply(prefs: Optional[dict], *, domain: str) -> SalesReply:
    result = _recommend_reply("recommend", domain=domain, prefs=prefs)
    patch = dict(result.prefs_patch or {})
    patch["visitor_resume_accepted"] = True
    result.prefs_patch = patch
    result.tools_used = (result.tools_used or []) + ["visitor.resume_picks"]
    return result


def _resume_reset_reply() -> SalesReply:
    return SalesReply(
        reply=_MENU_INTRO,
        intent=INTENT_GREETING,
        quick_replies=_menu_quick_replies(),
        tools_used=["visitor.resume_reset"],
        prefs_patch={
            "visitor_resume_declined": True,
            "visitor_resume_accepted": False,
            "visitor_resume_offered": False,
            "recommend_prefs": None,
            "pending_primary": "",
            "pending_product_url": "",
            "pending_pick_summary": "",
            "pending_tier_picks": [],
            "visitor_memory_hydrated": False,
            "awaiting_compare_pick": "",
            "pending_compare_pair": [],
            "awaiting_compare_recommend": False,
        },
        flow_stage="menu",
    )


#: Answers substantive enough to be worth rating. Asking after a menu prompt
#: or a clarifying question would measure nothing and add friction.
RATEABLE_INTENTS = frozenset(
    {
        INTENT_PREPURCHASE_POLICY,
        INTENT_PRICE,
        INTENT_SPECS,
        INTENT_STOCK,
        INTENT_COMPARE,
    }
)


def feedback_quick_replies() -> list[QuickReply]:
    return [
        QuickReply(label="That helped", payload="feedback:up"),
        QuickReply(label="Not what I needed", payload="feedback:down"),
    ]


def _feedback_reply(direction: str, prefs: Optional[dict]) -> Optional[SalesReply]:
    """Acknowledge a rating and hand the router a row to persist."""
    if direction not in {"up", "down"}:
        return None
    rating = "helpful" if direction == "up" else "not_helpful"
    rated_intent = str((prefs or {}).get("last_rateable_intent") or "") or None

    if rating == "helpful":
        text = "Glad that helped. Anything else I can check for you?"
        quick = _menu_quick_replies()
    else:
        text = (
            "Sorry that missed the mark — thanks for telling me. "
            "Want me to try a different angle, or put you with a specialist?"
        )
        quick = [
            QuickReply(label="Recommend a chair", payload="recommend"),
            QuickReply(label="Talk to a human", payload="human"),
        ]

    return SalesReply(
        reply=text,
        intent="feedback",
        quick_replies=quick,
        tools_used=[f"feedback.{rating}"],
        feedback={"rating": rating, "intent": rated_intent},
    )


def _prepurchase_policy_reply(message: str, *, domain: str) -> SalesReply:
    """Answer a published policy question instead of handing it to a human."""
    topic = detect_policy_topic(message)
    answer = policy_answer(topic, domain, message=message) if topic else None
    if answer is None:
        return _unclear_reply()

    followups = [QuickReply(label="Recommend a chair", payload="recommend")]
    if topic in (TOPIC_SHIPPING, TOPIC_WHITE_GLOVE, TOPIC_REMOTE_SHIPPING):
        followups.append(
            QuickReply(label="Check doorway fit", payload="recommend:space:narrow")
        )
    if topic == TOPIC_SHOWROOM:
        followups.insert(
            0, QuickReply(label="Request a visit", payload="cta:showroom:book")
        )
    else:
        followups.append(QuickReply(label="Visit showroom", payload="cta:showroom"))
    followups.append(QuickReply(label="Talk to a human", payload="human"))

    return SalesReply(
        reply=answer,
        intent=INTENT_PREPURCHASE_POLICY,
        quick_replies=followups,
        tools_used=[f"policy.{topic}"],
    )


def _unclear_reply() -> SalesReply:
    return SalesReply(
        reply=(
            "I can help with a recommendation, a price, specs, stock, or "
            "shipping — I just need a bit more. Tap one of these, or type a "
            "model name (e.g. *\"price of the OS-Pro Maestro\"*)."
        ),
        intent=INTENT_UNCLEAR,
        quick_replies=_menu_quick_replies(),
    )


def _list_reply() -> SalesReply:
    picks = [p for p in list_active_products() if is_public_browse_pick(p)][:8]
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
    "list": lambda _msg: _list_reply(),
    "compare": lambda msg: _compare_reply(msg),  # prefs/domain filled in _payload_reply
    "price": lambda msg: _price_reply(msg),
    "stock": lambda msg: _stock_reply(msg),
    "specs": lambda msg: _specs_reply(msg),
    "intensity": lambda msg: _intensity_reply(msg),
    "human": lambda _msg: _ask_question_before_human(),
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
        f"Thanks — I saved **{email}** with your pick (**{primary}**).",
        "You'll get a confirmation email, and someone from sales will follow up "
        "(usually next business day).",
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


def _showroom_cta_reply(prefs: Optional[dict], *, domain: str) -> SalesReply:
    primary = ((prefs or {}).get("pending_primary") or "").strip()
    url = ((prefs or {}).get("pending_product_url") or "").strip()
    quick = [
        QuickReply(label="Request a visit", payload="cta:showroom:book"),
        QuickReply(label="Talk to a human", payload="human"),
    ]
    if url.startswith("https://"):
        quick.insert(1, QuickReply(label="Shop this chair", payload=f"open:{url}"))
    return SalesReply(
        reply=showroom_blurb(domain=domain, primary=primary),
        intent=INTENT_PREPURCHASE_POLICY,
        quick_replies=quick,
        tools_used=["cta.showroom"],
        flow_stage="shop",
    )


def _showroom_book_windows_reply() -> SalesReply:
    hours = showroom_hours()
    return SalesReply(
        reply=(
            "Which window should I **request**?\n\n"
            f"Showroom hours: **{hours}**.\n"
            "Sales will confirm — this is not a locked appointment."
        ),
        intent=INTENT_PREPURCHASE_POLICY,
        quick_replies=[
            QuickReply(
                label="Weekday morning",
                payload="cta:showroom:window:weekday_am",
            ),
            QuickReply(
                label="Weekday afternoon",
                payload="cta:showroom:window:weekday_pm",
            ),
            QuickReply(
                label="Saturday",
                payload="cta:showroom:window:saturday",
            ),
            QuickReply(label="I'll type a time", payload="cta:showroom:type"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cta.showroom_book"],
        flow_stage="lead",
        prefs_patch={
            "awaiting_showroom_window": False,
            "awaiting_showroom_email": False,
        },
    )


def _showroom_type_time_reply() -> SalesReply:
    hours = showroom_hours()
    return SalesReply(
        reply=(
            f"Type a day and time that fits **{hours}**.\n\n"
            "I'll send it as a request — sales still confirms the slot."
        ),
        intent=INTENT_PREPURCHASE_POLICY,
        quick_replies=[
            QuickReply(label="Back to windows", payload="cta:showroom:book"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cta.showroom_book"],
        flow_stage="lead",
        prefs_patch={"awaiting_showroom_window": True, "awaiting_showroom_email": False},
    )


def _showroom_ask_email_reply(prefs: Optional[dict], *, window: str) -> SalesReply:
    window = (window or "").strip() or "a time sales will confirm"
    primary = ((prefs or {}).get("pending_primary") or "").strip()
    extra = f" for the **{primary}**" if primary else ""
    return SalesReply(
        reply=(
            f"Got it — I'll request **{window}**{extra} at the Carrollton showroom.\n\n"
            "Type your **email** so sales can confirm. "
            "This is a request, not a booked appointment."
        ),
        intent="lead_capture",
        quick_replies=[
            QuickReply(label="Back to windows", payload="cta:showroom:book"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cta.showroom_book"],
        flow_stage="lead",
        prefs_patch={
            "awaiting_showroom_window": False,
            "awaiting_showroom_email": True,
            "pending_showroom_window": window,
        },
    )


def _capture_showroom_visit(
    email: str, prefs: Optional[dict], *, domain: str
) -> SalesReply:
    window = str((prefs or {}).get("pending_showroom_window") or "").strip() or (
        "a time sales will confirm"
    )
    primary = ((prefs or {}).get("pending_primary") or "").strip()
    url = ((prefs or {}).get("pending_product_url") or "").strip()
    summary_lines = [
        "Showroom visit request (not a confirmed appointment)",
        f"Store: {domain}",
        f"Requested window: {window}",
        f"Address: {showroom_address()}",
    ]
    if primary:
        summary_lines.append(f"Chair: {primary}")
    if url:
        summary_lines.append(f"Product URL: {url}")
    summary = "\n".join(summary_lines)
    lines = [
        f"Sent a **visit request** for **{window}** at the Carrollton showroom.",
        f"I saved **{email}** so sales can confirm — this is **not** a locked appointment.",
    ]
    if is_sales_after_hours():
        lines.append(after_hours_blurb())
    return SalesReply(
        reply="\n".join(lines),
        intent="lead_capture",
        handoff=False,
        quick_replies=[
            QuickReply(label="Back to menu", payload="menu"),
            QuickReply(label="Recommend a chair", payload="recommend"),
            QuickReply(label="Talk to a human", payload="human"),
        ],
        tools_used=["cta.showroom_book", "lead.capture"],
        flow_stage="lead",
        prefs_patch={
            "awaiting_showroom_email": False,
            "awaiting_showroom_window": False,
        },
        lead_capture={
            "email": email,
            "interest_summary": summary,
            "reason": "showroom_visit",
        },
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
    if root == "menu":
        return _greeting_reply(prefs)
    if root == "resume":
        action = parts[1].strip().lower() if len(parts) > 1 else ""
        if action == "continue":
            return _resume_continue_reply(prefs, domain=domain)
        if action == "picks":
            return _resume_picks_reply(prefs, domain=domain)
        if action == "reset":
            return _resume_reset_reply()
        return _greeting_reply(prefs)
    if root == "tier" and len(parts) >= 2:
        try:
            n = int(re.sub(r"\D", "", parts[1]) or "0")
        except ValueError:
            n = 0
        if 1 <= n <= 3:
            return _tier_followup_reply(n - 1, prefs, domain=domain)
        return None
    if root == "compare":
        action = parts[1].strip().lower() if len(parts) > 1 else ""
        if action == "tiers" and len(parts) >= 4:
            try:
                left_n = int(parts[2])
                right_n = int(parts[3])
            except ValueError:
                return None
            return _compare_pending_tiers_reply(
                prefs, left_n, right_n, domain=domain
            )
        if action == "pick" and len(parts) >= 3:
            return _apply_compare_pick(
                ":".join(parts[2:]), prefs, domain=domain
            )
        if action == "recommend":
            return _compare_recommend_reply(
                message or "",
                payload=payload,
                prefs=prefs,
                domain=domain,
            )
        return _compare_reply(message or "", prefs=prefs, domain=domain)
    if root == "recommend":
        action = parts[1].strip().lower() if len(parts) > 1 else ""
        full_recommend = action in {"", "again"}
        if (prefs or {}).get("awaiting_compare_recommend") and not full_recommend:
            return _compare_recommend_reply(
                message or "",
                payload=payload,
                prefs=prefs,
                domain=domain,
            )
        patched = dict(prefs or {})
        if full_recommend:
            patched["awaiting_compare_pick"] = ""
            patched["awaiting_compare_recommend"] = False
            patched["pending_compare_candidates"] = []
        return _recommend_reply(
            message or "recommend",
            payload=payload,
            domain=domain,
            prefs=patched,
        )
    if root == "stock":
        return _stock_reply(message or payload, domain=domain)
    if root == "price":
        return _price_reply(message or payload, domain=domain)
    if root == "specs":
        return _specs_reply(message or payload, domain=domain)
    if root == "lead":
        action = parts[1].strip().lower() if len(parts) > 1 else "save_pick"
        if action in {"save_pick", "email", "email_pick"}:
            # If they already typed an email with the button tap, capture immediately.
            email = extract_email(message or "")
            if email:
                return _capture_pick_lead(email, prefs, domain=domain)
            return _ask_email_for_pick(prefs)
    if root == "ask" and len(parts) >= 2:
        topic = parts[1].strip().lower()
        if topic == "shipping":
            return _prepurchase_policy_reply(
                "how long does shipping take", domain=domain
            )
        if topic == "returns":
            return _prepurchase_policy_reply(
                "what is your return policy", domain=domain
            )
    if root == "human":
        confirm = len(parts) > 1 and parts[1].strip().lower() == "confirm"
        if confirm:
            return _handoff_reply(
                SalesIntent(label="human", confidence="high", handoff=True)
            )
        return _complete_then_offer_human(prefs=prefs, domain=domain, reason="human")
    if root == "feedback" and len(parts) >= 2:
        return _feedback_reply(parts[1].strip().lower(), prefs)
    if root == "open" and len(parts) >= 2:
        url = payload.split(":", 1)[1].strip()
        if url.startswith("https://"):
            return _open_product_reply(url, prefs)
    if root == "cta":
        action = parts[1].strip().lower() if len(parts) > 1 else ""
        if action == "showroom":
            sub = parts[2].strip().lower() if len(parts) > 2 else ""
            if sub == "book":
                return _showroom_book_windows_reply()
            if sub == "type":
                return _showroom_type_time_reply()
            if sub == "window":
                code = parts[3].strip().lower() if len(parts) > 3 else ""
                label = showroom_window_label(code)
                if not label:
                    return _showroom_book_windows_reply()
                email = extract_email(message or "")
                patched = dict(prefs or {})
                patched["pending_showroom_window"] = label
                if email:
                    return _capture_showroom_visit(email, patched, domain=domain)
                return _showroom_ask_email_reply(patched, window=label)
            return _showroom_cta_reply(prefs, domain=domain)
        if action == "financing":
            url = payload.split(":", 2)[2].strip() if len(parts) >= 3 else ""
            if not url.startswith("https://"):
                url = ((prefs or {}).get("pending_product_url") or "").strip()
            if url.startswith("https://"):
                return _financing_cta_reply(url, prefs)
            return _showroom_cta_reply(prefs, domain=domain)
    factory = _PAYLOAD_ROUTES.get(root)
    if factory is None:
        return None
    return factory(message or payload)


def _finalize_flow_stage(result: SalesReply) -> SalesReply:
    """Resolve the flow stage, then instrument the turn for analytics."""
    result = _resolve_flow_stage(result)

    # Stamp the stage into tools_used so per-question drop-off can be measured
    # from the message log without a schema change.
    stage = result.flow_stage or "menu"
    tools = result.tools_used or []
    if not any(t.startswith("stage:") for t in tools):
        result.tools_used = tools + [f"stage:{stage}"]

    # Offer a rating on answers substantial enough for it to mean something.
    if result.intent in RATEABLE_INTENTS and not result.handoff:
        existing = {q.payload for q in result.quick_replies}
        for quick in feedback_quick_replies():
            if quick.payload not in existing:
                result.quick_replies.append(quick)
        patch = dict(result.prefs_patch or {})
        patch["last_rateable_intent"] = result.intent
        result.prefs_patch = patch

    human_payloads = {"human", "human:confirm"}
    human = [
        q
        for q in result.quick_replies
        if q.payload.strip().lower() in human_payloads
    ]
    others = [
        q
        for q in result.quick_replies
        if q.payload.strip().lower() not in human_payloads
    ]
    if human:
        result.quick_replies = others + [human[0]]

    return result


def _resolve_flow_stage(result: SalesReply) -> SalesReply:
    """Fill flow_stage for Tidio static Decision branching when callers omit it."""
    if result.flow_stage and result.flow_stage not in {"", "menu"}:
        # Explicit ask_* / recommend / etc. already set.
        if result.flow_stage.startswith("ask_") or result.flow_stage in {
            "recommend",
            "recommend_nofit",
            "lead",
            "handoff",
            "warranty",
            "shop",
            "ask_doorway",
            "ask_doorway_fit",
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
        fit = _doorway_fit_mode(rec_prefs)
        fit_bit = (
            ", assembled"
            if fit == "assembled"
            else ", with disassembly"
            if fit == "disassembled"
            else ""
        )
        why_bits.append(
            f"checked against your **{rec_prefs['doorway_in']}\"** doorway{fit_bit}"
        )

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
    *,
    domain: str = "osakiusa.com",
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
    left_prod = (
        product_by_handle(str(left_pick.get("handle") or ""))
        or resolve_product(left_name)
    )
    right_prod = (
        product_by_handle(str(right_pick.get("handle") or ""))
        or resolve_product(right_name)
    )
    if left_prod is None or right_prod is None:
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

    result = compare_products(left_prod, right_prod)
    left = result["left"]
    right = result["right"]
    lt = (left_pick.get("tier") or "Option A").split("(")[0].strip()
    rt = (right_pick.get("tier") or "Option B").split("(")[0].strip()
    lines = [
        f"**{lt}: {left['model']}** — {_fmt_price(left['price_usd'])}",
        f"**{rt}: {right['model']}** — {_fmt_price(right['price_usd'])}",
        "",
        *_compare_spec_lines(result["diff"]),
        "",
        "Reply **1** for Value, **2** for Mid, or ask me which of these two "
        "fits better.",
    ]
    quick = [
        QuickReply(label=f"Choose {lt}", payload=f"tier:{left_n}"),
        QuickReply(label=f"Choose {rt}", payload=f"tier:{right_n}"),
        QuickReply(label="Recommend which one", payload="compare:recommend"),
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
            "pending_compare_pair": [left_prod.handle, right_prod.handle],
            "pending_compare_left": left_prod.handle,
            "pending_compare_right": right_prod.handle,
            "awaiting_compare_recommend": False,
        },
    )


def _product_from_tier_pick(pick: dict) -> Optional[ProductSpecs]:
    handle = str(pick.get("handle") or "").strip()
    name = str(pick.get("display") or pick.get("model") or "").strip()
    if handle:
        found = product_by_handle(handle)
        if found is not None:
            return found
        found = resolve_product(handle)
        if found is not None:
            return found
    return resolve_product(name) if name else None


def _pending_tier_index(message: str, prefs: Optional[dict]) -> Optional[int]:
    """Value / Mid / Premium aliases after a 3-pick list. Not a new search."""
    picks = (prefs or {}).get("pending_tier_picks") or []
    if not isinstance(picks, list) or not picks:
        return None
    raw = (message or "").strip().lower()
    if not raw:
        return None
    if re.search(
        r"\b(recommend|another|different|start over|new search)\b", raw
    ):
        return None
    rules: tuple[tuple[int, str], ...] = (
        (
            0,
            r"\b(the\s+)?(first|value|budget)\b|"
            r"\boption\s*1\b|\bnumber\s*1\b|#\s*1\b|"
            r"the\s+cheap(?:est)?\s+one",
        ),
        (
            1,
            r"\b(the\s+)?(second|mid(?:[- ]?range)?|middle)\b|"
            r"\boption\s*2\b|\bnumber\s*2\b|#\s*2\b",
        ),
        (
            2,
            r"\b(the\s+)?(third|premium|top)\b|"
            r"\boption\s*3\b|\bnumber\s*3\b|#\s*3\b|"
            r"the\s+expensive\s+one",
        ),
    )
    for idx, pattern in rules:
        if idx < len(picks) and re.search(pattern, raw, re.I):
            return idx
    return None


def _pending_tier_follow_reply(
    message: str,
    intent_label: str,
    prefs: Optional[dict],
    *,
    domain: str,
) -> Optional[SalesReply]:
    idx = _pending_tier_index(message, prefs)
    if idx is None:
        return None
    picks = (prefs or {}).get("pending_tier_picks") or []
    pick = picks[idx]
    if not isinstance(pick, dict):
        return None
    product = _product_from_tier_pick(pick)
    named = product.display_name if product is not None else ""
    if intent_label == INTENT_PRICE and named:
        return _price_reply(named, domain=domain)
    if intent_label == INTENT_STOCK and named:
        return _stock_reply(named, domain=domain)
    if intent_label == INTENT_SPECS and named:
        return _specs_reply(named, domain=domain)
    return _tier_followup_reply(idx, prefs, domain=domain)


def _tier_digit_reply(message: str, prefs: Optional[dict]) -> Optional[SalesReply]:
    """Map bare 1/2/3 to the matching Value/Mid/Premium follow-up card."""
    digit = re.fullmatch(r"([1-3])[).:\s]*", (message or "").strip())
    if not digit:
        return None
    picks = (prefs or {}).get("pending_tier_picks") or []
    if not isinstance(picks, list) or not picks:
        return None
    return _tier_followup_reply(int(digit.group(1)) - 1, prefs)


def _should_remember_question(message: str, payload: Optional[str]) -> bool:
    if payload:
        return False
    text = (message or "").strip()
    if len(text) < 4 or extract_email(text):
        return False
    if re.fullmatch(r"[1-3][).:\s]*", text):
        return False
    label = classify(text).label
    if label in {INTENT_HUMAN, INTENT_GREETING}:
        return False
    return True


def _remember_question(result: SalesReply, message: str, payload: Optional[str]) -> SalesReply:
    if not _should_remember_question(message, payload):
        return result
    patch = dict(result.prefs_patch or {})
    patch["last_shopper_question"] = message.strip()[:500]
    result.prefs_patch = patch
    return result


def respond(
    message: str,
    *,
    payload: Optional[str] = None,
    domain: str = "osakiusa.com",
    prefs: Optional[dict] = None,
    before_handoff: bool = False,
) -> SalesReply:
    """Return a SalesReply for one customer message (+ optional button payload).

    ``payload`` is the ``QuickReply.payload`` value emitted by a previous
    turn. When set, it overrides intent classification so button taps behave
    predictably — critical for hitting 100% satisfaction on menu paths.

    ``prefs`` is ``sales_sessions.collected_data`` (recommend answers accumulate
    across turns via ``prefs_patch`` on the reply).

    ``before_handoff`` replays a stored question so we can answer it fully
    without treating "talk to a human" as the question.
    """
    # Awaiting email after "Email me this pick" — capture before other intents.
    if (prefs or {}).get("awaiting_email_for_pick"):
        email = extract_email(message or "")
        if email:
            return _finalize_flow_stage(_capture_pick_lead(email, prefs, domain=domain))

    if (prefs or {}).get("awaiting_showroom_email"):
        email = extract_email(message or "")
        if email:
            return _finalize_flow_stage(
                _capture_showroom_visit(email, prefs, domain=domain)
            )

    if (
        (prefs or {}).get("awaiting_showroom_window")
        and not (payload or "").strip()
        and (message or "").strip()
    ):
        email = extract_email(message or "")
        window = (message or "").strip()[:200]
        patched = dict(prefs or {})
        patched["pending_showroom_window"] = window
        if email:
            return _finalize_flow_stage(
                _capture_showroom_visit(email, patched, domain=domain)
            )
        return _finalize_flow_stage(
            _showroom_ask_email_reply(patched, window=window)
        )

    # They asked for a person, then typed the real question.
    if (
        not before_handoff
        and (prefs or {}).get("awaiting_pre_human_question")
        and (message or "").strip()
        and (payload or "") not in {"human", "human:confirm"}
    ):
        intent = classify(message or "")
        if intent.label == INTENT_HUMAN:
            return _finalize_flow_stage(_handoff_reply(intent))
        if intent.label in HANDOFF_INTENTS and intent.label != INTENT_DISCOUNT:
            return _finalize_flow_stage(_handoff_reply(intent))
        if intent.label == INTENT_DISCOUNT:
            return _finalize_flow_stage(
                _attach_human_footer(
                    _discount_facts_reply(message, prefs, domain=domain)
                )
            )
        inner = respond(
            message,
            domain=domain,
            prefs=_prefs_for_inner(prefs),
            before_handoff=True,
        )
        return _finalize_flow_stage(_attach_human_footer(inner))

    if not payload:
        compare_digit = _compare_digit_reply(message, prefs, domain=domain)
        if compare_digit is not None:
            return _remember_question(
                _finalize_flow_stage(compare_digit), message, payload
            )

    if (
        not payload
        and (prefs or {}).get("awaiting_compare_pick")
        and (message or "").strip()
    ):
        if split_compare_terms(message or ""):
            return _remember_question(
                _finalize_flow_stage(
                    _compare_reply(message, prefs=prefs, domain=domain)
                ),
                message,
                payload,
            )
        looked = lookup_shop_models(message or "")
        if looked.unique is not None:
            return _remember_question(
                _finalize_flow_stage(
                    _apply_compare_pick(
                        looked.unique.handle, prefs, domain=domain
                    )
                ),
                message,
                payload,
            )
        if len(looked.matches) > 1:
            slot = str((prefs or {}).get("awaiting_compare_pick") or "left")
            return _remember_question(
                _finalize_flow_stage(
                    _ask_compare_which(
                        slot,
                        looked.query or (message or "").strip(),
                        list(looked.matches),
                        prefs=prefs,
                        left_q=str((prefs or {}).get("pending_compare_left_query") or ""),
                        right_q=str((prefs or {}).get("pending_compare_right_query") or ""),
                        left_handle=str((prefs or {}).get("pending_compare_left") or ""),
                        right_handle=str((prefs or {}).get("pending_compare_right") or ""),
                    )
                ),
                message,
                payload,
            )

    # After a tier list, bare "1"/"2"/"3" opens that chair (chat + Tidio).
    if not payload:
        tier_reply = _tier_digit_reply(message, prefs)
        if tier_reply is not None:
            return _finalize_flow_stage(tier_reply)

    if payload:
        forced = _payload_reply(payload, message, domain=domain, prefs=prefs)
        if forced is not None:
            return _remember_question(
                _finalize_flow_stage(forced), message, payload
            )

    intent = classify(message or "")

    # Regexes miss unusual phrasing, which showed up in production as a 27%
    # `unclear` rate. Re-route those turns before falling back to the menu.
    if intent.label == INTENT_UNCLEAR:
        recovered = resolve_unclear(message or "")
        if recovered is not None:
            intent = recovered
    elif intent.label == INTENT_RECOMMEND:
        revised = revise_recommend(intent, message or "")
        if revised is not None:
            intent = revised

    if not before_handoff and intent.label == INTENT_HUMAN:
        return _finalize_flow_stage(
            _complete_then_offer_human(
                prefs=prefs, domain=domain, reason="human", message=message
            )
        )

    if not before_handoff and intent.label == INTENT_DISCOUNT:
        return _remember_question(
            _finalize_flow_stage(
                _complete_then_offer_human(
                    prefs=prefs, domain=domain, reason="discount", message=message
                )
            ),
            message,
            payload,
        )

    if intent.label in HANDOFF_INTENTS:
        return _finalize_flow_stage(_handoff_reply(intent))

    waiting = _in_recommend_wait(prefs)

    if intent.label == INTENT_PREPURCHASE_POLICY:
        result = _prepurchase_policy_reply(message, domain=domain)
        if waiting:
            result = _resume_recommend_after_side(result, prefs, domain=domain)
        return _remember_question(
            _finalize_flow_stage(result), message, payload
        )

    if intent.label == INTENT_GREETING:
        return _finalize_flow_stage(_greeting_reply(prefs))

    if _EMAIL_PICK_RE.search(message or ""):
        if (prefs or {}).get("pending_primary"):
            result = _ask_email_for_pick(prefs)
        else:
            named_pick = _guess_model_from_text(message or "")
            if named_pick is not None:
                _extra, _quick, patch = _product_closeout(named_pick, domain=domain)
                merged = dict(prefs or {})
                merged.update(patch)
                result = _ask_email_for_pick(merged)
            else:
                result = SalesReply(
                    reply=(
                        "I can email a chair to you once we know which one. "
                        "Name a model, or tap **Recommend a chair** first, then "
                        "**Email me this pick**."
                    ),
                    intent=INTENT_RECOMMEND,
                    quick_replies=[
                        QuickReply(label="Recommend a chair", payload="recommend"),
                        QuickReply(label="Talk to a human", payload="human"),
                    ],
                    tools_used=["cta.email_pick"],
                )
        return _remember_question(
            _finalize_flow_stage(result), message, payload
        )

    if intent.label == INTENT_ORDER_STATUS:
        return _finalize_flow_stage(_order_status_reply(message))

    pending_tier = _pending_tier_follow_reply(
        message or "", intent.label, prefs, domain=domain
    )
    if pending_tier is not None:
        return _remember_question(
            _finalize_flow_stage(pending_tier), message, payload
        )

    color_ask = looks_like_color_question(message or "")
    door_in = _parse_doorway_inches_message(message or "")
    named = _guess_model_from_text(message or "")
    hints = parse_recommendation_hints(message or "")

    if color_ask and not (intent.label == INTENT_RECOMMEND and hints.height_in):
        if named is not None or waiting or intent.label in {
            INTENT_UNCLEAR,
            INTENT_STOCK,
            INTENT_SPECS,
        }:
            result = _color_reply(message, domain=domain, prefs=prefs)
            if waiting:
                result = _resume_recommend_after_side(result, prefs, domain=domain)
            return _remember_question(
                _finalize_flow_stage(result), message, payload
            )

    if door_in is not None and named is not None:
        result = _doorway_fit_reply(message, prefs=prefs, domain=domain)
        if waiting:
            result = _resume_recommend_after_side(result, prefs, domain=domain)
        return _remember_question(
            _finalize_flow_stage(result), message, payload
        )

    if waiting and door_in is not None:
        return _remember_question(
            _finalize_flow_stage(
                _recommend_reply(message, domain=domain, prefs=prefs)
            ),
            message,
            payload,
        )

    if (
        door_in is not None
        and named is None
        and re.search(r"\bdoor(?:way)?s?\b", message or "", re.I)
        and intent.label in {INTENT_UNCLEAR, INTENT_RECOMMEND, INTENT_SPECS}
    ):
        return _remember_question(
            _finalize_flow_stage(
                _doorway_fit_reply(message, prefs=prefs, domain=domain)
            ),
            message,
            payload,
        )

    if waiting and intent.label == INTENT_UNCLEAR:
        return _remember_question(
            _finalize_flow_stage(
                _recommend_reply(message or "recommend", domain=domain, prefs=prefs)
            ),
            message,
            payload,
        )

    if waiting and intent.label in {INTENT_PRICE, INTENT_STOCK, INTENT_SPECS}:
        if intent.label == INTENT_PRICE:
            inner = _price_reply(message, domain=domain)
        elif intent.label == INTENT_STOCK:
            inner = _stock_reply(message, domain=domain)
        else:
            inner = _specs_reply(message, domain=domain)
        return _remember_question(
            _finalize_flow_stage(
                _resume_recommend_after_side(inner, prefs, domain=domain)
            ),
            message,
            payload,
        )

    if intent.label == INTENT_PRICE:
        return _remember_question(
            _finalize_flow_stage(_price_reply(message, domain=domain)),
            message,
            payload,
        )
    if intent.label == INTENT_STOCK:
        return _remember_question(
            _finalize_flow_stage(_stock_reply(message, domain=domain)),
            message,
            payload,
        )
    if looks_like_compare_pair(message or ""):
        return _remember_question(
            _finalize_flow_stage(
                _compare_reply(message, prefs=prefs, domain=domain)
            ),
            message,
            payload,
        )
    if intent.label == INTENT_SPECS:
        return _remember_question(
            _finalize_flow_stage(_specs_reply(message, domain=domain)),
            message,
            payload,
        )

    if intent.label == INTENT_RECOMMEND:
        pair_handles = (prefs or {}).get("pending_compare_pair") or []
        if (prefs or {}).get("awaiting_compare_recommend") or (
            isinstance(pair_handles, list)
            and len(pair_handles) >= 2
            and is_which_of_pair(message or "")
        ):
            return _remember_question(
                _finalize_flow_stage(
                    _compare_recommend_reply(
                        message, prefs=prefs, domain=domain
                    )
                ),
                message,
                payload,
            )
        return _remember_question(
            _finalize_flow_stage(
                _recommend_reply(message, domain=domain, prefs=prefs)
            ),
            message,
            payload,
        )
    if intent.label == INTENT_COMPARE:
        return _remember_question(
            _finalize_flow_stage(
                _compare_reply(message, prefs=prefs, domain=domain)
            ),
            message,
            payload,
        )
    if intent.label == INTENT_INTENSITY:
        return _remember_question(
            _finalize_flow_stage(_intensity_reply(message)),
            message,
            payload,
        )

    return _remember_question(
        _finalize_flow_stage(_unclear_reply()), message, payload
    )
