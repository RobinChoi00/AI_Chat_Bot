"""
sales_agent.py
==============
Sales AI orchestrator — takes a raw customer message, runs the intent
classifier, applies guardrails, calls deterministic catalog tools, and
returns a structured response (text + quick-reply buttons + optional
handoff signal).

No LLM calls happen here. Everything the customer sees is either:
  1. Hard-coded copy tied to a specific intent, or
  2. Deterministic facts pulled from ``sales_catalog`` (price, specs, etc.).

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
    compare,
    list_active_products,
    parse_recommendation_hints,
    recommend,
    resolve_product,
)
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

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "intent": self.intent,
            "handoff": self.handoff,
            "handoff_reason": self.handoff_reason,
            "quick_replies": [{"label": q.label, "payload": q.payload} for q in self.quick_replies],
            "tools_used": self.tools_used,
            "products": self.products,
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
    "For **shipping, delivery, order tracking, or anything about a chair "
    "you already own**, please use the **Warranty chat** icon on the site.\n\n"
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


def _price_reply(message: str) -> SalesReply:
    product = _guess_model_from_text(message)
    if product is None:
        return SalesReply(
            reply=(
                "Sure — which model are you asking about? You can type the "
                "model name (e.g. *Osaki OS-Pro Maestro LE*) or tap "
                "**Recommend a chair** and I'll narrow it down."
            ),
            intent=INTENT_PRICE,
            quick_replies=[
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.resolve_product"],
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


def _stock_reply(message: str) -> SalesReply:
    product = _guess_model_from_text(message)
    if product is None:
        return SalesReply(
            reply=(
                "Happy to check — which model? Type the name or tap "
                "**See all models** and I'll list what's live in the catalog."
            ),
            intent=INTENT_STOCK,
            quick_replies=[
                QuickReply(label="See all models", payload="list"),
                QuickReply(label="Recommend a chair", payload="recommend"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.resolve_product"],
        )

    if product.status.lower() == "active":
        reply = (
            f"**{product.display_name}** is **active in our catalog**. "
            "For a live inventory count at the exact configuration you want, "
            "checkout will show the final availability — or I can hand you off "
            "to a rep for confirmation."
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
        tools_used=["catalog.resolve_product"],
        products=[product.as_public_dict()],
    )


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


_BUDGET_BAND_REPLIES = (
    QuickReply(label="Under $2,000", payload="recommend:budget:2000"),
    QuickReply(label="Around $3,000", payload="recommend:budget:3000"),
    QuickReply(label="Around $5,000", payload="recommend:budget:5000"),
    QuickReply(label="Around $6,000", payload="recommend:budget:6000"),
    QuickReply(label="Around $8,000+", payload="recommend:budget:8000"),
)


def _recommend_reply(message: str) -> SalesReply:
    request = parse_recommendation_hints(message)
    has_hints = any(
        [
            request.height_in,
            request.weight_lb,
            request.budget_usd,
            request.focus_areas,
        ]
    )
    picks = recommend(request, limit=3) if has_hints else []
    if not picks or not has_hints:
        # No signal — lead with price bands (most shoppers start there),
        # then offer body-focus shortcuts.
        return SalesReply(
            reply=(
                "Happy to recommend a chair. What's your **budget range**?\n\n"
                "Tap a price band below, or type something like *around 6000* / "
                "*under 3000*.\n\n"
                "You can also share **height** (e.g. *6'2\"*) or a focus area "
                "(back / neck / feet)."
            ),
            intent=INTENT_RECOMMEND,
            quick_replies=[
                *_BUDGET_BAND_REPLIES,
                QuickReply(label="Focus: back", payload="recommend:back"),
                QuickReply(label="Talk to a human", payload="human"),
            ],
            tools_used=["catalog.parse_hints"],
        )

    header_bits: list[str] = []
    if request.height_in:
        header_bits.append(f"height ~{request.height_in}\"")
    if request.weight_lb:
        header_bits.append(f"weight ~{request.weight_lb} lb")
    if request.focus_areas:
        header_bits.append("focus: " + ", ".join(request.focus_areas))
    if request.budget_usd:
        header_bits.append(f"budget around ${request.budget_usd:,}")

    header = (
        f"Based on {'; '.join(header_bits)}, here are my top picks:"
        if header_bits
        else "Here are strong matches from the current catalog:"
    )

    lines = [header]
    for product in picks:
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
        lines.append(
            f"- **{product.display_name}** — {_fmt_price(product.price_usd)}"
            + (f"  ({detail})" if detail else "")
        )
    lines.append(
        "\nWant specs on one of these, or should I hand you to a rep for a "
        "personal walkthrough?"
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
        tools_used=["catalog.recommend"],
        products=[p.as_public_dict() for p in picks],
    )


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
    ]
    if diff["price_delta_usd"] is not None:
        delta = diff["price_delta_usd"]
        direction = "more" if delta > 0 else "less"
        lines.append(
            f"- **Price gap**: the second is about ${abs(delta):,.0f} {direction}."
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
    "recommend": lambda msg: _recommend_reply(msg),
    "compare": lambda msg: _compare_reply(msg),
    "price": lambda msg: _price_reply(msg),
    "stock": lambda msg: _stock_reply(msg),
    "specs": lambda msg: _specs_reply(msg),
    "intensity": lambda msg: _intensity_reply(msg),
    "human": lambda _msg: _handoff_reply(
        SalesIntent(label="human", confidence="high", handoff=True)
    ),
}


def _recommend_message_from_payload(payload: str, message: str) -> str:
    """Map recommend:* button payloads to text the hint parser understands."""
    parts = [p.strip() for p in (payload or "").split(":") if p.strip()]
    if len(parts) >= 3 and parts[1].lower() == "budget":
        amount = parts[2].rstrip("+")
        if amount.isdigit():
            return f"budget around ${amount}"
    if len(parts) >= 2 and parts[1].lower() in {"back", "neck", "feet"}:
        return parts[1].lower()
    return (message or payload or "recommend").strip()


def _payload_reply(payload: str, message: str) -> Optional[SalesReply]:
    parts = (payload or "").split(":")
    root = parts[0].strip().lower() if parts else ""
    factory = _PAYLOAD_ROUTES.get(root)
    if factory is None:
        return None
    if root == "recommend":
        return _recommend_reply(_recommend_message_from_payload(payload, message))
    return factory(message or payload)


def respond(message: str, *, payload: Optional[str] = None) -> SalesReply:
    """Return a SalesReply for one customer message (+ optional button payload).

    ``payload`` is the ``QuickReply.payload`` value emitted by a previous
    turn. When set, it overrides intent classification so button taps behave
    predictably — critical for hitting 100% satisfaction on menu paths.
    """
    if payload:
        forced = _payload_reply(payload, message)
        if forced is not None:
            return forced

    intent = classify(message or "")

    if intent.label in HANDOFF_INTENTS:
        return _handoff_reply(intent)

    if intent.label == INTENT_GREETING:
        return _greeting_reply()

    if intent.label == INTENT_ORDER_STATUS:
        return _order_status_reply(message)

    if intent.label == INTENT_PRICE:
        return _price_reply(message)
    if intent.label == INTENT_STOCK:
        return _stock_reply(message)
    if intent.label == INTENT_SPECS:
        return _specs_reply(message)
    if intent.label == INTENT_RECOMMEND:
        return _recommend_reply(message)
    if intent.label == INTENT_COMPARE:
        return _compare_reply(message)
    if intent.label == INTENT_INTENSITY:
        return _intensity_reply(message)

    return _unclear_reply()
