"""
Dense shopper-phrase simulation.

Quality does not come from a million random strings. It comes from covering
the phrasings that actually dumped people into the unclear menu, then
checking the same invariants on every paraphrase.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

os.environ["SALES_INTENT_LLM"] = "0"

from sales_agent import SalesReply, respond  # noqa: E402
from sales_catalog import parse_recommendation_hints  # noqa: E402


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sales_agent.fetch_live_stock", lambda *a, **k: None)


def _merge(prefs: dict, reply: SalesReply) -> dict:
    out = dict(prefs)
    if reply.prefs_patch:
        for key, value in reply.prefs_patch.items():
            if key == "recommend_prefs" and isinstance(value, dict):
                rec = dict(out.get("recommend_prefs") or {})
                rec.update(value)
                out["recommend_prefs"] = rec
            else:
                out[key] = value
    return out


def _turn(message: str, prefs: dict | None = None) -> SalesReply:
    return respond(message, domain="osakiusa.com", prefs=prefs or {})


_UNCLEAR_MENU = "tap one of these"


_COLOR_PHRASES = [
    "Grande XL in black",
    "grande xl in brown",
    "Tital xl 3d in black",
    "tital xl 3d black",
    "do you have Grande XL in black",
    "is the Grande XL available in black",
    "Grande XL in Dark Brown",
    "can I get the Grande XL in beige",
]


_TALL_PHRASES = [
    "tall guy with back pain",
    "I'm a tall guy, back pain",
    "recommend a chair for a tall person",
    "good chair for a tall guy",
    "best chair for someone tall with back pain",
    "large guy, lower back",
]


_SUNDAY_PHRASES = [
    "are you open Sunday",
    "open on sunday?",
    "sunday hours",
    "do you open Sundays",
]


_PICKUP_PHRASES = [
    "can I pick up today",
    "pickup today",
    "can I pick up a chair today",
]


_DOOR_PHRASES = [
    "30 inch door",
    "will it fit through a 30 inch door",
    "32 inch doorway",
    "my door is 30 inches",
]


_NO_PHRASES = [
    ("do you sell refurbished chairs", ("new", "refurb")),
    ("any open box models", ("new", "open")),
    ("do you take trade-ins", ("trade",)),
    ("can I lease a chair", ("lease", "affirm")),
]


def test_color_and_typo_never_dump_the_menu():
    for message in _COLOR_PHRASES:
        reply = _turn(message)
        low = reply.reply.lower()
        assert reply.intent != "unclear", message
        assert _UNCLEAR_MENU not in low, message
        assert "checkout" in low, message
        assert "6'3" not in reply.reply


def test_tall_without_inches_asks_height():
    for message in _TALL_PHRASES:
        hints = parse_recommendation_hints(message)
        assert hints.height_in is None, message
        reply = _turn(message)
        low = reply.reply.lower()
        assert reply.intent == "recommend", message
        assert "height" in low, message
        assert "extra tall" not in low, message
        assert "6'3" not in reply.reply, message


def test_sunday_quotes_hours_and_does_not_invent_open():
    for message in _SUNDAY_PHRASES:
        reply = _turn(message)
        assert reply.intent == "prepurchase_policy", message
        assert "Sunday isn't listed" in reply.reply, message
        assert "9:30" in reply.reply or "Hours" in reply.reply, message


def test_pickup_does_not_confirm_same_day():
    for message in _PICKUP_PHRASES:
        reply = _turn(message)
        assert reply.intent == "prepurchase_policy", message
        assert "same-day pickup" in reply.reply.lower(), message


def test_doorway_without_model_stays_on_fit():
    for message in _DOOR_PHRASES:
        reply = _turn(message)
        low = reply.reply.lower()
        assert reply.intent != "unclear", message
        assert _UNCLEAR_MENU not in low, message
        assert "door" in low or "height" in low or "model" in low, message


def test_honest_nos_are_not_unclear():
    for message, needles in _NO_PHRASES:
        reply = _turn(message)
        low = reply.reply.lower()
        assert reply.intent == "prepurchase_policy", message
        assert reply.handoff is False, message
        assert any(n in low for n in needles), (message, low)


def test_compare_missing_and_unpublished_specs():
    family = _turn("Maestro vs Jupiter")
    low = family.reply.lower()
    assert "not on the current store catalog" in low
    assert "jupiter" in low
    assert "dont know" not in low

    pair = _turn("Maestro 4D vs Paragon")
    assert "dont know" not in pair.reply.lower()
    assert "don't know" not in pair.reply.lower()


def test_bare_color_does_not_guess_a_random_chair():
    reply = _turn("do you have black")
    low = reply.reply.lower()
    assert reply.intent != "unclear"
    assert "checkout" in low
    assert "haven" not in low
    assert "which **model**" in reply.reply.lower() or "which model" in low
    first = _turn("recommend a chair")
    prefs = _merge({}, first)
    assert "height" in first.reply.lower()
    second = _turn("do you have black", prefs)
    low = second.reply.lower()
    assert second.intent != "unclear"
    assert "checkout" in low
    assert "height" in low


def test_mid_flow_doorway_keeps_recommend():
    first = _turn("recommend a chair")
    prefs = _merge({}, first)
    second = _turn("30 inch door", prefs)
    assert second.intent == "recommend"
    assert second.intent != "unclear"
    assert "height" in second.reply.lower() or "door" in second.reply.lower()


def test_price_browse_skips_quest_3d():
    reply = _turn("how much is this chair")
    low = reply.reply.lower()
    assert "quest 3d" not in low
    assert "costco" not in low


def test_full_shopper_atlas_never_dumps_the_menu():
    """Every first-turn shopper family we know about gets a real route."""
    atlas = [
        # greet
        "hi",
        "hello",
        "hey there",
        "good morning",
        "thanks",
        "thank you",
        "bye",
        # price
        "how much is the Maestro 4D",
        "price of Paragon",
        "how much is this chair",
        "what's the cheapest chair",
        "most expensive chair",
        "Grande XL cost",
        "how much does the Grande XL weigh",
        # discount
        "any discounts",
        "discounts?",
        "can you do better on price",
        "price match Costco",
        "it's $1000 more than Costco",
        "military discount",
        "veteran discount",
        "coupon code",
        # financing / lease
        "do you offer financing",
        "can I use Affirm",
        "what's the APR",
        "monthly payments",
        "lease to own",
        "can I lease a chair",
        "do you take PayPal",
        # stock / color
        "is Maestro 4D in stock",
        "do you have the Paragon",
        "Grande XL in black",
        "Tital xl 3d in black",
        "do you have black",
        "backorder",
        "is the Maestro discontinued",
        # specs
        "specs for Maestro 4D",
        "does Paragon have heat",
        "does it have zero gravity",
        "foot rollers on Paragon",
        "weight capacity Grande XL",
        "what's 4D vs 3D",
        "bluetooth?",
        "SL-Track or L-Track",
        "does it recline",
        "how wide is the chair",
        "max user weight",
        "what's the difference between S-Track and SL-Track",
        # recommend / space
        "recommend a chair",
        "help me pick one",
        "tall guy with back pain",
        "I'm 6'2 230 lb lower back",
        "petite wife with neck pain",
        "chair for 5'2 and 110 lb",
        "best chair under $5k",
        "which chair for sciatica",
        "small apartment",
        "narrow hallway",
        "30 inch door",
        "will Grande XL fit a 30 inch door",
        "my door is 28 inches",
        "for my office",
        "two chairs for a couple",
        "his and hers",
        "I'm 400 pounds",
        "best seller",
        "first time buyer",
        # compare
        "Maestro 4D vs Paragon",
        "Maestro vs Jupiter",
        "compare Maestro and Champ",
        "which is better Maestro LE or Maestro 4D",
        "Maestro 4D and Paragon which one",
        # policy / showroom / ship
        "what's your return policy",
        "how long is the warranty",
        "how much is shipping",
        "do you ship to Hawaii",
        "ship to Alaska",
        "do you ship to Guam",
        "white glove",
        "do you assemble it",
        "are you open Sunday",
        "can I pick up today",
        "where is the showroom",
        "what are your hours",
        "I live on the third floor",
        "ship to Canada",
        "international shipping to Canada",
        "ship to Mexico",
        "sales tax",
        "how long do they last",
        "made in Japan",
        "can I use HSA",
        "PO Box delivery",
        # honest no
        "do you sell refurbished chairs",
        "open box",
        "do you take trade-ins",
        "is it pet friendly",
        "my cat will scratch it",
        "can you hide the price for a gift",
        "gift wrap",
        "do you speak Spanish",
        "puedo comprar una silla",
        "commercial use for a salon",
        "outdoor patio chair",
        "is it a medical device",
        "safe during pregnancy",
        "for kids",
        "rental program",
        "wholesale",
        "how loud is it",
        "what outlet / voltage",
        "does it need a special outlet",
        "pacemaker",
        # mid / handoff
        "email me this pick",
        "my chair is broken",
        "I want to cancel my order",
        "where is my order",
        "send a technician",
        "talk to a human",
        "when will it arrive",
    ]
    banned = ("dont know how many", "quest 3d", "extra tall (6'3")
    competitor = ("costco", "amazon", "walmart")
    failures: list[str] = []
    for message in atlas:
        reply = _turn(message)
        low = reply.reply.lower()
        if reply.intent == "unclear" or _UNCLEAR_MENU in low:
            failures.append(f"unclear:{message} [{reply.intent}]")
        for phrase in banned:
            if phrase in low:
                failures.append(f"{phrase}:{message}")
        if "apr" in message.lower() and ("%" in reply.reply or " apr" in low):
            failures.append(f"apr-leak:{message}")
        if re.search(r"\b(mexico|canada)\b", message, re.I):
            if reply.intent == "eta_shipping":
                failures.append(f"intl-warranty:{message}")
            if "curbside" in low and "canada" not in low and "mexico" not in low:
                failures.append(f"intl-us-copy:{message}")
        if any(brand in message.lower() for brand in competitor):
            if any(brand in low for brand in competitor):
                failures.append(f"competitor-named:{message}")
    assert not failures, failures
    """Every first-turn shopper family we know about gets a real route."""
    atlas = [
        "hi",
        "hey there",
        "how much is the Maestro 4D",
        "how much is this chair",
        "what's the cheapest chair",
        "most expensive chair",
        "any discounts",
        "what's the APR",
        "can I lease a chair",
        "lease to own",
        "Grande XL in black",
        "do you have black",
        "tall guy with back pain",
        "small apartment",
        "narrow hallway",
        "30 inch door",
        "Maestro vs Jupiter",
        "are you open Sunday",
        "can I pick up today",
        "ship to Canada",
        "ship to Mexico",
        "do you sell refurbished chairs",
        "is it pet friendly",
        "my cat will scratch it",
        "can you hide the price for a gift",
        "gift wrap",
        "do you speak Spanish",
        "puedo comprar una silla",
        "commercial use for a salon",
        "for my office",
        "outdoor patio chair",
        "is it a medical device",
        "safe during pregnancy",
        "for kids",
        "rental program",
        "wholesale",
        "how loud is it",
        "does it need a special outlet",
        "email me this pick",
        "does it recline",
        "how wide is the chair",
        "max user weight",
        "two chairs for a couple",
        "his and hers",
        "I'm 400 pounds",
        "my chair is broken",
        "talk to a human",
    ]
    banned = ("dont know how many", "quest 3d", "extra tall (6'3")
    failures: list[str] = []
    for message in atlas:
        reply = _turn(message)
        low = reply.reply.lower()
        if reply.intent == "unclear" or _UNCLEAR_MENU in low:
            failures.append(f"unclear:{message}")
        for phrase in banned:
            if phrase in low:
                failures.append(f"{phrase}:{message}")
        if "apr" in message.lower() and ("%" in reply.reply or " apr" in low):
            failures.append(f"apr-leak:{message}")
        if message == "ship to Mexico" and reply.intent == "eta_shipping":
            failures.append(f"mexico-warranty:{message}")
        if message == "ship to Canada" and "curbside" in low and "canada" not in low:
            failures.append(f"canada-us-copy:{message}")
    assert not failures, failures


def test_generated_shopper_matrix_stays_on_an_answer():
    """Cartesian-ish coverage of the failure modes — hundreds of turns, not millions."""
    colors = ("black", "brown", "beige")
    models = ("Grande XL", "Tital xl 3d", "Paragon")
    wrappers = (
        "{model} in {color}",
        "do you have the {model} in {color}",
        "is {model} available in {color}",
    )
    failures: list[str] = []
    count = 0
    for model in models:
        for color in colors:
            for wrap in wrappers:
                message = wrap.format(model=model, color=color)
                count += 1
                reply = _turn(message)
                if reply.intent == "unclear" or _UNCLEAR_MENU in reply.reply.lower():
                    failures.append(message)
                if "dont know" in reply.reply.lower():
                    failures.append(f"junk:{message}")
    for tall in _TALL_PHRASES:
        for extra in ("", " please", " — which one?"):
            count += 1
            message = tall + extra
            reply = _turn(message)
            if "extra tall" in reply.reply.lower() or "6'3" in reply.reply:
                failures.append(message)
    assert count >= 40
    assert not failures, failures


def test_oneshot_height_and_goal_shows_three_chairs():
    reply = _turn("I'm 5'10 with lower back pain")
    low = reply.reply.lower()
    assert reply.intent == "recommend"
    assert _UNCLEAR_MENU not in low
    assert "value" in low and "premium" in low
    assert "extra tall" not in low
    assert "assumed" in low


def test_mid_flow_typed_height_then_goal():
    first = _turn("recommend a chair")
    prefs = _merge({}, first)
    assert "height" in first.reply.lower()
    second = _turn("5'10", prefs)
    prefs = _merge(prefs, second)
    assert "focus" in second.reply.lower() or "neck" in second.reply.lower()
    third = _turn("lower back", prefs)
    low = third.reply.lower()
    assert third.intent == "recommend"
    assert "value" in low
    assert _UNCLEAR_MENU not in low


def test_mid_flow_not_sure_height_assumes_average():
    first = _turn("recommend a chair")
    prefs = _merge({}, first)
    second = _turn("not sure", prefs)
    prefs = _merge(prefs, second)
    assert second.intent != "unclear"
    assert "tap a range" not in second.reply.lower()
    assert "focus" in second.reply.lower() or "neck" in second.reply.lower()
    third = _turn("neck", prefs)
    low = third.reply.lower()
    assert "value" in low
    assert "average" in low or "5'4" in low or "assumed" in low


def test_opener_tall_guy_still_asks_height():
    reply = _turn("tall guy with back pain")
    assert "height" in reply.reply.lower()
    assert "value (under" not in reply.reply.lower()
    assert "extra tall" not in reply.reply.lower()


def test_after_picks_mid_one_and_price_alias():
    first = _turn("I'm 5'10 with lower back pain")
    prefs = _merge({}, first)
    assert prefs.get("pending_tier_picks")
    mid = _turn("the mid one", prefs)
    assert mid.intent != "unclear"
    assert _UNCLEAR_MENU not in mid.reply.lower()
    price = _turn("how much is the first one", prefs)
    assert price.intent == "price"
    assert "$" in price.reply or "price" in price.reply.lower()


def test_goal_and_doorway_buttons_fit_tidio_cap():
    from sales_tidio_buttons import prioritize_quick_replies

    first = _turn("recommend a chair")
    height_caps = prioritize_quick_replies(first.quick_replies, limit=5)
    assert len(height_caps) <= 5
    assert height_caps[-1]["payload"] == "human"
    prefs = _merge({}, first)
    goal_turn = _turn("5'10", prefs)
    goal_caps = prioritize_quick_replies(goal_turn.quick_replies, limit=5)
    assert len(goal_caps) <= 5
    payloads = [b["payload"] for b in goal_caps]
    assert "recommend:goal:lower_back" in payloads
    assert payloads[-1] == "human"


def test_production_unclear_phrases_never_dump_the_menu():
    """Phrases mined from live Tidio logs after the Sep 2026 quality ship."""
    cases = [
        (
            "hello, i am trying to make a purchase but my amex is not going through",
            "prepurchase_policy",
            ("checkout", "email"),
            ("amex", "american express", "visa", "apr"),
        ),
        ("👍👍👍😱", "greeting", ("height",), ()),
        ("{{visitor_question}}", "greeting", ("height",), ()),
        (
            "can tou look up the one my friend has",
            "prepurchase_policy",
            ("catalog", "model"),
            (),
        ),
        (
            "do you have exchange programs to replace previous models with newer models?",
            "prepurchase_policy",
            ("trade",),
            (),
        ),
        (
            "do you have an affiliate program",
            "prepurchase_policy",
            ("affiliate", "email"),
            (),
        ),
        (
            "please answer my question above.",
            "human_offer",
            ("tell me",),
            (),
        ),
        (
            "i just have feedback for you",
            "human",
            ("email",),
            (),
        ),
        (
            "96701 is my zip",
            "prepurchase_policy",
            ("freight", "you pay"),
            ("$",),
        ),
        (
            "tablet instalations",
            "prepurchase_policy",
            ("white glove", "assembly"),
            (),
        ),
        ("is this a brand new chair?", "prepurchase_policy", ("new",), ("open-box deal",)),
        ("which product is right for me?", "recommend", ("height",), ()),
        ("need manual", "warranty_redirect", ("service@osakititan.com",), ()),
        (
            "how to connect osaki app to the chair",
            "warranty_redirect",
            ("service@osakititan.com",),
            (),
        ),
        ("can i switch my order?", "cancel_refund", ("email",), ()),
        (
            "government 889 form",
            "prepurchase_policy",
            ("email",),
            ("compliant", "certified"),
        ),
        (
            "i want to create an invoice for massage chair",
            "human",
            ("email",),
            (),
        ),
        (
            "hi i was wondering how easy it is to move the chair once built. for example if i needed to periodically move it closer or further from a wall",
            "prepurchase_policy",
            ("wall clearance",),
            ("easy to roll",),
        ),
    ]
    failures: list[str] = []
    for message, intent, required, banned in cases:
        reply = _turn(message)
        low = reply.reply.lower()
        if reply.intent != intent:
            failures.append(f"intent:{message} [{reply.intent} != {intent}]")
        if reply.intent == "unclear" or _UNCLEAR_MENU in low:
            failures.append(f"unclear:{message}")
        for phrase in required:
            if phrase.lower() not in low:
                failures.append(f"missing:{phrase}:{message}")
        for phrase in banned:
            if phrase.lower() in low:
                failures.append(f"banned:{phrase}:{message}")
    assert not failures, failures
