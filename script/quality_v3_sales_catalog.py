"""Everyday shopper product/fit conversations through the real Sales HTTP API.

Audit only. Catalog prices/specs are checked against repository facts, inventory
is unavailable, and no actual model response or customer outcome is measured.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re

from script.quality_v2_support import CheckFailure, Scenario, expect, require


def scenarios() -> list[Scenario]:
    from sales_catalog import resolve_product
    from sales_models import get_session_collected

    rows = []
    descriptions = {}
    with (Path(__file__).resolve().parents[1] / "raw_data/products_export.csv").open() as source:
        for row in csv.DictReader(source):
            if row.get("Body (HTML)"):
                descriptions[row["Handle"]] = re.sub(r"<[^>]*>", " ", row["Body (HTML)"]).casefold()
    epic = "Titan TP-Epic 4D"
    champ = "Osaki OS-Champ"
    maestro = "Osaki OS-Pro Maestro 4D"
    le2 = "Osaki 4D Maestro LE 2.0"
    paragon = "Osaki Pro OS-4D Paragon"
    hypnos = "Osaki Hypnos 4D Pro AI"
    grande = "Grande XL"

    def fact(name):
        item = resolve_product(name)
        require(item is not None, "Known catalog fixture must exist", expected=name)
        return item.as_public_dict()

    def card(response, model):
        canonical = fact(model)["model"]
        items = [p for p in response.get("products", []) if p.get("model") == canonical]
        require(bool(items), "Answer must retain the customer's exact model identity",
                expected=canonical, observed=response)
        return items[0]

    def no_interview(response):
        reply = response.get("reply", "").casefold()
        require("what's the **user height**" not in reply and "tap a range, or type it" not in reply,
                "A concrete product question should be answered before restarting a fit interview",
                expected="answer the product question or clearly explain an unavailable fact", observed=response)

    def add(cid, category, family, title, messages, judge, severity="medium"):
        sid = "v3-catalog-" + cid
        def run(client):
            transcript = []
            responses = []
            saved = {}
            try:
                path = f"/api/v1/warranty/session/{sid}/consent"
                body = {"domain": "osakiusa.com", "policy_store": "osakiusa"}
                accepted = client.post(path, json=body)
                transcript.append({"phase": "consent", "endpoint": path, "request": body,
                                   "status": accepted.status_code, "response": accepted.json()})
                expect(accepted.status_code, 200, "Valid consent setup should succeed")
                for index, message in enumerate(messages, 1):
                    body = {"session_id": sid, "domain": "osakiusa.com", "message": message}
                    response = client.post("/api/v1/sales/chat", json=body)
                    payload = response.json()
                    saved = get_session_collected(sid)
                    transcript.append({"turn": index, "customer": message, "endpoint": "/api/v1/sales/chat",
                                       "request": body, "status": response.status_code, "response": payload,
                                       "saved_state": saved})
                    expect(response.status_code, 200, "A normal shopper question must produce a useful response")
                    require(bool(payload.get("reply", "").strip()), "Shopper response must not be empty")
                    responses.append(payload)
                judge(responses, saved)
            except CheckFailure as exc:
                raise CheckFailure(str(exc), expected=exc.expected,
                                   observed={"check_observed": exc.observed, "transcript": transcript,
                                             "saved_state": saved}) from exc
            return {"transcript": transcript, "saved_state": saved, "customer_turns": len(messages)}
        rows.append(Scenario("v3-catalog-" + cid, category, family, title, severity, run,
                             {"messages": messages, "domain": "osakiusa.com"}, "http_api"))

    def price_judge(model):
        def judge(responses, _):
            response = responses[-1]
            actual = card(response, model)
            source = fact(model)
            expect(actual.get("price_usd"), source["price_usd"], "Quoted amount must match the requested catalog model")
            require(f"${source['price_usd']:,.0f}" in response["reply"],
                    "The price must be answered in the customer-visible message", expected=source["price_usd"], observed=response)
            no_interview(response)
        return judge

    # 24 normal product-information goals, many asked after establishing context.
    for cid, model, messages in [
        ("price-epic", epic, ["What does the Titan TP-Epic 4D cost?"]),
        ("price-champ", champ, ["I am looking at the OS-Champ.", "What is the price of the OS-Champ?"]),
        ("price-le2-version", le2, ["What is the price of the Maestro LE 2.0?"]),
        ("price-followup", epic, ["Does Titan TP-Epic 4D have foot rollers?", "How much is that one?"]),
    ]:
        add(cid, "sales_product_questions", "catalog_price", "Answer the named product price using the correct version", messages, price_judge(model))

    def feature_judge(model, key, label):
        def judge(responses, _):
            response = responses[-1]
            actual = card(response, model)
            source = fact(model)["specs"][key]
            require(source not in {"", "-"}, "Published feature fixture must have an answer")
            expect(actual.get("specs", {}).get(key), source, "Product card must retain the authoritative feature")
            require(label.casefold() in response["reply"].casefold() and str(source).casefold() in response["reply"].casefold(),
                    "The requested published feature must be answered in the visible message",
                    expected={label: source}, observed=response)
            no_interview(response)
        return judge
    for cid, model, key, label, question in [
        ("foot-rollers", epic, "foot_roller", "roller", "Does Titan TP-Epic 4D have foot rollers?"),
        ("heating-area", epic, "heating", "heat", "Where does Titan TP-Epic 4D have heating?"),
        ("airbags", champ, "airbag", "airbag", "How many airbags does the Osaki OS-Champ have?"),
        ("zero-gravity-model", hypnos, "zero_gravity", "zero gravity", "Does the Hypnos have zero gravity?"),
        ("auto-programs", epic, "auto_programs", "program", "How many automatic programs does Titan TP-Epic 4D have?"),
        ("track-model", epic, "track_type", "track", "What is the track type of Titan TP-Epic 4D?"),
    ]:
        add(cid, "sales_product_questions", "published_feature_answers", "Answer a concrete product feature before fit interviewing",
            [f"Tell me about {model}.", question], feature_judge(model, key, label))

    def unlisted_feature(model, topic, published_marker=None):
        def judge(responses, _):
            response = responses[-1]
            canonical = fact(model)["model"]
            for turn in responses:
                require(all(item.get("model") == canonical for item in turn.get("products", [])),
                        "A feature question must not silently switch to another product generation",
                        expected=canonical, observed=turn)
            text = response["reply"].casefold()
            no_interview(response)
            if published_marker:
                source = descriptions.get(resolve_product(model).handle, "")
                require(published_marker in re.sub(r"\s+", " ", source), "Published description fixture must contain the independently checked feature",
                        expected=published_marker, observed=source[:500])
                require(topic in text and any(word in text for word in ("speaker", "scanning", "yes", "includes", "has")),
                        "Answer the specific feature available in the published product description",
                        expected=published_marker, observed=response)
            else:
                require(topic in text and any(word in text for word in ("not listed", "not verified", "can't confirm", "cannot confirm", "specialist", "check")),
                        "An unverified feature needs a direct uncertainty explanation rather than unrelated specs",
                        expected=f"address {topic} and explain verification is needed", observed=response)
        return judge
    for cid, model, topic, question, published_marker in [
        ("bluetooth", le2, "bluetooth", "Does the Maestro LE 2.0 have Bluetooth speakers?", "surround sound bluetooth speakers"),
        ("app-control", le2, "app", "Can I control the Maestro LE 2.0 with an app?", None),
        ("body-scan", le2, "scan", "Does the Maestro LE 2.0 scan my body?", "computerized body scanning"),
        ("voice-control", hypnos, "voice", "Does the Hypnos have voice control?", None),
    ]:
        add(cid, "sales_product_questions", "less_structured_features", "Use published description facts for specific features and state uncertainty when genuinely unverified",
            [f"What does {model} cost?", question], unlisted_feature(model, topic, published_marker))

    def colors_judge(model):
        def judge(responses, _):
            response = responses[-1]
            card(response, model)
            available = fact(model).get("colors", [])
            require(len(available) >= 2, "Color fixture needs multiple published choices")
            hits = [color for color in available if color.casefold() in response["reply"].casefold()]
            require(len(hits) >= 2, "A request for color choices needs actual catalog color options",
                    expected=available, observed=response)
            no_interview(response)
        return judge
    add("colors-grande", "sales_product_questions", "colors_and_versions", "Show color choices for a named chair",
        ["What does the Grande XL cost?", "What colors can I buy the Grande XL in?"], colors_judge(grande))
    add("colors-epic", "sales_product_questions", "colors_and_versions", "List the named chair's catalog colors without claiming live stock",
        ["Does Titan TP-Epic 4D have foot rollers?", "Which colors does the Titan TP-Epic 4D come in?"], colors_judge(epic))
    def versions(responses, _):
        response = responses[-1]
        card(response, le2)
        card(response, "Osaki OS-Pro Maestro LE")
        no_interview(response)
    add("compare-maestro-versions", "sales_product_questions", "colors_and_versions", "Distinguish Maestro LE from Maestro LE 2.0",
        ["Please compare Osaki OS-Pro Maestro LE and Osaki 4D Maestro LE 2.0."], versions)

    def unknown_model(responses, _):
        response = responses[-1]
        require(not response.get("products"), "Unknown product name must not become a guessed model", observed=response)
        text = response["reply"].casefold()
        require(any(word in text for word in ("which model", "model name", "couldn't find", "cannot find", "don't have", "not find")),
                "Ask for clarification or explain this model cannot be found", observed=response)
    add("unknown-model", "sales_product_questions", "product_context", "Clarify an unfamiliar model rather than guessing",
        ["Do you sell an Osaki Moonlight 9000?", "I mean the Moonlight 9000 model."], unknown_model)
    def own_weight(responses, _):
        response = responses[-1]
        card(response, champ)
        require("not verified" in response["reply"].casefold() and "different" in response["reply"].casefold(),
                "Distinguish the chair's own weight from maximum user weight when chair weight is unavailable", observed=response)
        no_interview(response)
    add("chair-weight", "sales_product_questions", "product_context", "Do not answer shipping/chair weight with user capacity",
        ["What is the price of the Osaki OS-Champ?", "How heavy is the chair itself?"], own_weight)
    add("heat-pronoun", "sales_product_questions", "product_context", "Retain the chair for a short feature follow-up",
        ["What does Titan TP-Epic 4D cost?", "Where does it have heat?"], feature_judge(epic, "heating", "heat"))
    add("feature-mid-interview", "sales_product_questions", "product_context", "Answer a named feature while a recommendation interview is paused",
        ["Recommend a massage chair.", "Does Titan TP-Epic 4D have foot rollers?"], feature_judge(epic, "foot_roller", "roller"))

    def education(words, excluded=()):
        def judge(responses, _):
            response = responses[-1]
            text = response["reply"].casefold()
            require(all(word.casefold() in text for word in words), "Explain the general concept instead of requiring a model or body measurement",
                    expected=words, observed=response)
            require(not response.get("handoff"), "A general product concept does not require a human handoff", observed=response)
            no_interview(response)
        return judge
    add("mechanism-education", "sales_product_questions", "general_product_education", "Explain 3D vs 4D before asking for a model",
        ["What is the difference between 3D and 4D massage?"], education(["3D", "4D", "speed"]))
    add("zero-gravity-education", "sales_product_questions", "general_product_education", "Explain zero gravity in ordinary shopping language",
        ["What does zero gravity mean in a massage chair?"], education(["zero gravity", "reclin"]))
    add("track-education", "sales_product_questions", "general_product_education", "Explain L-track and SL-track coverage",
        ["What is the difference between L-track and SL-track?"], education(["L-track", "SL-track"]))

    base = "I am 5 foot 8 and weigh 170 pounds. I want gentle shoulder massage. My budget is $4000. My doorway is 36 inches, assembled only. Recommend a chair."

    def fit_judge(expected, *, need_products=True, no_products=False, check_all=False, allow_no_fit=False):
        def judge(responses, saved):
            rec = saved.get("recommend_prefs", {})
            for key, value in expected.items():
                actual = rec.get(key)
                if isinstance(value, (int, float)):
                    require(actual is not None and abs(float(actual) - value) < .02,
                            "Shopper's own exact requirement must survive the conversation", expected={key: value}, observed=rec)
                else:
                    expect(actual, value, "Preserve the customer's affirmative preference")
            final = responses[-1]
            text = final["reply"].casefold()
            honest_no_fit = any(word in text for word in ("no ", "not ", "cannot", "couldn't", "can't", "unable")) and any(word in text for word in ("verify", "verified", "match", "fit", "catalog", "chair", "budget", "requirements"))
            if need_products and not (allow_no_fit and honest_no_fit):
                require(bool(final.get("products")), "With complete ordinary criteria, provide verified choices rather than repeat intake", observed=final)
            if no_products:
                require(not final.get("products"), "An impossible set of hard limits must not be silently relaxed", observed=final)
                require(honest_no_fit,
                        "Explain why no verified fit is available and offer an honest next step", observed=final)
            selected = responses if check_all else [final]
            for response in selected:
                for product in response.get("products", []):
                    fit = product.get("fit") or {}
                    if "budget_max_usd" in expected:
                        price = product.get("price_usd")
                        require(price is not None and price <= expected["budget_max_usd"],
                                "Shown recommendation must respect the saved maximum budget", expected=expected["budget_max_usd"], observed=product)
                    if "weight_lb" in expected:
                        cap = fit.get("maximum_user_weight_lb")
                        require(cap is not None and cap >= expected["weight_lb"], "Shown chair needs enough published weight capacity", expected=expected["weight_lb"], observed=product)
                    if "doorway_in" in expected:
                        width = fit.get("doorway_assembled_in")
                        if expected.get("doorway_fit") == "disassembled":
                            width = fit.get("doorway_disassembled_in") or width
                        require(width is not None and width <= expected["doorway_in"], "Shown chair needs to clear the stated access width", expected=expected["doorway_in"], observed=product)
        return judge

    # 24 fit/comparison goals. Ordinary business wording, predominantly multi-turn.
    add("interview-short-answers", "sales_fit_comparison", "fit_interview", "Collect height and focus once, then recommend",
        ["Please recommend a massage chair.", "I am five foot eight and 170 pounds.", "Gentle shoulder massage, under $4000."],
        fit_judge({"height_in": 68, "weight_lb": 170, "budget_max_usd": 4000, "intensity": "Gentle"}))
    add("interview-complete", "sales_fit_comparison", "fit_interview", "Use a complete shopper brief without asking the same basics again",
        [base], fit_judge({"height_in": 68, "weight_lb": 170, "budget_max_usd": 4000, "doorway_in": 36}))
    add("interview-parent", "sales_fit_comparison", "fit_interview", "Recommend a gentle option for a petite parent using her measurements",
        ["This is for my mother. She is five foot two and 130 pounds.", "Gentle neck and shoulder massage, with a $3000 budget."],
        fit_judge({"height_in": 62, "weight_lb": 130, "budget_max_usd": 3000, "intensity": "Gentle"}))
    add("interview-tall", "sales_fit_comparison", "fit_interview", "Carry a tall shopper's exact measurements through focus selection",
        ["I am six foot three and weigh 260 pounds. Can you recommend a chair?", "I want lower back massage, budget $7000."],
        fit_judge({"height_in": 75, "weight_lb": 260, "budget_max_usd": 7000}))

    add("budget-lower", "sales_fit_comparison", "budget_followups", "Recalculate after a smaller budget without losing fit criteria",
        [base, "Actually, my maximum budget is $2500."], fit_judge({"height_in": 68, "weight_lb": 170, "budget_max_usd": 2500}, check_all=False))
    add("budget-higher", "sales_fit_comparison", "budget_followups", "Accept an increased budget while keeping the same person",
        [base, "I can spend up to $6000 now."], fit_judge({"height_in": 68, "weight_lb": 170, "budget_max_usd": 6000}, check_all=False))
    add("budget-no-fit", "sales_fit_comparison", "budget_followups", "Tell the shopper honestly when a chair cannot fit the stated price cap",
        ["I am 5 foot 8, 170 pounds. I need gentle shoulder massage and can spend at most $500."],
        fit_judge({"budget_max_usd": 500}, need_products=False, no_products=True))
    add("product-weight-detour", "sales_fit_comparison", "budget_followups", "A product capacity question must not change the shopper's weight",
        [base, "Does the Hypnos support a 280 pound user?", "Recommend using my own measurements and the same budget."],
        fit_judge({"height_in": 68, "weight_lb": 170, "budget_max_usd": 4000}, check_all=False))

    add("access-wide", "sales_fit_comparison", "access_and_room", "Recommend only chairs that fit a measured ordinary doorway",
        [base], fit_judge({"doorway_in": 36, "doorway_fit": "assembled"}))
    add("access-narrow", "sales_fit_comparison", "access_and_room", "Retain a narrow assembled-only doorway through recommendations",
        ["I am 5 foot 8, 170 pounds and need gentle shoulder massage, budget $4000.", "My door is 29 inches wide. No disassembly. Please recommend."],
        fit_judge({"height_in": 68, "weight_lb": 170, "doorway_in": 29, "doorway_fit": "assembled", "budget_max_usd": 4000}, check_all=False, allow_no_fit=True))
    add("access-disassembly", "sales_fit_comparison", "access_and_room", "Use disassembled access only after the customer explicitly permits it",
        ["I am 5 foot 8, 170 pounds, want lower back massage, budget $7000.", "My doorway is 29 inches wide, and disassembly is allowed. Recommend a chair."],
        fit_judge({"doorway_in": 29, "doorway_fit": "disassembled", "budget_max_usd": 7000}, check_all=False))
    add("access-tight-room", "sales_fit_comparison", "access_and_room", "Retain narrow access after a room-space concern",
        [base, "This is a small apartment. My doorway opening is actually 28 inches, assembled only.", "Please use the dimensions I gave you."],
        fit_judge({"doorway_in": 28, "doorway_fit": "assembled"}, check_all=False, allow_no_fit=True))

    def comparison_judge(models, *, cheaper=False):
        def judge(responses, _):
            response = responses[-1]
            no_interview(response)
            if cheaper:
                lower = min((fact(model) for model in models), key=lambda item: item["price_usd"])
                lo = lower["price_usd"]
                require(f"${lo:,.0f}" in response["reply"] or str(int(lo)) in response["reply"].replace(",", ""),
                        "Relative-price answer must identify the lower priced catalog option", expected=lo, observed=response)
                require("Paragon" in response["reply"], "Identify the less expensive model rather than only a dollar amount", observed=response)
            else:
                for model in models:
                    card(response, model)
                    require(fact(model)["model"] in response["reply"], "Visible comparison must keep both exact models", observed=response)
        return judge
    add("compare-cost-followup", "sales_fit_comparison", "comparison_followups", "Keep the compared pair for a relative-price follow-up",
        ["Compare the Maestro 4D and Paragon.", "Which of these two is less expensive?"], comparison_judge([maestro, paragon], cheaper=True))
    add("compare-feet-followup", "sales_fit_comparison", "comparison_followups", "Answer foot-massage differences between the same pair",
        ["Compare Titan TP-Epic 4D with Osaki OS-Champ.", "How do the foot rollers compare between those two?"], comparison_judge([epic, champ]))
    add("compare-exact-models", "sales_fit_comparison", "comparison_followups", "Compare two clearly named models without restarting intake",
        ["Please compare Titan TP-Epic 4D and Osaki OS-Champ."], comparison_judge([epic, champ]))

    def shared_capacity(responses, saved):
        if not responses[-1].get("products"):
            text = responses[-1]["reply"].casefold()
            require(any(word in text for word in ("both", "each", "two users", "two people"))
                    and any(word in text for word in ("verify", "confirm", "specialist", "clarif", "cannot", "can't")),
                    "Shared-use shopper needs verified choices or an explicit explanation for checking both users", observed=responses[-1])
            return
        for product in responses[-1]["products"]:
            cap = (product.get("fit") or {}).get("maximum_user_weight_lb")
            require(cap is not None and cap >= 320, "A shared chair must support the heavier of both stated users", expected=320, observed=product)
    add("couple-shared-weight", "sales_fit_comparison", "shared_and_gift_users", "Protect both users in a shared-chair recommendation",
        ["My father is 6 feet tall and 320 pounds. My mother is 5 foot 3 and 150 pounds. They will share the chair.",
         "They want gentle full body massage, budget $12000. Please recommend something suitable for both."], shared_capacity, "high")
    add("gift-parent-gentle", "sales_fit_comparison", "shared_and_gift_users", "A parent's gift preference survives an accessory feature detour",
        ["My mother is five foot four and 140 pounds. Recommend a chair with gentle shoulder massage under $4000.",
         "Does Titan TP-Epic 4D have heat?", "Please recommend using my mother's measurements and budget."],
        fit_judge({"height_in": 64, "weight_lb": 140, "budget_max_usd": 4000, "intensity": "Gentle"}, check_all=False))
    add("gift-budget-recipient", "sales_fit_comparison", "shared_and_gift_users", "Keep gift recipient measurements separate from the buyer's purchase budget",
        ["This is a gift for my dad. He is five foot ten and 200 pounds.", "I can spend $3500. He likes lower back massage."],
        fit_judge({"height_in": 70, "weight_lb": 200, "budget_max_usd": 3500}))

    def defect_boundary(responses, _):
        final = responses[-1]
        require(final.get("handoff") and final.get("intent") in {"warranty_redirect", "parts_technician"},
                "A broken feature on an owned chair needs warranty/service routing, not a new-chair interview", observed=final)
        require(not final.get("products"), "Do not offer new chair cards as the response to a defect report", observed=final)
        no_interview(final)
    for cid, first, second in [
        ("fault-foot-rollers", "Does Titan TP-Epic 4D have foot rollers?", "Titan TP-Epic 4D foot rollers stopped working."),
        ("fault-remote", "What does the Osaki OS-Champ cost?", "The remote on my OS-Champ is dead."),
        ("fault-heating", "Does Titan TP-Epic 4D have heat?", "The heater on my Titan TP-Epic 4D is not warming up."),
    ]:
        add(cid, "sales_fit_comparison", "feature_vs_existing_fault", "Distinguish asking whether a feature exists from reporting it broken",
            [first, second], defect_boundary, "high")

    add("preference-strong-to-gentle", "sales_fit_comparison", "preference_and_assistance", "A later gentle preference replaces strong pressure",
        ["I am five foot eight, 170 pounds, budget $4000. Recommend strong lower back massage.", "Actually I prefer gentle massage, not strong."],
        fit_judge({"height_in": 68, "weight_lb": 170, "budget_max_usd": 4000, "intensity": "Gentle"}, check_all=False))
    add("preference-focus-change", "sales_fit_comparison", "preference_and_assistance", "A new target area replaces the earlier shopping focus",
        [base, "Actually, foot and calf massage matters most to me now."],
        fit_judge({"height_in": 68, "weight_lb": 170, "budget_max_usd": 4000, "goal": "Foot & Calf"}, check_all=False))
    def human_judge(responses, saved):
        require(responses[-1].get("handoff") is True, "Explicit request for a person should stop the automated interview", observed=responses[-1])
        no_interview(responses[-1])
        require(saved.get("recommend_prefs", {}).get("budget_max_usd") == "4000", "Keep shopping context for the human", observed=saved)
    add("person-after-shopping", "sales_fit_comparison", "preference_and_assistance", "Hand off the shopping context when a customer wants a person",
        [base, "I would like to talk to a person about these options."], human_judge)

    return rows
