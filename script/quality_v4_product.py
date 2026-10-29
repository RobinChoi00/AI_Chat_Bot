"""Continued Sales product conversations: record misunderstandings and try recovery.

All assistant messages come from the real isolated HTTP API. Catalog prices are
repository reference prices, not live offers. Semantic failures never stop the
remaining customer messages.
"""
from __future__ import annotations

from copy import deepcopy
import re

from script.quality_v4_support import Journey


DOMAIN = "osakiusa.com"


class SalesLog:
    def __init__(self, client, sid):
        self.client, self.sid = client, sid
        self.transcript, self.checks, self.recovery_attempts = [], [], []
        self.customer_turns = 0
        self.saved = {}
        response = client.post(f"/api/v1/warranty/session/{sid}/consent",
                               json={"domain": DOMAIN, "policy_store": "osakiusa"})
        self.transcript.append({"setup": True, "endpoint": f"/api/v1/warranty/session/{sid}/consent",
                                "request": {"domain": DOMAIN, "policy_store": "osakiusa"},
                                "status": response.status_code, "response": response.json()})
        self.setup_ok = response.status_code == 200

    def say(self, message):
        from sales_models import get_session_collected

        self.customer_turns += 1
        body = {"session_id": self.sid, "domain": DOMAIN, "message": message}
        try:
            response = self.client.post("/api/v1/sales/chat", json=body)
            try:
                payload = response.json()
            except ValueError:
                payload = {"text": response.text}
            status = response.status_code
        except Exception as exc:
            # Preserve an actual exception; do not invent an assistant response.
            status, payload = None, {"request_exception": type(exc).__name__, "message": str(exc)}
        self.saved = deepcopy(get_session_collected(self.sid))
        self.transcript.append({"setup": False, "turn": self.customer_turns, "customer": message,
                                "endpoint": "/api/v1/sales/chat", "request": body, "status": status,
                                "response": payload, "saved_state": deepcopy(self.saved)})
        self.check("understanding", "Customer receives a usable reply", status == 200 and bool(payload.get("reply", "").strip()),
                   "HTTP200 and a nonempty real assistant reply", {"status": status, "response": payload})
        return payload

    def check(self, dimension, name, condition, expected=None, observed=None, *, turn=None):
        self.checks.append({"dimension": dimension, "name": name,
                            "status": "unmeasured" if condition is None else "pass" if condition else "fail",
                            "turn": self.customer_turns if turn is None else turn,
                            "expected": expected, "observed": deepcopy(observed)})
        return condition

    def recovery(self, action, before_ok, after_ok, evidence):
        recovered = bool(after_ok) if before_ok is False else None
        self.recovery_attempts.append({"turn": self.customer_turns, "action": action,
                                       "recovered": recovered, "evidence": deepcopy(evidence)})
        self.check("recovery", action, recovered,
                   "The explicit correction repairs the earlier misunderstanding" if before_ok is False else "No earlier failure to recover from",
                   evidence)

    def finish(self, goals, reason):
        conditions = [bool(value) for value in goals.values()]
        usable = sum(t.get("status") == 200 for t in self.transcript if not t["setup"])
        if not self.setup_ok or not usable:
            outcome = "blocked"
        elif conditions and all(conditions):
            outcome = "goal_met"
        elif any(conditions) or any(human_route(t["response"]) for t in self.transcript if not t["setup"]):
            outcome = "partially_met"
        else:
            outcome = "not_met"
        first = next((row for row in self.checks if row["status"] == "fail"), None)
        return {"transcript": self.transcript, "checks": self.checks,
                "first_misunderstanding": deepcopy(first), "recovery_attempts": self.recovery_attempts,
                "final_outcome": outcome, "goal_reason": reason, "goal_conditions": goals,
                "customer_turns": self.customer_turns, "saved_state": deepcopy(self.saved),
                "mode": "actual isolated Sales HTTP API; deterministic/model-unavailable path",
                "catalog_prices_are_live": False}


def facts(model):
    from sales_catalog import load_product_index

    matches = [p for p in load_product_index() if p.display_name == model]
    if not matches:
        raise ValueError(f"Independent exact-name catalog fixture missing: {model}")
    return matches[0].as_public_dict()


def exact_card(payload, model):
    return next((p for p in payload.get("products", []) if p.get("model") == model), None)


def correct_model_price(payload, model):
    source = facts(model)
    item = exact_card(payload, model)
    return bool(item and item.get("price_usd") == source["price_usd"]
                and f"${source['price_usd']:,.0f}" in payload.get("reply", ""))


def direct_fact(payload, model, field):
    source = facts(model)["specs"][field]
    item = exact_card(payload, model)
    return bool(source and item and item.get("specs", {}).get(field) == source
                and str(source).casefold() in payload.get("reply", "").casefold())


def prefs(log):
    return log.saved.get("recommend_prefs", {})


def numeric(value, expected, tolerance=.01):
    try:
        return abs(float(value) - expected) <= tolerance
    except (ValueError, TypeError):
        return False


def interview(payload):
    text = payload.get("reply", "").casefold()
    return "what's the **user height**" in text or "tap a range, or type it" in text


def human_route(payload):
    if payload.get("handoff"):
        return True
    raw = str(payload.get("buttons", [])) + str(payload.get("quick_replies", [])) + str(payload.get("actions", []))
    return payload.get("intent") == "human" and ("confirm" in raw.casefold()
        or any(word in payload.get("reply", "").casefold() for word in ("sales", "connect", "specialist")))


def warranty_route(payload):
    return payload.get("intent") in {"warranty_redirect", "parts_technician"} or payload.get("next_action") == "warranty_redirect"


def independently_eligible(payload, *, cap=None, weight=None, door=None):
    products = payload.get("products", [])
    if not products:
        return None
    for item in products:
        fit = item.get("fit", {})
        if cap is not None and (item.get("price_usd") is None or float(item["price_usd"]) > cap):
            return False
        if weight is not None and (fit.get("maximum_user_weight_lb") is None or float(fit["maximum_user_weight_lb"]) < weight):
            return False
        if door is not None and (fit.get("doorway_assembled_in") is None or float(fit["doorway_assembled_in"]) > door):
            return False
    return True


def honest_no_fit(payload):
    text = payload.get("reply", "").casefold()
    boundary = any(t in text for t in ("cannot verify", "can't verify", "couldn't verify", "could not verify", "no verified", "no matching", "no chair matches", "not enough verified"))
    topic = any(t in text for t in ("fit", "requirements", "doorway", "height", "capacity", "weight"))
    return not payload.get("products") and boundary and topic


def uncertain(text):
    text = text.casefold()
    return any(t in text for t in ("not verified", "not listed", "can't confirm", "cannot confirm", "can't verify", "cannot verify", "need to confirm", "needs confirmation", "not published"))


def published_feature_answer(payload, model, topic, detail):
    text = payload.get("reply", "").casefold()
    feature = r"(?:bluetooth(?:\s+(?:audio|speakers?))?|(?:computerized\s+)?body[ -]?scann\w*)"
    negated = re.search(
        r"\b(?:no|without|lacks?|not)\s+(?:(?:built[ -]?in|any|computerized)\s+)?" + feature
        + r"|\b(?:does\s+not|doesn't|cannot|can't)\s+(?:have|include|offer|support|feature)\b[^.\n]{0,35}" + feature
        + r"|" + feature + r"\s+(?:is|are)\s+(?:not|never)\b", text,
    )
    return bool(exact_card(payload, model)) and topic in text and detail in text and not uncertain(text) and not negated


def visible_doorway(payload, model):
    card = exact_card(payload, model)
    expected = facts(model)["fit"]["doorway_assembled_in"]
    text = payload.get("reply", "").casefold()
    aliases = (model.casefold(), re.sub(r"^(?:osaki|titan)\s+", "", model, flags=re.I).casefold())
    return bool(card and card.get("fit", {}).get("doorway_assembled_in") == expected
                and any(alias in text for alias in aliases)
                and re.search(r"(?<![\d.])" + re.escape(f"{expected:g}") + r"(?![\d.])", text)
                and any(label in text for label in ("doorway", "door width", "door clearance")))


def journeys():
    output = []

    def add(cid, persona, goal, messages, run):
        output.append(Journey("product-v4-" + cid, "sales_product", persona, goal, run, messages))

    gift_messages = [
        "I'm buying a massage chair for my father, not for myself. He is 5 foot 8 and weighs 195 pounds. Budget is $5000 and he likes gentle shoulder massage. Recommend a chair.",
        "The doorway in his home is 32 inches wide. Assembled only.",
        "Those measurements belong to my father: 5 foot 8 and 195 pounds. Please keep them.",
        "Which of your options has foot rollers as well?",
        "Please compare Titan TP-Epic 4D and Osaki OS-Champ for my father.",
        "He wants gentle massage, not strong. Keep the $5000 budget and 32 inch doorway.",
        "Please show one suitable option using my father's requirements, or say which fit information still needs confirmation.",
        "If you cannot verify that fit, connect me to a Sales specialist.",
    ]

    def gift(client):
        log = SalesLog(client, "product-v4-gift")
        first = log.say(gift_messages[0])
        first_ok = numeric(prefs(log).get("weight_lb"), 195)
        log.check("context", "Recipient's body measurements are retained", first_ok, 195, prefs(log))
        log.say(gift_messages[1]); log.say(gift_messages[2])
        now = numeric(prefs(log).get("weight_lb"), 195)
        log.recovery("Restate who the measurements belong to", first_ok, now, prefs(log))
        foot = log.say(gift_messages[3])
        foot_ok = "roller" in foot.get("reply", "").casefold() and not interview(foot)
        log.check("understanding", "Answer a follow-up foot-roller question", foot_ok, "Foot-roller facts or a relevant clarification", foot)
        compared = log.say(gift_messages[4])
        models = {p.get("model") for p in compared.get("products", [])}
        compare_ok = {"Titan TP-Epic 4D", "Osaki OS-Champ"} <= models
        log.check("accuracy", "Comparison retains both exact named candidates", compare_ok, ["Titan TP-Epic 4D", "Osaki OS-Champ"], sorted(models))
        log.say(gift_messages[5])
        final = log.say(gift_messages[6])
        eligibility = independently_eligible(final, cap=5000, weight=195, door=32)
        log.check("accuracy", "Shortlist obeys independently read price/capacity/door facts", eligibility,
                  {"price_max": 5000, "weight_min": 195, "assembled_door_max": 32}, final)
        last = log.say(gift_messages[7])
        log.check("recovery", "A specialist route remains available when facts are insufficient", human_route(last), "Sales specialist confirmation or handoff", last)
        goals = {"father_requirements_kept": numeric(prefs(log).get("weight_lb"),195) and numeric(prefs(log).get("budget_max_usd"),5000),
                 "requested_comparison_answered": compare_ok, "verified_shortlist_or_honest_limit": eligibility is True or honest_no_fit(final)}
        return log.finish(goals, "A gift shopper needs a verified shortlist for the actual recipient, not just a generic recommendation or human button.")
    add("gift", "Adult child buying for father", "Keep the recipient's facts while explaining and comparing suitable options", gift_messages, gift)

    couple_messages = [
        "My father is 6 foot 2 and weighs 320 pounds. My mother is 5 foot 2 and weighs 150 pounds. They will share one chair. Budget $12000.",
        "They both want gentle full body massage. Please recommend something suitable for both.",
        "The heavier user weighs 320 pounds. Please do not use 150 pounds as the weight requirement.",
        "The doorway is 36 inches, assembled only. Please check the capacity of every recommendation.",
        "Which chairs actually have a published capacity of at least 320 pounds?",
        "Does the Grande XL support a 320 pound user? Please check the published maximum weight.",
        "Keep the 320 pound requirement when showing any other choices. My mother is still the second user.",
        "If both users' height fit is not published, please tell me it needs confirmation from Sales.",
    ]

    def couple(client):
        log = SalesLog(client, "product-v4-couple")
        log.say(couple_messages[0])
        first = log.say(couple_messages[1])
        first_ok = numeric(prefs(log).get("weight_lb"), 320)
        log.check("context", "A shared chair retains the heavier user's requirement", first_ok, 320, prefs(log))
        log.check("accuracy", "First displayed options support the stated heavier user", independently_eligible(first, weight=320), "Each displayed capacity >=320lb", first)
        log.say(couple_messages[2])
        corrected = numeric(prefs(log).get("weight_lb"),320)
        log.recovery("Explicitly restate heavier user's weight", first_ok, corrected, prefs(log))
        log.say(couple_messages[3])
        shortlist = log.say(couple_messages[4])
        safe = independently_eligible(shortlist, cap=12000, weight=320, door=36)
        log.check("accuracy", "Post-correction shortlist observes independent capacity facts", safe, "Capacity>=320lb; price<=12000; door<=36in", shortlist)
        grande = log.say(couple_messages[5])
        actual = exact_card(grande, "Grande XL- Big and Tall (Up to 375lbs, 6'5\")")
        grande_ok = bool(actual and actual.get("fit", {}).get("maximum_user_weight_lb",0) >= 320)
        log.check("accuracy", "Named big-and-tall option uses the published weight capacity", grande_ok, "Grande XL card with capacity>=320lb", grande)
        late = log.say(couple_messages[6])
        late_corrected = numeric(prefs(log).get("weight_lb"), 320)
        log.recovery("Restate320lb again after the failed first correction", corrected, late_corrected, {"saved_state": prefs(log), "response": late})
        final = log.say(couple_messages[7])
        text=final.get("reply", "").casefold()
        transparent = "height" in text and any(t in text for t in ("both", "father", "mother", "two users", "shared")) and uncertain(text)
        log.check("accuracy", "Unpublished individual height fit is not treated as confirmed", transparent, "State height-fit confirmation is needed", final)
        log.check("accuracy", "Both individual users' height fit remains unmeasured in this audit", None, "Independent height-range facts for both users", "Public fit cards do not expose verified minimum/maximum heights")
        return log.finish({"heavier_user_retained":numeric(prefs(log).get("weight_lb"),320),
                           "weight_suitable_option_or_honest_limit":safe is True or honest_no_fit(shortlist),
                           "named_capacity_answered":grande_ok,"unknown_height_fit_disclosed":transparent},
                          "Explicit correction can repair a weight slot without proving that both users' full ergonomic fit was verified.")
    add("couple", "Couple with substantially different body sizes", "Respect both users and recover from a lost heavier-user requirement", couple_messages, couple)

    apartment_messages = [
        "I am 5 foot 7, 165 pounds, and want gentle back massage. Budget $4500. My doorway is 36 inches, assembled only. Recommend a chair.",
        "Correction: my doorway is 28 inches, not 36. I cannot remove the chair's side panels.",
        "I am correcting my home's opening. My own doorway is 28 inches wide, assembled only.",
        "Please recommend again using my 28 inch doorway and the other limits I gave.",
        "What assembled doorway width does Titan TP-Epic 4D require?",
        "So Titan TP-Epic 4D needs more than my 28 inch opening, correct?",
        "Please try another option that is verified to fit 28 inches, or explain that you cannot verify a match.",
        "If the dimensions still need confirmation, connect me to a Sales specialist before I order.",
    ]

    def apartment(client):
        log=SalesLog(client,"product-v4-apartment")
        log.say(apartment_messages[0]); log.say(apartment_messages[1])
        first_ok=numeric(prefs(log).get("doorway_in"),28)
        log.check("context","Initial doorway correction replaces the old36in",first_ok,28,prefs(log))
        log.say(apartment_messages[2])
        after=numeric(prefs(log).get("doorway_in"),28)
        log.recovery("Clarify that28in is the customer's own doorway",first_ok,after,prefs(log))
        recommendation=log.say(apartment_messages[3])
        log.check("accuracy","Post-correction displayed choices fit stated28in assembled opening",independently_eligible(recommendation,cap=4500,weight=165,door=28),"Verified door<=28in",recommendation)
        fact_reply=log.say(apartment_messages[4])
        epic=exact_card(fact_reply,"Titan TP-Epic 4D")
        expected=facts("Titan TP-Epic 4D")["fit"]["doorway_assembled_in"]
        dimension_ok=bool(epic and epic.get("fit",{}).get("doorway_assembled_in")==expected and str(expected) in fact_reply.get("reply", ""))
        log.check("accuracy","Answer actual assembled dimension for explicitly named chair",dimension_ok,expected,fact_reply)
        interpretation=log.say(apartment_messages[5])
        log.check("understanding","Do not restart a fit interview when interpreting a known dimension",not interview(interpretation),"Address28in versus published dimension",interpretation)
        final=log.say(apartment_messages[6])
        eligible=independently_eligible(final,cap=4500,weight=165,door=28)
        truthful=honest_no_fit(final)
        last=log.say(apartment_messages[7])
        log.check("recovery","Sales confirmation is available before ordering on uncertain dimensions",human_route(last),"Sales review/confirmation",last)
        return log.finish({"latest_door_retained":numeric(prefs(log).get("doorway_in"),28),
                           "correct_named_dimension":dimension_ok,"final_fit_or_honest_limit":eligible is True or truthful},
                          "A human escape does not itself prove a chair fits the corrected opening.")
    add("apartment","Apartment shopper measuring a narrow entry","Correct an earlier door measurement and avoid an oversized purchase",apartment_messages,apartment)

    le2_messages = [
        "What is the price of the Maestro LE 2.0?",
        "No, I mean the Osaki 4D Maestro LE 2.0, not the older OS-Pro Maestro LE.",
        "What does Osaki 4D Maestro LE 2.0 cost? Please use the 2.0 version.",
        "Does the Maestro LE 2.0 have Bluetooth speakers?",
        "I am asking about built-in Bluetooth speakers on LE 2.0, not a new-chair recommendation.",
        "Does the LE 2.0 have computerized body scanning?",
        "What colors can I buy the LE 2.0 in? Keep the exact model please.",
        "Please confirm which exact model and catalog price you used before I decide.",
        "If you cannot find the exact LE 2.0, connect me to Sales rather than guessing another model.",
    ]

    def le2(client):
        log=SalesLog(client,"product-v4-le2")
        model="Osaki 4D Maestro LE 2.0"
        first=log.say(le2_messages[0]); first_ok=correct_model_price(first,model)
        log.check("accuracy","Initial quote uses exact2.0 product and reference price",first_ok,facts(model),first)
        log.say(le2_messages[1]); corrected=log.say(le2_messages[2]); price_ok=correct_model_price(corrected,model)
        log.recovery("Explicitly distinguish2.0 from olderMaestroLE",first_ok,price_ok,corrected)
        bluetooth=log.say(le2_messages[3])
        bluetooth_ok=published_feature_answer(bluetooth,model,"bluetooth","speaker")
        log.check("understanding","Published Bluetooth feature is answered on the exact model",bluetooth_ok,"LE2.0 Bluetooth speakers answer",bluetooth)
        rephrased=log.say(le2_messages[4])
        after_bt=published_feature_answer(rephrased,model,"bluetooth","speaker") and not interview(rephrased)
        log.recovery("Repeat the specific Bluetooth question without requesting a recommendation",bluetooth_ok,after_bt,rephrased)
        scanning=log.say(le2_messages[5]); scan_ok=published_feature_answer(scanning,model,"body","scann")
        log.check("understanding","Published body-scanning question receives a related answer",scan_ok,"Computerized body scanning onLE2.0",scanning)
        color=log.say(le2_messages[6]); available=facts(model).get("colors",[])
        color_ok=sum(c.casefold() in color.get("reply", "").casefold() for c in available)>=min(2,len(available)) and bool(exact_card(color,model))
        log.check("understanding","Color answer gives actual choices for the same generation",color_ok,available,color)
        summary=log.say(le2_messages[7]); summary_ok=correct_model_price(summary,model)
        log.check("context","Final decision summary retains the intended exact product",summary_ok,facts(model)["price_usd"],summary)
        last=log.say(le2_messages[8]);log.check("recovery","Unverified model can be sent toSales for review",human_route(last),"Sales specialist option",last)
        return log.finish({"correct_quote_after_correction":price_ok,"Bluetooth_answer":after_bt,"body_scan_answer":scan_ok,"colors_answer":color_ok,"final_model_correct":summary_ok},
                          "A general Sales button is only partial progress when the exact product's known facts remain unanswered.")
    add("le2","Detail-oriented buyer comparing product generations","Get the exactLE2.0 price and features after correcting a mistaken model",le2_messages,le2)

    gentle_messages = [
        "Is Titan TP-Epic 4D very strong? I want to understand its strength, not start a recommendation.",
        "I have not asked for a chair recommendation. Does Titan TP-Epic 4D have adjustable massage intensity?",
        "Does Titan TP-Epic 4D have foot rollers?",
        "Where does Titan TP-Epic 4D have heating?",
        "I am 5 foot 7 and 165 pounds, but please keep discussing the same Titan TP-Epic 4D.",
        "Can I choose different massage strength for my calves than for my back? If not verified, please say so.",
        "Does strong massage mean it will cure my back pain?",
        "Please connect me to a Sales specialist for the specific comfort-adjustability question.",
    ]

    def gentle(client):
        log=SalesLog(client,"product-v4-gentle")
        first=log.say(gentle_messages[0]); first_ok=not interview(first) and any(t in first.get("reply", "").casefold() for t in ("intensity","strength","strong","adjust")) and (uncertain(first.get("reply", "")) or bool(exact_card(first,"Titan TP-Epic 4D")))
        log.check("understanding","A named strength question is handled before an interview",first_ok,"Relevant strength facts or truthful verification boundary",first)
        corrected=log.say(gentle_messages[1]); corrected_ok=not interview(corrected) and any(t in corrected.get("reply", "").casefold() for t in ("intensity","strength","adjust")) and (uncertain(corrected.get("reply", "")) or bool(exact_card(corrected,"Titan TP-Epic 4D")))
        log.recovery("Explicitly reject an unwanted recommendation interview",first_ok,corrected_ok,corrected)
        foot=log.say(gentle_messages[2]); foot_ok=direct_fact(foot,"Titan TP-Epic 4D","foot_roller")
        log.check("accuracy","Direct foot-roller feature uses published source",foot_ok,facts("Titan TP-Epic 4D")["specs"]["foot_roller"],foot)
        heat=log.say(gentle_messages[3]); heat_ok=direct_fact(heat,"Titan TP-Epic 4D","heating")
        log.check("accuracy","Direct heating question uses the same published model",heat_ok,facts("Titan TP-Epic 4D")["specs"]["heating"],heat)
        log.say(gentle_messages[4]); unknown=log.say(gentle_messages[5])
        unknown_text=unknown.get("reply", "").casefold()
        unknown_ok=uncertain(unknown_text) and any(t in unknown_text for t in ("intensity","strength","adjust")) and any(t in unknown_text for t in ("calf","calves","back","independent","separate"))
        log.check("accuracy","Unverified independent-intensity feature is not invented",unknown_ok,"Explicit verification boundary",unknown)
        medical=log.say(gentle_messages[6]);medical_ok=any(t in medical.get("reply", "").casefold() for t in ("clinician","doctor","cannot promise","can't promise")) and "will cure" not in medical.get("reply", "").casefold()
        log.check("accuracy","Product strength is not a medical cure promise",medical_ok,"No cure promise; appropriate medical boundary",medical)
        last=log.say(gentle_messages[7]);log.check("recovery","Specific comfort question has an appropriateSalesroute",human_route(last),"Sales specialist",last)
        return log.finish({"strength_question_addressed":corrected_ok,"foot_fact_answered":foot_ok,"heat_fact_answered":heat_ok,"unknown_feature_truthful":unknown_ok,"medical_boundary":medical_ok},
                          "Known feature answers and a truthful unknown-feature boundary are separate from merely offering a specialist.")
    add("gentle","Buyer concerned about massage comfort","Answer direct features and intensity questions without forcing an unrelated interview",gentle_messages,gentle)

    comparison_messages = [
        "I want to compare Osaki OS-Champ and Titan TP-Epic 4D.",
        "What are the prices and the main differences between those two?",
        "The two exact models are Osaki OS-Champ and Titan TP-Epic 4D. Please compare only those.",
        "Which of the two has more foot rollers?",
        "Which needs the smaller assembled doorway?",
        "Please keep the same two models rather than choosing another chair.",
        "I weigh 250 pounds. Which of these two has a published capacity that supports that weight?",
        "Please put their prices and assembled doorway widths together before I decide.",
    ]

    def comparison(client):
        log=SalesLog(client,"product-v4-compare")
        names={"Osaki OS-Champ","Titan TP-Epic 4D"}
        first=log.say(comparison_messages[0]); first_ok=names<={p.get("model") for p in first.get("products",[])}
        log.check("context","Initial comparison preserves both named models",first_ok,sorted(names),first)
        log.say(comparison_messages[1]); corrected=log.say(comparison_messages[2]); correct_ok=names<={p.get("model") for p in corrected.get("products",[])}
        log.recovery("Repeat both full model names",first_ok,correct_ok,corrected)
        foot=log.say(comparison_messages[3]);foot_ok=all(direct_fact(foot,n,"foot_roller") for n in names)
        log.check("accuracy","Foot-roller comparison retains both models' facts",foot_ok,{n:facts(n)["specs"]["foot_roller"] for n in names},foot)
        doorway=log.say(comparison_messages[4]);door_ok=all(visible_doorway(doorway,n) for n in names)
        log.check("accuracy","Doorway comparison retains field-bound dimensions",door_ok,{n:facts(n)["fit"]["doorway_assembled_in"] for n in names},doorway)
        log.say(comparison_messages[5]); weight=log.say(comparison_messages[6])
        cap_ok=all(exact_card(weight,n) and exact_card(weight,n).get("fit",{}).get("maximum_user_weight_lb",0)>=250 for n in names)
        log.check("accuracy","Known capacity question is answered for the same two chairs",cap_ok,"Both displayed known capacities >=250lb",weight)
        final=log.say(comparison_messages[7]);final_ok=all(correct_model_price(final,n) and visible_doorway(final,n) for n in names)
        log.check("context","Final purchase comparison retains model-bound prices and dimensions",final_ok,{n:facts(n) for n in names},final)
        return log.finish({"models_retained_after_clarification":correct_ok,"foot_comparison":foot_ok,"doorway_comparison":door_ok,"capacity_comparison":cap_ok,"decision_summary":final_ok},
                          "A long conversation must retain both sides of a comparison, not substitute an unrelated recommendation.")
    add("compare","Buyer comparing two specific affordable chairs","Keep two model identities and compare requested fields over multiple followups",comparison_messages,comparison)

    education_messages = [
        "What does zero gravity mean in a massage chair?",
        "I mean the general term, not a particular model. What is the reclined position for?",
        "How does the zero-gravity recline position work? Please explain the concept first.",
        "Does the Osaki Hypnos 4D Pro AI have zero gravity?",
        "How many zero-gravity stages does that Hypnos have?",
        "Is zero gravity the same thing as an SL-Track, or are those different features?",
        "Please explain the difference, or clearly say if you need a Sales specialist to confirm it.",
        "Connect me to a Sales specialist if you cannot answer that feature explanation.",
    ]

    def education(client):
        log=SalesLog(client,"product-v4-education")
        def explained(p):
            text=p.get("reply", "").casefold()
            return "zero gravity" in text and any(t in text for t in ("reclin","position","weightless")) and "which model" not in text
        first=log.say(education_messages[0]);first_ok=explained(first)
        log.check("understanding","General concept is explained without demanding a model",first_ok,"Zero-gravity reclining concept explanation",first)
        log.say(education_messages[1]); retry=log.say(education_messages[2]); retry_ok=explained(retry)
        log.recovery("Restate a general conceptual question",first_ok,retry_ok,retry)
        specific=log.say(education_messages[3]);spec_ok=direct_fact(specific,"Osaki Hypnos 4D Pro AI","zero_gravity")
        log.check("accuracy","Switching to a concrete model uses its published zero-gravity facts",spec_ok,facts("Osaki Hypnos 4D Pro AI")["specs"]["zero_gravity"],specific)
        stages=log.say(education_messages[4]);stage_ok=direct_fact(stages,"Osaki Hypnos 4D Pro AI","zero_gravity")
        log.check("context","A pronoun follow-up retains Hypnos and its stage count",stage_ok,"Same model's verified zero-gravity stages",stages)
        difference=log.say(education_messages[5]);dtext=difference.get("reply", "").casefold()
        diff_ok=("zero gravity" in dtext and ("sl-track" in dtext or "sl track" in dtext)
                 and any(t in dtext for t in ("reclin", "position"))
                 and any(t in dtext for t in ("roller", "path", "massage mechanism"))
                 and any(t in dtext for t in ("different", "whereas", "while", "instead", "rather"))
                 and not interview(difference))
        log.check("understanding","Related feature concepts stay on the customer's educational task",diff_ok,"Different features: recline position and massage track",difference)
        final=log.say(education_messages[6]);last=log.say(education_messages[7]);log.check("recovery","Missing explanation can be escalated truthfully",human_route(last),"Sales specialist route",last)
        return log.finish({"general_explanation_recovered":retry_ok,"specific_model_fact":spec_ok,"pronoun_context":stage_ok,"concept_difference":diff_ok},
                          "Accurate product cards alone do not answer a general feature explanation that the customer explicitly requested.")
    add("education","First-time shopper learning feature terminology","Move between general education and a specific model without losing the question",education_messages,education)

    owner_messages = [
        "I own a Titan TP-Epic 4D and its foot rollers stopped working.",
        "This chair is already in my house. It is broken. I need technical support, not a new chair.",
        "The broken model is Titan TP-Epic 4D. Please give me the right team contact.",
        "Separately, I also want to buy a second chair for my office. That is a new purchase.",
        "Recommend a new office chair under $3000. I am 5 foot 8, 170 pounds and prefer gentle back massage.",
        "My office doorway is 36 inches wide, assembled only.",
        "Does the Osaki OS-Champ have foot rollers? This question is about the new purchase.",
        "The broken Epic is a service case. The OS-Champ question is a new purchase. Please keep those separate.",
        "Please route the old Epic repair to Warranty and the new-purchase question to Sales.",
    ]

    def owner(client):
        log=SalesLog(client,"product-v4-owner")
        first=log.say(owner_messages[0]);first_ok=warranty_route(first)
        log.check("understanding","Owned faulty feature is identified as a service inquiry",first_ok,"Warranty/service route",first)
        retry=log.say(owner_messages[1]);retry_ok=warranty_route(retry)
        log.recovery("Explicitly state ownership and a repair request",first_ok,retry_ok,retry)
        contact=log.say(owner_messages[2]);contact_ok=warranty_route(contact) or "technical support" in contact.get("reply", "").casefold()
        log.check("understanding","The old chair gets a technical-support next step",contact_ok,"Warranty is the responsible service route",contact)
        log.say(owner_messages[3]);purchase=log.say(owner_messages[4]);purchase_ok=purchase.get("intent")=="recommend" and not warranty_route(purchase)
        log.check("context","An explicit new purchase is separate from the prior service case",purchase_ok,"New-chair recommendation task",purchase)
        dimensions=log.say(owner_messages[5]);log.check("context","New-purchase constraints are retained",numeric(prefs(log).get("budget_max_usd"),3000) and numeric(prefs(log).get("weight_lb"),170),{"budget":3000,"weight":170},prefs(log))
        feature=log.say(owner_messages[6]);feature_ok=direct_fact(feature,"Osaki OS-Champ","foot_roller")
        log.check("accuracy","The new-model feature question keeps the new product identity",feature_ok,facts("Osaki OS-Champ")["specs"]["foot_roller"],feature)
        log.say(owner_messages[7]);final=log.say(owner_messages[8]);text=final.get("reply", "").casefold()
        primary = re.split(r"\*\*sales\s*\(delivery", text, flags=re.I)[0]
        department_ok="warranty" in primary and "sales" in primary and any(t in primary for t in ("new purchase", "new chair", "buy"))
        log.check("understanding","Separate business tasks get appropriate department guidance",department_ok,"Warranty for old-chair repair and Sales for new purchase",final)
        return log.finish({"service_route_recovered":retry_ok and contact_ok,"new_purchase_separated":purchase_ok,"new_model_fact_answered":feature_ok,"departments_explained":department_ok},
                          "Recovering the initial fault classification must not make every later shopping question a warranty case.")
    add("owner","Existing owner who is also considering a second chair","Recover a missed defect and then switch correctly to a distinct new-purchase task",owner_messages,owner)
    return output
