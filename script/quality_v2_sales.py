"""Broader Sales audit: frozen customer contracts, no production fixes."""
from __future__ import annotations

from types import SimpleNamespace

from script.quality_v2_support import Scenario, expect, require


def _equal_number(actual, expected, label, tolerance=.002):
    require(actual is not None, label + " was not retained", expected=expected, observed=actual)
    require(abs(float(actual) - expected) <= tolerance, label + " changed the customer's exact quantity", expected=expected, observed=actual)


def _merge(prefs, patch):
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(prefs.get(key), dict):
            prefs[key] = {**prefs[key], **value}
        else:
            prefs[key] = value


def _reply_evidence(reply):
    return {"reply": reply.reply, "intent": reply.intent, "handoff": reply.handoff,
            "handoff_reason": reply.handoff_reason, "flow_stage": reply.flow_stage,
            "products": reply.products, "prefs_patch": reply.prefs_patch}


def scenarios():
    from sales_constraints import extract_constraints, intensity_preference, product_constraint_issues
    from sales_catalog import parse_recommendation_hints
    import sales_agent
    from sales_models import get_session_collected

    out = []

    def add(cid, category, family, title, run, inputs=None, severity="high", level="function"):
        out.append(Scenario(cid, category, family, title, severity, run, inputs, level))

    def parsed(cid, family, text, field, expected, previous=None, absent=False, tolerance=.002):
        def run(_):
            actual = extract_constraints(text, previous)
            if absent:
                require(field not in actual, "Product metadata/invalid quantity must not become a shopper limit", expected="absent", observed=actual)
            elif isinstance(expected, (int, float)):
                _equal_number(actual.get(field), expected, field, tolerance)
            else:
                expect(actual.get(field), expected, "Customer constraint was interpreted incorrectly")
            return {"message": text, "previous": previous, "constraints": actual}
        add(cid, "sales_limits", family, "Retain the stated shopper constraint without substituting a nearby number", run,
            {"message": text, "field": field, "expected": expected, "previous": previous},
            severity="medium" if family == "budget_notation" else "high")

    # Monetary precision, alternate numeric notation, actual hard-cap boundaries.
    for cid, text, expected in [
        ("cents", "My budget is $2,999.99.", 2999.99),
        ("decimal-k", "My budget is $2.75k.", 2750),
        ("zero", "My maximum budget is $0.", 0),
        ("code-prefix", "My budget is USD 4,250.", 4250),
        ("code-suffix", "My budget is 4,250 USD.", 4250),
        ("upper-bound", "My budget is $50,001.", 50001),
        ("spoken-money", "My budget is three thousand dollars.", 3000),
        ("money-correction", "My budget is $2800, rather than $4000.", 2800),
    ]:
        parsed("sl-budget-" + cid, "budget_notation", text, "budget_max_usd", expected)

    for cid, text, expected in [
        ("m-decimal", "I am 1.625 metres tall.", 64),
        ("cm-decimal", "My height is 162.5 cm.", 64),
        ("feet-zero", "I am five feet zero inches tall.", 60),
        ("feet-eleven", "I am six feet eleven inches tall.", 83),
        ("inch-only", "My height is 68 inches.", 68),
        ("unicode-prime", "I am 5′8″ tall.", 68),
    ]:
        parsed("sl-height-" + cid, "body_height_units", text, "height_in", expected)

    for cid, text, expected in [
        ("decimal-kg", "I weigh 80.5 kilograms.", 80.5 * 2.20462),
        ("metric-short", "My weight is 95 kgs.", 95 * 2.20462),
        ("imperial-decimal", "I weigh 199.5 pounds.", 199.5),
        ("stone", "I weigh 14 stone.", 196),
        ("correction", "I weigh 180 pounds, not 220 pounds.", 180),
    ]:
        parsed("sl-weight-" + cid, "body_weight_units", text, "weight_lb", expected)

    def shared_users(_):
        from sales_catalog import resolve_product
        from sales_spec_index import lookup_fit_spec
        message = "The chair is for my father who weighs 320 pounds and my mother who weighs 150 pounds."
        constraints = extract_constraints(message)
        product = resolve_product("Osaki OS-Champ")
        require(product is not None, "The checked-in catalog fixture must exist")
        fit = lookup_fit_spec(product.display_name)
        require(fit is not None and fit.max_user_lb is not None, "The checked-in chair capacity must be published")
        require(fit.max_user_lb < 320, "Shared-user case must use a chair inadequate for the heavier shopper")
        issues = product_constraint_issues(product, constraints)
        observed = {"saved_constraints": constraints, "candidate": product.display_name,
                    "candidate_maximum_user_weight_lb": fit.max_user_lb, "rejection_reasons": issues}
        require("weight_capacity" in issues, "A chair used by both customers must reject a capacity below the heavier user's weight",
                expected="reject capacity below 320 lb", observed=observed)
        return {"message": message, **observed}
    add("sl-weight-multi-person-safe-max", "sales_limits", "body_weight_units", "Protect both users of a shared chair rather than only the most recently mentioned lighter user",
        shared_users, "The chair is for my father who weighs 320 pounds and my mother who weighs 150 pounds.")

    for cid, text, expected in [
        ("mm", "My doorway is 740 mm wide.", 740 / 25.4),
        ("cm", "My doorway is 82.5 centimetres wide.", 82.5 / 2.54),
        ("m", "My doorway is 0.825 metres wide.", 82.5 / 2.54),
        ("imperial", "My doorway is 29.5 inches wide.", 29.5),
        ("narrowest-passage", "My front doorway is 36 inches wide, but the hallway doorway is 29 inches wide.", 29),
        ("explicit-correction", "My doorway is 28 inches, not 35 inches.", 28),
    ]:
        parsed("sl-door-" + cid, "access_geometry", text, "doorway_in", expected)

    for cid, text, field in [
        ("capacity", "This chair supports up to 350 pounds.", "weight_lb"),
        ("shipping-weight", "The package weighs 300 pounds.", "weight_lb"),
        ("chair-height", "The chair is 180 cm tall.", "height_in"),
        ("room-height", "My ceiling is 2.1 metres high.", "height_in"),
        ("feet-count", "Does it have 3 foot rollers?", "height_in"),
        ("capacity-not-money", "I need a chair for up to 350 pounds.", "budget_max_usd"),
    ]:
        parsed("sl-number-context-" + cid, "product_person_disambiguation", text, field, None, absent=True)

    for cid, text, expected in [
        ("disassemble-allowed", "We can take the chair apart for delivery.", "disassembled"),
        ("disassemble-prohibited", "Please fit it through my door without taking the chair apart.", "assembled"),
        ("disassemble-revoked", "Actually, no disassembly is allowed.", "assembled"),
    ]:
        parsed("sl-door-mode-" + cid, "access_permissions", text, "doorway_fit", expected,
               previous={"doorway_fit": "disassembled"})

    def fit_gate(cid, constraints, price, should_reject, unknown=False):
        def run(_):
            product = SimpleNamespace(display_name="Synthetic unknown chair", title="Synthetic unknown chair", price_usd=price,
                                      massage_mechanism="3D")
            issues = product_constraint_issues(product, constraints, price_usd=price)
            require(bool(issues) is should_reject, "Hard gate must reject only a failed or unverified requirement", expected=should_reject, observed=issues)
            return {"synthetic_product": {"price_usd": price, "mechanism": "3D", "fit_metadata": None}, "constraints": constraints, "issues": issues}
        add("sl-gates-" + cid, "sales_limits", "hard_gate_boundaries", "Enforce price and missing-fit boundaries without treating unknown facts as a fit", run,
            {"constraints": constraints, "price_usd": price, "should_reject": should_reject})
    fit_gate("equal-cap", {"budget_max_usd": "3000"}, 3000, False)
    fit_gate("one-cent-over", {"budget_max_usd": "3000"}, 3000.01, True)
    fit_gate("one-cent-under", {"budget_max_usd": "3000"}, 2999.99, False)
    fit_gate("missing-weight", {"weight_lb": "180"}, 2000, True)
    fit_gate("missing-door", {"doorway_in": "32", "doorway_fit": "assembled"}, 2000, True)
    fit_gate("excluded-mechanism", {"excluded_mechanisms": "3D,4D"}, 2000, True)

    # Dialogues run through the same wrapper as storefront conversations. They
    # retain every emitted response and patch rather than replacing them with canned success.
    base = "I am 5 foot 8 and weigh 170 pounds. I want gentle shoulder massage. My budget is $4000. My doorway is 36 inches, assembled only. Recommend a chair."

    def dialogue(cid, family, messages, expected, *, payloads=None, final_no_products=False,
                 expected_handoff=None, level="agent", initial=None, domain="osakiusa.com"):
        def run(client):
            prefs = dict(initial or {})
            evidence = []
            sid = "v2-sales-" + cid
            if level == "http_api":
                consent = client.post(f"/api/v1/warranty/session/{sid}/consent", json={"domain": domain, "policy_store": domain})
                expect(consent.status_code, 200, "Valid Sales simulation consent setup must succeed")
            for index, message in enumerate(messages):
                payload = (payloads or {}).get(index)
                if level == "http_api":
                    body = {"session_id": sid, "domain": domain, "message": message}
                    if payload:
                        body["payload"] = payload
                    response = client.post("/api/v1/sales/chat", json=body)
                    expect(response.status_code, 200, "Valid conversational turn must not crash")
                    observed = response.json()
                    prefs = get_session_collected(sid)
                else:
                    reply = sales_agent.respond(message, payload=payload, prefs=prefs, domain=domain)
                    _merge(prefs, reply.prefs_patch)
                    observed = _reply_evidence(reply)
                evidence.append({"message": message, "payload": payload, "response": observed, "saved_preferences": dict(prefs.get("recommend_prefs") or {})})
            rec = prefs.get("recommend_prefs") or {}
            for field, value in expected.items():
                if isinstance(value, (int, float)):
                    _equal_number(rec.get(field), value, field)
                else:
                    expect(rec.get(field), value, "Earlier shopping conditions must survive corrections and interruptions")
            if final_no_products:
                expect(len(observed.get("products") or []), 0, "An impossible combination must not be silently relaxed")
            if expected_handoff is not None:
                expect(bool(observed.get("handoff")), expected_handoff, "Human assistance requests must interrupt the interview")
            # Independent numerical checks inspect the customer-facing cards;
            # they do not reuse the production constraint validator as oracle.
            for product in observed.get("products") or []:
                if "budget_max_usd" in expected and expected["budget_max_usd"] != "":
                    require(product.get("price_usd") is not None and product["price_usd"] <= float(expected["budget_max_usd"]), "Recommended price exceeded a retained cap", expected=expected["budget_max_usd"], observed=product)
                if "weight_lb" in expected:
                    cap = (product.get("fit") or {}).get("maximum_user_weight_lb")
                    require(cap is not None and cap >= float(expected["weight_lb"]), "Recommended chair cannot verify the customer's weight", expected=expected["weight_lb"], observed=product)
                if "doorway_in" in expected and expected.get("doorway_fit", "assembled") == "assembled":
                    width = (product.get("fit") or {}).get("doorway_assembled_in")
                    require(width is not None and width <= float(expected["doorway_in"]), "Recommended assembled chair cannot clear the stated doorway", expected=expected["doorway_in"], observed=product)
            return {"turns": evidence, "final_preferences": rec}
        add("sd-" + cid, "sales_dialogue", family, "Maintain stated customer requirements throughout a real dialogue", run,
            {"messages": messages, "expected_final_constraints": expected, "domain": domain, "initial": initial}, level=level)

    # Different corrections touch independent exact quantities.
    for cid, correction, expected in [
        ("budget-drops", "My budget is now $350, not $4000.", {"budget_max_usd": 350}),
        ("budget-rises", "Actually my maximum budget is now $6000.", {"budget_max_usd": 6000}),
        ("weight-corrects", "Correction: I weigh 290 pounds, not 170 pounds.", {"weight_lb": 290}),
        ("height-corrects", "Actually the buyer is six feet four inches.", {"height_in": 76}),
        ("door-narrows", "My doorway is actually 28 inches wide.", {"doorway_in": 28}),
        ("gentle-to-strong", "I prefer firm massage now, not gentle pressure.", {"intensity": "Strong"}),
        ("strong-to-medium", "I prefer medium pressure, not firm pressure.", {"intensity": "Balanced"}),
    ]:
        dialogue(cid, "correction_retention", [base, correction, "Please recommend using my updated conditions."], expected,
                 final_no_products=cid in {"budget-drops", "door-narrows"}, level="http_api" if cid in {"budget-drops", "weight-corrects"} else "agent")

    for cid, interruption in [
        ("white-glove", "How does White Glove delivery work?"),
        ("returns", "What is your return policy?"),
        ("stock", "Is the Osaki Solo Flex in stock?"),
        ("comparison", "Compare Maestro and Paragon."),
        ("showroom", "Where is your showroom?"),
        ("part", "Can I buy a replacement remote?"),
    ]:
        dialogue("interrupt-" + cid, "task_interruptions", [base, interruption, "Continue my chair recommendation using my saved limits."],
                 {"height_in": 68, "weight_lb": 170, "budget_max_usd": 4000, "doorway_in": 36, "doorway_fit": "assembled"},
                 level="http_api" if cid in {"returns", "showroom"} else "agent")

    for cid, fact_question in [
        ("other-price", "Is the Maestro's price $8999?"),
        ("other-weight", "Does the Maestro have a 350 pound capacity?"),
        ("delivery-fee", "Is White Glove shipping $500?"),
        ("chair-height", "Is the chair 190 centimeters high?"),
        ("product-width", "Does the chair require a minimum doorway of 35 inches?"),
    ]:
        dialogue("facts-" + cid, "product_facts_do_not_override", [base, fact_question, "Use my own measurements and budget for recommendations."],
                 {"height_in": 68, "weight_lb": 170, "budget_max_usd": 4000, "doorway_in": 36})

    for cid, narrow in [("below-products", "$100"), ("zero-budget", "$0"), ("under-fivehundred", "$499")]:
        dialogue("nofit-" + cid, "no_fit_integrity", [base, "My new maximum budget is " + narrow + ".", "Can you make an exception?", "Then show only chairs that meet my budget."],
                 {"budget_max_usd": float(narrow[1:])}, final_no_products=True, level="http_api")

    for cid, rejected, wanted in [
        ("exclude-four-five", "Do not recommend 4D or 5D. I want 3D.", "3D"),
        ("change-to-four", "I want 4D now, not 3D.", "4D"),
        ("change-back-three", "Actually I want 3D rather than 4D.", "3D"),
    ]:
        dialogue("mechanism-" + cid, "mechanism_dialogue", [base, rejected, "Please keep my mechanism preference when choosing."], {"mechanism": wanted})

    for cid, text in [
        ("person", "Please connect me to a person."),
        ("specialist", "I want to talk to a sales specialist."),
        ("callback", "Can a person call me back?"),
    ]:
        dialogue("handoff-" + cid, "human_handoff", [base, text], {"budget_max_usd": 4000, "weight_lb": 170}, expected_handoff=True)

    # More than five customer turns: later changes cannot erase untouched constraints.
    for cid, later, expected in [
        ("six-turn-narrow", "Actually my doorway is only 27 inches, assembled only.", {"height_in": 70, "weight_lb": 185, "budget_max_usd": 3500, "doorway_in": 27, "doorway_fit": "assembled"}),
        ("six-turn-budget", "Actually my maximum budget is now $350.", {"height_in": 70, "weight_lb": 185, "budget_max_usd": 350, "doorway_in": 34, "doorway_fit": "assembled"}),
        ("six-turn-weight", "Actually the buyer weighs 380 pounds.", {"height_in": 70, "weight_lb": 380, "budget_max_usd": 3500, "doorway_in": 34, "doorway_fit": "assembled"}),
    ]:
        dialogue(cid, "long_interview_state", ["Recommend a chair with a maximum $3500 budget.", "I am five feet ten inches tall.", "I weigh 185 pounds.", "My doorway is 34 inches, assembled only.", "I want shoulder massage.", "How does delivery work?", later, "Recommend with all my exact limits."], expected,
                 final_no_products=True, level="http_api")

    for cid, change, expected in [
        ("allow-disassembly", "Disassembly is allowed now.", "disassembled"),
        ("no-disassembly", "No disassembly. It must stay fully assembled.", "assembled"),
    ]:
        dialogue("access-" + cid, "access_permission_changes", [base, change, "Use the current delivery access preference."],
                 {"doorway_in": 36, "doorway_fit": expected})

    # Optional demands must not be silently converted to a single hard demand.
    def optional_mechanism(_):
        actual = extract_constraints("Either 3D or 4D would be fine; I do not have a preference.")
        checked = []
        for mechanism in ("3D", "4D"):
            candidate = SimpleNamespace(display_name="Synthetic option " + mechanism, title="Synthetic option " + mechanism,
                                        price_usd=2000, massage_mechanism=mechanism)
            issues = product_constraint_issues(candidate, actual)
            checked.append({"candidate_mechanism": mechanism, "rejection_reasons": issues})
        require(not any("mechanism" in row["rejection_reasons"] or "excluded_mechanism" in row["rejection_reasons"] for row in checked),
                "Either allowed mechanism must remain eligible for comparison; selecting one actual recommendation is permitted",
                expected="neither allowed mechanism excluded by hard constraint", observed={"saved_constraints": actual, "candidate_checks": checked})
        return {"constraints": actual, "candidate_checks": checked}
    add("sl-either-mechanism", "sales_limits", "preference_scope", "Preserve explicitly alternative product preferences", optional_mechanism,
        "Either 3D or 4D would be fine; I do not have a preference.", severity="medium")

    def no_longer_budget(_):
        actual = extract_constraints("My budget is no longer a restriction.", {"budget_max_usd": "3000"})
        require(actual.get("budget_max_usd") in (None, ""), "An explicitly withdrawn limit must not remain enforced",
                expected="no active cap", observed=actual.get("budget_max_usd"))
        return {"constraints": actual}
    add("sl-withdraw-budget", "sales_limits", "preference_scope", "Allow the customer to remove an earlier hard budget", no_longer_budget,
        {"message": "My budget is no longer a restriction.", "previous": {"budget_max_usd": "3000"}}, severity="medium")
    return out
