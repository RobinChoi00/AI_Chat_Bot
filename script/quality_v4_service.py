"""Continued Sales service role-plays: wrong replies remain in the trace.

These conversations exercise actual isolated local APIs with no external model,
stock or notification delivery. Customer rephrases are scripted/adaptive; no bot
answer is invented and no semantic failure terminates the remaining turns.
"""
from __future__ import annotations

import re

from script.quality_v4_support import Journey


def contract(*groups, allowed=(), department=None, no_interview=True, fact=None):
    return {"groups": groups, "allowed": allowed, "department": department,
            "no_interview": no_interview, "fact": fact}


def sales_step(*groups):
    return contract(*groups, department="sales")


def warranty_step(*groups):
    return contract(*groups, allowed=("warranty_redirect", "parts_technician"), department="warranty")


class ServiceJourney:
    def __init__(self, client, sid, planned, expected_model=None):
        self.client, self.sid, self.planned = client, sid, planned
        self.transcript, self.checks, self.recovery_attempts = [], [], []
        self.results = []
        self.covered = set()
        self.first_misunderstanding = None
        self.expected_model = expected_model
        self.final_model_correct = True
        response = client.post(f"/api/v1/warranty/session/{sid}/consent", json={
            "domain": "osakiusa.com", "policy_store": "osakiusa"})
        self.transcript.append({"setup": True, "method": "POST",
            "endpoint": f"/api/v1/warranty/session/{sid}/consent",
            "request": {"domain": "osakiusa.com", "policy_store": "osakiusa"},
            "http_status": response.status_code, "response": response.json()})
        if response.status_code != 200:
            raise RuntimeError("Valid synthetic consent setup failed")

    def record_check(self, dimension, name, passed, expected, observed, turn):
        self.checks.append({"dimension": dimension, "name": name,
            "status": "pass" if passed else "fail", "turn": turn,
            "expected": expected, "observed": observed})

    def say(self, step):
        from sales_models import get_session_collected

        turn = len(self.results) + 1
        message = step["message"]
        prior_failed = False
        previous = step.get("recovery_from")
        if previous is not None and len(self.results) >= previous:
            prior_failed = not self.results[previous - 1]["on_task"]
            # The customer makes a focused correction after a bad answer; when
            # already understood, ask the equivalent confirmation instead.
            if prior_failed and step.get("rephrase"):
                message = step["rephrase"]
        body = {"session_id": self.sid, "domain": "osakiusa.com", "message": message}
        offered = True
        if step.get("payload"):
            previous_payload = (self.transcript[-1].get("response") or {}) if self.transcript else {}
            offered = any(choice.get("payload") == step["payload"]
                          for choice in previous_payload.get("quick_replies", []))
            if offered:
                body["payload"] = step["payload"]
            else:
                # Do not manufacture an unavailable UI action or force hidden
                # state. Ask for the missing option as a real customer would.
                message = "I cannot find that visit option. How can I request the showroom visit window?"
                body["message"] = message
        response = self.client.post("/api/v1/sales/chat", json=body)
        try:
            payload = response.json()
        except ValueError:
            payload = {"reply": response.text}
        text = str(payload.get("reply") or "")
        lower = text.casefold()
        saved = get_session_collected(self.sid)
        spec = step["contract"]
        reasons = []
        if response.status_code != 200 or not text.strip():
            reasons.append("No usable customer response")
        if not offered:
            reasons.append("The customer-selected button was not offered by the previous actual response")
        if spec["allowed"] and payload.get("intent") not in spec["allowed"]:
            reasons.append("The reply follows a different business task")
        for group in spec["groups"]:
            if not any(term.casefold() in lower for term in group):
                reasons.append("Missing answer to requested topic: " + " / ".join(group))
        if spec["no_interview"] and re.search(r"what'?s the \*\*user height|tap a range|main (?:focus|goal).*massage", lower):
            reasons.append("Restarted an unrelated new-chair fit interview")
        # Both phone numbers are in a directory at the end of many replies.
        # Judge the primary next step, rather than crediting that directory.
        primary = re.split(r"\*\*Sales\s*\(delivery", text, flags=re.I)[0].casefold()
        if spec["department"] == "sales" and not any(term in primary for term in
                ("sales", "cancel / refund", "order number", "ventas", "영업", "세일즈")):
            reasons.append("The primary next step did not identify Sales")
        if spec["department"] == "warranty" and not any(term in primary for term in
                ("warranty", "service@osakititan.com", "technical support", "garantía", "garantia", "보증")):
            reasons.append("The primary next step did not identify Warranty")
        fact = spec.get("fact")
        if fact == "return_shipping":
            cost_responsibility = bool(re.search(r"pay|responsib|deduct|non.?refundable|expense|cost", lower))
            both_directions = bool(re.search(r"both (?:ways|directions)|original.{0,100}return|return.{0,100}original", lower, re.S))
            if not ("shipping" in lower and cost_responsibility and both_directions):
                reasons.append("Public return-cost answer must explain the original and return shipping responsibility")
        elif fact == "labor_coverage":
            if not ("labor" in lower and re.search(r"first year|year\s*1\b|1st year", lower)):
                reasons.append("Public coverage answer must identify the published first-year labor limit")
        elif fact == "label_location":
            if not (re.search(r"model.{0,40}(?:label|sticker)|serial.{0,40}(?:label|sticker)|nameplate", lower)
                    and re.search(r"chair|back|rear|under|side", lower)):
                reasons.append("Owner needs actual model/serial label guidance, not only a generic support referral")
        elif fact == "showroom_floor":
            if not (re.search(r"stock|display|floor|availability", lower)
                    and re.search(r"confirm|check|specialist|call|sales", lower)):
                reasons.append("Showroom floor availability needs a specific staff-check route, not generic visit hours")
        elif fact == "preferred_window":
            retained = str(saved.get("pending_showroom_window") or "").casefold()
            if not ("saturday" in retained or re.search(r"(?:request|got it).{0,100}saturday|saturday.{0,100}(?:request|confirm|email)", lower, re.S)):
                reasons.append("The stated Saturday window must be acknowledged or retained; a generic showroom brochure is insufficient")
        on_task = not reasons
        self.transcript.append({"setup": False, "turn": turn, "customer": message,
            "method": "POST", "endpoint": "/api/v1/sales/chat", "request": body,
            "http_status": response.status_code, "response": payload, "saved_state": saved})
        result = {"turn": turn, "message": message, "on_task": on_task,
                  "intent": payload.get("intent"), "reply": text, "reasons": reasons,
                  "goal_key": step["goal_key"]}
        self.results.append(result)
        self.record_check("understanding", step["goal_key"], on_task,
            {"customer_goal": step["goal_key"], "contract": spec},
            {"intent": payload.get("intent"), "reply": text, "reasons": reasons}, turn)
        if on_task:
            self.covered.add(step["goal_key"])
        elif self.first_misunderstanding is None:
            self.first_misunderstanding = {"turn": turn, "customer": message,
                "bot_reply": text, "expected_goal": step["goal_key"], "reasons": reasons}

        unverified = re.search(
            r"\byour (?:refund|return|replacement|claim) (?:is|has been) (?:approved|issued)\b|"
            r"\byour order has been cancel[le]+d\b|\ba technician has been booked\b|"
            r"\byour appointment is confirmed\b|\bdelivery is guaranteed (?:tomorrow|by)\b", text, re.I)
        if unverified and self.first_misunderstanding is None:
            self.first_misunderstanding = {"turn": turn, "customer": message, "bot_reply": text,
                "expected_goal": "No unverified completed action or approval", "dimension": "accuracy"}
        self.record_check("accuracy", "No unverified approval, booking or completed action",
            not bool(unverified), "A stated policy, limit or pending staff-confirmation step",
            text, turn)
        if self.expected_model:
            displayed = [str(item.get("model") or "") for item in payload.get("products", [])]
            primary = str(saved.get("pending_primary") or "")
            model_correct = all(self.expected_model.casefold() in name.casefold() for name in displayed)
            if primary:
                model_correct = model_correct and self.expected_model.casefold() in primary.casefold()
            self.final_model_correct = model_correct
            self.record_check("context", "Preserve the customer's named showroom model", model_correct,
                self.expected_model, {"displayed_models": displayed, "pending_primary": primary}, turn)
            if not model_correct and self.first_misunderstanding is None:
                self.first_misunderstanding = {"turn": turn, "customer": message, "bot_reply": text,
                    "expected_goal": "Preserve the requested product identity", "dimension": "context"}

        if step.get("context"):
            self.record_check("context", step["context"], on_task,
                "Keep the previous service topic or explicit correction", result, turn)
        if previous is not None:
            recovered = on_task if prior_failed else None
            self.recovery_attempts.append({"turn": turn, "from_turn": previous,
                "action": message, "needed": prior_failed, "recovered": recovered,
                "evidence": {"intent": payload.get("intent"), "reply": text,
                             "goal_key": step["goal_key"], "reasons": reasons}})
            if prior_failed:
                self.record_check("recovery", step["goal_key"], on_task,
                    "A focused rephrase restores the requested business task", result, turn)
        return payload

    def finish(self, required):
        missing = [key for key in required if key not in self.covered]
        if self.expected_model and not self.final_model_correct:
            missing.append("correct_named_showroom_model")
        accuracy_failures = [row for row in self.checks if row["dimension"] == "accuracy" and row["status"] == "fail"]
        if not missing and not accuracy_failures:
            outcome = "goal_met"
            reason = "Every required customer goal was addressed at least once; earlier wrong replies remain recorded."
        elif self.covered:
            outcome = "partially_met"
            reason = "Some goals or a useful responsible-team next step were reached, but required goals remain: " + ", ".join(missing)
        else:
            outcome = "not_met"
            reason = "No required business goal received an on-topic answer or valid responsible-team next step."
        return {"transcript": self.transcript, "checks": self.checks,
            "first_misunderstanding": self.first_misunderstanding,
            "final_outcome": outcome, "goal_reason": reason,
            "required_goals": required, "covered_goals": sorted(self.covered), "unmet_goals": missing,
            "customer_turns": len(self.results), "planned_customer_turns": len(self.planned),
            "unrun_customer_turns": len(self.planned) - len(self.results),
            "recovery_attempts": self.recovery_attempts}


def step(message, goal_key, spec, *, context=None, recovery_from=None, rephrase=None, payload=None):
    return {"message": message, "goal_key": goal_key, "contract": spec,
            "context": context, "recovery_from": recovery_from, "rephrase": rephrase, "payload": payload}


def journeys():
    rows = []

    def add(cid, persona, goal, steps, required, *, expected_model=None):
        def run(client):
            conversation = ServiceJourney(client, "v4-service-" + cid, steps, expected_model=expected_model)
            for item in steps:
                conversation.say(item)
            return conversation.finish(required)
        rows.append(Journey("v4-service-" + cid, "sales_service", persona, goal, run, steps))

    add("stairs-hawaii", "A shopper planning delivery to a third-floor Hawaii apartment",
        "Understand delivery services, access costs and the boundary on a promised arrival date", [
        step("I am thinking of buying a massage chair. How long does delivery normally take?", "shipping_times", contract(("2 weeks", "two weeks"), ("3 weeks", "three weeks"))),
        step("Does White Glove bring it inside and assemble it?", "white_glove", contract(("inside", "in-home"), ("assembl",))),
        step("My apartment is on the third floor without an elevator. Are stairs an extra charge?", "stairs_cost", contract(("stair", "third person"), ("extra", "charge", "fee"))),
        step("I mean the stair-carry fee, not the chair price. Does White Glove include that?", "stairs_cost", contract(("stair", "third person"), ("extra", "charge", "fee")), context="Retain the stair-access question", recovery_from=3),
        step("The address is in Honolulu. Do you ship to Hawaii?", "hawaii_freight", contract(("hawaii",), ("pay", "freight"))),
        step("Is that freight free there?", "hawaii_freight", contract(("pay", "not", "quote", "freight")), context="There means Hawaii from the previous turn"),
        step("Can you guarantee delivery to that apartment by next Friday?", "eta_boundary", contract(("can't", "cannot", "confirm", "promise"))),
        step("Please tell me how to have Sales check the Hawaii freight, stairs and date before I order.", "sales_delivery_step", sales_step(("email", "call", "order", "confirm"))),
        ], ["shipping_times", "white_glove", "stairs_cost", "hawaii_freight", "eta_boundary", "sales_delivery_step"])

    add("return-to-refund", "A buyer who first checks the policy, then explains the chair was delivered",
        "Learn return costs and obtain the correct unconfirmed Sales return/refund next step", [
        step("Before buying, what is the return policy for a chair I do not like?", "return_window", contract(("30",), ("shipping",))),
        step("Would I pay shipping both ways if I return it?", "return_shipping", contract(("shipping",), fact="return_shipping"), context="Keep the return-cost topic"),
        step("I am asking about return shipping costs, not delivery options. What costs are deducted?", "return_shipping", contract(("shipping",), fact="return_shipping"), recovery_from=2),
        step("Actually I received the chair yesterday and want to return it.", "actual_return", sales_step(("return", "refund", "RMA"), ("email", "sales", "order"))),
        step("I want a refund for my delivered chair, not a new recommendation.", "actual_return", sales_step(("refund", "return"), ("email", "sales", "order")), recovery_from=4),
        step("How soon will the money be back?", "refund_timing", sales_step(("refund", "review", "confirm", "agent")), context="Keep the actual refund topic"),
        step("I mean my refund processing time. Can Sales check it rather than ask my height?", "refund_timing", sales_step(("refund", "review", "confirm", "agent")), recovery_from=6),
        step("What should I provide to Sales so they can review my return request?", "return_verification", sales_step(("order", "email", "return", "refund"))),
        ], ["return_window", "return_shipping", "actual_return", "refund_timing", "return_verification"])

    add("coverage-remote", "An owner who moves from warranty coverage questions to a broken remote",
        "Understand labor coverage and obtain Warranty service without an automatic replacement approval", [
        step("What does the warranty on a new massage chair cover?", "coverage", contract(("parts",), ("labor",))),
        step("Does that mean free labor for all three years?", "labor_limit", contract(("labor",), fact="labor_coverage"), context="Keep the warranty labor topic"),
        step("I mean the labor coverage specifically, not a repair claim. Which years cover labor?", "labor_limit", contract(("labor",), fact="labor_coverage"), recovery_from=2),
        step("I already own an OS-Champ. The remote screen is black and the buttons do not work.", "remote_fault", warranty_step()),
        step("The remote is broken. I am not asking whether you have a black chair in stock.", "remote_fault", warranty_step(), recovery_from=4),
        step("Can I get a replacement remote for the chair?", "parts_review", warranty_step(("service", "parts", "warranty"))),
        step("Is the replacement approved and free already?", "approval_boundary", contract(("confirm", "review", "warranty", "cannot", "not")), context="A service route does not approve replacement coverage"),
        step("Please give me the Warranty team's contact so they can check the remote and coverage.", "warranty_contact", warranty_step(("service@", "freshdesk", "ext. 3"))),
        ], ["coverage", "labor_limit", "remote_fault", "parts_review", "approval_boundary", "warranty_contact"])

    add("order-changes", "An existing buyer trying to correct an order and get an invoice",
        "Keep existing-order actions with Sales and distinguish review from completed changes", [
        step("I need to change the shipping address on my order.", "address_change", sales_step(("order number",), ("email",))),
        step("I also want to change the ordered chair from brown to black.", "color_change", sales_step(("order",), ("review", "change", "confirm", "email")), context="Black is an existing-order change, not new stock browsing"),
        step("This is an order change before shipping, not a repair or warranty claim. Can Sales review it?", "color_change", sales_step(("review", "order", "change", "confirm")), recovery_from=2),
        step("If the change is impossible, please cancel my order.", "cancel_review", sales_step(("cancel", "refund"), ("email", "review", "agent"))),
        step("So has my order been cancelled now?", "cancel_boundary", sales_step(("not", "confirm", "review", "cancel", "agent")), context="Keep the cancellation-status question"),
        step("I am asking whether Sales must approve the cancellation, not shopping for another chair.", "cancel_boundary", sales_step(("confirm", "review", "cancel", "agent", "email")), recovery_from=5),
        step("Could Sales send me a copy of the invoice for the existing order?", "invoice_step", sales_step(("invoice", "order", "email"))),
        step("Please connect me to a Sales person to review the address, cancellation and invoice request.", "sales_order_review", sales_step(("email", "order", "review", "confirm"))),
        ], ["address_change", "color_change", "cancel_review", "cancel_boundary", "invoice_step", "sales_order_review"])

    add("showroom-visit", "A local shopper checking a specific chair and arranging a visit",
        "Get location and a truthful showroom availability/appointment confirmation route", [
        step("Where is your showroom so I can try a chair?", "showroom_location", contract(("Carrollton", "1001 W Crosby"))),
        step("Will Solo Flex definitely be there to try today?", "floor_availability", contract(("confirm", "check", "availability", "specialist"), fact="showroom_floor"), context="Keep the showroom-floor availability topic"),
        step("How would I confirm the showroom has Solo Flex before visiting?", "floor_availability", contract(("confirm", "check", "availability", "specialist"), fact="showroom_floor"), recovery_from=2,
             rephrase="I mean showroom floor availability for Solo Flex, not the chair's specs. Can a specialist confirm it?"),
        step("Can I book a showroom visit on Saturday afternoon?", "visit_request", contract(("request", "confirm", "specialist", "window"))),
        step("Saturday afternoon is my preferred visit window.", "preferred_window", contract(("request", "confirm", "email", "window"), fact="preferred_window"), context="Retain the proposed visit window"),
        step("Please show me the showroom visit options again.", "visit_request", contract(("request", "showroom"))),
        step("I will tap Request a visit.", "visit_request", contract(("window",), ("confirm", "request")), payload="cta:showroom:book"),
        step("I will tap the offered Saturday window.", "preferred_window", contract(("Saturday",), ("request", "email"), fact="preferred_window"), recovery_from=5, payload="cta:showroom:window:saturday"),
        step("Is that a confirmed appointment or only a request so far?", "appointment_boundary", contract(("not", "request", "confirm", "specialist"))),
        step("Can Sales confirm whether Solo Flex is on display and available for same-day pickup before I drive there?", "floor_availability", contract(("confirm", "check", "specialist"), department="sales", fact="showroom_floor"), context="The visit still requires a live showroom-stock check"),
        ], ["showroom_location", "floor_availability", "visit_request", "preferred_window", "appointment_boundary"], expected_model="Solo Flex")

    add("payment-promotion", "A prospective buyer asking financing, a declined card and a promotion",
        "Provide published payment guidance without inventing APR, bank reasons or discounts", [
        step("Can I pay monthly for a massage chair?", "monthly_checkout", contract(("Affirm", "monthly"), ("checkout",))),
        step("What APR will I personally get?", "apr_boundary", contract(("approval", "checkout", "depend", "won't quote", "specialist"))),
        step("I mean whether financing approval decides the rate; please do not guess my APR.", "apr_boundary", contract(("approval", "checkout", "depend", "won't", "specialist")), recovery_from=2),
        step("I tried buying but my credit card was declined. What should I do?", "declined_card", contract(("payment", "checkout", "specialist", "sales"))),
        step("I am not asking for a different chair. Can Sales help check the checkout problem?", "declined_card", sales_step(("checkout", "payment", "specialist", "email")), recovery_from=4),
        step("Do you have a coupon code or authorized promotion?", "promotion_boundary", contract(("coupon", "discount", "promotion"), ("specialist", "confirm", "sales", "email"))),
        step("The Costco price looked lower. Can someone compare the exact model rather than assume my budget?", "price_gap", sales_step(("compare", "difference", "model", "email", "confirm"))),
        step("Please have a Sales person review the payment issue and any eligible promotion.", "payment_staff_step", sales_step(("email", "payment", "sales", "review"))),
        ], ["monthly_checkout", "apr_boundary", "declined_card", "promotion_boundary", "price_gap", "payment_staff_step"])

    add("gifted-chair-support", "Someone given a used chair who needs instructions and replacement parts",
        "Keep owner-help and parts with Warranty rather than selling a new chair", [
        step("My father gave me his Hypnos chair. Can I get its user manual?", "owner_manual", warranty_step(("manual", "service", "warranty"))),
        step("I own it already. I just need the manual link for the same Hypnos.", "owner_manual", warranty_step(("manual", "service", "warranty")), recovery_from=1),
        step("Where should I look for the serial or model label?", "label_help", contract(("label", "serial", "model"), fact="label_location"), context="Retain the existing-owner support task"),
        step("The tablet remote is dead, so this is a repair question.", "dead_remote", warranty_step()),
        step("Can I buy a replacement tablet remote?", "buy_remote", warranty_step(("parts", "service", "warranty"))),
        step("Would a technician be needed to install the replacement?", "technician_review", warranty_step(("technician", "warranty", "service"))),
        step("So a technician has not been booked yet, correct?", "visit_boundary", contract(("not", "confirm", "review", "support", "warranty", "service")), context="A support route is not a completed technician booking"),
        step("Please tell me the Warranty support contact for the manual and dead remote.", "warranty_owner_contact", warranty_step(("service@", "ext. 3", "freshdesk"))),
        ], ["owner_manual", "label_help", "dead_remote", "buy_remote", "technician_review", "visit_boundary", "warranty_owner_contact"])

    add("bilingual-refund", "A Spanish-speaking owner who switches to English after confusion",
        "Recover the existing-order refund task across a customer-requested language change", [
        step("¿Dónde está mi pedido? Quiero revisar la entrega.", "spanish_order_route", contract(("pedido", "entrega", "ventas", "sales"))),
        step("Ya recibí la silla, pero quiero devolverla.", "spanish_return", contract(("devol", "reemb", "ventas", "sales")), context="The customer now wants a delivered-chair return"),
        step("Necesito un reembolso de mi silla, no comprar otra.", "spanish_return", contract(("reemb", "devol", "ventas", "sales")), recovery_from=2),
        step("Please answer in English. I want to return my delivered chair.", "english_return", sales_step(("return", "refund"), ("email", "sales", "order"))),
        step("How long would my refund take?", "refund_followup", sales_step(("refund", "review", "confirm", "agent")), context="Keep the refund topic after switching language"),
        step("I am asking about my refund status, not a new chair recommendation. Can Sales check?", "refund_followup", sales_step(("refund", "review", "confirm", "email", "agent")), recovery_from=5),
        step("Could I speak to a Sales agent about this return?", "sales_refund_contact", sales_step(("email", "sales", "return", "refund"))),
        step("What order details should I provide for Sales to review the refund?", "refund_verification", sales_step(("order", "email", "refund", "review"))),
        ], ["spanish_order_route", "spanish_return", "english_return", "refund_followup", "sales_refund_contact", "refund_verification"])
    return rows
