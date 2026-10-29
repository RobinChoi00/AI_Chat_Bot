"""Continued ordinary Warranty journeys: observe mistakes and try to recover.

Every reply is returned by the real isolated API. The scripted customer can
rephrase or press an actually offered button; neither action changes app state
outside that public API. No model, shipment, notification, or repair is faked.
"""
from __future__ import annotations

import re

from script.quality_v4_support import Journey

DOMAIN = "osakiusa.com"


class Conversation:
    def __init__(self, client, sid, model="OS-4000T"):
        self.client, self.sid, self.expected_model = client, sid, model
        self.transcript, self.checks, self.recovery_attempts = [], [], []
        self.payload, self.ticket_id = {}, None
        self.turn = 0

    def stored(self):
        from warranty_workflow import WarrantyEngine
        ticket = WarrantyEngine.get_ticket(self.ticket_id) if self.ticket_id else None
        if ticket is None:
            return None
        return {"ticket_id": str(ticket.ticket_id), "model": str(ticket.model_name or ""),
                "issue_type": str(ticket.issue_type or ""), "status": str(ticket.status),
                "collected": ticket.get_collected()}

    def text(self):
        ticket = self.payload.get("ticket") or {}
        smart = self.payload.get("smart_start") or {}
        routing = smart.get("routing_confirmation") or self.payload.get("routing_confirmation") or {}
        return str(self.payload.get("assistant_message") or ticket.get("customer_message")
                   or routing.get("message") or (ticket.get("current_node") or {}).get("prompt")
                   or self.payload.get("detail") or "")

    def node(self):
        return str((((self.payload.get("ticket") or {}).get("current_node") or {}).get("node_id")) or "")

    def options(self):
        node = (self.payload.get("ticket") or {}).get("current_node") or {}
        return {str(row.get("answer_key")): row for row in node.get("options") or []}

    def proposed(self):
        return str((self.payload.get("suggested_option") or self.payload.get("interpreted_option") or {}).get("answer_key") or "")

    def check(self, dimension, name, condition, expected=None, observed=None):
        self.checks.append({"turn": self.turn, "dimension": dimension, "name": name,
                            "status": "pass" if condition else "fail", "expected": expected,
                            "observed": observed})
        return bool(condition)

    def request(self, action, body=None, *, customer=None, setup=False, method="post", mode="typed", allowed=(200,)):
        if action.startswith("/"):
            path = action
        elif action in {"answer", "back", "troubleshooting-outcome"}:
            path = f"/api/v1/warranty/{self.ticket_id}/{action}"
        else:
            path = f"/api/v1/warranty/session/{self.sid}" + (f"/{action}" if action else "")
        if customer is not None and not setup:
            self.turn += 1
        response = getattr(self.client, method)(path, **({"json": body} if body is not None else {}))
        try:
            payload = response.json()
        except ValueError:
            payload = {"text": response.text}
        if isinstance(payload, dict) and payload.get("ticket"):
            self.ticket_id = payload["ticket"]["ticket_id"]
        self.payload = payload if isinstance(payload, dict) else {"response": payload}
        self.transcript.append({"turn": self.turn, "customer": customer, "input_mode": mode,
            "setup": setup, "method": method, "endpoint": path, "request": body,
            "status": response.status_code, "response": self.payload,
            "customer_visible_api_text": self.text(), "stored_state": self.stored()})
        if not setup:
            self.check("understanding", "Customer request remains actionable", response.status_code in allowed,
                       list(allowed), {"status": response.status_code, "reply": self.text()})
            snapshot = self.stored()
            if self.expected_model and snapshot:
                self.check("context", "Confirmed model remains attached to this case",
                           snapshot["model"] == self.expected_model, self.expected_model, snapshot["model"])
            self.check("accuracy", "No unverified approval, shipment, or booked callback",
                not re.search(r"your (?:warranty|claim|replacement) (?:is|has been) approved|"
                              r"we (?:have shipped|will ship|have sent)|callback (?:is|has been) booked", self.text(), re.I),
                "Review or pending status; no automatic approval", self.text())
        return self.payload

    def identity(self, model=True):
        self.request("consent", {"domain": DOMAIN, "policy_store": "osakiusa"}, setup=True)
        self.request("contact-email", {"customer_email": "continued.warranty@example.test"}, setup=True)
        if model and self.expected_model:
            self.request("register-model", {"model": self.expected_model, "domain": DOMAIN}, setup=True)

    def answer(self, text, *, mode="typed"):
        return self.request("answer", {"answer": text}, customer=text, mode=mode)

    def recover(self, action, recovered, evidence):
        self.recovery_attempts.append({"turn": self.turn, "action": action,
                                       "recovered": bool(recovered), "evidence": evidence})

    def begin(self, symptom, rephrase):
        self.request("smart-start", {"message": symptom, "domain": DOMAIN}, customer=symptom)
        suggested = (self.payload.get("smart_start") or {}).get("suggested_issue_type")
        recognized = self.check("understanding", "Initial ordinary symptom is recognized as a malfunction",
                                suggested == "defect", "defect suggestion with confirmation", suggested)
        if not recognized:
            self.request("natural-start", {"message": rephrase, "domain": DOMAIN}, customer=rephrase)
            found = self.payload.get("suggested_issue_type") == "defect"
            self.check("recovery", "A plain-language rephrase recovers the malfunction intent", found,
                       "defect suggestion", self.payload.get("suggested_issue_type"))
            self.recover("Customer rephrases the symptom", found, self.text())
        if self.node() in {"root", "issue_type"}:
            self.request("quick-start", {"issue_type": "defect", "domain": DOMAIN},
                         customer="I choose the chair malfunction / Defect option.", mode="button")
        self.check("context", "Customer's explicit Defect confirmation reaches the component menu",
                   self.node() == "defect_problem_type", "defect_problem_type", self.node())

    def button(self, key):
        offered = self.options()
        if key not in offered:
            self.recover("The planned customer button is not offered; no unavailable action was sent", False,
                         {"requested": key, "node": self.node(), "offered": list(offered)})
            return False
        label = str(offered[key].get("label") or key)
        self.request("answer", {"answer": key}, customer=label, mode="button")
        return True

    def observed_answer(self, message, key, correction):
        from warranty_workflow import _NODES

        previous_node = self.node()
        expected_available = key in self.options()
        expected_next = next((option.get("next") for option in _NODES.get(previous_node, {}).get("options", [])
                              if option.get("answer_key") == key), None)
        self.answer(message)
        proposed = self.proposed()
        self.check("accuracy", "A proposed or advanced answer must not reverse the customer's reported result",
                   (not proposed or proposed == key) and self.node() in {previous_node, expected_next},
                   {"key": key, "next_node": expected_next},
                   {"node": self.node(), "suggested": proposed, "reply": self.text()})
        matched = self.node() == expected_next or (self.node() == previous_node and proposed == key)
        self.check("understanding", "The completed check has an appropriate interpretation", matched,
                   {"key": key, "next_node": expected_next}, {"node": self.node(), "suggested": proposed, "reply": self.text()})
        if self.node() == previous_node and (not proposed or proposed != key):
            self.answer(correction)
            corrected = self.node() == expected_next or (self.node() == previous_node and self.proposed() == key)
            self.check("recovery", "A direct correction recovers the reported check", corrected,
                       key, {"node": self.node(), "suggested": self.proposed(), "reply": self.text()})
            self.recover("Customer corrects a wrong or unmapped interpretation", corrected, self.text())
        if self.node() == previous_node and expected_available:
            self.button(key)
            self.recover("Customer presses the actually offered correct option", self.node() == expected_next, self.node())

    def outcome(self, value, customer):
        if not self.node():
            # Outcome acknowledgements may be a small result object rather
            # than a full session. Read the real state before offering the
            # next control; do not infer a node or post a stale UI action.
            self.refresh()
        terminal = bool((((self.payload.get("ticket") or {}).get("current_node") or {}).get("is_terminal")))
        if not terminal:
            self.recover("Outcome control is not offered at this node; no unoffered action was sent", False,
                         {"outcome": value, "node": self.node()})
            return False
        self.request("troubleshooting-outcome", {"outcome": value}, customer=customer, mode="button")
        return True

    def support(self, message="I still need help. Please have the warranty team review this case."):
        self.request("handoff", {"message": message, "domain": DOMAIN}, customer=message, mode="support_request")
        requested = (self.payload.get("handoff") or {}).get("requested") is True
        self.check("recovery", "The explicit request reaches a recorded team-review request", requested,
                   "saved handoff requested", self.payload.get("handoff"))
        self.recover("Customer explicitly requests team review", requested, self.payload.get("handoff"))
        return requested

    def refresh(self):
        self.request("", method="get")

    def finish(self, goal_met, reason):
        first = next((row for row in self.checks if row["status"] == "fail"), None)
        meaningful = bool(goal_met)
        saved = self.stored() or {}
        partial = bool(saved.get("collected", {}).get("human_support_requested") or
                       (saved.get("issue_type") == "defect" and self.node().startswith("defect_")))
        return {"transcript": self.transcript, "checks": self.checks,
                "first_misunderstanding": first, "final_outcome": "goal_met" if meaningful else "partially_met" if partial else "not_met" if self.ticket_id else "blocked",
                "goal_reason": reason, "customer_turns": self.turn,
                "recovery_attempts": self.recovery_attempts, "stored_state": self.stored(),
                "mode": "no-OpenAI deterministic/fallback conversation; no actual chair repair measured"}


def power_journey(client):
    c = Conversation(client, "v4-w-power")
    c.identity()
    c.begin("My chair will not turn on.", "It has no power and will not start.")
    c.button("power")
    c.button("remote_off")
    c.observed_answer("I already checked the outlet and reconnected the cord. There is no clicking sound.",
                      "no_clicking", "No clicking sound and nothing resolved")
    correct_stage = c.node() == "defect_power_no_click_terminal"
    c.check("accuracy", "The case records no-clicking rather than a positive clicking diagnosis", correct_stage,
            "defect_power_no_click_terminal", c.node())
    terminal_before = c.node()
    requested = c.support()
    c.refresh()
    saved = c.stored() or {}
    return c.finish(correct_stage and requested and saved.get("model") == "OS-4000T",
                    f"Reached {terminal_before}; review request saved={requested}. This is not a confirmed repair or approval.")


def remote_journey(client):
    c = Conversation(client, "v4-w-remote")
    c.identity()
    c.begin("The remote screen is black and the chair does not respond.", "The remote has no power.")
    c.button("remote")
    c.button("no_power")
    c.observed_answer("The cable is firmly connected and the fuse is intact. It still will not turn on.",
                      "all_checked_ok", "All checked — nothing obvious found")
    correct_stage = c.node() == "defect_remote_pcb_check_terminal"
    c.check("context", "Intact fuse and completed cable checks end in review, not fuse replacement", correct_stage,
            "defect_remote_pcb_check_terminal", c.node())
    c.outcome("steps_completed", "I have completed the listed checks.")
    c.outcome("unresolved", "It is still not working.")
    unresolved = (c.stored() or {}).get("collected", {}).get("troubleshooting_outcome") == "unresolved"
    c.check("context", "The customer's still-broken result is saved", unresolved,
            "unresolved", (c.stored() or {}).get("collected", {}).get("troubleshooting_outcome"))
    requested = c.support("Please review the remote. I am not saying the fuse is blown.")
    return c.finish(correct_stage and requested and unresolved, "Correct completed-check branch, unresolved result and team-review request are required; no replacement has been approved.")


def airbag_journey(client):
    c = Conversation(client, "v4-w-air")
    c.identity()
    c.begin("The air bags in the arms stopped inflating.", "The arm airbags are not inflating.")
    c.button("air")
    c.button("arms")
    c.observed_answer("They worked before, but now neither arm inflates.", "yes_worked", "Yes, it worked before but stopped")
    relevant = c.node() == "defect_air_arms_hissing_q"
    c.check("understanding", "The next question concerns arm inflation rather than another chair part", relevant,
            "defect_air_arms_hissing_q", c.node())
    c.answer("I cannot safely remove a side panel or disconnect the hose. Can I have a person help?")
    not_fabricated = c.node() != "defect_air_pump_terminal"
    c.check("accuracy", "Being unable to perform a hose check must not become a reported no-air test result",
            not_fabricated, "No invented hissing/airflow observation", {"node": c.node(), "reply": c.text()})
    c.check("accuracy", "Unable to safely attempt internal checks should stop repeating panel/hose instructions",
            not re.search(r"please remove[^.\n]{0,60}panel|disconnect[^.\n]{0,60}hose", c.text(), re.I),
            "Pause unattemptable internal checks and offer team help", c.text())
    automatic = (c.payload.get("handoff") or {}).get("requested") is True
    c.check("recovery", "Unable-to-attempt plus a direct person request interrupts DIY checks", automatic,
            "saved human request", c.payload.get("handoff"))
    if not automatic:
        c.support("Please connect me with a warranty specialist; I cannot safely do that check.")
    c.request("handoff", {"message": "Please keep the arm-airbag problem and my model in the request.", "domain": DOMAIN},
              customer="Please keep the arm-airbag problem and my model in the request.", mode="support_request")
    requested = (c.payload.get("handoff") or {}).get("requested") is True
    c.refresh()
    return c.finish(relevant and requested and not_fabricated,
                    "A team request was saved, but an invented no-air/pump result leaves the goal only partially met." if not not_fabricated else
                    "Reached the relevant arm-air question, then recorded inability to perform it and a team-review request. No physical test or repair was performed.")


def heating_journey(client):
    c = Conversation(client, "v4-w-heat")
    c.identity()
    c.begin("The heater is not getting warm.", "The heating feature is not working.")
    c.button("heat")
    c.button("not_heating")
    c.observed_answer("I ran a program for twenty minutes with heat on, and it is still cold.",
                      "still_no_heat", "Yes, ran for 10+ minutes with heat ON — still no warmth")
    relevant = c.node() == "defect_heating_not_heating_terminal"
    c.outcome("steps_completed", "I already completed that warm-up test.")
    c.outcome("unresolved", "There is still no warmth, so it is not resolved.")
    unresolved = (c.stored() or {}).get("collected", {}).get("troubleshooting_outcome") == "unresolved"
    c.check("context", "The completed warm-up with no improvement remains unresolved", unresolved,
            "unresolved", (c.stored() or {}).get("collected", {}).get("troubleshooting_outcome"))
    c.check("accuracy", "A still-cold observation must not mark the chair self-resolved",
            (c.stored() or {}).get("status") != "self_resolved", "not self_resolved", (c.stored() or {}).get("status"))
    requested = c.support("Please have the team review the heating problem after my completed warm-up check.")
    return c.finish(relevant and requested and unresolved, "Correct heating path and unresolved observation retained for team review, not a claim that the heater was fixed.")


def rollers_journey(client):
    c = Conversation(client, "v4-w-rollers")
    c.identity()
    c.begin("The rollers make a clicking noise during the massage.", "The rolling massage mechanism is not working properly.")
    c.button("rolling")
    c.observed_answer("It pops and clicks while it is massaging my back.", "pops", "Pops or clicks during the massage")
    relevant = c.node() == "defect_rolling_pops_terminal"
    c.check("understanding", "Massage-time popping stays distinct from whole-chair power clicking", relevant,
            "defect_rolling_pops_terminal", c.node())
    c.outcome("unable_to_attempt", "I cannot safely do any more mechanical checks myself.")
    requested = c.support("Please review the roller popping. I can describe it but I have not repaired it.")
    c.refresh()
    return c.finish(relevant and requested, "Correct roller-noise review route and explicit team request; no internal repair or successful fix was simulated.")


def cosmetic_journey(client):
    c = Conversation(client, "v4-w-cosmetic")
    c.identity()
    c.request("smart-start", {"message": "The leather is torn on the armrest.", "domain": DOMAIN},
              customer="The leather is torn on the armrest.")
    initial = (c.payload.get("smart_start") or {}).get("suggested_issue_type")
    c.check("understanding", "Torn upholstery is recognized as a defect rather than ignored", initial == "defect",
            "defect suggestion", initial)
    c.request("quick-start", {"issue_type": "defect", "domain": DOMAIN},
              customer="I choose the chair malfunction / Defect option.", mode="button")
    c.answer("The leather is torn on the armrest, not an airbag problem.")
    recognized = c.node() == "defect_cosmetic_location" or c.proposed() == "cosmetic"
    c.check("understanding", "A torn surface is treated as cosmetic damage", recognized,
            "cosmetic option or location question", {"node": c.node(), "suggested": c.proposed(), "reply": c.text()})
    if not recognized:
        c.answer("No, that is cosmetic damage to the appearance, not an inflation issue.")
        c.recover("Customer corrects the wrong Air interpretation in natural language", c.proposed() == "cosmetic" or c.node() == "defect_cosmetic_location", c.text())
    if c.node() == "defect_air_location":
        c.request("back", {}, customer="I press Go back so I can choose the damaged surface instead.", mode="button")
        c.recover("Customer uses the actual Go back action after the wrong branch", c.node() == "defect_problem_type", c.node())
    if c.node() == "defect_problem_type":
        c.button("cosmetic")
    c.button("other")
    c.button("visible_at_unboxing")
    c.button("yes_white_glove")
    relevant = c.node() == "defect_cosmetic_wg_terminal"
    c.check("context", "Surface damage ends in evidence/review for the correct case", relevant,
            "defect_cosmetic_wg_terminal", c.node())
    requested = c.support("Please review the torn upholstery. I can provide photos; is coverage still subject to review?")
    return c.finish(relevant and requested, "Cosmetic evidence/review stage is reached after rephrasing or a genuine offered selection; no free repair or coverage approval is asserted.")


def unknown_model_journey(client):
    c = Conversation(client, "v4-w-unknown", model=None)
    c.identity(model=False)
    c.request("natural-start", {"message": "My chair is not working and I don't know the model.", "domain": DOMAIN},
              customer="My chair is not working and I don't know the model.", allowed=(200, 422))
    c.check("understanding", "The missing model gets a concrete information request", "model" in c.text().lower(),
            "model information request", c.text())
    c.request("natural-start", {"message": "Where can I find the model number on my chair?", "domain": DOMAIN},
              customer="Where can I find the model number on my chair?", allowed=(200, 422))
    c.check("understanding", "A model-location question gives useful locating guidance", bool(re.search(r"sticker|label|serial|back|underside", c.text(), re.I)),
            "where to look for the model label", c.text())
    c.expected_model = "OS-4000T"
    c.request("register-model", {"model": "OS-4000T", "domain": DOMAIN},
              customer="I found the label myself. It says OS-4000T.")
    c.recover("Customer independently finds and supplies the model label", (c.stored() or {}).get("model") == "OS-4000T", c.stored())
    c.begin("It shuts off after a few minutes.", "The chair has a power problem and stops working.")
    c.button("power")
    c.button("remote_on")
    c.button("no_response")
    relevant = c.node() == "defect_power_remote_replace_terminal"
    requested = c.support("Please review the power problem now that I have provided the model.")
    return c.finish(relevant and requested, "After independently finding the model, the customer reaches a power/remote review. Any missing label-location help remains a failed checkpoint.")


def person_after_loop_journey(client):
    c = Conversation(client, "v4-w-person")
    c.identity()
    c.begin("The chair's Bluetooth will not pair with my phone.", "I need help with a Bluetooth connection fault.")
    c.answer("None of these parts describes the Bluetooth issue.")
    c.answer("I have tried pairing again; I do not understand which option to choose.")
    c.answer("담당자와 상담하고 싶어요.")
    korean_requested = (c.payload.get("handoff") or {}).get("requested") is True
    c.check("understanding", "A Korean request for a person interrupts the repeated diagnostic menu", korean_requested,
            "saved human request", {"handoff": c.payload.get("handoff"), "reply": c.text()})
    c.answer("I want to talk to a person, please.")
    english_requested = (c.payload.get("handoff") or {}).get("requested") is True
    c.check("recovery", "Rephrasing the person request in English recovers a human route", english_requested,
            "saved human request", c.payload.get("handoff"))
    c.recover("Customer retries the human request in English", english_requested, c.text())
    c.request("handoff", {"message": "Could someone call me back about the same Bluetooth problem?", "domain": DOMAIN, "reason": "callback_request"},
              customer="Could someone call me back about the same Bluetooth problem?", mode="support_request")
    c.refresh()
    snapshot = c.stored() or {}
    requested = snapshot.get("collected", {}).get("human_support_requested") is True
    return c.finish(requested and snapshot.get("model") == "OS-4000T",
                    "A real support request and callback request are recorded for the same model; a callback is not booked or a Bluetooth fault repaired.")


def journeys():
    definitions = [
        ("power-rephrase", "Owner of an OS-4000T that will not start", "Record the completed no-clicking checks and reach the correct review step", power_journey,
         ["My chair will not turn on.", "It has no power and will not start.", "Defect", "Power", "Remote off", "There is no clicking sound.", "No clicking sound and nothing resolved", "Please have the team review."]),
        ("remote-completed-checks", "Owner with a blank remote and an intact fuse", "Keep intact-fuse facts and request appropriate remote review", remote_journey,
         ["The remote screen is black.", "Defect", "Remote", "No power", "Fuse intact and cable connected", "All checked; nothing found", "Steps completed", "Unresolved", "Please review remote"]),
        ("arm-airbag-unable", "Owner whose arm airbags stopped inflating", "Reach the arm-inflation question and hand off a check the customer cannot safely attempt", airbag_journey,
         ["Arm airbags stopped inflating", "Defect", "Air", "Arms", "Worked before", "I cannot safely remove a panel", "Please keep the model and problem"]),
        ("heating-still-cold", "Owner who already completed a 20-minute warm-up", "Record a still-unresolved heating complaint after the warm-up and request review", heating_journey,
         ["Heat does not warm up", "Heating not working", "Defect", "Heat", "Not heating", "Twenty minutes already; still cold", "Steps completed", "Unresolved", "Team review"]),
        ("roller-popping", "Owner hearing popping only while massage runs", "Keep massage-time popping distinct from power clicking and reach the correct review", rollers_journey,
         ["Rollers click during massage", "Rolling mechanism problem", "Defect", "Rolling", "Pops during massage", "Unable to attempt more", "Team review"]),
        ("cosmetic-rephrase", "Newly delivered chair has torn armrest upholstery", "Get to cosmetic evidence/review without claiming coverage or replacement approval", cosmetic_journey,
         ["Torn armrest leather", "Defect", "Torn leather, not airbag", "No, cosmetic not inflation", "Go back if wrong Air branch", "Cosmetic", "Other area", "Visible at unboxing", "White Glove", "Please review photos"]),
        ("unknown-model", "Owner initially does not know the model number", "Request model information, preserve a later supplied model, then reach the appropriate review", unknown_model_journey,
         ["Chair not working; model unknown", "Where is model label?", "Found OS-4000T", "Shuts off after minutes", "Power problem", "Defect", "Power", "Remote on", "No response", "Review"]),
        ("person-after-loop", "Owner cannot match a Bluetooth fault to the listed parts", "Escape repeated clarification and record a truthful support/callback request", person_after_loop_journey,
         ["Bluetooth will not pair", "Bluetooth connection fault", "Defect", "None match", "I do not understand choices", "담당자와 상담하고 싶어요", "Talk to a person", "Request callback"]),
    ]
    return [Journey("warranty-v4-" + cid, "warranty", persona, goal, run, planned)
            for cid, persona, goal, run, planned in definitions]
