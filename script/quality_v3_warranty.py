"""Everyday Warranty customer journeys, tested through real isolated HTTP APIs.

The no-OpenAI fallback is the measured mode. An unrecognized symptom here is
not evidence that a configured production model would misunderstand it.
"""
from __future__ import annotations

import json
import re

from script.quality_v2_support import CheckFailure, Scenario


DOMAIN = "osakiusa.com"


class WarrantyConversation:
    def __init__(self, client, sid):
        self.client, self.sid = client, sid
        self.transcript = []
        self.payload = {}
        self.ticket_id = None

    def check(self, condition, message, *, expected=None, observed=None):
        if not condition:
            raise CheckFailure(message, expected=expected, observed={
                "actual": observed,
                "transcript": self.transcript,
                "stored_state": self.stored(),
                "mode": "no-OpenAI keyword/rule fallback",
            })

    def stored(self):
        from warranty_workflow import WarrantyEngine

        ticket = WarrantyEngine.get_ticket(self.ticket_id) if self.ticket_id else None
        if ticket is None:
            return None
        return {"ticket_id": str(ticket.ticket_id), "model": str(ticket.model_name or ""),
                "issue_type": str(ticket.issue_type or ""), "status": str(ticket.status),
                "collected": ticket.get_collected()}

    def request(self, action, body=None, *, customer=None, setup=False, method="post", statuses=(200,)):
        if action.startswith("/"):
            path = action
        elif action in {"answer", "back", "troubleshooting-outcome"}:
            path = f"/api/v1/warranty/{self.ticket_id}/{action}"
        else:
            path = f"/api/v1/warranty/session/{self.sid}" + (f"/{action}" if action else "")
        response = getattr(self.client, method)(path, **({"json": body} if body is not None else {}))
        try:
            data = response.json()
        except ValueError:
            data = {"text": response.text}
        if isinstance(data, dict) and data.get("ticket"):
            self.ticket_id = data["ticket"]["ticket_id"]
        self.payload = data
        self.transcript.append({"customer": customer, "endpoint": path, "request": body,
                                "setup": setup, "status": response.status_code,
                                "response": data, "stored_state": self.stored()})
        self.check(response.status_code in statuses, "Customer request should receive an actionable response",
                   expected=list(statuses), observed={"status": response.status_code, "body": data})
        return data

    def identity(self, model="OS-4000T"):
        self.request("consent", {"domain": DOMAIN, "policy_store": "osakiusa"},
                     customer="I agree to the privacy notice.", setup=True)
        self.request("contact-email", {"customer_email": "everyday.warranty@example.test"},
                     customer="everyday.warranty@example.test", setup=True)
        if model is not None:
            self.request("register-model", {"model": model, "domain": DOMAIN},
                         customer=f"The model label says {model}.", setup=True)

    def defect(self, symptom=None):
        self.request("quick-start", {"issue_type": "defect", "domain": DOMAIN},
                     customer="I confirm this is a chair malfunction.")
        if symptom:
            self.answer(symptom)

    def answer(self, text):
        return self.request("answer", {"answer": text}, customer=text)

    def text(self):
        ticket = self.payload.get("ticket") or {}
        node = ticket.get("current_node") or {}
        return str(self.payload.get("assistant_message") or ticket.get("customer_message") or node.get("prompt") or "")

    def node(self):
        return str(((self.payload.get("ticket") or {}).get("current_node") or {}).get("node_id") or "")

    def evidence(self):
        return {"mode": "no-OpenAI keyword/rule fallback", "transcript": self.transcript,
                "stored_state": self.stored(), "customer_request_count": sum(not r["setup"] for r in self.transcript)}


def scenarios():
    rows = []

    def add(cid, category, family, title, run, inputs, severity="medium"):
        rows.append(Scenario("w3-" + cid, category, family, title, severity, run,
                             inputs=inputs, level="http_api"))

    # A clear symptom should at least be recognized as a malfunction suggestion.
    # Confirmation is retained; these do not demand automatic diagnosis/repair.
    symptoms = [
        ("no-power", "power_intake", "My chair will not turn on."),
        ("stopped", "power_intake", "The chair stopped working yesterday."),
        ("intermittent", "power_intake", "The chair shuts off after about five minutes."),
        ("remote-black", "remote_intake", "The remote screen is black."),
        ("remote-buttons", "remote_intake", "The remote buttons do not respond."),
        ("air", "air_intake", "The air bags do not inflate."),
        ("one-arm", "air_intake", "The right arm airbag stopped inflating."),
        ("backrest", "movement_intake", "The backrest will not recline."),
        ("footrest", "movement_intake", "The footrest is stuck."),
        ("heat", "function_intake", "The heater is not getting warm."),
        ("rollers", "function_intake", "The rollers make a clicking noise."),
        ("voice", "function_intake", "The voice commands stopped responding."),
        ("ko-power", "multilingual_intake", "의자가 작동하지 않아요."),
        ("ko-remote", "multilingual_intake", "리모컨 화면이 안 켜져요."),
        ("es-power", "multilingual_intake", "Mi silla no enciende."),
        ("es-air", "multilingual_intake", "Los airbags no se inflan."),
    ]
    for cid, family, symptom in symptoms:
        def run(client, cid=cid, symptom=symptom):
            c = WarrantyConversation(client, "w3-i-" + cid)
            c.identity()
            original = c.ticket_id
            data = c.request("smart-start", {"message": symptom, "domain": DOMAIN}, customer=symptom)
            smart = data.get("smart_start") or {}
            c.check(smart.get("suggested_issue_type") == "defect",
                    "A clear everyday malfunction should receive a malfunction confirmation in the offline fallback",
                    expected="defect suggestion, with customer confirmation retained", observed=smart)
            c.check(c.ticket_id == original and data["ticket"]["model_name"] == "OS-4000T",
                    "Describing symptoms must not discard the registered chair", observed=data["ticket"])
            c.check(data["ticket"]["current_node"]["node_id"] == "issue_type",
                    "Symptom recognition must not silently start an unconfirmed diagnostic path", observed=data["ticket"])
            return c.evidence()
        add("intake-" + cid, "warranty_intake", family, "Recognize: " + symptom, run, {"model": "OS-4000T", "message": symptom})

    for cid, message, expected in [
        ("assembly", "I just unpacked it. How do I assemble the footrest?", "installation"),
        ("loose-cable-setup", "I need help installing the side panels.", "installation"),
        ("box-damage", "The delivery box arrived crushed and the chair is damaged.", "delivery"),
        ("tracking", "Where is my order? I have not received the chair yet.", "delivery"),
    ]:
        def run(client, cid=cid, message=message, expected=expected):
            c = WarrantyConversation(client, "w3-i-" + cid)
            c.identity()
            data = c.request("smart-start", {"message": message, "domain": DOMAIN}, customer=message)
            actual = (data.get("smart_start") or {}).get("suggested_issue_type") or data["ticket"]["issue_type"]
            c.check(actual == expected, "Setup and delivery inquiries must remain distinct from product malfunction",
                    expected=expected, observed={"route": actual, "response": data})
            return c.evidence()
        add("intake-" + cid, "warranty_intake", "setup_delivery_boundary", message, run, {"message": message, "expected": expected})

    def unknown_model(client):
        c = WarrantyConversation(client, "w3-i-unknown")
        c.identity(model=None)
        d = c.request("natural-start", {"message": "My chair is not working and I don't know the model.", "domain": DOMAIN},
                      customer="My chair is not working and I don't know the model.", statuses=(422,))
        c.check("model" in str(d.get("detail", "")).lower(), "Missing model should receive a concrete model-information request", observed=d)
        return c.evidence()
    add("intake-unknown-model", "warranty_intake", "model_information", "Ask for a model before model-specific diagnosis", unknown_model,
        {"message": "My chair is not working and I don't know the model."})

    def combined_model(client):
        c = WarrantyConversation(client, "w3-i-model-in-message")
        c.identity(model=None)
        message = "My OS-4000T chair won't turn on."
        d = c.request("smart-start", {"message": message, "domain": DOMAIN}, customer=message)
        c.check("4000" in d["ticket"]["model_name"], "An explicitly stated model should not need to be typed again", expected="OS-4000T", observed=d["ticket"])
        c.check(bool(d.get("model_confirmation") or d["ticket"]["model_confirmed"]),
                "Extracted model identity must be confirmed before use", observed=d)
        return c.evidence()
    add("intake-model-in-message", "warranty_intake", "model_information", "Retain the model provided with the symptom", combined_model,
        {"message": "My OS-4000T chair won't turn on."})

    for cid, message in [("unclear", "Something feels different but I cannot tell what."),
                         ("very-vague", "There is something wrong with the chair.")]:
        def run(client, cid=cid, message=message):
            c = WarrantyConversation(client, "w3-i-" + cid)
            c.identity()
            d = c.request("natural-start", {"message": message, "domain": DOMAIN}, customer=message)
            c.check(c.node() == "issue_type" and bool(c.text().strip()),
                    "An unspecified problem should ask for clarification without inventing a failed part", observed=d)
            c.check(not ((d.get("ticket") or {}).get("current_node") or {}).get("is_terminal"),
                    "A vague problem is not a completed diagnosis", observed=d)
            return c.evidence()
        add("intake-" + cid, "warranty_intake", "clarify_vague_symptom", message, run, {"message": message})

    # The customer explicitly confirms Defect first; the following free-text
    # answer should identify the relevant family without demanding a key name.
    for cid, symptom, expected_node in [
        ("power", "The chair won't turn on.", "defect_power_remote_on_q"),
        ("remote", "The remote screen is black.", "defect_remote_power_q"),
        ("air", "The airbags are not inflating.", "defect_air_location"),
        ("recline", "The backrest will not recline.", "defect_recline_which"),
        ("footrest", "The footrest is stuck.", "defect_footrest_which"),
        ("heat", "The heater is not getting warm.", "defect_heating_which"),
        ("rolling", "The rollers make a clicking noise.", "defect_rolling_problem_type"),
        ("cosmetic", "The leather is torn on the armrest.", "defect_cosmetic_location"),
        ("voice", "The voice control does not respond.", "defect_voice_which"),
    ]:
        def run(client, cid=cid, symptom=symptom, expected_node=expected_node):
            c = WarrantyConversation(client, "w3-f-component-" + cid)
            c.identity()
            c.defect(symptom)
            d = c.payload
            # Non-exact language is deliberately allowed to produce a proposed
            # option, provided it is the right option and asks confirmation.
            suggested = d.get("suggested_option") or {}
            interpreted = d.get("interpreted_option") or {}
            expected_key = {"rolling": "rolling", "cosmetic": "cosmetic"}.get(cid, cid)
            c.check(c.node() == expected_node or suggested.get("answer_key") == expected_key or interpreted.get("answer_key") == expected_key,
                    "A normal symptom should lead to the relevant next diagnostic question or a correct confirmable choice",
                    expected={"diagnostic_question": expected_node, "or_suggested_key": expected_key}, observed=d)
            return c.evidence()
        add("followup-component-" + cid, "warranty_followup", "symptom_to_component", symptom, run,
            {"confirmed_issue": "defect", "message": symptom})

    # Three ordinary non-matches must lead to a person rather than an endless
    # repeated symptom menu; no impossible remote diagnosis is required.
    for cid, messages in [
        ("bluetooth", ["Bluetooth will not pair with my phone."] * 3),
        ("unknown-labels", ["I don't understand these choices."] * 3),
        ("no-matching-choice", ["None of those options match my issue."] * 3),
    ]:
        def run(client, cid=cid, messages=messages):
            c = WarrantyConversation(client, "w3-f-clarify-" + cid)
            c.identity()
            c.defect()
            for message in messages:
                c.answer(message)
            handoff = c.payload.get("handoff") or {}
            c.check(bool(handoff.get("requested") or handoff.get("offered")),
                    "Repeatedly unmatched everyday descriptions need a visible human-support escape", observed=c.payload)
            c.check(c.stored()["model"] == "OS-4000T", "Clarification must preserve model context", observed=c.stored())
            return c.evidence()
        add("followup-clarify-" + cid, "warranty_followup", "avoid_repeated_menu_loop", "Recover from unmatched input: " + cid,
            run, {"messages": messages}, severity="high")

    for cid, message in [("human", "Can I talk to a person?"),
                         ("callback", "Could someone call me back?"),
                         ("ko-human", "담당자와 상담하고 싶어요.")]:
        def run(client, cid=cid, message=message):
            c = WarrantyConversation(client, "w3-f-human-" + cid)
            c.identity()
            c.defect("The remote screen is black.")
            c.answer(message)
            handoff = c.payload.get("handoff") or {}
            c.check(handoff.get("requested") is True, "A direct request for a person should interrupt the diagnostic interview", observed=c.payload)
            c.check(c.stored()["model"] == "OS-4000T" and c.stored()["collected"].get("human_support_requested"),
                    "The saved human request must retain chair identity and context", observed=c.stored())
            c.check(not re.search(r"(?:callback is booked|callback has been booked|we have called|replacement is approved)", c.text(), re.I),
                    "Saved contact requests must not claim an actual booking or warranty approval", observed=c.text())
            return c.evidence()
        add("followup-human-" + cid, "warranty_followup", "human_support_request", "Interrupt with " + message, run, {"message": message})

    for cid, messages, expected_key, expected_node in [
        ("power-already-tested", ["power", "remote_off", "I already checked the outlet and reconnected the power cord. There is no clicking sound."], "no_clicking", "defect_power_no_click_terminal"),
        ("remote-already-tested", ["remote", "no_power", "The cable is firmly connected and the fuse is intact. It still will not turn on."], "all_checked_ok", "defect_remote_pcb_check_terminal"),
        ("heat-already-tested", ["heat", "not_heating", "I have run it for twenty minutes with heat switched on, and it is still cold."], "still_no_heat", "defect_heating_not_heating_terminal"),
    ]:
        def run(client, cid=cid, messages=messages, expected_key=expected_key, expected_node=expected_node):
            c = WarrantyConversation(client, "w3-f-tried-" + cid)
            c.identity()
            c.defect()
            for message in messages[:-1]:
                c.answer(message)
            old_node, old_prompt = c.node(), c.text()
            c.answer(messages[-1])
            # Retaining a node with a specific proposed answer is valid, while
            # returning the identical generic checklist without acknowledgement
            # leaves the customer's already-completed work unusable.
            d = c.payload
            suggested = (d.get("suggested_option") or {}).get("answer_key")
            interpreted = (d.get("interpreted_option") or {}).get("answer_key")
            c.check(not suggested or suggested == expected_key,
                    "A suggested confirmation must not reverse a completed check or explicit denial",
                    expected=expected_key, observed=d)
            c.check(not interpreted or interpreted == expected_key,
                    "An interpreted answer must preserve the customer's completed-check facts",
                    expected=expected_key, observed=d)
            change = c.node() == expected_node or suggested == expected_key or interpreted == expected_key
            unquoted = re.sub(r'["“][^"”\n]*["”]', '', c.text())
            acknowledgement = bool(re.search(r"already|completed|you(?:'ve| have)? checked|please confirm whether", unquoted, re.I)) and c.text() != old_prompt
            c.check(change or acknowledgement,
                    "A description of completed checks needs an acknowledged next step or confirmable interpretation",
                    expected="acknowledge completed checks, ask a focused clarification, or progress", observed=d)
            return c.evidence()
        add("followup-" + cid, "warranty_followup", "already_performed_checks", "Use completed troubleshooting: " + cid,
            run, {"messages": messages}, severity="high")

    def retains_model(client):
        c = WarrantyConversation(client, "w3-f-model-context")
        c.identity("Hypnos 4D")
        original = c.ticket_id
        c.defect("The airbags are not inflating.")
        c.answer("The arm airbags.")
        c.request("", method="get", customer="I reopen my existing conversation.")
        c.check(c.ticket_id == original and "hypnos" in c.payload["ticket"]["model_name"].lower(),
                "A normal diagnostic follow-up and resume must preserve model and ticket identity", observed=c.payload)
        return c.evidence()
    add("followup-resume-model", "warranty_followup", "model_and_resume_context", "Resume without asking the chair model again", retains_model,
        {"model": "Hypnos 4D", "messages": ["The airbags are not inflating.", "The arm airbags."]})

    def correct_choice(client):
        c = WarrantyConversation(client, "w3-f-back")
        c.identity()
        c.defect("remote")
        before = c.ticket_id
        c.request("back", {}, customer="I picked the wrong part; go back.")
        c.answer("The airbags are not inflating.")
        proposed = (c.payload.get("suggested_option") or {}).get("answer_key")
        c.check(c.ticket_id == before and (c.node() == "defect_air_location" or proposed == "air"),
                "Correcting a wrong part selection should reuse the ticket and move to the corrected topic", observed=c.payload)
        return c.evidence()
    add("followup-back-correction", "warranty_followup", "model_and_resume_context", "Correct the selected faulty part", correct_choice,
        {"messages": ["remote", "go back", "The airbags are not inflating."]})

    def repeated_smart(client):
        c = WarrantyConversation(client, "w3-f-repeat")
        c.identity()
        c.defect("The remote screen is black.")
        original_id, original_node = c.ticket_id, c.node()
        c.request("smart-start", {"message": "It is still the same remote problem.", "domain": DOMAIN}, customer="It is still the same remote problem.")
        c.check(c.ticket_id == original_id and c.node() == original_node,
                "Repeating the problem must not restart the entire intake or discard the diagnostic stage", observed=c.payload)
        return c.evidence()
    add("followup-repeat-symptom", "warranty_followup", "model_and_resume_context", "Keep progress when the customer repeats a symptom", repeated_smart,
        {"messages": ["The remote screen is black.", "It is still the same remote problem."]})

    for cid, answers, expected_node in [
        ("remote-review", ["remote", "no_power", "all_checked_ok"], "defect_remote_pcb_check_terminal"),
        ("power-review", ["power", "remote_off", "no_clicking"], "defect_power_no_click_terminal"),
        ("heat-review", ["heat", "not_heating", "still_no_heat"], "defect_heating_not_heating_terminal"),
    ]:
        def run(client, cid=cid, answers=answers, expected_node=expected_node):
            c = WarrantyConversation(client, "w3-f-terminal-" + cid)
            c.identity()
            c.defect()
            for answer in answers:
                c.answer(answer)
            c.check(c.node() == expected_node and bool(c.payload["ticket"]["current_node"]["is_terminal"]),
                    "Valid completed diagnostic choices must reach the intended review stage",
                    expected=expected_node, observed=c.payload)
            c.check(not re.search(r"(?:your (?:claim|warranty|replacement) (?:is|has been) approved|we (?:will|have) (?:ship|shipped|send|sent)|free replacement)", c.text(), re.I),
                    "Troubleshooting a common malfunction must not automatically approve or confirm shipment of a replacement", observed=c.text())
            c.check(c.payload["ticket"]["status"] != "self_resolved",
                    "The customer must confirm resolution rather than the bot declaring the problem solved", observed=c.payload)
            return c.evidence()
        add("followup-" + cid, "warranty_followup", "review_not_automatic_coverage", "Finish checks with truthful review status: " + cid,
            run, {"confirmed_answers": answers}, severity="high")
    return rows
