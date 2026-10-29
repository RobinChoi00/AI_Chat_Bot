"""Everyday purchase-policy and existing-order conversations, with full transcripts.

These customer contracts audit existing behavior; they do not change the app,
invoke a model, or claim that a handoff means a real agent was notified.
"""
from __future__ import annotations

import re

from script.quality_v2_support import Scenario, CheckFailure


class Conversation:
    def __init__(self, client, sid, messages):
        self.client, self.sid, self.turns = client, sid, []
        self.messages = messages
        self.request("post", f"/api/v1/warranty/session/{sid}/consent",
                     {"domain": "osakiusa.com", "policy_store": "osakiusa"}, setup=True)

    def request(self, method, path, body=None, *, setup=False):
        response = getattr(self.client, method)(path, **({"json": body} if body is not None else {}))
        try:
            data = response.json()
        except ValueError:
            data = {"text": response.text}
        from sales_models import get_session_collected
        self.turns.append({"setup": setup, "method": method, "endpoint": path,
                           "customer_request": body, "http_status": response.status_code,
                           "response": data, "saved_state": get_session_collected(self.sid)})
        if response.status_code != 200:
            raise CheckFailure("The customer conversation must remain usable", expected="HTTP 200",
                               observed=self.evidence())
        return data

    def say(self, message):
        return self.request("post", "/api/v1/sales/chat", {"session_id": self.sid,
                            "message": message, "domain": "osakiusa.com"})

    def fail(self, message, expected):
        raise CheckFailure(message, expected=expected, observed=self.evidence())

    def evidence(self):
        executed = sum(not turn["setup"] for turn in self.turns)
        return {"transcript": self.turns, "planned_messages": self.messages,
                "customer_turns_executed": executed,
                "unrun_customer_turns": max(0, len(self.messages) - executed)}

    def check(self, payload, spec):
        text = str(payload.get("reply") or "")
        lower = text.casefold()
        if not text.strip():
            self.fail("Customer received an empty answer", "A useful reply")
        if spec.get("public") and payload.get("handoff"):
            self.fail("An ordinary published policy question was unnecessarily handed off", "Answer the published policy directly")
        allowed = spec.get("intents")
        if allowed and payload.get("intent") not in allowed:
            self.fail("The reply follows the wrong business task", sorted(allowed))
        actionable = bool(spec.get("allow_sales_confirmation") and any(
            str(choice.get("payload") or "").startswith("human")
            for choice in payload.get("quick_replies", [])))
        if spec.get("handoff") and not (payload.get("handoff") or actionable):
            self.fail("The requested existing-order action needs the responsible team", "A truthful handoff to the responsible team")
        for group in spec.get("any_groups", ()):
            if not any(term.casefold() in lower for term in group):
                self.fail("The answer did not address the customer's specific policy/service question", {"one_of": group})
        for term in spec.get("all", ()):
            if term.casefold() not in lower:
                self.fail("A material published fact or service next step is missing", {"contains": term})
        primary = re.split(r"\*\*Sales\s*\(delivery", text, flags=re.I)[0].casefold()
        if spec.get("department") == "sales" and not any(term in primary for term in ("sales", "cancel / refund", "order number", "ventas", "영업", "세일즈")):
            self.fail("Existing orders and returns need a Sales contact, not only a warranty directory", "Sales is the primary next step")
        if spec.get("department") == "warranty" and not any(term in primary for term in ("warranty", "service@osakititan.com", "technical support", "garantía", "garantia", "보증", "워런티")):
            self.fail("A fault or replacement-part request needs Warranty service", "Warranty service is the primary next step")
        for phrase in ("your refund is approved", "your order has been cancelled", "your order has been canceled",
                       "your refund has been issued", "a technician has been booked", "delivery is guaranteed tomorrow"):
            if phrase in lower:
                self.fail("The chat claimed an unverified approval, action or appointment", "No unverified operational confirmation")


def scenarios():
    out = []

    def add(cid, category, family, goal, steps, severity="medium"):
        def run(client):
            conversation = Conversation(client, "v3-service-" + cid, [message for message, _ in steps])
            for message, contract in steps:
                conversation.check(conversation.say(message), contract)
            return {"goal": goal, **conversation.evidence()}
        out.append(Scenario("service-v3-" + cid, category, family, goal, severity, run,
                            {"messages": [message for message, _ in steps], "contracts": [spec for _, spec in steps]}, "http_api"))

    def policy(*groups):
        return {"public": True, "any_groups": groups}

    def confirm(*groups):
        # Individual availability, staff authorization and unpublished terms
        # may be answered with a truthful limit and specialist next step.
        return {"any_groups": groups}

    def handoff(intents, department, *groups):
        return {"handoff": True, "intents": list(intents), "department": department, "any_groups": groups}

    price = {"any_groups": (("$", "price", "current catalog"),)}
    # Each itinerary is an ordinary shopper task, not a numeric paraphrase grid.
    purchases = [
        ("showroom-visit", "showroom_visit", "Know where to try a chair before buying", [
            ("Where can I try a chair in person?", policy(("1001 W Crosby", "Carrollton"))),
            ("Are you open Sunday?", confirm(("Sunday", "published hours", "confirm"))),
        ]),
        ("showroom-model", "showroom_visit", "Do not promise a specific chair is on the showroom floor", [
            ("Where is your showroom?", policy(("Carrollton",))),
            ("Will the Solo Flex definitely be there to try today?", {"any_groups": (("check", "confirm", "specialist", "availability"),)}),
        ]),
        ("showroom-appointment", "showroom_visit", "Explain a visit request without booking a calendar slot", [
            ("Can I book a showroom visit for Saturday afternoon?", confirm(("confirm", "preferred", "request", "specialist"))),
            ("So is that appointment confirmed now?", {"any_groups": (("not", "confirm", "request", "specialist"),)}),
        ]),
        ("showroom-pickup", "showroom_visit", "Same-day pickup needs showroom stock confirmation", [
            ("Can I pick up a chair today?", confirm(("won't confirm", "check", "specialist"))),
        ]),
        ("curbside-assembly", "delivery_options", "Differentiate standard delivery from in-home assembly", [
            ("Does standard delivery include assembly?", policy(("assembly",), ("White Glove",))),
            ("What is the difference between curbside and White Glove?", policy(("curb",), ("assembly",))),
        ]),
        ("white-glove-fee", "delivery_options", "White Glove fee needs a real quote, not an invented flat amount", [
            ("How does White Glove delivery work?", policy(("in-home", "inside"), ("assembly",))),
            ("How much is the White Glove fee for my address?", {"any_groups": (("fee", "quote", "specialist", "address"),)}),
        ]),
        ("stairs-access", "delivery_options", "Identify stair carry charges and access planning before purchase", [
            ("I live on the third floor with no elevator. Can you bring the chair upstairs?", policy(("stair", "third person"), ("extra", "charge", "fee"))),
            ("Can the delivery team remove a door if it is too narrow?", policy(("cannot", "can't", "clear", "remove doors"))),
        ]),
        ("haul-old", "delivery_options", "Answer old-chair removal instead of treating it as a new defect", [
            ("I am shopping for a new chair. Will you haul away my old one?", policy(("don't haul", "cannot", "can't", "no haul"))),
        ]),
        ("shipping-time", "shipping_locations", "Give published lead times without a guaranteed arrival date", [
            ("How long does shipping normally take?", policy(("2 weeks", "two weeks"), ("3 weeks", "three weeks"))),
            ("Can you guarantee it will arrive by next Friday?", {"any_groups": (("can't", "cannot", "confirm", "sales"),)}),
        ]),
        ("shipping-address", "shipping_locations", "Explain quote prerequisites for shipping cost", [
            ("How much is delivery to ZIP 75006?", policy(("zip", "address", "model"), ("confirm", "quote", "depends"))),
        ]),
        ("shipping-hawaii", "shipping_locations", "Hawaii freight is customer-paid and address-specific", [
            ("Do you ship to Honolulu?", policy(("Hawaii",), ("pay", "freight"))),
            ("Is the freight free there?", {"any_groups": (("pay", "not", "quote", "freight"),)}),
        ]),
        ("shipping-canada", "shipping_locations", "An international address needs confirmation rather than a fabricated rate", [
            ("Can you deliver to Canada?", confirm(("specialist", "confirm"), ("don't have", "can't", "published"))),
        ]),
        ("return-window", "returns_before_purchase", "Answer the return window before purchase", [
            ("Before I buy, what is your return policy?", policy(("30",), ("shipping",))),
            ("Would I pay the shipping both ways?", {"any_groups": (("both", "original", "return shipping"),)}),
        ]),
        ("return-comfort", "returns_before_purchase", "A hypothetical comfort return is a sales policy question", [
            ("If I buy it and don't like the massage, can I return it?", policy(("30",), ("RMA", "authorization"))),
        ]),
        ("return-not-bought", "returns_before_purchase", "Do not mistake 'haven't bought yet' for an existing order", [
            ("I haven't bought anything yet. What is the return policy?", policy(("30",))),
        ]),
        ("warranty-before", "warranty_before_purchase", "Describe standard coverage before purchase without starting a claim", [
            ("How long is the warranty on a new chair?", policy(("3 years", "three", "3)"))),
            ("Do you offer an extended warranty?", policy(("4-year", "5-year", "extended"))),
        ]),
        ("warranty-parts-labor", "warranty_before_purchase", "Explain parts and labor rather than treating coverage as a fault", [
            ("What does the warranty cover?", policy(("parts",), ("labor",))),
            ("Does that mean free labor for all three years?", {"any_groups": (("Year 1", "year 1", "first year", "confirm", "specialist"),)}),
        ]),
        ("finance-monthly", "payments_and_promotions", "Financing terms depend on checkout and approval", [
            ("Can I pay monthly?", policy(("Affirm", "monthly"), ("checkout",))),
            ("What APR would I get?", confirm(("won't quote", "approval", "checkout", "depend"))),
        ]),
        ("discount-code", "payments_and_promotions", "A coupon request needs authorized Sales review", [
            ("Do you have a coupon code I can use?", confirm(("discount", "coupon"), ("specialist", "sales", "can't quote"))),
        ]),
        ("competitor-gap", "payments_and_promotions", "A competitor price gap is not the customer's budget", [
            ("Why is your chair $1000 more than Costco?", confirm(("different", "compare", "published", "specialist", "confirm"))),
            ("Can a person check the price difference?", {**handoff(("human", "discount"), "sales"), "allow_sales_confirmation": True}),
        ]),
        ("checkout-card", "checkout_and_special_sales", "A declined card needs checkout support without guessing a bank reason", [
            ("I am trying to buy but my credit card was declined.", confirm(("specialist", "payment", "checkout"), ("can't see", "won't guess", "follow"))),
        ]),
        ("commercial-use", "checkout_and_special_sales", "Commercial-use requirements need confirmation beyond home-chair policy", [
            ("Can I buy one for my salon's customers?", confirm(("home", "commercial"), ("specialist", "confirm"))),
        ]),
        ("new-or-used", "checkout_and_special_sales", "Clarify refurbished/open-box inventory limits", [
            ("Do you sell new or refurbished chairs?", confirm(("new",), ("don't have", "specialist", "confirm"))),
        ]),
        ("gift-tax", "checkout_and_special_sales", "Gift orders and tax exemptions are distinct supported-sales questions", [
            ("Can you hide the price on a gift receipt?", confirm(("can't", "specialist"), ("gift",))),
            ("Can a specialist help with tax exempt purchasing?", {"any_groups": (("specialist", "sales", "email", "commercial"),)}),
        ]),
    ]
    for cid, family, goal, steps in purchases:
        add(cid, "sales_purchase_policy", family, goal, steps)

    sales_order = lambda *groups: handoff(("order_status", "eta_shipping"), "sales", *groups)
    change = handoff(("order_change", "order_status", "human"), "sales", ("order number",), ("email",))
    cancel = handoff(("cancel_refund",), "sales", ("email",))
    warranty = handoff(("warranty_redirect", "parts_technician"), "warranty")
    orders = [
        ("track-basic", "order_status", "Existing shipment tracking belongs to Sales", [
            ("Where is my order?", sales_order(("delivery", "tracking", "order status"))),
            ("I just need the tracking number, not a new chair.", sales_order()),
        ]),
        ("track-number", "order_status", "Providing an order number must not trigger a fresh recommendation", [
            ("Can you track order OSKMC1234?", sales_order()),
            ("The order is under buyer@example.test.", sales_order()),
        ]),
        ("late-delivery", "order_status", "A missed delivery estimate needs current order review", [
            ("My delivery is late. What happened?", sales_order()),
        ]),
        ("delivered-missing", "order_status", "Delivered-but-missing is an order investigation, not a product warranty claim", [
            ("Tracking says delivered but nothing arrived.", sales_order()),
        ]),
        ("shipping-after-buy", "existing_delivery_changes", "An order already placed needs a delivery upgrade review", [
            ("I already ordered. Can I add White Glove?", change),
            ("Has the upgrade been added now?", {"any_groups": (("not", "review", "sales", "confirm"),)}),
        ]),
        ("address-change", "existing_delivery_changes", "Sales must review an address change without claiming it is complete", [
            ("Can I change the shipping address on my order?", change),
        ]),
        ("color-change", "existing_delivery_changes", "Changing the ordered chair's color needs Sales review", [
            ("I already ordered a black chair. Can I switch it to brown?", change),
        ]),
        ("delivery-day", "existing_delivery_changes", "Do not guarantee a time slot for an existing order", [
            ("Can you move my delivery to Saturday morning?", handoff(("order_change", "order_status", "eta_shipping"), "sales")),
        ]),
        ("cancel-basic", "cancellation_and_refund", "An order cancellation request needs Sales authorization", [
            ("Please cancel my order.", cancel),
            ("Is it cancelled now?", {"any_groups": (("review", "agent", "email", "not", "confirm"),)}),
        ]),
        ("cancel-unshipped", "cancellation_and_refund", "Unshipped cancellation cannot be approved automatically", [
            ("I want to cancel my purchase before it ships.", cancel),
        ]),
        ("refund-basic", "cancellation_and_refund", "A refund request needs an agent without a refund promise", [
            ("I need a refund for my chair.", cancel),
            ("How soon will the money be back?", {"any_groups": (("confirm", "agent", "sales", "review", "refund"),)}),
        ]),
        ("return-delivered", "cancellation_and_refund", "An actual received-chair return is not a new pre-purchase policy question", [
            ("I received the chair but want to return it.", cancel),
        ]),
        ("fault-remote", "post_purchase_fault", "A broken remote belongs to Warranty, not a feature interview", [
            ("My remote screen is black and the buttons don't work.", warranty),
            ("Can you send a replacement remote?", warranty),
        ]),
        ("fault-recline", "post_purchase_fault", "A stuck recliner needs Warranty service", [
            ("My chair won't recline anymore.", warranty),
            ("The footrest is stuck too.", warranty),
        ]),
        ("fault-airbag", "post_purchase_fault", "Airbag failure is post-purchase service", [
            ("I bought a Solo Flex and the airbags stopped inflating.", warranty),
            ("The airbag is still not working after a restart.", warranty),
        ]),
        ("fault-new-delivery", "post_purchase_fault", "Newly delivered physical damage needs Warranty evidence review", [
            ("My chair arrived with a cracked armrest.", warranty),
            ("Can I send photos of the damage to the warranty team?", warranty),
        ]),
        ("manual-request", "owner_help_and_parts", "An owner's manual request needs the support/manual route", [
            ("I own a Hypnos. Can you email me the user manual?", warranty),
            ("Can warranty support send me the manual link?", warranty),
        ]),
        ("app-connect", "owner_help_and_parts", "An existing owner's app connection trouble belongs to technical support", [
            ("How do I connect my Osaki app to the chair I bought?", warranty),
        ]),
        ("technician-visit", "owner_help_and_parts", "A home repair visit must not be booked automatically", [
            ("Can you send a technician to fix my chair?", warranty),
            ("Has somebody been scheduled to come out?", {"any_groups": (("review", "contact", "support", "confirm", "not"),)}),
        ]),
        ("part-purchase", "owner_help_and_parts", "Replacement-part purchases go to the responsible service team", [
            ("Where can I buy a replacement footrest?", warranty),
        ]),
        ("invoice-copy", "account_and_service_continuity", "An existing purchase's invoice requires Sales support", [
            ("Can I get a copy of the invoice for my order?", handoff(("order_status", "human", "order_change"), "sales", ("sales", "email", "order"))),
            ("Can I talk to a sales agent about that invoice?", handoff(("human", "order_status"), "sales")),
        ]),
        ("warranty-status", "account_and_service_continuity", "An old owner's coverage needs review rather than blanket approval", [
            ("I bought a chair last year. Is it still under warranty?", warranty),
        ]),
        ("order-ko", "multilingual_service", "A Korean existing-order query reaches Sales rather than Warranty", [
            ("제 주문 배송이 어디쯤 왔나요?", handoff(("order_status", "eta_shipping"), "sales")),
        ]),
        ("refund-es", "multilingual_service", "A Spanish refund request reaches the cancellation/refund team", [
            ("Quiero devolver mi silla y pedir un reembolso.", handoff(("cancel_refund",), "sales")),
        ]),
    ]
    for cid, family, goal, steps in orders:
        add(cid, "sales_existing_orders", family, goal, steps, severity="high")
    return out
