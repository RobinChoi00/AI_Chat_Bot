"""Broader isolated public-API privacy and resilience audit; no app patches."""
from __future__ import annotations

import json
from unittest.mock import patch

from script.quality_v2_support import Scenario, expect, require


def scenarios() -> list[Scenario]:
    rows = []

    def add(cid, category, family, title, run, severity="high", inputs=None):
        rows.append(Scenario("privacy-" + cid, category, family, title, severity,
                             run, inputs=inputs, level="http_api"))

    def accepted(client, sid, email=False):
        response = client.post(f"/api/v1/warranty/session/{sid}/consent",
                               json={"domain": "osakiusa.com", "policy_store": "osakiusa"})
        expect(response.status_code, 200, "Synthetic consent setup must succeed")
        if email:
            contact = client.post(f"/api/v1/warranty/session/{sid}/contact-email",
                                  json={"customer_email": "audit.buyer@example.test"})
            expect(contact.status_code, 200, "Synthetic contact setup must succeed")

    def registered(client, sid):
        accepted(client, sid, email=True)
        response = client.post(f"/api/v1/warranty/session/{sid}/register-model",
                               json={"model": "OS-4000T", "domain": "osakiusa.com"})
        expect(response.status_code, 200, "Exact model setup must succeed")
        require(bool(response.json().get("ticket", {}).get("ticket_id")),
                "Model setup must return a ticket", observed=response.json())
        return response.json()["ticket"]["ticket_id"]

    def sales(client, sid, text="Hello", domain="osakiusa.com"):
        return client.post("/api/v1/sales/chat", json={
            "session_id": sid, "message": text, "domain": domain})

    def request_case(cid, category, family, title, method, path, body, expected,
                     setup="none", severity="high"):
        sid = "v2p-" + cid
        def run(client):
            if setup == "model":
                registered(client, sid)
            elif setup != "none":
                accepted(client, sid, email=setup == "identity")
            actual_body = ({**body, "session_id": sid} if isinstance(body, dict)
                           and body.get("session_id") == "SID" else body)
            response = getattr(client, method)(path.format(sid=sid),
                **({"json": actual_body} if actual_body is not None else {}))
            observed = {"status": response.status_code, "body": response.json()}
            expect(response.status_code, expected, title + ": " + json.dumps(observed, ensure_ascii=False))
            return observed
        add(cid, category, family, title, run, severity, {
            "method": method, "path": path, "body": body, "setup": setup})

    for suffix, path, body in [
        ("sales", "/api/v1/sales/chat", {"session_id": "SID", "message": "Hello"}),
        ("button", "/api/v1/sales/chat", {"session_id": "SID", "payload": "recommend"}),
        ("phone-channel", "/api/v1/sales/chat", {"session_id": "SID", "message": "Hello", "channel": "phone"}),
        ("lead", "/api/v1/sales/lead", {"session_id": "SID", "email": "buyer@example.test"}),
        ("model", "/api/v1/warranty/session/{sid}/register-model", {"model": "OS-4000T"}),
        ("handoff", "/api/v1/warranty/session/{sid}/handoff", {"message": "Please connect me to a person"}),
    ]:
        request_case("consent-" + suffix, "privacy_isolation", "consent_entrypoints",
                     "Unconsented " + suffix + " stops before customer processing",
                     "post", path, body, 403)

    def rejected_not_stored(client):
        sid = "v2p-no-storage"
        response = sales(client, sid, "My private concern is SYNTHETIC-SECRET-754")
        expect(response.status_code, 403)
        resumed = client.get(f"/api/v1/sales/session/{sid}")
        expect(resumed.status_code, 404, "Rejected chat must not create a stored Sales session")
        return {"chat_status": response.status_code, "resume_status": resumed.status_code}
    add("rejected-not-stored", "privacy_isolation", "consent_entrypoints",
        "Unconsented messages do not create stored Sales conversations", rejected_not_stored)

    def one_consent_only(client):
        accepted(client, "v2p-consent-a", email=True)
        response = sales(client, "v2p-consent-b")
        expect(response.status_code, 403, "Session A's consent cannot authorize B")
        return {"other_session_status": response.status_code}
    add("consent-session-isolation", "privacy_isolation", "session_store_isolation",
        "Recorded consent is isolated between sessions", one_consent_only)

    def separate_history(client):
        for sid, text in (("v2p-history-a", "I need advice for lower back massage"),
                          ("v2p-history-b", "Does Titan TP-Epic 4D have foot rollers?")):
            accepted(client, sid)
            expect(sales(client, sid, text).status_code, 200)
        resumed_a = client.get("/api/v1/sales/session/v2p-history-a")
        resumed_b = client.get("/api/v1/sales/session/v2p-history-b")
        expect(resumed_a.status_code, 200, "A's saved Sales session must be resumable")
        expect(resumed_b.status_code, 200, "B's saved Sales session must be resumable")
        a, b = resumed_a.json(), resumed_b.json()
        require("lower back massage" in json.dumps(a), "A must receive its own saved customer text", observed=a)
        require("Does Titan TP-Epic" in json.dumps(b), "B must receive its own saved customer text", observed=b)
        require("Does Titan TP-Epic" not in json.dumps(a), "A must not inherit B's transcript", observed=a)
        require("lower back massage" not in json.dumps(b), "B must not inherit A's transcript", observed=b)
        return {"a": a, "b": b}
    add("distinct-history", "privacy_isolation", "session_store_isolation",
        "Independent shoppers resume only their own transcripts", separate_history)

    def storefront_migration(client):
        sid = "v2p-store-migration"
        accepted(client, sid)
        expect(sales(client, sid, "Hello", "osakiusa.com").status_code, 200)
        changed = sales(client, sid, "Hello", "titanchair.com")
        expect(changed.status_code, 409, "Existing Sales session cannot migrate to another storefront")
        state = client.get(f"/api/v1/sales/session/{sid}").json()
        expect(state["session"]["domain"], "osakiusa.com")
        return {"changed_status": changed.status_code, "state": state}
    add("storefront-migration", "privacy_isolation", "session_store_isolation",
        "A conversation remains bound to its original storefront", storefront_migration)

    def separate_ticket(client):
        ticket = registered(client, "v2p-ticket-a")
        response = client.get("/api/v1/warranty/session/v2p-ticket-b")
        expect(response.status_code, 200, "A fresh warranty session must return explicit public state")
        other = response.json()
        expect(other.get("ticket"), None)
        require(ticket not in json.dumps(other), "A fresh session cannot leak another ticket identifier", observed=other)
        return {"other_session": other}
    add("ticket-session-isolation", "privacy_isolation", "session_store_isolation",
        "A fresh warranty session cannot inherit another shopper's ticket", separate_ticket)

    for suffix, value in [
        ("prefix", "Email me at buyer@example.test"),
        ("two-addresses", "one@example.test,two@example.test"),
        ("newline", "buyer@example.test\nBcc: other@example.test"),
        ("trailing-host", "buyer@example.test.evil trailing"),
    ]:
        request_case("contact-" + suffix, "privacy_isolation", "contact_validation",
                     "Contact field rejects surrounding or multi-address input: " + suffix,
                     "post", "/api/v1/warranty/session/{sid}/contact-email",
                     {"customer_email": value}, 422, setup="consent")
    request_case("skip-email-enforced", "privacy_isolation", "contact_validation",
                 "Required warranty contact cannot be silently skipped", "post",
                 "/api/v1/warranty/session/{sid}/contact-email", {"skipped": True}, 422, setup="consent")

    def contact_masking(client):
        sid = "v2p-contact-mask"
        accepted(client, sid)
        response = client.post(f"/api/v1/warranty/session/{sid}/contact-email",
                               json={"customer_email": "  Audit.Buyer+chair@Example.Test  "})
        expect(response.status_code, 200)
        payload = response.json()
        expect(payload["email_gate_status"], "provided")
        require("audit.buyer+chair@example.test" not in json.dumps(payload).lower(),
                "Public acknowledgement must mask the full contact email", observed=payload)
        return payload
    add("contact-mask-normalized", "privacy_isolation", "public_pii_minimization",
        "Normalized contact is saved but masked in public acknowledgement", contact_masking)

    def contact_metadata(client):
        sid = "v2p-sales-contact-mask"
        accepted(client, sid)
        expect(sales(client, sid).status_code, 200)
        from sales_models import SalesSession
        from warranty_models import warranty_db_session
        with warranty_db_session() as db:
            row = db.query(SalesSession).filter_by(session_id=sid).one()
            row.contact_email, row.contact_phone = "private.buyer@example.test", "+12145551234"
        payload = client.get(f"/api/v1/sales/session/{sid}").json()
        for private in ("private.buyer@example.test", "+12145551234"):
            require(private not in json.dumps(payload), "Public contact metadata must mask PII", observed=payload)
        return payload
    add("sales-contact-masked", "privacy_isolation", "public_pii_minimization",
        "Sales resume metadata masks stored email and phone", contact_metadata)

    def privacy_metadata(client):
        sid = "v2p-public-privacy"
        accepted(client, sid, email=True)
        payload = client.get(f"/api/v1/warranty/session/{sid}").json()
        require("audit.buyer@example.test" not in json.dumps(payload),
                "Pre-ticket privacy state exposes flags rather than the full email", observed=payload)
        expect(payload["privacy"]["contact_email_available"], True)
        return payload
    add("session-privacy-flags", "privacy_isolation", "public_pii_minimization",
        "Pre-ticket privacy metadata exposes only contact availability", privacy_metadata)

    for suffix, path in [
        ("sales-list", "/admin/sales/sessions"),
        ("sales-detail", "/admin/sales/sessions/nonexistent"),
        ("leads", "/admin/sales/leads"),
        ("warranty-list", "/admin/warranty/tickets"),
        ("warranty-detail", "/admin/warranty/tickets/nonexistent"),
    ]:
        request_case("admin-" + suffix, "privacy_isolation", "admin_fail_closed",
                     "Missing server credentials fail closed for admin " + suffix,
                     "get", path, None, 503)

    def wrong_admin(client):
        response = client.get("/admin/sales/sessions", headers={"X-Admin-Key": "synthetic-guessed-key"})
        expect(response.status_code, 503)
        require("synthetic-guessed-key" not in response.text, "Admin errors must not echo supplied secrets")
        return {"status": response.status_code, "body": response.json()}
    add("admin-no-secret-echo", "privacy_isolation", "admin_fail_closed",
        "Guessed admin credentials neither unlock history nor get echoed", wrong_admin)

    def status_case(client, variant):
        sid = "v2p-status-" + variant
        ticket_id = registered(client, sid)
        from warranty_case_ref import case_reference_for_ticket
        from warranty_models import WarrantyTicket, warranty_db_session
        with warranty_db_session() as db:
            ticket = db.query(WarrantyTicket).filter_by(ticket_id=ticket_id).one()
            ticket.admin_notes = "SYNTHETIC-PRIVATE-ADMIN-NOTE"
            reference = case_reference_for_ticket(ticket)
        email = "different.buyer@example.test" if variant == "wrong-email" else "audit.buyer@example.test"
        if variant == "unknown-reference":
            reference = "WR-20260101-FFFF00"
        response = client.post("/api/v1/warranty/status", json={"case_reference": reference, "email": email})
        expect(response.status_code, 200 if variant == "safe-fields" else 404)
        require("SYNTHETIC-PRIVATE-ADMIN-NOTE" not in response.text, "Public case lookup must hide admin notes")
        require(ticket_id not in response.text, "Public case lookup must hide the internal UUID")
        return {"variant": variant, "status": response.status_code, "body": response.json()}
    for variant in ("safe-fields", "wrong-email", "unknown-reference"):
        add("status-" + variant, "privacy_isolation", "case_lookup_identity",
            "Reference/email proof and safe fields: " + variant,
            lambda client, v=variant: status_case(client, v))

    def restart_clear(client):
        sid = "v2p-reset-clear"
        registered(client, sid)
        reset = client.post(f"/api/v1/warranty/session/{sid}/restart", json={})
        expect(reset.status_code, 200)
        expect(reset.json()["ticket"], None)
        reloaded = client.get(f"/api/v1/warranty/session/{sid}").json()
        expect(reloaded.get("ticket"), None, "Reload cannot resurrect an abandoned case")
        history = client.get(f"/api/v1/warranty/session/{sid}/messages")
        expect(history.status_code, 404, "Reset must hide abandoned public conversation")
        return {"reset": reset.json(), "reloaded": reloaded, "history_status": history.status_code}
    add("restart-clears-history", "privacy_isolation", "restart_isolation",
        "Explicit reset hides abandoned workflow and history on reload", restart_clear)

    def restart_identity(client):
        sid = "v2p-reset-identity"
        registered(client, sid)
        payload = client.post(f"/api/v1/warranty/session/{sid}/restart", json={}).json()
        expect(payload["privacy"]["consent_recorded"], True)
        expect(payload["privacy"]["contact_email_available"], True)
        return payload
    add("restart-retains-identity", "privacy_isolation", "restart_isolation",
        "Restart clears workflow while retaining recorded consent/contact", restart_identity)

    def restart_other(client):
        a, b = registered(client, "v2p-reset-a"), registered(client, "v2p-reset-b")
        expect(client.post("/api/v1/warranty/session/v2p-reset-a/restart", json={}).status_code, 200)
        other = client.get("/api/v1/warranty/session/v2p-reset-b").json()
        expect(other["ticket"]["ticket_id"], b)
        require(a != b, "Two independent shoppers must have distinct cases")
        return {"unchanged_b": other}
    add("restart-one-session", "privacy_isolation", "restart_isolation",
        "Restarting A preserves the case belonging to B", restart_other)

    for suffix, body in [
        ("missing-sid", {"message": "Hello"}),
        ("empty-sid", {"session_id": "", "message": "Hello"}),
        ("sid-too-long", {"session_id": "s" * 201, "message": "Hello"}),
        ("message-object", {"session_id": "SID", "message": {"text": "Hello"}}),
        ("payload-object", {"session_id": "SID", "payload": {"choice": "hello"}}),
        ("message-too-long", {"session_id": "SID", "message": "x" * 2001}),
        ("payload-too-long", {"session_id": "SID", "payload": "x" * 201}),
    ]:
        request_case("validation-" + suffix, "api_robustness", "request_schema",
                     "Invalid Sales request shape is a client error: " + suffix,
                     "post", "/api/v1/sales/chat", body, 422, setup="consent", severity="medium")

    for suffix, body in [
        ("empty", {"session_id": "SID", "message": ""}),
        ("whitespace", {"session_id": "SID", "message": " \n\t "}),
        ("empty-button", {"session_id": "SID", "payload": " "}),
    ]:
        request_case("content-" + suffix, "api_robustness", "content_validation",
                     "Empty customer content receives actionable validation: " + suffix,
                     "post", "/api/v1/sales/chat", body, 422, setup="consent", severity="medium")

    def unicode_case(client, text):
        sid = "v2p-unicode"
        accepted(client, sid)
        response = sales(client, sid, text)
        expect(response.status_code, 200, "Valid Unicode customer input must not crash")
        require(bool(response.json().get("reply", "").strip()), "Unclear text requires nonempty guidance", observed=response.json())
        return {"message": text, "response": response.json()}
    for suffix, text in (("emoji", "😕"), ("combining", "cafe\u0301?"), ("mixed-script", "hello 안녕하세요 hola")):
        add("unicode-" + suffix, "api_robustness", "content_validation",
            "Unicode input remains processable: " + suffix,
            lambda client, t=text: unicode_case(client, t), "medium", {"message": text})

    request_case("model-empty", "api_robustness", "model_intake_validation",
                 "An empty model cannot become a confirmed chair", "post",
                 "/api/v1/warranty/session/{sid}/register-model", {"model": "   "}, 422, setup="identity")
    request_case("issue-unsupported", "api_robustness", "model_intake_validation",
                 "Unsupported issue codes are rejected explicitly", "post",
                 "/api/v1/warranty/session/{sid}/quick-start", {"issue_type": "refund_approved"}, 422, setup="model")
    request_case("confirm-no-session", "api_robustness", "model_intake_validation",
                 "Model confirmation cannot invent a missing case", "post",
                 "/api/v1/warranty/session/{sid}/confirm-model", {"confirmed": True}, 404, setup="identity")

    def model_declined(client):
        sid = "v2p-confirm-declined"
        registered(client, sid)
        response = client.post(f"/api/v1/warranty/session/{sid}/confirm-model", json={"confirmed": False})
        expect(response.status_code, 422, "Declining a model without correction cannot confirm it")
        return {"status": response.status_code, "body": response.json()}
    add("model-declined", "api_robustness", "model_intake_validation",
        "Declining a chair model requires clarification", model_declined)

    def ambiguous_model(client):
        sid = "v2p-ambiguous-model"
        accepted(client, sid, email=True)
        response = client.post(f"/api/v1/warranty/session/{sid}/register-model",
                               json={"model": "OS-4000T or Hypnos", "domain": "osakiusa.com"})
        require(response.status_code < 500, "Ambiguous model input must not crash", observed=response.text)
        if response.status_code == 200:
            require(not response.json().get("ticket", {}).get("model_confirmed"),
                    "Two alternative models need clarification rather than a confirmed choice",
                    expected="model clarification", observed=response.json())
        else:
            require(response.status_code in (400, 422), "Ambiguous models need a client clarification", observed=response.text)
        return {"status": response.status_code, "body": response.json()}
    add("model-ambiguous", "api_robustness", "model_intake_validation",
        "A question between two models must not silently confirm one", ambiguous_model)

    for suffix, method, path, body, status in [
        ("unknown-answer", "post", "/api/v1/warranty/not-a-real-ticket/answer", {"answer": "yes"}, 404),
        ("unknown-back", "post", "/api/v1/warranty/not-a-real-ticket/back", {}, 404),
        ("missing-answer", "post", "/api/v1/warranty/not-a-real-ticket/answer", {}, 422),
        ("history-negative-limit", "get", "/api/v1/warranty/session/{sid}/messages?limit=-1", None, 422),
        ("history-large-limit", "get", "/api/v1/warranty/session/{sid}/messages?limit=101", None, 422),
        ("history-invalid-cursor", "get", "/api/v1/warranty/session/{sid}/messages?before_id=0", None, 422),
    ]:
        request_case(suffix, "api_robustness", "workflow_input_errors", suffix.replace("-", " "),
                     method, path, body, status, setup="identity", severity="medium")

    for suffix, value in (("unknown-domain", "evil.example.test"),
                          ("javascript-domain", "javascript:alert(1)"),
                          ("credentials-domain", "https://buyer:secret@evil.example.test")):
        request_case(suffix, "api_robustness", "storefront_validation",
                     "Unsupported storefront must not create a conversation: " + suffix,
                     "post", "/api/v1/sales/chat", {"session_id": "SID", "message": "Hello", "domain": value},
                     422, setup="consent")

    def bad_json(client):
        accepted(client, "v2p-bad-json")
        response = client.post("/api/v1/sales/chat", content='{"session_id":',
                               headers={"Content-Type": "application/json"})
        expect(response.status_code, 422)
        return {"status": response.status_code, "body": response.json()}
    add("invalid-json", "api_robustness", "http_protocol_errors",
        "Malformed JSON receives validation rather than an internal exception", bad_json, "medium")
    request_case("wrong-method", "api_robustness", "http_protocol_errors",
                 "GET on the POST-only chat endpoint is method-not-allowed",
                 "get", "/api/v1/sales/chat", None, 405, severity="medium")
    request_case("unknown-route", "api_robustness", "http_protocol_errors",
                 "An unknown customer endpoint returns not-found",
                 "get", "/api/v1/sales/does-not-exist", None, 404, severity="medium")

    def repeated_consent(client):
        sid = "v2p-consent-repeat"
        accepted(client, sid, email=True)
        accepted(client, sid)
        from warranty_consent import WarrantyChatConsent
        from warranty_models import warranty_db_session
        with warranty_db_session() as db:
            count = db.query(WarrantyChatConsent).filter_by(session_id=sid).count()
        expect(count, 1, "Repeated consent updates rather than duplicates the consent row")
        payload = client.get(f"/api/v1/warranty/session/{sid}").json()
        expect(payload["privacy"]["contact_email_available"], True)
        return {"consent_rows": count, "state": payload}
    add("consent-idempotent", "api_robustness", "repeated_actions",
        "Repeated I Agree preserves valid contact without duplicate records", repeated_consent)

    def repeated_restart(client):
        sid = "v2p-repeated-restart"
        registered(client, sid)
        first = client.post(f"/api/v1/warranty/session/{sid}/restart", json={})
        second = client.post(f"/api/v1/warranty/session/{sid}/restart", json={})
        expect(first.status_code, 200)
        expect(second.status_code, 200)
        expect(first.json()["closed_ticket_count"], 1)
        expect(second.json()["closed_ticket_count"], 0)
        return {"first": first.json(), "second": second.json()}
    add("restart-idempotent", "api_robustness", "repeated_actions",
        "Repeated start-over closes only the original active case", repeated_restart)

    def repeated_registration(client):
        sid = "v2p-repeated-model"
        original = registered(client, sid)
        response = client.post(f"/api/v1/warranty/session/{sid}/register-model", json={"model": "OS-4000T"})
        expect(response.status_code, 200)
        expect(response.json()["ticket"]["ticket_id"], original, "Same model must reuse the active case")
        return response.json()
    add("registration-idempotent", "api_robustness", "repeated_actions",
        "Repeated identical chair registration cannot create a second active case", repeated_registration)

    for suffix, body in [
        ("invalid-email", {"case_reference": "WR-20261029-ABCDEF", "email": "not-an-email"}),
        ("invalid-case", {"case_reference": "approved-for-free", "email": "buyer@example.test"}),
        ("missing-case", {"email": "buyer@example.test"}),
    ]:
        request_case("status-" + suffix, "api_robustness", "status_lookup_validation",
                     "Case status rejects invalid identity input: " + suffix, "post",
                     "/api/v1/warranty/status", body, 422, severity="medium")

    def failed_notification(client):
        sid = "v2p-notification-failed"
        registered(client, sid)
        with patch("warranty_email.maybe_send_warranty_transcript",
                   return_value=("audit.buyer@example.test", False)):
            response = client.post(f"/api/v1/warranty/session/{sid}/notify-email",
                                   json={"message": "audit.buyer@example.test"})
        expect(response.status_code, 200)
        expect(response.json().get("sent"), False,
               "A failed notification adapter must not become a successful email acknowledgement")
        return {"injected_boundary": "notification adapter returned sent=False", "response": response.json()}
    add("notification-failure", "api_robustness", "unavailable_integrations",
        "A failed notification is reported truthfully without claiming a sent email", failed_notification)

    def missing_notification_email(client):
        sid = "v2p-notification-no-email"
        registered(client, sid)
        response = client.post(f"/api/v1/warranty/session/{sid}/notify-email",
                               json={"message": "I need more support"})
        expect(response.status_code, 200)
        expect(response.json().get("sent"), False)
        expect(response.json().get("reason"), "no_email_found")
        return response.json()
    add("notification-no-recipient", "api_robustness", "unavailable_integrations",
        "A notification request without a recipient cannot claim delivery", missing_notification_email)

    def wrong_notification_email(client):
        sid = "v2p-notification-wrong-recipient"
        registered(client, sid)
        response = client.post(f"/api/v1/warranty/session/{sid}/notify-email",
                               json={"message": "wrong.recipient@example.test"})
        expect(response.status_code, 422, "Notification cannot be redirected from the saved contact address")
        return {"status": response.status_code, "body": response.json()}
    add("notification-wrong-recipient", "api_robustness", "unavailable_integrations",
        "A transcript cannot be sent to an address different from the saved case contact", wrong_notification_email)

    def tracking_recovery(client):
        sid = "v2p-tracking-stale"
        accepted(client, sid, email=True)
        from delivery_lookup import TrackingSnapshot
        current = TrackingSnapshot(source="shopify", available=True, status="IN_TRANSIT",
                                   carrier="FixtureCarrier", tracking_number="FIXTURE123")
        unavailable = TrackingSnapshot(source="unavailable", available=False, error="synthetic upstream outage")
        body = {"order_number": "OSKMC1234", "email": "audit.buyer@example.test",
                "domain": "osakiusa.com"}
        with patch("delivery_lookup.lookup_by_order_or_email", side_effect=[current, unavailable]):
            first = client.post(f"/api/v1/warranty/session/{sid}/delivery-tracking", json=body)
            second = client.post(f"/api/v1/warranty/session/{sid}/delivery-tracking", json=body)
        expect(first.status_code, 200)
        expect(second.status_code, 200)
        expect(first.json()["tracking_summary"]["available"], True)
        latest = second.json()["tracking_summary"]
        expect(latest["available"], False, "An outage after success must not reuse the previous tracking result")
        expect(latest["status"], "unavailable")
        require("FixtureCarrier" not in latest["message"],
                "A failed refresh must not show stale carrier data as current", observed=latest)
        return {"injected_boundary": "tracking adapter success then unavailable",
                "first": first.json()["tracking_summary"], "second": latest}
    add("tracking-outage-after-success", "api_robustness", "unavailable_integrations",
        "An upstream outage after a successful lookup cannot replay stale tracking details", tracking_recovery)
    return rows
