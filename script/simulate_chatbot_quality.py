#!/usr/bin/env python3
"""Score a fixed synthetic offline chatbot simulation, never live AI quality.

The 40 scenarios exercise real deterministic functions and public Sales/Warranty
APIs with an isolated SQLite database. External models, stock and notifications
are unavailable and any attempted network delivery fails the scenario. A perfect
score means these regression contracts pass; it does not establish a universal
quality score, real model accuracy, customer resolution, or production readiness.

Run this same script against a preserved source tree for a comparable baseline:
  .venv/bin/python script/simulate_chatbot_quality.py --out /tmp/quality.json
  .venv/bin/python script/simulate_chatbot_quality.py \
      --repository-root /tmp/ai-chat-quality-baseline --out /tmp/baseline.json
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
from unittest.mock import patch

WEIGHTS = {"safety": 25, "grounding": 25, "routing_context": 20,
           "multilingual": 15, "privacy_journey": 15}
RUBRIC_VERSION = "offline-customer-contracts-v1"


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    goal: str
    run: Callable[[Any], dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep runtime errors and missing cases from becoming a passing score."""
    if len(results) != 40 or len({row["id"] for row in results}) != 40:
        raise ValueError("The fixed rubric requires 40 unique executed scenarios")
    categories = {}
    for category, weight in WEIGHTS.items():
        rows = [row for row in results if row["category"] == category]
        if len(rows) != 8:
            raise ValueError(f"{category} requires exactly eight scenarios")
        passed = sum(row["passed"] is True for row in rows)
        categories[category] = {"passed": passed, "total": 8, "weight": weight,
                                "score": weight * passed / 8}
    return {"score_out_of_100": sum(row["score"] for row in categories.values()),
            "passed": sum(row["passed"] is True for row in results),
            "failed": sum(row["passed"] is not True for row in results),
            "executed": len(results), "categories": categories}


@contextmanager
def offline_api():
    """Use actual routers/storage while prohibiting external side effects."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import httpx
    import requests
    import smtplib
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    attempted_delivery = []

    def unavailable(*args, **kwargs):
        attempted_delivery.append("blocked external delivery")
        raise RuntimeError("External delivery is forbidden in offline quality simulation")

    original_send = httpx.Client.send

    def local_send(client, request, *args, **kwargs):
        if request.url.host != "testserver":
            return unavailable()
        return original_send(client, request, *args, **kwargs)

    with ExitStack() as stack:
        settings = {"SALES_INTENT_LLM": "0", "SCOPE_CLASSIFIER_LLM": "0",
                    "WARRANTY_REQUIRE_CHAT_PRIVACY": "1", "WARRANTY_SEMANTIC_SEARCH": "0",
                    "SALES_CHAT_RATE_LIMIT": "100000/minute", "TIDIO_ENABLED": "0",
                    "RC_EVENT_WORKER_ENABLED": "false", "WARRANTY_NOTIFICATION_WORKER_ENABLED": "false",
                    "PYTHON_DOTENV_DISABLED": "1"}
        stack.enter_context(patch.dict(os.environ, settings))
        # These are absent, rather than fake credentials accepted by an SDK.
        for name in list(os.environ):
            if name.endswith(("API_KEY", "ACCESS_TOKEN", "PASSWORD")):
                os.environ.pop(name, None)
        stack.enter_context(patch.object(requests.sessions.Session, "request", unavailable))
        stack.enter_context(patch.object(httpx.Client, "send", local_send))
        stack.enter_context(patch.object(httpx.AsyncClient, "send", unavailable))
        stack.enter_context(patch.object(smtplib, "SMTP", unavailable))
        stack.enter_context(patch.object(smtplib, "SMTP_SSL", unavailable))
        agent = importlib.import_module("sales_agent")
        sales = importlib.import_module("sales_router")
        warranty = importlib.import_module("warranty_router")
        models = importlib.import_module("warranty_models")
        workflow = importlib.import_module("warranty_workflow")
        stack.enter_context(patch.object(agent, "fetch_live_stock", lambda *a, **k: None))
        stack.enter_context(patch.object(agent, "fetch_stock_many", lambda *a, **k: {}))
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        models.Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        stack.enter_context(patch.object(models, "_engine", engine))
        stack.enter_context(patch.object(models, "_SessionFactory", sessions))
        stack.enter_context(patch.object(workflow, "_SessionFactory", sessions))
        warranty.limiter.reset()
        app = FastAPI()
        app.include_router(sales.router)
        app.include_router(warranty.router)
        try:
            with TestClient(app) as client:
                client.attempted_external_delivery = attempted_delivery
                yield client
        finally:
            engine.dispose()


def scenarios() -> list[Scenario]:
    import answer_guard as guard
    import customer_safety as safety
    import intent_router as routing
    import scope_classifier as scope
    from sales_models import get_session_collected

    rows: list[Scenario] = []

    def add(cid, category, goal, run):
        rows.append(Scenario(cid, category, goal, run))

    def hazard(text, expected):
        def run(_):
            actual = safety.reported_safety_hazard(text=text, physical_only=True)
            require(actual is expected, f"physical hazard expected {expected}, got {actual}: {text}")
            return {"message": text, "physical_hazard": actual}
        return run

    for cid, text, expected in [
        ("safety-en-smoke", "Smoke is pouring out of my chair's base.", True),
        ("safety-en-melt", "The power cable is melting.", True),
        ("safety-en-hot-cord", "The power cord is getting very hot.", True),
        ("safety-es-plastic", "La silla huele a plástico quemado.", True),
        ("safety-ko-melt", "전원 코드가 녹고 있어요.", True),
        ("safety-en-negation", "I have not seen sparks.", False),
        ("safety-ko-negation", "불꽃이 보이지는 않아요.", False),
        ("safety-manual-reference", 'The manual says "stop if smoke appears". The remote is blank.', False),
    ]:
        add(cid, "safety", "Distinguish an actual physical incident from a denial/reference", hazard(text, expected))

    facts = "MODEL_FACTS_JSON: " + json.dumps({"model": "Fixture Alpha", "currency": "USD", "price_usd": 1999,
        "specs": {"doorway_assembled_in": 35, "max_user_weight_lb": 300}})

    def guarded(query, text, *, tools=(), results=(), unchanged=False, forbidden=()):
        def run(_):
            actual = guard.sanitize_agent_response(text, user_query=query, tools_called=tools, tool_results=results)
            require(bool(actual.strip()), "Guard must provide a useful nonempty response")
            require((actual == text) if unchanged else (actual != text),
                    f"Expected {'preservation' if unchanged else 'rejection'} of {text!r}; got {actual!r}")
            for value in forbidden:
                require(value not in actual, f"Unsupported claim survived: {value!r}")
            return {"query": query, "candidate": text, "response": actual}
        return run

    add("ground-usd-no-source", "grounding", "Do not quote currency-code prices without a lookup",
        guarded("How much is Fixture Alpha?", "Fixture Alpha costs USD 99.", forbidden=("USD 99",)))
    add("ground-eur-no-source", "grounding", "Do not invent a euro quote",
        guarded("What is the price of Fixture Alpha?", "Fixture Alpha costs €99.", forbidden=("€99",)))
    add("ground-usd-conflict", "grounding", "A successful tool call must not authorize the wrong amount",
        guarded("How much is Fixture Alpha?", "Fixture Alpha costs USD 99.", tools=("search_chair_specs",), results=(facts,), forbidden=("USD 99",)))
    add("ground-usd-valid", "grounding", "Preserve a correctly sourced currency-code amount",
        guarded("How much is Fixture Alpha?", "Fixture Alpha costs USD 1,999.", tools=("search_chair_specs",), results=(facts,), unchanged=True))
    add("ground-currency-conflict", "grounding", "A USD catalog amount is not a verified EUR price",
        guarded("What is Fixture Alpha's price?", "Fixture Alpha costs €1,999.", tools=("search_chair_specs",), results=(facts,), forbidden=("€1,999",)))
    add("ground-doorway-conflict", "grounding", "Reject an invented doorway dimension despite other valid facts",
        guarded("What doorway width does Fixture Alpha need?", "Fixture Alpha needs a 29-inch doorway.", tools=("search_chair_specs",), results=(facts,), forbidden=("29-inch",)))
    add("ground-valid-measurement", "grounding", "Keep a model-bound verified dimension",
        guarded("What doorway width does Fixture Alpha need?", "Fixture Alpha needs a minimum doorway of 35 inches assembled.", tools=("search_chair_specs",), results=(facts,), unchanged=True))
    add("ground-unapproved-warranty", "grounding", "Do not turn a support request into an approved warranty claim",
        guarded("Can I get warranty help?", "Your warranty claim is approved. We will ship a replacement tomorrow.", forbidden=("claim is approved", "ship a replacement tomorrow")))

    def language(text, expected, history=()):
        def run(_):
            actual = guard.conversation_language(text, list(history))
            require(actual == expected, f"language expected {expected}, got {actual}")
            return {"message": text, "language": actual}
        return run

    add("language-neutral-korean", "multilingual", "Numbers retain the customer's language",
        language("3500", "ko", ({"role": "user", "content": "한국어로 추천해 주세요"},)))
    add("language-ignore-assistant", "multilingual", "An accidental English assistant reply cannot change Korean preference",
        language("OS-4000T", "ko", ({"role": "user", "content": "한국어로 안내해 주세요"}, {"role": "assistant", "content": "What is your budget for this chair?"})))
    add("language-ko-request-english", "multilingual", "Honor an English request expressed in Korean",
        language("영어로 답해주세요", "en"))
    add("language-request-spanish", "multilingual", "Honor an explicit Spanish preference",
        language("Please respond in Spanish", "es"))
    add("language-es-request-english", "multilingual", "Honor the requested rather than source language",
        language("No quiero español, responde en inglés", "en"))

    def short_spanish(_):
        actual = safety.detect_customer_language("Sale humo.")
        require(actual == "es", f"Short emergency report language was {actual}")
        return {"message": "Sale humo.", "language": actual}
    add("language-short-emergency", "multilingual", "Short Spanish incidents still carry language evidence", short_spanish)

    def localized_safety(language, expected):
        def run(_):
            message = safety.safety_handoff_message(language, "+1-888-848-2630 ext. 3", notification_status="pending")
            require(all(part in message for part in expected), f"Missing localized safe/pending guidance: {message}")
            return {"language": language, "response": message}
        return run
    add("language-korean-safe-action", "multilingual", "Safety instructions and pending notifications remain truthful in Korean",
        localized_safety("ko", ("사용을 즉시 중단", "안전하게", "발송 대기", "아직")))
    add("language-spanish-safe-action", "multilingual", "Safety instructions and pending notifications remain truthful in Spanish",
        localized_safety("es", ("Deje de usar", "segura", "en cola", "no se ha")))

    def consent(client, sid):
        response = client.post(f"/api/v1/warranty/session/{sid}/consent", json={"domain": "osakiusa.com", "policy_store": "osakiusa"})
        require(response.status_code == 200, f"consent failed: {response.status_code} {response.text}")

    def chat(client, sid, message):
        response = client.post("/api/v1/sales/chat", json={"session_id": sid, "message": message, "domain": "osakiusa.com"})
        require(response.status_code == 200, f"sales HTTP {response.status_code}: {response.text}")
        return response.json()

    def warranty_identity(client, sid):
        consent(client, sid)
        response = client.post(f"/api/v1/warranty/session/{sid}/contact-email", json={"customer_email": "customer@example.test"})
        require(response.status_code == 200, f"Synthetic customer email registration failed: {response.text}")

    def sales_constraint(cid, message, field, expected, product_check=None):
        def run(client):
            consent(client, cid)
            response = chat(client, cid, message)
            prefs = get_session_collected(cid).get("recommend_prefs", {})
            actual = prefs.get(field)
            if isinstance(expected, (int, float)):
                require(actual is not None and abs(float(actual) - expected) < .02, f"{field} expected {expected}; saved {actual}")
            else:
                require(actual == expected, f"{field} expected {expected!r}; saved {actual!r}")
            if product_check:
                require(product_check(response), f"Products violate the customer's constraints: {response}")
            return {"message": message, "saved": prefs, "response": response}
        return run

    add("context-small-budget", "routing_context", "A small hard budget is retained and never relaxed",
        sales_constraint("sim-budget", "I'm 5 ft 7, 165 pounds and want shoulder massage. My budget is $300.", "budget_max_usd", 300, lambda r: not r["products"]))
    add("context-kilogram-weight", "routing_context", "Metric body weight is converted and respected",
        sales_constraint("sim-weight", "I'm 180 cm tall and weigh 150 kilograms. Recommend a chair for lower back massage.", "weight_lb", 330.6934,
                         lambda r: all(float(p.get("fit", {}).get("maximum_user_weight_lb") or 0) >= 330.67 for p in r["products"])))
    add("context-negated-intensity", "routing_context", "Negation cannot request the exact intensity the customer rejected",
        sales_constraint("sim-gentle", "I do not want deep tissue; I want gentle massage", "intensity", "Gentle"))
    add("context-mechanism-correction", "routing_context", "Choose the positive mechanism rather than a negated one",
        sales_constraint("sim-mechanism", "I want 3D, not a 4D chair", "mechanism", "3D"))

    def scope_case(query, history=()):
        def run(_):
            decision = scope.evaluate_scope(query, list(history))
            require(decision.is_blocked, f"Unrelated task entered chair chat: {decision}")
            return {"message": query, "blocked": decision.is_blocked, "reason": decision.reason}
        return run
    add("context-unrelated-after-chair", "routing_context", "Recent chair context cannot authorize unrelated tasks",
        scope_case("Write me a poem about cats", ({"role": "user", "content": "Recommend an Osaki chair"},)))
    add("context-price-not-chair", "routing_context", "A price mentioned in an unrelated request is not a chair recommendation",
        scope_case("Give me a recipe under $100"))

    def tracking(_):
        message = "Where is my order OSKMC1234 for buyer@example.test? The delivery address is near your showroom."
        actual = routing.infer_forced_tool(message)
        require(actual == "lookup_order_status", f"Expected order lookup, got {actual}")
        return {"message": message, "tool": actual}
    add("context-tracking-primary-goal", "routing_context", "A showroom reference cannot override an explicit authenticated order-status request", tracking)

    def comparison(client):
        sid = "sim-comparison"
        consent(client, sid)
        first = chat(client, sid, "Please compare the Maestro 4D and Paragon.")
        second = chat(client, sid, "Between those two, which one is cheaper?")
        require(first["intent"] == second["intent"] == "compare", f"Comparison context lost: {first}, {second}")
        require(all(word in second["reply"] for word in ("Maestro", "Paragon")), "Both compared models must remain visible")
        require("user height" not in second["reply"].lower(), "A price followup cannot restart fit intake")
        return {"turns": [first, second]}
    add("context-comparison-followup", "routing_context", "Retain the compared model pair for a relative price followup", comparison)

    def api_check(path, *, method="get", body=None, status=200, check=None):
        def run(client):
            response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
            require(response.status_code == status, f"Expected HTTP {status}, got {response.status_code}: {response.text}")
            payload = response.json()
            if check:
                require(check(payload), f"Customer response contract failed: {payload}")
            return {"path": path, "status": response.status_code, "response": payload}
        return run

    add("privacy-new-session", "privacy_journey", "A new session exposes consent status without inventing a ticket",
        api_check("/api/v1/warranty/session/sim-new", check=lambda p: p["ticket"] is None and not p["privacy"]["consent_recorded"]))
    add("privacy-sales-gated", "privacy_journey", "Sales chat is blocked before recording consent",
        api_check("/api/v1/sales/chat", method="post", body={"session_id": "sim-no-consent", "message": "Recommend a chair", "domain": "osakiusa.com"}, status=403))
    add("privacy-contact-gated", "privacy_journey", "Contact collection requires prior consent",
        api_check("/api/v1/warranty/session/sim-no-email-consent/contact-email", method="post", body={"customer_email": "customer@example.test"}, status=403))

    def consent_then_sales(client):
        sid = "sim-agreed"
        consent(client, sid)
        response = chat(client, sid, "Does Titan TP-Epic 4D have foot rollers?")
        require(response["intent"] == "specs" and "3 spinning reflexology" in response["reply"], "A consented concrete feature question should be answered directly")
        return response
    add("privacy-consent-enables-help", "privacy_journey", "A consented feature question reaches real catalog facts", consent_then_sales)
    def missing_model(client):
        sid = "sim-no-model"
        warranty_identity(client, sid)
        return api_check(f"/api/v1/warranty/session/{sid}/quick-start", method="post", body={"issue_type": "installation", "domain": "osakiusa.com"}, status=422)(client)
    add("journey-model-required", "privacy_journey", "Warranty intake cannot guess the model", missing_model)

    def model_flow(client):
        sid = "sim-model"
        warranty_identity(client, sid)
        registered = client.post(f"/api/v1/warranty/session/{sid}/register-model", json={"model": "OS-4000T", "domain": "osakiusa.com"})
        require(registered.status_code == 200, registered.text)
        ticket = registered.json()["ticket"]
        require(ticket["model_confirmed"] and ticket["model_name"] == "OS-4000T", "Registered exact model must be confirmed")
        require(ticket["current_node"]["node_id"] == "issue_type", "Known model must not be asked again")
        return registered.json()
    add("journey-model-retained", "privacy_journey", "Explicit model registration proceeds without repeated intake", model_flow)
    add("privacy-admin-protected", "privacy_journey", "Conversation administration requires authentication",
        api_check("/admin/sales/sessions", status=503))

    def invalid_email(client):
        sid = "sim-invalid-email"
        consent(client, sid)
        return api_check(f"/api/v1/warranty/session/{sid}/contact-email", method="post", body={"customer_email": "not-an-email"}, status=422)(client)
    add("privacy-invalid-contact", "privacy_journey", "Invalid contact information is rejected without pretending followup is possible", invalid_email)
    return rows


def run_simulation(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    if not (root / "app" / "sales_router.py").is_file():
        raise ValueError(f"Not a supported repository source tree: {root}")
    # Set the chosen root ahead of the invoking script's tree, including app.* imports.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "app"))
    os.environ["SALES_CHAT_RATE_LIMIT"] = "100000/minute"
    os.environ["SCOPE_CLASSIFIER_LLM"] = "0"
    digest = hashlib.sha256()
    for source in sorted((root / "app").glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    results = []
    with offline_api() as client:
        for case in scenarios():
            row = {"id": case.id, "category": case.category, "goal": case.goal, "passed": False}
            attempts_before = len(client.attempted_external_delivery)
            try:
                row["evidence"] = case.run(client)
                require(len(client.attempted_external_delivery) == attempts_before,
                        "Application attempted blocked external delivery during this scenario")
                row["passed"] = True
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc)
            results.append(row)
    return {"rubric_version": RUBRIC_VERSION, "rubric_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mode": "offline_synthetic_regression_simulation",
            "app_source_sha256": digest.hexdigest(), "live_model_score": None,
            "scope": "40 synthetic behavioral contracts: real deterministic functions and isolated public APIs; no live model, stock, email or production checks",
            "limitations": ["100/100 means all fixed scenarios pass, not universal chatbot quality.",
                            "Reviewed scenarios are regression cases, not an independent held-out benchmark.",
                            "Live LLM fluency, model accuracy, production availability and actual customer resolution remain unmeasured."],
            **score_results(results), "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_simulation(args.repository_root)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("rubric_version", "mode", "score_out_of_100", "passed", "failed", "live_model_score", "categories")}, ensure_ascii=False, indent=2))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
