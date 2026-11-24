"""Sales Warranty review opt-in: ask before saving a team request."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import sales_agent as agent


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("SALES_INTENT_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "fetch_live_stock", lambda *a, **k: None)


def _merge(prefs, reply):
    out = dict(prefs or {})
    out.update(reply.prefs_patch or {})
    return out


def test_defect_asks_to_submit_before_contact_dump():
    first = agent.respond("It's told me a number two code. It won't let the chair operate")
    assert first.intent == "warranty_redirect"
    assert first.handoff is False
    lower = first.reply.lower()
    assert "submit" in lower
    assert "does **not** approve" in first.reply.lower() or "does not approve" in lower
    assert "service@osakititan.com" not in lower
    payloads = [q.payload for q in first.quick_replies]
    assert "warranty_submit:yes" in payloads


def test_yes_then_email_saves_review_without_approval():
    first = agent.respond("I need a tablet for OS-Pro Alpina")
    prefs = _merge({}, first)
    ask = agent.respond("yes", prefs=prefs, payload="warranty_submit:yes")
    assert "email" in ask.reply.lower()
    prefs = _merge(prefs, ask)
    saved = agent.respond("owner@example.com", prefs=prefs)
    assert saved.handoff is True
    assert saved.lead_capture is not None
    assert saved.lead_capture["email"] == "owner@example.com"
    assert saved.lead_capture["reason"] == "warranty_submit"
    assert "warranty" in saved.lead_capture["interest_summary"].lower()
    lower = saved.reply.lower()
    assert "saved" in lower
    assert "has **not** approved" in saved.reply.lower() or "has not approved" in lower
    assert "repair or replacement" in lower


def test_decline_returns_self_serve_warranty_contacts():
    first = agent.respond("looking for osaki tablet for Alpina")
    prefs = _merge({}, first)
    declined = agent.respond("no", prefs=prefs, payload="warranty_submit:no")
    assert declined.handoff is True
    assert "service@osakititan.com" in declined.reply.lower()
    assert "freshdesk.com" in declined.reply.lower()
    assert "has not approved a repair" in declined.reply.lower()


def test_owned_model_defect_offers_submit_not_contact_dump():
    first = agent.respond(
        "I own a Titan TP-Epic 4D and its foot rollers stopped working."
    )
    assert first.intent == "warranty_redirect"
    assert first.handoff is False
    assert "Titan TP-Epic 4D" in first.reply
    assert "submit" in first.reply.lower()
    assert "service@osakititan.com" not in first.reply.lower()
    assert (first.prefs_patch or {}).get("owner_service_case", {}).get("model") == (
        "Titan TP-Epic 4D"
    )


def test_explicit_phone_ask_still_gives_warranty_contacts():
    first = agent.respond(
        "I own a Titan TP-Epic 4D and its foot rollers stopped working."
    )
    prefs = _merge({}, first)
    phone = agent.respond(
        "Back to my broken chair, what number should I call?", prefs=prefs
    )
    assert phone.intent == "warranty_redirect"
    assert "ext. 3" in phone.reply
    assert "haven't submitted a service ticket" in phone.reply
    assert "service@osakititan.com" in phone.reply.lower()
