"""Answer-first warranty intake regressions from Admin failures."""

from __future__ import annotations

import pytest

from script.quality_v3_warranty import WarrantyConversation
from script.simulate_chatbot_quality import offline_api


@pytest.fixture
def bare_client():
    with offline_api() as client:
        yield client
        assert not client.attempted_external_delivery


def _privacy(client, session_id: str, domain: str = "titanchair.com") -> None:
    assert client.post(
        f"/api/v1/warranty/session/{session_id}/consent",
        json={"domain": domain, "policy_store": domain.split(".")[0]},
    ).status_code == 200
    assert client.post(
        f"/api/v1/warranty/session/{session_id}/contact-email",
        json={"customer_email": "answer.first@example.test"},
    ).status_code == 200


def test_quantum_roller_skips_issue_menu_and_enters_mechanism(bare_client):
    sid = "answer-first-quantum"
    _privacy(bare_client, sid)
    message = (
        "Titan 3D Quantum Massage Chair. Purchased 12/23/24. "
        "The roller on chair stopped moving up and down. "
        "It just sits at the butt and doesn't move up and down."
    )
    data = bare_client.post(
        f"/api/v1/warranty/session/{sid}/smart-start",
        json={"message": message, "domain": "titanchair.com"},
    ).json()
    ticket = data["ticket"]
    smart = data["smart_start"]
    assert smart.get("auto_entered") is True
    assert smart.get("suggested_issue_type") == "defect"
    assert ticket["issue_type"] == "defect"
    assert ticket["model_name"] == "Titan 3D Quantum"
    assert ticket["current_node"]["node_id"] != "issue_type"
    assert "rolling" in ticket["current_node"]["node_id"]
    assert "Which area do you need help with" not in (data.get("assistant_message") or "")


def test_up_and_down_motion_maps_without_option_mismatch_loop():
    with offline_api() as client:
        chat = WarrantyConversation(client, "answer-first-up-down")
        chat.identity(model="Titan 3D Quantum")
        chat.defect()
        data = chat.answer("rolling")
        assert chat.node() == "defect_rolling_problem_type"
        data = chat.answer("the up and down motion has stopped.")
        assert data.get("interpreted_option", {}).get("answer_key") == "heads_not_moving"
        assert chat.node() == "defect_rolling_heads_q"
        assert not (data.get("handoff") or {}).get("requested")
        assert not client.attempted_external_delivery


def test_second_option_mismatch_hands_off_instead_of_looping():
    with offline_api() as client:
        chat = WarrantyConversation(client, "answer-first-mismatch")
        chat.identity()
        chat.defect()
        first = chat.answer("zzz totally unrelated phrase one")
        assert not (first.get("handoff") or {}).get("requested")
        second = chat.answer("zzz totally unrelated phrase two")
        assert second["handoff"]["requested"] is True
        assert second["handoff"]["reason"] == "clarification_loop"
        assert not client.attempted_external_delivery


def test_auto_check_howto_does_not_paste_unrelated_air_hose_guide():
    with offline_api() as client:
        chat = WarrantyConversation(client, "answer-first-autocheck")
        chat.identity(model="Osaki Platinum - Escape Duo 4D")
        chat.request(
            "quick-start",
            {"issue_type": "installation", "domain": "osakiusa.com"},
            customer="Installation help",
        )
        data = chat.answer("how do i start auto-check on escape duo")
        text = (data.get("assistant_message") or "").lower()
        assert "air hose" not in text
        assert "under the footrest" not in text
        assert "automatic body check" in text or "published" in text
        assert "won't guess" in text or "do not have" in text or "don't have" in text
        assert not client.attempted_external_delivery


def test_delivery_status_opens_lookup_panel_first(bare_client):
    sid = "answer-first-delivery"
    _privacy(bare_client, sid, domain="osakiusa.com")
    data = bare_client.post(
        f"/api/v1/warranty/session/{sid}/smart-start",
        json={
            "message": "Where is my order? It has not arrived yet.",
            "domain": "osakiusa.com",
        },
    ).json()
    assert data["ticket"]["issue_type"] == "delivery"
    assert data.get("delivery_lookup_required") is True
    assert "order number" in (data.get("assistant_message") or "").lower()
    assert "email" in (data.get("assistant_message") or "").lower()
