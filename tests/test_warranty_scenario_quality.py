"""
End-to-end warranty scenario quality checks (local TestClient, no live LLM).

These walk representative customer paths and assert on customer-facing copy.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as wm  # noqa: E402
import warranty_workflow as wf  # noqa: E402
from warranty_router import router  # noqa: E402

DOMAIN = "osakiusa.com"


@pytest.fixture
def warranty_client(monkeypatch):
    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    mem_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=mem_engine,
        expire_on_commit=False,
    )
    wm.Base.metadata.create_all(bind=mem_engine)
    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", mem_session_factory)
    monkeypatch.setattr(wf, "_SessionFactory", mem_session_factory)
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _customer_message(payload: dict) -> str:
    text = str(payload.get("assistant_message") or "").strip()
    if text:
        return text
    node = (payload.get("ticket") or {}).get("current_node") or {}
    return str(node.get("prompt") or "").strip()


def _post(client: TestClient, path: str, body: dict) -> dict:
    resp = client.post(path, json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_installation_path_is_clear(warranty_client):
    sid = str(uuid.uuid4())
    _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/register-model",
        {"model": "Hypnos 4D", "domain": DOMAIN},
    )
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/quick-start",
        {"issue_type": "installation", "domain": DOMAIN},
    )
    text = _customer_message(payload)
    assert "installation" in text.lower()
    assert "Refer to: Page" not in text
    assert "Replace the main PCB" not in text


def test_delivery_path_asks_for_tracking(warranty_client):
    sid = str(uuid.uuid4())
    _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/register-model",
        {"model": "OS-4000T", "domain": DOMAIN},
    )
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/quick-start",
        {"issue_type": "delivery", "domain": DOMAIN},
    )
    text = _customer_message(payload)
    assert "delivery" in text.lower()
    assert payload["ticket"]["current_node"]["node_id"] == "delivery_intent_q"


def test_defect_air_path_avoids_unrelated_symptom_labels(warranty_client):
    sid = str(uuid.uuid4())
    _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/register-model",
        {"model": "3D LTX", "domain": DOMAIN},
    )
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/quick-start",
        {"issue_type": "defect", "domain": DOMAIN},
    )
    tid = payload["ticket"]["ticket_id"]
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/{tid}/answer",
        {"answer": "air"},
    )
    text = _customer_message(payload)
    assert "What you can try" in text or "air" in text.lower()
    assert "Red blinking light" not in text
    assert "Refer to: Page" not in text


def test_defect_power_path_uses_category_not_qa_subject(warranty_client):
    sid = str(uuid.uuid4())
    _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/register-model",
        {"model": "OS-4000T", "domain": DOMAIN},
    )
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/quick-start",
        {"issue_type": "defect", "domain": DOMAIN},
    )
    tid = payload["ticket"]["ticket_id"]
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/{tid}/answer",
        {"answer": "power"},
    )
    text = _customer_message(payload)
    assert "Red blinking light" not in text
    assert "power" in text.lower()


def test_shoulders_free_text_maps_to_option(warranty_client):
    sid = str(uuid.uuid4())
    _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/register-model",
        {"model": "3D LTX", "domain": DOMAIN},
    )
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/quick-start",
        {"issue_type": "defect", "domain": DOMAIN},
    )
    tid = payload["ticket"]["ticket_id"]
    _post(warranty_client, f"/api/v1/warranty/{tid}/answer", {"answer": "air"})
    stuck = _post(
        warranty_client,
        f"/api/v1/warranty/{tid}/answer",
        {"answer": "my shoulders"},
    )
    # Menu free text is buttons-only — do not interpret or advance.
    assert stuck.get("side_question") is True
    assert stuck["ticket"]["current_node"]["node_id"] == "defect_air_location"
    assert "tap" in _customer_message(stuck).lower()
    assert "confirm" not in _customer_message(stuck).lower()

    payload = _post(
        warranty_client,
        f"/api/v1/warranty/{tid}/answer",
        {"answer": "shoulders_hips"},
    )
    # Exact answer_key tap still advances.
    assert payload["ticket"]["current_node"]["node_id"] == "defect_air_shoulders_hissing_q"


def test_clarifying_message_when_answer_is_ambiguous(warranty_client):
    sid = str(uuid.uuid4())
    _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/register-model",
        {"model": "3D LTX", "domain": DOMAIN},
    )
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/quick-start",
        {"issue_type": "defect", "domain": DOMAIN},
    )
    tid = payload["ticket"]["ticket_id"]
    _post(warranty_client, f"/api/v1/warranty/{tid}/answer", {"answer": "air"})
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/{tid}/answer",
        {"answer": "yes"},
    )
    text = _customer_message(payload)
    assert "please tap" in text.lower() or "tap one" in text.lower()


def test_smart_start_footrest_air_advances_flow(warranty_client, monkeypatch):
    import warranty_intake as wi

    monkeypatch.setattr(
        wi,
        "extract_workflow_prefill",
        lambda **kwargs: {
            "answer_keys": ["warranty", "defect", "air", "footrest"],
            "model_name": "OS-4000T",
            "confidence": "high",
            "summary": "Footrest air not inflating on OS-4000T.",
            "source": "llm",
        },
    )
    sid = str(uuid.uuid4())
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/smart-start",
        {"message": "OS-4000T footrest air not inflating", "domain": DOMAIN},
    )
    assert payload["ticket"]["model_name"] == "OS-4000T"
    # Suggest defect, but stay on the issue menu until the customer taps.
    assert payload["smart_start"]["applied_keys"] == ["warranty"]
    assert payload["ticket"]["current_node"]["node_id"] == "issue_type"
    assert not payload["ticket"].get("issue_type")
    confirm = payload["smart_start"]["routing_confirmation"]
    assert confirm["requires_confirmation"] is True
    assert "defect" in confirm["message"].lower() or "warranty" in confirm["message"].lower()
    text = _customer_message(payload)
    assert "Red blinking light" not in text


def test_smart_start_vague_message_stays_on_issue_menu(warranty_client, monkeypatch):
    import warranty_intake as wi

    monkeypatch.setattr(
        wi,
        "extract_workflow_prefill",
        lambda **kwargs: {"answer_keys": [], "confidence": "low", "source": "empty"},
    )
    sid = str(uuid.uuid4())
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/smart-start",
        {"message": "hello there", "domain": DOMAIN},
    )
    assert payload["smart_start"]["applied_keys"] == ["warranty"]
    assert payload["ticket"]["current_node"]["node_id"] == "issue_type"
    assert "defect" not in payload["smart_start"]["applied_keys"]


def test_smart_start_blocks_sales_pricing_question(warranty_client):
    sid = str(uuid.uuid4())
    payload = _post(
        warranty_client,
        f"/api/v1/warranty/session/{sid}/smart-start",
        {"message": "How much is the Osaki Solo Flex?", "domain": DOMAIN},
    )
    assert payload.get("side_question") is True
    assert "warranty support" in payload["assistant_message"].lower()
    assert payload["ticket"]["current_node"]["node_id"] == "issue_type"
