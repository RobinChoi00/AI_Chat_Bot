"""Tests for outbound Freshdesk case creation from warranty tickets."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


class _FakeTicket:
    ticket_id = "tid-123"
    session_id = "sess-1"
    domain = "osaki.com"
    status = "awaiting_admin_review"
    issue_type = "delivery"
    model_name = "Titan Nido 3D"
    current_node_id = "delivery_replace_claim_terminal"
    collected_data = "{}"

    def get_collected(self):
        import json

        return json.loads(self.collected_data)


def test_maybe_create_freshdesk_case_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "0")
    from warranty_freshdesk_case import maybe_create_freshdesk_case  # noqa: W402

    result = maybe_create_freshdesk_case("tid-123", engine=MagicMock())
    assert result["created"] is False
    assert result["skipped"] is True


def test_maybe_create_freshdesk_case_creates_and_persists(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")

    fake_ticket = _FakeTicket()
    engine = MagicMock()
    engine.get_ticket.return_value = fake_ticket
    engine.get_turns.return_value = []

    class FakeResponse:
        status_code = 201

        @staticmethod
        def json():
            return {"id": 99901}

    monkeypatch.setattr(
        "warranty_freshdesk_case.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    mem_engine = __import__("sqlalchemy").create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    from sqlalchemy.orm import sessionmaker

    import warranty_models as wm

    wm.Base.metadata.create_all(bind=mem_engine)
    session_factory = sessionmaker(bind=mem_engine)
    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", session_factory)

    with session_factory() as db:
        row = wm.WarrantyTicket(
            ticket_id="tid-123",
            session_id="sess-1",
            domain="osaki.com",
            status="awaiting_admin_review",
            current_node_id="delivery_replace_claim_terminal",
            collected_data="{}",
        )
        db.add(row)
        db.commit()

    from warranty_freshdesk_case import maybe_create_freshdesk_case  # noqa: W402

    result = maybe_create_freshdesk_case("tid-123", engine=engine)
    assert result["created"] is True
    assert result["freshdesk_ticket_id"] == "99901"
    assert "freshdesk.com/a/tickets/99901" in result["freshdesk_url"]

    with session_factory() as db:
        saved = (
            db.query(wm.WarrantyTicket)
            .filter(wm.WarrantyTicket.ticket_id == "tid-123")
            .first()
        )
        collected = saved.get_collected()
        assert collected["freshdesk_ticket_id"] == "99901"
        assert collected["case_reference"].startswith("WR-")


def test_maybe_create_freshdesk_case_phone_ivr_uses_caller_phone(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")

    fake_ticket = _FakeTicket()
    fake_ticket.status = "in_progress"
    fake_ticket.collected_data = json.dumps(
        {"channel": "phone", "caller_phone": "+15551234567"}
    )
    engine = MagicMock()
    engine.get_ticket.return_value = fake_ticket
    engine.get_turns.return_value = []

    captured: list[dict] = []

    class FakeResponse:
        status_code = 201

        @staticmethod
        def json():
            return {"id": 88801}

    def _fake_post(url, *args, **kwargs):
        captured.append(kwargs.get("json") or {})
        return FakeResponse()

    monkeypatch.setattr("warranty_freshdesk_case.requests.post", _fake_post)

    mem_engine = __import__("sqlalchemy").create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    from sqlalchemy.orm import sessionmaker

    import warranty_models as wm

    wm.Base.metadata.create_all(bind=mem_engine)
    session_factory = sessionmaker(bind=mem_engine)
    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", session_factory)

    with session_factory() as db:
        row = wm.WarrantyTicket(
            ticket_id="tid-123",
            session_id="sess-1",
            domain="phone",
            status="in_progress",
            current_node_id="issue_type",
            collected_data=fake_ticket.collected_data,
        )
        db.add(row)
        db.commit()

    from warranty_freshdesk_case import maybe_create_freshdesk_case  # noqa: W402

    result = maybe_create_freshdesk_case("tid-123", engine=engine, allow_any_status=True)
    assert result["created"] is True
    assert captured
    payload = captured[0]
    assert payload["phone"] == "+15551234567"
    assert "phone-ivr" in payload["tags"]
    assert "Caller phone" in payload["description"]


def test_maybe_sync_admin_decision_posts_private_note(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")

    fake_ticket = _FakeTicket()
    fake_ticket.collected_data = json.dumps({"freshdesk_ticket_id": "555"})
    engine = MagicMock()
    engine.get_ticket.return_value = fake_ticket

    posted: list[dict] = []

    def _fake_post(url, *args, **kwargs):
        posted.append({"url": url, "json": kwargs.get("json")})
        return MagicMock(status_code=201, text="ok")

    monkeypatch.setattr("warranty_freshdesk_case.requests.post", _fake_post)

    from warranty_freshdesk_case import maybe_sync_admin_decision_to_freshdesk  # noqa: W402

    result = maybe_sync_admin_decision_to_freshdesk(
        "tid-123",
        decision="approved",
        note="Looks good",
        customer_message="Your claim is approved.",
        decided_by="admin",
        engine=engine,
    )
    assert result["synced"] is True
    assert posted
    assert "Admin decision recorded" in posted[0]["json"]["body"]
    assert "approved" in posted[0]["json"]["body"]


def test_evidence_to_dict_public_hides_file_path():
    from warranty_models import WarrantyEvidence  # noqa: W402

    ev = WarrantyEvidence(
        ticket_id="tid",
        evidence_type="damage_photos",
        file_path="/secret/path/photo.jpg",
        original_filename="photo.jpg",
    )
    public = ev.to_dict_public()
    assert "file_path" not in public
    assert public["original_filename"] == "photo.jpg"
