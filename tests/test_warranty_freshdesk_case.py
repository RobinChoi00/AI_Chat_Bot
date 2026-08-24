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


def test_case_description_strips_role_prefix_and_markdown_from_timeline():
    """Fred reported tickets like ``[assistant/enrichment] For your **Titan
    3D Prestige**, **a remote where some blackberries do not respond**``.
    The description helper must:
      1. drop the raw ``[role/kind]`` prefix,
      2. strip ``**bold**`` markdown, and
      3. dedupe repeated enrichment tips.
    """
    from warranty_freshdesk_case import _build_case_description  # noqa: WPS433

    ticket = _FakeTicket()
    ticket.issue_type = "defect"
    ticket.model_name = "Titan 3D Prestige"
    ticket.current_node_id = "defect_remote_partial_terminal"
    ticket.collected_data = json.dumps(
        {
            "intake_summary": "Remote / controller issue.",
            "chat_timeline": [
                {
                    "role": "assistant",
                    "kind": "enrichment",
                    "text": (
                        "For your **Titan 3D Prestige**, **a remote where "
                        "some buttons do not respond** can often be improved "
                        "by checking the **cable, fuse, and connections**."
                    ),
                    "node_id": "defect_remote_partial",
                },
                {
                    "role": "assistant",
                    "kind": "enrichment",
                    "text": (
                        "For your **Titan 3D Prestige**, **a remote where "
                        "some buttons do not respond** can often be improved "
                        "by checking the **cable, fuse, and connections**."
                    ),
                    "node_id": "defect_remote_partial",
                },
                {
                    "role": "user",
                    "kind": "side_question",
                    "text": "is the remote replaceable?",
                    "node_id": "defect_remote_partial",
                },
            ],
        }
    )

    description = _build_case_description(ticket, case_ref="WR-TEST", turns=[])

    assert "[assistant/enrichment]" not in description
    assert "**" not in description
    # Freshdesk needs HTML or newlines collapse into a wall of text.
    assert "<table" in description
    assert "<h3" in description
    assert "Case summary" in description
    assert description.count("Bot shared tip") == 1  # dedupe worked
    assert "is the remote replaceable?" in description
    assert "Bot / customer notes" in description
    assert "Extra chat tips / side questions:" not in description


def test_case_description_includes_soft_eligibility_and_failed_lookup():
    from warranty_freshdesk_case import _build_case_description  # noqa: WPS433

    ticket = _FakeTicket()
    ticket.collected_data = json.dumps(
        {
            "warranty_eligibility_status": "possibly_expired",
            "purchase_date": "2020-01-10",
            "delivery_lookup_failed": "1",
            "tracking_number": "1Z999AA10123456784",
            "delivery_lookup_error": "Carrier lookup failed (API error)",
        }
    )
    description = _build_case_description(ticket, case_ref="WR-TEST", turns=[])
    assert "possibly_expired" in description
    assert "2020-01-10" in description
    assert "Failed" in description
    assert "1Z999AA10123456784" in description
    assert "Carrier lookup failed" in description


def test_case_description_truncation_stops_on_word_boundary():
    """A message with a long tail must be trimmed on whitespace + ellipsis,
    never mid-word (regression: ``blackberries`` came from a hard slice)."""
    from warranty_freshdesk_case import _smart_truncate  # noqa: WPS433

    text = (
        "For your Titan 3D Prestige a remote where some buttons do not "
        "respond can often be improved by checking the cable fuse and "
        "connections and finally the main harness under the seat cover"
    )
    trimmed = _smart_truncate(text, 80)
    assert trimmed.endswith("…")
    assert " " in trimmed[-30:] or trimmed.endswith(" …") or trimmed[-2] != "r"
    assert len(trimmed) <= 82
    # Must never mid-cut a word: last non-ellipsis char is a full word.
    tail = trimmed.rstrip("…").rstrip()
    assert not tail.endswith("connectio")


def test_case_description_strips_baked_in_role_prefix_and_drops_offtopic_tips():
    """Fred's WR ticket had ``[assistant/enrichment]`` mid-body AND a
    footrest-backorder tip on a remote/controller case. Both must be gone."""
    from warranty_freshdesk_case import _build_case_description  # noqa: WPS433

    ticket = _FakeTicket()
    ticket.issue_type = "defect"
    ticket.model_name = "Titan 3D Prestige"
    ticket.current_node_id = "defect_remote_partial_terminal"
    ticket.collected_data = json.dumps(
        {
            "intake_summary": "Remote / controller issue.",
            "chat_timeline": [
                {
                    "role": "assistant",
                    "kind": "enrichment",
                    "text": (
                        "[assistant/enrichment] For your Titan 3D Prestige, "
                        "a remote where some buttons do not respond can often "
                        "be improved by checking the cable, fuse, and connections."
                    ),
                },
                {
                    "role": "assistant",
                    "kind": "enrichment",
                    "text": (
                        "Our parts department confirmed that the footrest part "
                        "for this chair is currently very backordered."
                    ),
                },
                {
                    "role": "user",
                    "kind": "side_question",
                    "text": "is the remote replaceable?",
                },
            ],
        }
    )

    description = _build_case_description(ticket, case_ref="WR-TEST", turns=[])

    assert "[assistant/enrichment]" not in description
    assert "footrest" not in description.lower()
    assert "backordered" not in description.lower()
    assert "Bot shared tip" in description
    assert "remote" in description.lower()
    assert "is the remote replaceable?" in description
    assert "<table" in description
    assert "Customer intake" in description


def test_case_description_uses_html_sections_not_plain_newlines():
    """Regression for Fred's wall-of-text ticket: Freshdesk renders HTML,
    so the description must be structured HTML (table + headings), not a
    single plain-text blob joined by ``\\n``."""
    from warranty_freshdesk_case import _build_case_description  # noqa: WPS433

    class _Turn:
        node_prompt = "Which part of the chair is the problem?"
        customer_answer = "Remote / controller"
        answer_key = "remote"

    ticket = _FakeTicket()
    ticket.issue_type = "defect"
    ticket.model_name = "Titan 3D Prestige"
    ticket.collected_data = json.dumps(
        {"intake_summary": "Remote / controller issue."}
    )

    description = _build_case_description(
        ticket, case_ref="WR-20260731-EE75F8", turns=[_Turn()]
    )

    assert description.strip().startswith("<")
    assert "Case summary" in description
    assert "Workflow answers" in description
    assert "Customer intake" in description
    assert "WR-20260731-EE75F8" in description
    assert "Titan 3D Prestige" in description
    assert "Remote / controller" in description
    # Must not be a newline-joined plain blob (Freshdesk collapses those).
    assert "\nCase reference:" not in description


def test_resolve_freshdesk_ticket_type_mapping(monkeypatch):
    monkeypatch.delenv("FRESHDESK_WARRANTY_TYPE", raising=False)
    monkeypatch.delenv("FRESHDESK_WARRANTY_DEFAULT_TYPE", raising=False)

    from warranty_freshdesk_case import resolve_freshdesk_ticket_type  # noqa: W402

    damage = _FakeTicket()
    assert resolve_freshdesk_ticket_type(damage) == "Issue with Product"

    status = _FakeTicket()
    status.current_node_id = "delivery_status_terminal"
    assert resolve_freshdesk_ticket_type(status) == "Inquiry"

    class _Turn:
        def __init__(self, node_id, answer_key):
            self.node_id = node_id
            self.answer_key = answer_key

    intent_ticket = _FakeTicket()
    intent_ticket.current_node_id = "delivery_get_tracking_number"
    assert (
        resolve_freshdesk_ticket_type(
            intent_ticket,
            turns=[_Turn("delivery_intent_q", "status_check")],
        )
        == "Inquiry"
    )

    defect = _FakeTicket()
    defect.issue_type = "defect"
    assert resolve_freshdesk_ticket_type(defect) == "Issue with Product"

    phone = _FakeTicket()
    phone.issue_type = None
    phone.current_node_id = "issue_type"
    phone.collected_data = json.dumps({"channel": "phone"})
    assert resolve_freshdesk_ticket_type(phone) == "Inquiry"

    monkeypatch.setenv("FRESHDESK_WARRANTY_TYPE", "Service Task")
    assert resolve_freshdesk_ticket_type(damage) == "Service Task"


def test_maybe_create_freshdesk_case_creates_and_persists(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")
    monkeypatch.delenv("FRESHDESK_WARRANTY_TYPE", raising=False)

    fake_ticket = _FakeTicket()
    engine = MagicMock()
    engine.get_ticket.return_value = fake_ticket
    engine.get_turns.return_value = []

    captured: list[dict] = []

    class FakeResponse:
        status_code = 201

        @staticmethod
        def json():
            return {"id": 99901}

    def _fake_post(*args, **kwargs):
        captured.append(kwargs.get("json") or {})
        return FakeResponse()

    monkeypatch.setattr(
        "warranty_freshdesk_case.requests.post",
        _fake_post,
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
    assert captured
    assert captured[0]["type"] == "Issue with Product"

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
    assert payload["type"] in {
        "Issue with Product",
        "Inquiry",
        "Purchase Parts",
        "Service Task",
        "IDNT",
    }


def test_maybe_create_freshdesk_case_persists_http_failure(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")
    monkeypatch.delenv("FRESHDESK_WARRANTY_TYPE", raising=False)

    fake_ticket = _FakeTicket()
    engine = MagicMock()
    engine.get_ticket.return_value = fake_ticket
    engine.get_turns.return_value = []

    class FakeResponse:
        status_code = 400
        text = '{"description":"Validation failed","errors":[{"field":"type","message":"Unexpected/invalid field value"}]}'

        @staticmethod
        def json():
            return {}

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
    assert result["created"] is False
    assert result["error"] == "http_400"
    assert "Validation failed" in (result.get("detail") or "")
    assert result.get("failed_at")
    assert result.get("attempt_count") == 1

    with session_factory() as db:
        saved = (
            db.query(wm.WarrantyTicket)
            .filter(wm.WarrantyTicket.ticket_id == "tid-123")
            .first()
        )
        collected = saved.get_collected()
        assert collected["freshdesk_create_error"] == "http_400"
        assert "Validation failed" in collected["freshdesk_create_error_detail"]
        assert collected["freshdesk_create_attempt_count"] == 1
        assert collected["case_reference"].startswith("WR-")
        assert "freshdesk_ticket_id" not in collected


def test_maybe_create_freshdesk_case_clears_error_on_success(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")

    fake_ticket = _FakeTicket()
    fake_ticket.collected_data = json.dumps(
        {
            "freshdesk_create_error": "http_400",
            "freshdesk_create_error_detail": "old failure",
            "freshdesk_create_failed_at": "2026-01-01T00:00:00Z",
            "freshdesk_create_attempt_count": 2,
        }
    )
    engine = MagicMock()
    engine.get_ticket.return_value = fake_ticket
    engine.get_turns.return_value = []

    class FakeResponse:
        status_code = 201

        @staticmethod
        def json():
            return {"id": 777}

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
            collected_data=fake_ticket.collected_data,
        )
        db.add(row)
        db.commit()

    from warranty_freshdesk_case import maybe_create_freshdesk_case  # noqa: W402

    result = maybe_create_freshdesk_case("tid-123", engine=engine)
    assert result["created"] is True

    with session_factory() as db:
        saved = (
            db.query(wm.WarrantyTicket)
            .filter(wm.WarrantyTicket.ticket_id == "tid-123")
            .first()
        )
        collected = saved.get_collected()
        assert collected["freshdesk_ticket_id"] == "777"
        assert "freshdesk_create_error" not in collected
        assert "freshdesk_create_error_detail" not in collected
        assert "freshdesk_create_failed_at" not in collected
        assert collected["freshdesk_create_attempt_count"] == 3


def test_schedule_freshdesk_case_creation_runs_sync_by_default(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_CREATE_CASE", "1")
    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")
    monkeypatch.delenv("WARRANTY_FRESHDESK_ASYNC_CREATE", raising=False)

    calls: list[str] = []

    def _fake_maybe(ticket_id, **kwargs):
        calls.append(ticket_id)
        return {"created": True, "freshdesk_ticket_id": "99"}

    monkeypatch.setattr(
        "warranty_freshdesk_case.maybe_create_freshdesk_case",
        _fake_maybe,
    )

    from warranty_freshdesk_case import schedule_freshdesk_case_creation  # noqa: W402

    result = schedule_freshdesk_case_creation("tid-sync")
    assert calls == ["tid-sync"]
    assert result["freshdesk_ticket_id"] == "99"


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
