"""Tests for warranty transcript email capture."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_email import (  # noqa: E402
    build_admin_decision_customer_body,
    build_evidence_notification_body,
    build_phone_ivr_team_email_body,
    build_transcript_body,
    extract_email,
    maybe_send_admin_decision_customer_email,
    maybe_send_warranty_transcript,
    resolve_customer_email,
    send_admin_decision_customer_email,
    send_evidence_upload_notification,
    send_phone_ivr_team_email,
)


class FakeTurn:
    def __init__(self, node_id, prompt, answer):
        self.node_id = node_id
        self.node_prompt = prompt
        self.customer_answer = answer


class FakeTicket:
    def __init__(self):
        self.ticket_id = "T-123"
        self.session_id = "sess-abc"
        self.domain = "osaki.com"
        self.status = "in_progress"
        self.issue_type = "delivery"
        self.model_name = ""
        self._collected = {}

    def get_collected(self):
        return dict(self._collected)

    def set_collected(self, key, value):
        self._collected[key] = value


def test_extract_email_from_text():
    assert extract_email("Reach me at Buyer.Name@Example.com please") == "buyer.name@example.com"
    assert extract_email("no email here") is None


def test_build_transcript_body_includes_turns():
    body = build_transcript_body(
        ticket_id="T-1",
        session_id="sess-1",
        customer_email="buyer@example.com",
        domain="osaki.com",
        ticket_status="awaiting_admin_review",
        issue_type="defect",
        model_name="OS-4000T",
        turns=[FakeTurn("install_model", "What model?", "OS-4000T")],
        chat_messages=[{"role": "user", "content": "My chair is broken"}],
    )
    assert "buyer@example.com" in body
    assert "T-1" in body
    assert "What model?" in body
    assert "My chair is broken" in body


def test_build_phone_ivr_team_email_body_includes_caller_and_turns():
    body = build_phone_ivr_team_email_body(
        caller_phone="+15551234567",
        session_id="rc-session-1",
        ticket_id="T-IVR-1",
        case_reference="WR-20260701-ABC123",
        ticket_status="in_progress",
        issue_type="defect",
        model_name="OS-4000T",
        current_node_id="defect_air",
        turns=[FakeTurn("defect_problem_type", "What type of problem?", "1")],
        sms_sent=True,
    )
    assert "+15551234567" in body
    assert "WR-20260701-ABC123" in body
    assert "What type of problem?" in body
    assert "Sent to caller's phone number." in body
    assert "After-hours warranty phone IVR" in body


def test_send_phone_ivr_team_email_skips_without_smtp_config(monkeypatch):
    monkeypatch.setattr("warranty_email.EMAIL_SENDER", "")
    monkeypatch.setattr("warranty_email.EMAIL_PASSWORD", "")
    assert (
        send_phone_ivr_team_email(
            caller_phone="+15551234567",
            session_id="rc-session-2",
            ticket_id="T-IVR-2",
        )
        is False
    )


def test_maybe_send_records_collected_data(monkeypatch):
    ticket = FakeTicket()
    sent_calls = []

    def fake_send(**kwargs):
        sent_calls.append(kwargs)
        return True

    monkeypatch.setattr("warranty_email.send_warranty_transcript_email", fake_send)

    email, sent = maybe_send_warranty_transcript(
        ticket=ticket,
        answer_text="buyer@example.com",
        turns=[],
    )
    assert email == "buyer@example.com"
    assert sent is True
    assert ticket.get_collected()["transcript_emailed"] == "1"
    assert ticket.get_collected()["customer_contact_email"] == "buyer@example.com"
    assert len(sent_calls) == 1


def test_maybe_send_skips_duplicate(monkeypatch):
    ticket = FakeTicket()
    ticket.set_collected("transcript_emailed", "1")

    monkeypatch.setattr(
        "warranty_email.send_warranty_transcript_email",
        lambda **_k: pytest.fail("should not send twice"),
    )

    email, sent = maybe_send_warranty_transcript(
        ticket=ticket,
        answer_text="buyer@example.com",
        turns=[],
    )
    assert email == "buyer@example.com"
    assert sent is False


def test_maybe_send_stores_email_even_when_smtp_fails(monkeypatch):
    ticket = FakeTicket()
    monkeypatch.setattr(
        "warranty_email.send_warranty_transcript_email",
        lambda **_k: False,
    )

    email, sent = maybe_send_warranty_transcript(
        ticket=ticket,
        answer_text="buyer@example.com",
        turns=[],
    )
    assert email == "buyer@example.com"
    assert sent is False
    assert ticket.get_collected()["customer_contact_email"] == "buyer@example.com"


def test_resolve_customer_email_from_collected_and_turns():
    ticket = FakeTicket()
    ticket.set_collected("order_or_email", "akhattakster@gmail.com")
    assert resolve_customer_email(ticket) == "akhattakster@gmail.com"

    ticket2 = FakeTicket()
    turn = FakeTurn("delivery_get_name", "Order?", "buyer@example.com")
    assert resolve_customer_email(ticket2, turns=[turn]) == "buyer@example.com"
    body = build_evidence_notification_body(
        ticket_id="T-99",
        customer_email="buyer@example.com",
        evidence_type="video_of_issue",
        original_filename="issue.mp4",
        file_size_bytes=12345,
        issue_type="defect",
        model_name="OS-4000T",
    )
    assert "buyer@example.com" in body
    assert "T-99" in body
    assert "issue.mp4" in body
    assert "OS-4000T" in body


def test_send_evidence_upload_notification(monkeypatch, tmp_path):
    attachment = tmp_path / "photo.jpg"
    attachment.write_bytes(b"\xff\xd8\xff")

    sent_messages = []

    class FakeSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def ehlo(self):
            return None

        def starttls(self):
            return None

        def login(self, *_args, **_kwargs):
            return None

        def send_message(self, msg):
            sent_messages.append(msg)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("warranty_email.EMAIL_SENDER", "bot@example.com")
    monkeypatch.setattr("warranty_email.EMAIL_PASSWORD", "secret")
    monkeypatch.setattr("warranty_email.smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr(
        "warranty_email._resolve_case_summary",
        lambda **_k: {
            "summary": "Test summary for evidence upload.",
            "suggested_subject": "OS-4000T test",
            "source": "deterministic",
        },
    )

    ok = send_evidence_upload_notification(
        ticket_id="T-55",
        customer_email="buyer@example.com",
        evidence_type="damage_photos",
        original_filename="photo.jpg",
        file_path=str(attachment),
        mime_type="image/jpeg",
        file_size_bytes=4,
        recipients=[("Test User", "team@example.com")],
    )
    assert ok is True
    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert msg["To"] == "team@example.com"
    assert msg["Reply-To"] == "buyer@example.com"
    assert "T-55" in msg["Subject"]
    part = msg.get_payload()[0]
    body_text = part.get_payload(decode=True).decode("utf-8")
    assert "Test summary for evidence upload." in body_text


def test_build_admin_decision_customer_body_excludes_internal_note():
    body = build_admin_decision_customer_body(
        ticket_id="WR-100",
        customer_message="Your replacement has been approved.",
        model_name="OS-4000T",
        issue_type="defect",
    )
    assert "Your replacement has been approved." in body
    assert "WR-100" in body
    assert "OS-4000T" in body
    assert "defect" in body
    assert "Internal" not in body


def test_maybe_send_admin_decision_skips_without_message():
    ticket = FakeTicket()
    ticket.set_collected("customer_contact_email", "buyer@example.com")
    sent, reason = maybe_send_admin_decision_customer_email(
        ticket=ticket,
        decision="approved",
        customer_message="",
    )
    assert sent is False
    assert reason == "no_customer_message"


def test_maybe_send_admin_decision_skips_admin_reviewing():
    ticket = FakeTicket()
    ticket.set_collected("customer_contact_email", "buyer@example.com")
    sent, reason = maybe_send_admin_decision_customer_email(
        ticket=ticket,
        decision="admin_reviewing",
        customer_message="We are reviewing your case.",
    )
    assert sent is False
    assert reason == "decision_not_notifiable"


def test_maybe_send_admin_decision_sends_when_ready(monkeypatch):
    ticket = FakeTicket()
    ticket.set_collected("customer_contact_email", "buyer@example.com")
    calls = []

    def fake_send(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr("warranty_email.send_admin_decision_customer_email", fake_send)

    sent, reason = maybe_send_admin_decision_customer_email(
        ticket=ticket,
        decision="approved",
        customer_message="Your claim is approved.",
    )
    assert sent is True
    assert reason is None
    assert len(calls) == 1
    assert calls[0]["to_email"] == "buyer@example.com"
    assert calls[0]["customer_message"] == "Your claim is approved."
    assert calls[0]["decision"] == "approved"


def test_send_admin_decision_customer_email_sets_reply_to_team(monkeypatch):
    sent_messages = []

    class FakeSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def ehlo(self):
            return None

        def starttls(self):
            return None

        def login(self, *_args, **_kwargs):
            return None

        def send_message(self, msg):
            sent_messages.append(msg)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("warranty_email.EMAIL_SENDER", "bot@example.com")
    monkeypatch.setattr("warranty_email.EMAIL_PASSWORD", "secret")
    monkeypatch.setattr("warranty_email.WARRANTY_TEAM_EMAIL", "service@osakititan.com")
    monkeypatch.setattr("warranty_email.smtplib.SMTP", FakeSMTP)

    ok = send_admin_decision_customer_email(
        to_email="buyer@example.com",
        ticket_id="WR-77",
        decision="rejected",
        customer_message="We cannot approve this claim.",
    )
    assert ok is True
    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert msg["To"] == "buyer@example.com"
    assert msg["Reply-To"] == "service@osakititan.com"
    assert "Not approved" in msg["Subject"]
    assert "WR-77" in msg["Subject"]
