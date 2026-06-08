"""Tests for warranty transcript email capture."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_email import (  # noqa: E402
    build_evidence_notification_body,
    build_transcript_body,
    extract_email,
    maybe_send_warranty_transcript,
    send_evidence_upload_notification,
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


def test_build_evidence_notification_body():
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
