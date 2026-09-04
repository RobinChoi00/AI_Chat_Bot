"""
tests/test_evidence_upload.py
==============================
Tests for the evidence upload endpoint:
    POST /api/v1/warranty/{ticket_id}/evidence

Uses FastAPI's TestClient with:
  - An in-memory SQLite DB (never touches production DB)
  - A temporary upload directory (cleaned up after each test)
  - No LLM calls
  - No email sending

Scenarios
---------
  1. valid jpg  → 200, metadata stored, file saved
  2. valid pdf  → 200, metadata stored
  3. invalid extension (.exe) rejected → 422
  4. unsafe filename (path traversal) → sanitised and stored safely
  5. metadata stored correctly in WarrantyEvidence table
  6. evidence notification queued (emailed flag updated when send succeeds)
  7. unknown ticket_id → 404
  8. oversized file → 413
  9. missing or invalid customer_email → 422
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Path + in-memory DB setup (must happen before app imports)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as _wm
import warranty_workflow as _wf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def in_memory_db(tmp_path, monkeypatch):
    """
    Replace the warranty DB engine with a shared in-memory SQLite for every test.

    Key points:
    - StaticPool: ensures all SQLAlchemy connections share the same SQLite
      in-memory connection (required for async FastAPI TestClient which may
      create connections on different threads/coroutines).
    - Patches both warranty_models._SessionFactory AND warranty_workflow._SessionFactory
      because warranty_workflow.py imports _SessionFactory as a direct module-level
      reference (not looked up via warranty_models globals at call time).
    - Redirects _UPLOAD_ROOT to tmp_path so test files never touch the real folder.
    """
    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _wm.Base.metadata.create_all(bind=mem_engine)
    mem_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=mem_engine,
        expire_on_commit=False,
    )
    monkeypatch.setattr(_wm, "_engine", mem_engine)
    monkeypatch.setattr(_wm, "_SessionFactory", mem_factory)
    # Also patch warranty_workflow's direct reference (mirrors test_warranty_flow.py)
    monkeypatch.setattr(_wf, "_SessionFactory", mem_factory)

    # Redirect _UPLOAD_ROOT to tmp_path so tests never write to the real folder
    import warranty_router as wr
    monkeypatch.setattr(wr, "_UPLOAD_ROOT", tmp_path)

    yield


@pytest.fixture(autouse=True)
def block_evidence_email_notification(monkeypatch):
    """Prevent background SMTP threads during most upload tests."""
    import warranty_email as we

    monkeypatch.setattr(we, "notify_evidence_upload_async", lambda **kwargs: None)


# ---------------------------------------------------------------------------
# FastAPI TestClient — built after DB is patched
# ---------------------------------------------------------------------------

_CUSTOMER_EMAIL = "customer@example.com"


def _upload_form(evidence_type: str) -> dict:
    return {
        "evidence_type": evidence_type,
        "customer_email": _CUSTOMER_EMAIL,
    }

@pytest.fixture()
def client(in_memory_db):
    """Return a TestClient for the FastAPI app (warranties router only)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import warranty_router as wr

    test_app = FastAPI()
    test_app.include_router(wr.router)
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Helper: create a real warranty ticket in the in-memory DB
# ---------------------------------------------------------------------------

def _make_ticket(session_id: str = "test-session") -> str:
    """Start a warranty session and return its ticket_id."""
    from warranty_workflow import WarrantyEngine
    ticket_id, _ = WarrantyEngine.start_session(session_id, "test.com")
    return ticket_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvidenceUploadEndpoint:
    """HTTP-level tests for POST /api/v1/warranty/{ticket_id}/evidence."""

    def test_valid_jpg_upload(self, client):
        """Uploading a valid JPEG returns 200 and stores metadata."""
        ticket_id = _make_ticket("jpg-test")
        jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # minimal JPEG header

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("damage_photos"),
            files={"file": ("photo.jpg", io.BytesIO(jpg_data), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ticket_id"] == ticket_id
        assert body["evidence_type"] == "damage_photos"
        assert body["original_filename"] == "photo.jpg"
        assert body["file_size_bytes"] == len(jpg_data)
        # Evidence ID must be a positive integer
        assert isinstance(body["evidence_id"], int)
        assert body["evidence_id"] > 0

    def test_valid_pdf_upload(self, client):
        """Uploading a valid PDF returns 200."""
        ticket_id = _make_ticket("pdf-test")
        pdf_data = b"%PDF-1.4 " + b"\x00" * 50

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("proof_of_purchase"),
            files={"file": ("receipt.pdf", io.BytesIO(pdf_data), "application/pdf")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["original_filename"] == "receipt.pdf"

    def test_webp_upload_accepted(self, client):
        ticket_id = _make_ticket("webp-test")
        webp_data = b"RIFF" + b"\x00" * 4 + b"WEBP"

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("photo_of_defect"),
            files={"file": ("defect.webp", io.BytesIO(webp_data), "image/webp")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["original_filename"] == "defect.webp"
        assert "saved_path" not in body

    @pytest.mark.parametrize(
        ("filename", "content_type", "payload"),
        [
            ("issue.avi", "video/x-msvideo", b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 8),
            ("issue.webm", "video/webm", b"\x1aE\xdf\xa3" + b"\x00" * 16),
        ],
    )
    def test_advertised_video_formats_are_accepted(
        self, client, filename, content_type, payload
    ):
        ticket_id = _make_ticket(f"video-{filename}")
        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("video_of_issue"),
            files={"file": (filename, io.BytesIO(payload), content_type)},
        )
        assert response.status_code == 200, response.text

    def test_invalid_extension_rejected(self, client):
        """Uploading a .exe file is rejected with HTTP 422."""
        ticket_id = _make_ticket("ext-test")

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("other"),
            files={"file": ("malware.exe", io.BytesIO(b"MZ" + b"\x00" * 50), "application/octet-stream")},
        )
        assert response.status_code == 422, response.text
        assert ".exe" in response.json()["detail"].lower() or "not allowed" in response.json()["detail"].lower()

    def test_invalid_extension_zip_rejected(self, client):
        """Uploading a .zip file is rejected with HTTP 422."""
        ticket_id = _make_ticket("zip-test")

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("other"),
            files={"file": ("archive.zip", io.BytesIO(b"PK" + b"\x00" * 50), "application/zip")},
        )
        assert response.status_code == 422

    def test_spoofed_file_contents_rejected_and_deleted(self, client, tmp_path):
        ticket_id = _make_ticket("spoof-test")
        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("damage_photos"),
            files={"file": ("fake.jpg", io.BytesIO(b"not-a-jpeg"), "image/jpeg")},
        )
        assert response.status_code == 422
        assert not [path for path in tmp_path.rglob("*") if path.is_file()]

    def test_unsafe_filename_path_traversal_sanitised(self, client, tmp_path):
        """
        A filename like '../../etc/passwd.jpg' must be sanitised before saving.
        The file should end up inside the ticket's upload folder, never outside.
        """
        ticket_id = _make_ticket("traversal-test")
        malicious_name = "../../etc/passwd.jpg"
        jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 20

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("damage_photos"),
            files={"file": (malicious_name, io.BytesIO(jpg_data), "image/jpeg")},
        )
        # Should succeed (sanitisation, not rejection)
        assert response.status_code == 200, response.text
        body = response.json()
        assert "saved_path" not in body

        from warranty_workflow import WarrantyEngine

        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 1
        saved_path = Path(str(evidences[0].file_path))
        assert str(saved_path).startswith(str(tmp_path)), (
            f"File was saved outside upload root! saved_path={saved_path}"
        )
        assert ".." not in saved_path.parts, (
            f"Saved path contains '..' components: {saved_path}"
        )

    def test_metadata_stored_in_db(self, client):
        """After upload, WarrantyEvidence row is retrievable from the engine."""
        from warranty_workflow import WarrantyEngine

        ticket_id = _make_ticket("meta-test")
        jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 30

        client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("photo_of_defect"),
            files={"file": ("defect.jpg", io.BytesIO(jpg_data), "image/jpeg")},
        )

        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 1
        ev = evidences[0]
        assert str(ev.evidence_type) == "photo_of_defect"
        assert str(ev.original_filename) == "defect.jpg"
        assert str(ev.file_path) != ""     # path was stored
        assert Path(str(ev.file_path)).exists()  # file actually exists on disk

        ticket = WarrantyEngine.get_ticket(ticket_id)
        assert ticket is not None
        assert ticket.get_collected().get("customer_contact_email") == _CUSTOMER_EMAIL
        assert str(evidences[0].customer_email) == _CUSTOMER_EMAIL

    def test_no_email_sent_when_notification_disabled(self, client):
        """When notification is blocked, emailed flag stays 0."""
        from warranty_workflow import WarrantyEngine
        from typing import cast as _cast

        ticket_id = _make_ticket("email-test")
        jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 20

        client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("damage_photos"),
            files={"file": ("img.jpg", io.BytesIO(jpg_data), "image/jpeg")},
        )

        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 1
        assert _cast(int, evidences[0].emailed) == 0

    def test_evidence_notification_queued(self, client, monkeypatch):
        """Successful upload queues an evidence notification with customer email."""
        import warranty_email as we

        calls = []
        monkeypatch.setattr(
            we,
            "notify_evidence_upload_async",
            lambda **kwargs: calls.append(kwargs),
        )

        ticket_id = _make_ticket("notify-test")
        jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 20

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data={"evidence_type": "damage_photos", "customer_email": "buyer@example.com"},
            files={"file": ("img.jpg", io.BytesIO(jpg_data), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        assert response.json()["customer_email"] == "b***r@example.com"
        assert len(calls) == 1
        assert calls[0]["customer_email"] == "buyer@example.com"
        assert calls[0]["ticket_id"] == ticket_id

    def test_missing_customer_email_rejected(self, client):
        """Upload without customer_email is rejected with HTTP 422."""
        ticket_id = _make_ticket("no-email-test")
        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data={"evidence_type": "damage_photos"},
            files={"file": ("img.jpg", io.BytesIO(b"\xff\xd8"), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_invalid_customer_email_rejected(self, client):
        """Upload with an invalid customer_email is rejected with HTTP 422."""
        ticket_id = _make_ticket("bad-email-test")
        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data={"evidence_type": "damage_photos", "customer_email": "not-an-email"},
            files={"file": ("img.jpg", io.BytesIO(b"\xff\xd8"), "image/jpeg")},
        )
        assert response.status_code == 422
        assert "email" in response.json()["detail"].lower()

    def test_unknown_ticket_returns_404(self, client):
        """Uploading evidence for a nonexistent ticket returns HTTP 404."""
        response = client.post(
            "/api/v1/warranty/ghost-ticket-9999/evidence",
            data=_upload_form("damage_photos"),
            files={"file": ("img.jpg", io.BytesIO(b"\xff\xd8"), "image/jpeg")},
        )
        assert response.status_code == 404

    def test_oversized_file_rejected(self, client):
        """A file exceeding 20 MB is rejected with HTTP 413."""
        ticket_id = _make_ticket("size-test")
        # 20 MB + 1 byte
        big_data = b"\x00" * (20 * 1024 * 1024 + 1)

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data=_upload_form("video_of_issue"),
            files={"file": ("huge.mp4", io.BytesIO(big_data), "video/mp4")},
        )
        assert response.status_code == 413

    def test_list_evidence_endpoint(self, client):
        """GET /api/v1/warranty/{ticket_id}/evidence lists uploaded files."""
        from warranty_workflow import WarrantyEngine

        ticket_id = _make_ticket("list-test")
        for i in range(2):
            client.post(
                f"/api/v1/warranty/{ticket_id}/evidence",
                data=_upload_form("damage_photos"),
            files={"file": (f"img{i}.jpg", io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 10), "image/jpeg")},
            )

        response = client.get(f"/api/v1/warranty/{ticket_id}/evidence")
        assert response.status_code == 200
        body = response.json()
        assert body["ticket_id"] == ticket_id
        assert len(body["evidence"]) == 2

    def test_multiple_uploads_same_ticket(self, client):
        """Multiple files can be uploaded to the same ticket independently."""
        ticket_id = _make_ticket("multi-test")

        files = [
            ("photo1.jpg", b"\xff\xd8\xff" + b"\x00" * 10, "image/jpeg", "damage_photos"),
            ("receipt.pdf", b"%PDF-" + b"\x00" * 10, "application/pdf", "proof_of_purchase"),
        ]
        ids = []
        for fname, data, mime, ev_type in files:
            r = client.post(
                f"/api/v1/warranty/{ticket_id}/evidence",
                data=_upload_form(ev_type),
                files={"file": (fname, io.BytesIO(data), mime)},
            )
            assert r.status_code == 200, r.text
            ids.append(r.json()["evidence_id"])

        # Each upload gets a distinct DB ID
        assert len(set(ids)) == len(ids), "Duplicate evidence IDs returned"


class TestWarrantyContactEndpoint:
    """HTTP-level tests for POST /api/v1/warranty/{ticket_id}/contact (email-only / N/A)."""

    def _terminal_ticket(self) -> str:
        from warranty_workflow import WarrantyEngine

        ticket_id, _ = WarrantyEngine.start_session("contact-test", "test.com")
        WarrantyEngine.submit_answer(ticket_id, "warranty")
        WarrantyEngine.submit_answer(ticket_id, "installation")
        WarrantyEngine.submit_answer(ticket_id, "OS-4000T")
        WarrantyEngine.submit_answer(ticket_id, "footrest_or_no_air")
        return ticket_id

    def test_email_only_contact_on_terminal(self, client, monkeypatch):
        from warranty_workflow import WarrantyEngine

        transcript_calls = []
        notify_calls = []
        monkeypatch.setattr(
            "warranty_email.send_warranty_transcript_email",
            lambda **kwargs: transcript_calls.append(kwargs) or True,
        )
        monkeypatch.setattr(
            "warranty_email.notify_email_only_contact_async",
            lambda **kwargs: notify_calls.append(kwargs),
        )
        monkeypatch.setattr(
            "warranty_email.send_customer_receipt_email",
            lambda **_k: False,
        )

        ticket_id = self._terminal_ticket()
        response = client.post(
            f"/api/v1/warranty/{ticket_id}/contact",
            json={"customer_email": "buyer@example.com", "evidence_na": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["customer_email"] == "b***r@example.com"
        assert body["evidence_type"] == "not_available"
        assert body["evidence_na"] is True
        assert "case_summary" in body
        assert body["case_summary_source"] in {"llm", "deterministic"}

        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 1
        assert str(evidences[0].evidence_type) == "not_available"
        assert str(evidences[0].original_filename) == "N/A"
        assert str(evidences[0].customer_email) == "buyer@example.com"

        ticket = WarrantyEngine.get_ticket(ticket_id)
        assert ticket is not None
        assert ticket.get_collected().get("customer_contact_email") == "buyer@example.com"
        assert ticket.get_collected().get("evidence_na") == "1"
        assert len(transcript_calls) == 1
        assert len(notify_calls) == 1
        assert str(body.get("case_reference") or "").startswith("WR-")

    def test_email_only_contact_sends_customer_receipt(self, client, monkeypatch):
        receipt_calls = []
        monkeypatch.setattr(
            "warranty_email.send_warranty_transcript_email",
            lambda **_k: True,
        )
        monkeypatch.setattr(
            "warranty_email.notify_email_only_contact_async",
            lambda **_k: None,
        )
        monkeypatch.setattr(
            "warranty_email.send_customer_receipt_email",
            lambda **kwargs: receipt_calls.append(kwargs) or True,
        )
        monkeypatch.setattr("warranty_email.EMAIL_SENDER", "bot@example.com")
        monkeypatch.setattr("warranty_email.EMAIL_PASSWORD", "secret")

        ticket_id = self._terminal_ticket()
        response = client.post(
            f"/api/v1/warranty/{ticket_id}/contact",
            json={"customer_email": "buyer@example.com", "evidence_na": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["receipt_email_sent"] is True
        assert body["case_reference"].startswith("WR-")
        assert len(receipt_calls) == 1
        assert receipt_calls[0]["to_email"] == "buyer@example.com"
        assert receipt_calls[0]["case_reference"].startswith("WR-")

    def test_contact_rejected_before_terminal(self, client):
        from warranty_workflow import WarrantyEngine

        ticket_id, _ = WarrantyEngine.start_session("contact-early", "test.com")
        WarrantyEngine.submit_answer(ticket_id, "warranty")
        WarrantyEngine.submit_answer(ticket_id, "defect")

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/contact",
            json={"customer_email": "buyer@example.com", "evidence_na": True},
        )
        assert response.status_code == 422

    def test_contact_invalid_email_rejected(self, client):
        ticket_id = self._terminal_ticket()
        response = client.post(
            f"/api/v1/warranty/{ticket_id}/contact",
            json={"customer_email": "not-an-email", "evidence_na": True},
        )
        assert response.status_code == 422
