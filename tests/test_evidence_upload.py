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
  6. no email sent (emailed flag stays 0)
  7. unknown ticket_id → 404
  8. oversized file → 413
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


# ---------------------------------------------------------------------------
# FastAPI TestClient — built after DB is patched
# ---------------------------------------------------------------------------

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
            data={"evidence_type": "damage_photos"},
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
            data={"evidence_type": "proof_of_purchase"},
            files={"file": ("receipt.pdf", io.BytesIO(pdf_data), "application/pdf")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["original_filename"] == "receipt.pdf"

    def test_invalid_extension_rejected(self, client):
        """Uploading a .exe file is rejected with HTTP 422."""
        ticket_id = _make_ticket("ext-test")

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data={"evidence_type": "other"},
            files={"file": ("malware.exe", io.BytesIO(b"MZ" + b"\x00" * 50), "application/octet-stream")},
        )
        assert response.status_code == 422, response.text
        assert ".exe" in response.json()["detail"].lower() or "not allowed" in response.json()["detail"].lower()

    def test_invalid_extension_zip_rejected(self, client):
        """Uploading a .zip file is rejected with HTTP 422."""
        ticket_id = _make_ticket("zip-test")

        response = client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data={"evidence_type": "other"},
            files={"file": ("archive.zip", io.BytesIO(b"PK" + b"\x00" * 50), "application/zip")},
        )
        assert response.status_code == 422

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
            data={"evidence_type": "damage_photos"},
            files={"file": (malicious_name, io.BytesIO(jpg_data), "image/jpeg")},
        )
        # Should succeed (sanitisation, not rejection)
        assert response.status_code == 200, response.text
        body = response.json()
        saved_path = Path(body["saved_path"])
        # Verify the saved path is inside tmp_path (the mocked _UPLOAD_ROOT)
        assert str(saved_path).startswith(str(tmp_path)), (
            f"File was saved outside upload root! saved_path={saved_path}"
        )
        # The filename must NOT contain '..' components
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
            data={"evidence_type": "photo_of_defect"},
            files={"file": ("defect.jpg", io.BytesIO(jpg_data), "image/jpeg")},
        )

        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 1
        ev = evidences[0]
        assert str(ev.evidence_type) == "photo_of_defect"
        assert str(ev.original_filename) == "defect.jpg"
        assert str(ev.file_path) != ""     # path was stored
        assert Path(str(ev.file_path)).exists()  # file actually exists on disk

    def test_no_email_sent(self, client):
        """After upload, emailed flag must remain 0 (email not sent in D-lite)."""
        from warranty_workflow import WarrantyEngine
        from typing import cast as _cast

        ticket_id = _make_ticket("email-test")
        jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 20

        client.post(
            f"/api/v1/warranty/{ticket_id}/evidence",
            data={"evidence_type": "damage_photos"},
            files={"file": ("img.jpg", io.BytesIO(jpg_data), "image/jpeg")},
        )

        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 1
        assert _cast(int, evidences[0].emailed) == 0, (
            "emailed flag must be 0 — email sending is not implemented in Phase D-lite"
        )

    def test_unknown_ticket_returns_404(self, client):
        """Uploading evidence for a nonexistent ticket returns HTTP 404."""
        response = client.post(
            "/api/v1/warranty/ghost-ticket-9999/evidence",
            data={"evidence_type": "damage_photos"},
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
            data={"evidence_type": "video_of_issue"},
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
                data={"evidence_type": "damage_photos"},
                files={"file": (f"img{i}.jpg", io.BytesIO(b"\xff\xd8" + b"\x00" * 10), "image/jpeg")},
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
            ("photo1.jpg", b"\xff\xd8" + b"\x00" * 10, "image/jpeg", "damage_photos"),
            ("receipt.pdf", b"%PDF" + b"\x00" * 10, "application/pdf", "proof_of_purchase"),
        ]
        ids = []
        for fname, data, mime, ev_type in files:
            r = client.post(
                f"/api/v1/warranty/{ticket_id}/evidence",
                data={"evidence_type": ev_type},
                files={"file": (fname, io.BytesIO(data), mime)},
            )
            assert r.status_code == 200, r.text
            ids.append(r.json()["evidence_id"])

        # Each upload gets a distinct DB ID
        assert len(set(ids)) == len(ids), "Duplicate evidence IDs returned"
