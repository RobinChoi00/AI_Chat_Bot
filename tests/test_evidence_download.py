"""
tests/test_evidence_download.py
================================
Tests for the admin evidence download endpoint:
    GET /admin/warranty/{ticket_id}/evidence/{evidence_id}/download

Security scenarios covered
--------------------------
1.  Valid download     → 200, correct bytes returned
2.  Wrong ticket_id    → 404  (cross-ticket access blocked)
3.  Unknown evidence_id → 404
4.  File missing on disk → 404
5.  No X-Admin-Key header → 401
6.  Wrong X-Admin-Key → 401
7.  ADMIN_API_KEY not configured on server → 503

Also validates:
- Content-Type header is set correctly for images
- raw server file path is NOT exposed in any response body
- Path-traversal safety (file must be inside _UPLOAD_ROOT)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast, Generator

import pytest
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Ensure app/ directory is on sys.path before any app imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_models as _wm
import warranty_workflow as _wf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Admin key used in tests — must match what the fixture patches
ADMIN_KEY = "test-admin-key-dl"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def in_memory_db(tmp_path, monkeypatch) -> Generator:
    """
    Shared in-memory SQLite DB + tmp upload dir for every test in this module.

    Mirrors the setup from test_evidence_upload.py:
    - StaticPool keeps all connections on the same in-memory SQLite connection.
    - Patches both _SessionFactory references so warranty_workflow's module-level
      import is also redirected (avoids "split-module" issues in tests).
    - Redirects _UPLOAD_ROOT to tmp_path so tests never touch real files.
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
    monkeypatch.setattr(_wf, "_SessionFactory", mem_factory)

    import warranty_router as wr
    monkeypatch.setattr(wr, "_UPLOAD_ROOT", tmp_path)

    yield


@pytest.fixture()
def admin_client(in_memory_db, monkeypatch):
    """
    TestClient with ADMIN_API_KEY configured.

    The warrant_router's module-level _ADMIN_API_KEY variable is patched
    so _require_admin() picks up the test key at call time.
    """
    import warranty_router as wr
    monkeypatch.setattr(wr, "_ADMIN_API_KEY", ADMIN_KEY)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    test_app = FastAPI()
    test_app.include_router(wr.router)
    return TestClient(test_app)


@pytest.fixture()
def unconfigured_client(in_memory_db, monkeypatch):
    """TestClient where ADMIN_API_KEY is intentionally not set (empty string)."""
    import warranty_router as wr
    monkeypatch.setattr(wr, "_ADMIN_API_KEY", "")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    test_app = FastAPI()
    test_app.include_router(wr.router)
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_ticket(session_id: str = "dl-session") -> str:
    """Start a warranty session and return the ticket_id."""
    from warranty_workflow import WarrantyEngine
    ticket_id, _ = WarrantyEngine.start_session(session_id, "test.com")
    return ticket_id


def _write_evidence(
    tmp_path: Path,
    ticket_id: str,
    filename: str = "photo.jpg",
    content: bytes = b"\xff\xd8\xff" + b"\x00" * 50,
    mime_type: str = "image/jpeg",
    evidence_type: str = "damage_photos",
) -> int:
    """
    Write a file to tmp_path (the patched _UPLOAD_ROOT) and record a DB row.

    Returns the primary key (evidence_id) of the new WarrantyEvidence row.
    The DB record's file_path points to the real temp file so the download
    endpoint can validate path safety and serve the bytes.
    """
    from warranty_workflow import WarrantyEngine

    dest_dir = tmp_path / "warranty" / ticket_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / filename
    file_path.write_bytes(content)

    ev = WarrantyEngine.record_evidence(
        ticket_id=ticket_id,
        evidence_type=evidence_type,
        file_path=str(file_path),
        original_filename=filename,
        mime_type=mime_type,
        file_size_bytes=len(content),
    )
    return cast(int, ev.id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvidenceDownloadEndpoint:
    """HTTP-level tests for GET /admin/warranty/{ticket_id}/evidence/{id}/download."""

    def test_valid_download_returns_file_bytes(self, admin_client, tmp_path):
        """Happy-path: valid ticket + valid evidence → 200 with correct bytes."""
        content = b"\xff\xd8\xff\xe0" + b"\x99" * 128  # synthetic JPEG
        ticket_id = _make_ticket("valid-dl")
        ev_id = _write_evidence(tmp_path, ticket_id, content=content)

        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 200, resp.text
        assert resp.content == content

    def test_valid_download_sets_content_type(self, admin_client, tmp_path):
        """Content-Type header must match the stored MIME type."""
        ticket_id = _make_ticket("ct-test")
        ev_id = _write_evidence(
            tmp_path, ticket_id, filename="receipt.pdf",
            content=b"%PDF-1.4 fake", mime_type="application/pdf",
        )

        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 200
        assert "application/pdf" in resp.headers.get("content-type", "")

    def test_raw_file_path_not_in_response(self, admin_client, tmp_path):
        """The raw server-side file path must never appear in any response body."""
        ticket_id = _make_ticket("path-exposure")
        ev_id = _write_evidence(tmp_path, ticket_id)

        # Successful download — no file_path in the binary response
        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 200
        # Attempt to find the tmp_path string in the response body
        assert str(tmp_path).encode() not in resp.content

    def test_wrong_ticket_id_returns_404(self, admin_client, tmp_path):
        """Requesting evidence under the wrong ticket_id is rejected with 404."""
        real_ticket = _make_ticket("real-owner")
        other_ticket = _make_ticket("other-owner")
        ev_id = _write_evidence(tmp_path, real_ticket)

        resp = admin_client.get(
            # evidence belongs to real_ticket, not other_ticket
            f"/admin/warranty/{other_ticket}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 404, resp.text
        # No file path in error body
        assert str(tmp_path) not in resp.text

    def test_unknown_evidence_id_returns_404(self, admin_client, tmp_path):
        """A non-existent evidence ID returns 404."""
        ticket_id = _make_ticket("unknown-ev")

        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/999999/download",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 404

    def test_file_missing_on_disk_returns_404(self, admin_client, tmp_path):
        """If the DB record exists but the file was deleted, return 404."""
        ticket_id = _make_ticket("missing-file")
        ev_id = _write_evidence(tmp_path, ticket_id, filename="will_be_deleted.jpg")

        # Delete the file after recording it
        file_on_disk = tmp_path / "warranty" / ticket_id / "will_be_deleted.jpg"
        file_on_disk.unlink()

        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 404

    def test_no_admin_key_header_returns_401(self, admin_client, tmp_path):
        """Omitting X-Admin-Key returns 401."""
        ticket_id = _make_ticket("no-key")
        ev_id = _write_evidence(tmp_path, ticket_id)

        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download"
            # No X-Admin-Key header
        )
        assert resp.status_code == 401

    def test_wrong_admin_key_returns_401(self, admin_client, tmp_path):
        """Supplying the wrong X-Admin-Key returns 401."""
        ticket_id = _make_ticket("wrong-key")
        ev_id = _write_evidence(tmp_path, ticket_id)

        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": "definitely-wrong-key"},
        )
        assert resp.status_code == 401

    def test_unconfigured_admin_key_returns_503(self, unconfigured_client, tmp_path):
        """When ADMIN_API_KEY is not set on the server, return 503."""
        ticket_id = _make_ticket("no-config")
        ev_id = _write_evidence(tmp_path, ticket_id)

        resp = unconfigured_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": "any-key"},
        )
        assert resp.status_code == 503

    def test_path_traversal_via_db_rejected(self, admin_client, tmp_path, monkeypatch):
        """
        If a DB record somehow contains a path outside _UPLOAD_ROOT (e.g. a
        manually corrupted row), the endpoint rejects it with 404 — not 500 or
        a file disclosure.
        """
        import warranty_router as wr
        from warranty_workflow import WarrantyEngine

        ticket_id = _make_ticket("traversal")

        # Record evidence pointing to a path OUTSIDE _UPLOAD_ROOT
        ev = WarrantyEngine.record_evidence(
            ticket_id=ticket_id,
            evidence_type="damage_photos",
            file_path="/etc/passwd",  # outside _UPLOAD_ROOT
            original_filename="etc_passwd.txt",
            mime_type="text/plain",
            file_size_bytes=0,
        )
        ev_id = cast(int, ev.id)

        resp = admin_client.get(
            f"/admin/warranty/{ticket_id}/evidence/{ev_id}/download",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        # Must be rejected — 404, not 200
        assert resp.status_code == 404
        # The path itself must not appear in the response
        assert "/etc/passwd" not in resp.text
