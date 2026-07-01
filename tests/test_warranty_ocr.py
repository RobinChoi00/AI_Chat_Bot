"""
tests/test_warranty_ocr.py
==========================
Unit + HTTP tests for the serial-label OCR endpoint.

We never hit the real OpenAI API. Instead we replace ``_openai_client`` with
a stub that returns a canned ``chat.completions`` payload, verify the
extraction/normalization pipeline, and confirm the FastAPI handler surfaces
the right errors for oversized / malformed uploads.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_ocr  # noqa: E402


def _make_completion(content: str):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class _FakeOpenAI:
    def __init__(self, canned: str, capture: dict | None = None):
        self._canned = canned
        self._capture = capture
        # Mirror the OpenAI SDK surface used by warranty_ocr.
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        if self._capture is not None:
            self._capture.update(kwargs)
        return _make_completion(self._canned)


@pytest.fixture
def patch_client(monkeypatch):
    """Return a helper that installs a fake OpenAI client with the given text."""

    def _install(payload: str, *, capture: dict | None = None):
        fake = _FakeOpenAI(payload, capture=capture)
        monkeypatch.setattr(warranty_ocr, "_openai_client", lambda: fake)

    return _install


@pytest.fixture
def resolver_stub(monkeypatch):
    """Force resolve_model_name to a deterministic mapping for tests."""
    known = {"OS-4000T": "OS-4000T", "PRO JUPITER": "Titan Pro Jupiter LE"}

    def fake_resolve(raw: str):
        norm = (raw or "").strip().upper()
        for key, canonical in known.items():
            if key in norm:
                return canonical
        return None

    import product_catalog

    monkeypatch.setattr(product_catalog, "resolve_model_name", fake_resolve)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(warranty_ocr.router)
    return TestClient(app)


def test_extract_returns_normalized_model(patch_client, resolver_stub):
    patch_client(
        json.dumps(
            {
                "model_name": "OSAKI OS-4000T",
                "serial_number": "OS4T20220512",
                "raw_text": "OSAKI OS-4000T\nS/N: OS4T20220512",
                "confidence": "high",
            }
        )
    )
    result = warranty_ocr.extract_serial_from_image_bytes(b"fake-image-bytes")
    # Catalog resolver normalizes the noisy vision output.
    assert result.model_name == "OS-4000T"
    assert result.serial_number == "OS4T20220512"
    assert "OS-4000T" in result.raw_text
    assert result.confidence == "high"


def test_extract_handles_prose_around_json(patch_client, resolver_stub):
    patch_client(
        "Here you go:\n"
        '{"model_name":"Pro Jupiter LE","serial_number":"","raw_text":"Titan","confidence":"medium"}\n'
        "Hope this helps!"
    )
    result = warranty_ocr.extract_serial_from_image_bytes(b"x")
    assert result.model_name == "Titan Pro Jupiter LE"
    assert result.serial_number is None
    assert result.confidence == "medium"


def test_extract_handles_unreadable_label(patch_client, resolver_stub):
    patch_client(
        json.dumps(
            {"model_name": "", "serial_number": "", "raw_text": "", "confidence": "low"}
        )
    )
    result = warranty_ocr.extract_serial_from_image_bytes(b"x")
    assert result.model_name is None
    assert result.serial_number is None
    assert result.confidence == "low"


def test_extract_rejects_malformed_json(patch_client, resolver_stub):
    patch_client("Sorry, I can't read the sticker.")
    result = warranty_ocr.extract_serial_from_image_bytes(b"x")
    assert result.model_name is None
    assert result.confidence == "low"
    assert result.raw_text == ""


def test_http_endpoint_happy_path(patch_client, resolver_stub, client):
    patch_client(
        json.dumps(
            {
                "model_name": "OS-4000T",
                "serial_number": "SN123",
                "raw_text": "OSAKI OS-4000T\nSN123",
                "confidence": "high",
            }
        )
    )
    resp = client.post(
        "/api/v1/warranty/ocr/serial",
        files={"file": ("sticker.jpg", b"jpg-bytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "OS-4000T"
    assert body["serial_number"] == "SN123"
    assert body["confidence"] == "high"


def test_http_endpoint_rejects_empty_upload(patch_client, resolver_stub, client):
    patch_client("{}")
    resp = client.post(
        "/api/v1/warranty/ocr/serial",
        files={"file": ("sticker.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_http_endpoint_rejects_oversized_upload(patch_client, resolver_stub, client, monkeypatch):
    monkeypatch.setattr(warranty_ocr, "_MAX_IMAGE_BYTES", 128)
    patch_client("{}")
    resp = client.post(
        "/api/v1/warranty/ocr/serial",
        files={"file": ("sticker.jpg", b"x" * 200, "image/jpeg")},
    )
    assert resp.status_code == 413


def test_http_endpoint_rejects_bad_extension(patch_client, resolver_stub, client):
    patch_client("{}")
    resp = client.post(
        "/api/v1/warranty/ocr/serial",
        files={"file": ("label.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


def test_http_endpoint_502_when_api_key_missing(monkeypatch, client):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = client.post(
        "/api/v1/warranty/ocr/serial",
        files={"file": ("s.jpg", b"data", "image/jpeg")},
    )
    assert resp.status_code == 503


def test_extract_sends_image_and_prompt(patch_client, resolver_stub):
    capture: dict = {}
    patch_client(
        json.dumps(
            {"model_name": "OS-4000T", "serial_number": "S", "raw_text": "", "confidence": "high"}
        ),
        capture=capture,
    )
    warranty_ocr.extract_serial_from_image_bytes(b"xyz", mime="image/png")
    assert capture["model"] == warranty_ocr._OCR_MODEL
    # user message contains both a text part and an image_url part
    user_msg = next(m for m in capture["messages"] if m["role"] == "user")
    part_types = {p["type"] for p in user_msg["content"]}
    assert part_types == {"text", "image_url"}
    image_part = next(p for p in user_msg["content"] if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
