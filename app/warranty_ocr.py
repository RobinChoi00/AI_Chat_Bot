"""
warranty_ocr.py
===============
Photo-of-the-serial-number → chair model auto-detection.

Design
------
Customers often can't remember the exact model name of their chair. The
warranty sticker on the base of the chair contains:
  * Model name  (e.g. "Osaki OS-4000T" or "Titan Pro-Jupiter LE")
  * Serial number
  * Manufacture date
  * Voltage / wattage

We take a single JPEG/PNG upload, ask OpenAI's ``gpt-4o-mini`` vision model
to read the label as structured JSON, and pipe the resulting model string
through the existing ``resolve_model_name`` catalog matcher so the frontend
gets a canonical name it can drop straight into the warranty intake box.

Contract
--------
    POST /api/v1/warranty/ocr/serial
        multipart/form-data
            file:  image/*  (≤ 8 MB, jpg/png/heic/webp)
    →   200 OK
        {
          "model_name":   "OS-4000T"  |  null,
          "serial_number":"...."       |  null,
          "raw_text":     "Osaki OS-4000T\nS/N: 4T20220512\n...",
          "confidence":   "high" | "medium" | "low"
        }

Failure modes
-------------
- No API key configured → 503
- Image too big / wrong type → 415 / 413
- Vision call raises → 502 with the OpenAI error message

Cost note: gpt-4o-mini vision is ~$0.15/1M input tokens; each label call is
roughly 1-2k tokens (mostly the image tile embedding), so ~$0.0003 / photo.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["warranty-ocr"])

_MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB — plenty for a phone photo of a sticker
_ALLOWED_MIME_PREFIXES = ("image/",)
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

_OCR_MODEL = os.getenv("WARRANTY_OCR_MODEL", "gpt-4o-mini")
_OCR_MAX_TOKENS = int(os.getenv("WARRANTY_OCR_MAX_TOKENS", "400"))

_SYSTEM_PROMPT = (
    "You read warranty labels/stickers off massage chairs (Osaki, Titan and related brands). "
    "The customer's photo may be blurry, glare-y, or angled. "
    "Return ONLY a compact JSON object with these keys (strings, no arrays):\n"
    '  "model_name": the chair model text as printed (e.g. "OS-4000T", "Pro Jupiter LE"). '
    "Do NOT invent one; if you cannot read it, use \"\".\n"
    '  "serial_number": the serial / SN / S/N value, or "" if not visible.\n'
    '  "raw_text": every readable line you saw, joined with "\\n". '
    "Keep it short — do not paraphrase.\n"
    '  "confidence": "high", "medium", or "low" — your certainty that model_name is correct.\n'
    "If the photo is not a warranty label at all, return "
    '{"model_name":"","serial_number":"","raw_text":"","confidence":"low"}.'
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class OcrResponse(BaseModel):
    model_name: Optional[str] = None
    serial_number: Optional[str] = None
    raw_text: str = ""
    confidence: str = "low"


def _openai_client():
    from openai import OpenAI  # noqa: WPS433

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=api_key)


def _validate_upload(upload: UploadFile) -> None:
    if not upload.filename:
        raise HTTPException(status_code=422, detail="Missing filename.")
    ext = os.path.splitext(upload.filename)[1].lower()
    if ext and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension: {ext}. Use jpg / png / webp / heic.",
        )
    if upload.content_type and not upload.content_type.startswith(_ALLOWED_MIME_PREFIXES):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {upload.content_type}",
        )


def _parse_llm_json(text: str) -> dict:
    """Extract the first JSON object from a model response, tolerating chatter."""
    if not text:
        return {}
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_confidence(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    return "low"


def _resolve_catalog_name(raw_model: str) -> Optional[str]:
    """
    Pass the vision extraction through the product catalog so the ticket
    inherits a canonical, admin-friendly model name whenever possible.
    """
    if not raw_model:
        return None
    try:
        # Prefer the app-level import path so tests that monkey-patch
        # ``resolve_model_name`` observe the same object.
        from product_catalog import resolve_model_name  # noqa: WPS433
    except ImportError:  # pragma: no cover
        return raw_model.strip() or None
    resolved = resolve_model_name(raw_model)
    if resolved:
        return resolved
    stripped = raw_model.strip()
    return stripped or None


def extract_serial_from_image_bytes(data: bytes, *, mime: str = "image/jpeg") -> OcrResponse:
    """
    Core OCR pipeline. Split out from the FastAPI handler so tests can mock
    the OpenAI client directly.
    """
    client = _openai_client()

    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    try:
        completion = client.chat.completions.create(
            model=_OCR_MODEL,
            max_tokens=_OCR_MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Read this warranty label and reply with the JSON object "
                                "described in the system prompt. No prose, no code fences."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR vision call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Vision model error: {exc}") from exc

    text = ""
    try:
        text = completion.choices[0].message.content or ""
    except (AttributeError, IndexError):
        text = ""

    parsed = _parse_llm_json(text)
    raw_model = str(parsed.get("model_name") or "").strip()
    serial = str(parsed.get("serial_number") or "").strip()
    raw_text = str(parsed.get("raw_text") or "").strip()
    confidence = _normalize_confidence(parsed.get("confidence"))

    canonical = _resolve_catalog_name(raw_model)
    return OcrResponse(
        model_name=canonical,
        serial_number=serial or None,
        raw_text=raw_text,
        confidence=confidence,
    )


@router.post("/api/v1/warranty/ocr/serial", response_model=OcrResponse)
async def ocr_serial_label(file: UploadFile = File(...)) -> OcrResponse:
    """
    Read a serial-label photo and return a normalized model name.

    The endpoint is intentionally stateless — it does not create tickets or
    write anything to the DB. The frontend calls this endpoint, previews the
    detected model name to the customer, and only then submits a warranty
    intake using the confirmed value.
    """
    _validate_upload(file)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty file upload.")
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(data)} bytes; max {_MAX_IMAGE_BYTES}).",
        )

    mime = file.content_type or "image/jpeg"
    return extract_serial_from_image_bytes(data, mime=mime)
