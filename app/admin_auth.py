"""Shared fail-closed authentication for internal FastAPI endpoints."""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import HTTPException


def configured_admin_key() -> str:
    return os.getenv("ADMIN_API_KEY", "").strip()


def require_admin_key(
    received: Optional[str],
    configured: Optional[str] = None,
) -> None:
    expected = configured_admin_key() if configured is None else configured.strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin API is not configured. Set ADMIN_API_KEY.",
        )
    supplied = (received or "").strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key.")
