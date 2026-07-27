"""Helpers to keep customer PII out of logs and public API responses."""

from __future__ import annotations

import re


_EMAIL_RE = re.compile(
    r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,})",
    re.IGNORECASE,
)


def mask_email(value: str | None) -> str:
    """Mask a single email for logs/UI (keeps domain)."""
    text = (value or "").strip()
    if not text or "@" not in text:
        return "***"
    local, _, domain = text.partition("@")
    if not local or not domain:
        return "***"
    if len(local) <= 1:
        masked_local = "*"
    elif len(local) == 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[0] + ("*" * min(len(local) - 2, 6)) + local[-1]
    return f"{masked_local}@{domain}"


def mask_phone(value: str | None) -> str:
    """Mask a phone number, keeping the last 4 digits when present."""
    digits = re.sub(r"\D", "", value or "")
    if len(digits) < 4:
        return "***"
    return f"***-***-{digits[-4:]}"


def mask_emails_in_text(value: str | None) -> str:
    """Replace every email-looking token in free text."""
    text = value or ""
    if not text:
        return text

    def _repl(match: re.Match[str]) -> str:
        return mask_email(match.group(0))

    return _EMAIL_RE.sub(_repl, text)
