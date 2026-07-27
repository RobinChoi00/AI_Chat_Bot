"""Unit tests for PII masking helpers."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from pii_redact import mask_email, mask_emails_in_text, mask_phone  # noqa: E402


def test_mask_email_keeps_domain():
    assert mask_email("buyer@example.com") == "b***r@example.com"
    assert "@example.com" in mask_email("ab@example.com")


def test_mask_phone_keeps_last_four():
    assert mask_phone("+1 (555) 123-4567") == "***-***-4567"


def test_mask_emails_in_text():
    text = "Contact me at buyer@example.com please"
    out = mask_emails_in_text(text)
    assert "buyer@example.com" not in out
    assert "@example.com" in out
