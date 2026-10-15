"""
Shared pytest defaults for the warranty API suite.

Production enforces live-chat privacy consent by default
(``WARRANTY_REQUIRE_CHAT_PRIVACY=1``). Most unit/API tests exercise
workflow behavior and should not have to record consent + email first.
Opt back in inside individual tests with monkeypatch when needed.
"""

from __future__ import annotations

import os

import pytest

# CI has no .env. main.py refuses to import without these, and the sales
# lead path imports it while sending a summary email.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SHOPIFY_WEBHOOK_SECRET", "test-webhook-secret")
# The full suite sends more than 60 sales chats in one minute. The limit is
# read when the sales routers are imported, so this has to be set first.
os.environ.setdefault("SALES_CHAT_RATE_LIMIT", "100000/minute")


@pytest.fixture(autouse=True)
def _disable_chat_privacy_by_default(monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")
    yield
