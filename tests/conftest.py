"""
Shared pytest defaults for the warranty API suite.

Production enforces live-chat privacy consent by default
(``WARRANTY_REQUIRE_CHAT_PRIVACY=1``). Most unit/API tests exercise
workflow behavior and should not have to record consent + email first.
Opt back in inside individual tests with monkeypatch when needed.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_chat_privacy_by_default(monkeypatch):
    monkeypatch.setenv("WARRANTY_REQUIRE_CHAT_PRIVACY", "0")
    yield
