"""
tests/test_warranty_step_paraphrase.py
======================================
LLM rewrite layer for non-terminal workflow messages (mocked OpenAI).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_step_paraphrase as paraphrase  # noqa: E402


class _FakeOpenAI:
    def __init__(self, canned: str):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self._canned = canned

    def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._canned)
                )
            ]
        )


def test_paraphrase_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    draft = "Tip\n\nDo you hear a click?"
    out, ok = paraphrase.paraphrase_step_message(
        draft,
        base_prompt="Do you hear a click?",
    )
    assert out == draft
    assert ok is False


def test_paraphrase_disabled_by_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(paraphrase, "_paraphrase_enabled", lambda: False)

    draft = "Tip\n\nDo you hear a click?"
    out, ok = paraphrase.paraphrase_step_message(
        draft,
        base_prompt="Do you hear a click?",
    )
    assert out == draft
    assert ok is False


def test_paraphrase_accepts_valid_rewrite(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(paraphrase, "_ENABLED", True)

    base = "When you toggle the back switch, do you hear a click?"
    rewritten = (
        "Thanks for sticking with us — here is a quick tip from similar cases.\n\n"
        "1. Toggle the back switch OFF for 10 seconds, then ON.\n\n"
        f"{base}"
    )
    monkeypatch.setattr(
        paraphrase,
        "_openai_client",
        lambda: _FakeOpenAI(rewritten),
    )

    draft = f"Summary\n\n{base}"
    out, ok = paraphrase.paraphrase_step_message(draft, base_prompt=base)
    assert ok is True
    assert out == rewritten
    assert base in out


def test_paraphrase_rejects_missing_question(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(paraphrase, "_ENABLED", True)
    monkeypatch.setattr(
        paraphrase,
        "_openai_client",
        lambda: _FakeOpenAI("Friendly intro with no question."),
    )

    base = "Do you hear a click?"
    draft = f"Tip\n\n{base}"
    out, ok = paraphrase.paraphrase_step_message(draft, base_prompt=base)
    assert ok is False
    assert out == draft


def test_paraphrase_rejects_new_error_code_claim(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(paraphrase, "_ENABLED", True)
    base = "Does the remote have power?"
    rewritten = f"Your chair is showing a 4000CS error code.\n\n{base}"
    monkeypatch.setattr(
        paraphrase,
        "_openai_client",
        lambda: _FakeOpenAI(rewritten),
    )

    draft = f"This looks like a remote issue.\n\n{base}"
    out, ok = paraphrase.paraphrase_step_message(draft, base_prompt=base)

    assert ok is False
    assert out == draft


def test_question_preserved_allows_whitespace_diff():
    base = "Do you hear a click?"
    output = "Intro\n\nDo   you hear a click?"
    assert paraphrase._question_preserved(output, base)


def test_build_paraphrase_system_prompt_includes_question():
    base = "Which part is affected?"
    prompt = paraphrase.build_paraphrase_system_prompt(base_prompt=base)
    assert base in prompt
    assert "verbatim" in prompt.lower()
