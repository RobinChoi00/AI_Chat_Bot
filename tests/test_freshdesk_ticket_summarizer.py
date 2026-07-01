"""
tests/test_freshdesk_ticket_summarizer.py
=========================================
LLM rescue of Freshdesk tickets that the regex extractor drops.

All tests mock the OpenAI client — no live API traffic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import freshdesk_ticket_summarizer as summ  # noqa: E402
import warranty_knowledge as wk  # noqa: E402


class _FakeOpenAI:
    """Return a canned JSON payload from ``chat.completions.create``."""

    def __init__(self, payload):
        if isinstance(payload, list):
            self._queue = list(payload)
        else:
            self._queue = [payload]
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.calls: list[dict] = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


@pytest.fixture(autouse=True)
def _enable_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WARRANTY_FRESHDESK_LLM_SUMMARY", "1")


def test_summarize_ticket_returns_safe_steps(monkeypatch):
    fake = _FakeOpenAI(
        {
            "summary": "Chair will not turn on",
            "category": "power",
            "steps": [
                "Verify the power cord is firmly seated at the wall and chair.",
                "Toggle the back power switch OFF for 10 seconds, then ON.",
                "Test a different wall outlet with another device.",
            ],
        }
    )
    result = summ.summarize_ticket(
        "OS-4000T won't power on",
        "Customer says the chair does not respond.",
        "Long agent reply full of prose without imperative bullets.",
        client=fake,
    )
    assert result is not None
    assert len(result.steps) == 3
    assert "back power switch" in " ".join(result.steps).lower()
    assert result.category == "power"


def test_summarize_ticket_strips_unsafe_steps(monkeypatch):
    fake = _FakeOpenAI(
        {
            "summary": "",
            "category": "remote",
            "steps": [
                "Verify the remote cable is connected.",
                "We will ship a replacement remote to your address today.",
                "Call our support team at 888-848-2630 for approval.",
                "Toggle the remote power button OFF and ON.",
            ],
        }
    )
    result = summ.summarize_ticket("Remote dead", "no response", "reply", client=fake)
    assert result is not None
    joined = " ".join(result.steps).lower()
    assert "ship a replacement" not in joined
    assert "888-848-2630" not in joined
    # The two safe steps survive.
    assert any("cable" in s.lower() for s in result.steps)
    assert any("toggle" in s.lower() for s in result.steps)


def test_summarize_ticket_returns_none_on_empty_json(monkeypatch):
    fake = _FakeOpenAI({"summary": "", "category": "general", "steps": []})
    result = summ.summarize_ticket("S", "Q", "A", client=fake)
    assert result is None


def test_summarize_ticket_returns_none_on_bad_output(monkeypatch):
    fake = _FakeOpenAI("Sorry, I can't summarize this ticket.")
    assert summ.summarize_ticket("S", "Q", "A", client=fake) is None


def test_summarize_ticket_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("WARRANTY_FRESHDESK_LLM_SUMMARY", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert summ.summarize_ticket("S", "Q", "A") is None


def test_content_hash_stable_across_whitespace():
    a = summ.content_hash("S", "Q", "A")
    b = summ.content_hash("S", "Q", "A")
    assert a == b
    assert summ.content_hash("S", "Q", "A") != summ.content_hash("S", "Q", "A2")


def test_cache_roundtrip(tmp_path):
    path = tmp_path / "summaries.json"
    cache = {
        "hash1": summ.SummarizedTicket(
            summary="s",
            category="power",
            steps=("Verify the wall outlet with another device.",),
            model="gpt-test",
            created_at=0.0,
        )
    }
    summ.save_summary_cache(cache, path=path)
    loaded = summ.load_summary_cache(path=path)
    assert set(loaded.keys()) == {"hash1"}
    assert loaded["hash1"].steps == cache["hash1"].steps
    assert loaded["hash1"].category == "power"


def test_summarize_missing_tickets_only_hits_llm_for_no_step_tickets(monkeypatch, tmp_path):
    # First ticket already has a safe step in the reply (skipped).
    # Second ticket is prose only — regex fails, LLM rescues.
    tickets = [
        {
            "subject": "Air not working",
            "question": "leg air stopped",
            "answer": "Check the air hose connection between the base and footrest.",
        },
        {
            "subject": "Chair intermittent",
            "question": "sometimes it wakes up, sometimes it does not",
            "answer": "Hey — long friendly agent prose without imperative bullets, "
            "just chatty text.",
        },
    ]

    fake = _FakeOpenAI(
        {
            "summary": "Intermittent power",
            "category": "power",
            "steps": [
                "Unplug the chair from the wall for 30 seconds, then plug back in.",
                "Toggle the back power switch OFF and ON.",
                "Check the fuse on the back of the chair for damage.",
            ],
        }
    )

    cache_path = tmp_path / "summaries.json"
    stats = summ.summarize_missing_tickets(
        tickets,
        cache_path=cache_path,
        client=fake,
    )
    assert stats["skipped"] == 1
    assert stats["processed"] == 1
    assert stats["rescued"] == 1
    assert stats["errors"] == 0
    assert len(fake.calls) == 1

    # Second run reuses cache — no new LLM call.
    fake2 = _FakeOpenAI({"summary": "", "category": "general", "steps": []})
    stats2 = summ.summarize_missing_tickets(
        tickets,
        cache_path=cache_path,
        client=fake2,
    )
    assert stats2["cached"] == 1
    assert stats2["processed"] == 0
    assert len(fake2.calls) == 0


def test_knowledge_loader_uses_llm_rescue(monkeypatch, tmp_path):
    """
    warranty_knowledge should ingest LLM-rescued steps for tickets whose
    agent reply produced zero regex-extractable steps.
    """
    tickets_path = tmp_path / "freshdesk_tickets.json"
    summaries_path = tmp_path / "freshdesk_summaries.json"

    tickets_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": 42,
                    "subject": "Air not inflating on left arm",
                    "question": "left arm airbags stopped filling",
                    "answer": "Prose-only reply with lots of chatter and no imperative bullets at all.",
                }
            ]
        ),
        encoding="utf-8",
    )

    key = summ.content_hash(
        "Air not inflating on left arm",
        "left arm airbags stopped filling",
        "Prose-only reply with lots of chatter and no imperative bullets at all.",
    )
    summaries_path.write_text(
        json.dumps(
            {
                key: {
                    "summary": "Left-arm air bags not inflating.",
                    "category": "air",
                    "steps": [
                        "Check the air hose connection between the base and left arm.",
                        "Reconnect any loose hose fittings you find.",
                    ],
                    "model": "test",
                    "created_at": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(wk, "_FRESHDESK_PATH", tickets_path)
    monkeypatch.setattr(summ, "_CACHE_PATH", summaries_path)
    wk.load_knowledge_entries.cache_clear()

    entries = [e for e in wk.load_knowledge_entries() if e.source == "freshdesk"]
    assert entries, "LLM rescue should surface at least one Freshdesk entry"
    assert any(
        "left arm" in " ".join(e.customer_steps).lower() for e in entries
    )
    assert any(e.category == "air" for e in entries)
