"""Tests for append-only warranty chat timeline."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_chat_timeline import append_chat_event, get_chat_timeline  # noqa: E402


def test_append_chat_event_in_memory():
    collected: dict = {}
    ticket = SimpleNamespace(
        ticket_id="",
        get_collected=lambda: collected,
        set_collected=lambda key, value: collected.__setitem__(key, value),
    )
    append_chat_event(
        ticket,
        role="user",
        kind="side_question",
        text="Does the footrest need air?",
        node_id="defect_air_location",
        persist=False,
    )
    append_chat_event(
        ticket,
        role="assistant",
        kind="side_question",
        text="Yes — choose Footrest if the legs do not inflate.",
        node_id="defect_air_location",
        persist=False,
    )
    events = get_chat_timeline(ticket)
    assert len(events) == 2
    assert events[0]["role"] == "user"
    assert events[1]["kind"] == "side_question"


def test_append_chat_event_dedupes_identical():
    collected: dict = {}
    ticket = SimpleNamespace(
        ticket_id="",
        get_collected=lambda: collected,
        set_collected=lambda key, value: collected.__setitem__(key, value),
    )
    for _ in range(3):
        append_chat_event(
            ticket,
            role="assistant",
            kind="enrichment",
            text="Try power cycling the chair.",
            node_id="defect_power",
            persist=False,
        )
    assert len(get_chat_timeline(ticket)) == 1
