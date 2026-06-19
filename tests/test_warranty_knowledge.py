"""Tests for unified warranty knowledge search."""

from __future__ import annotations

import json
from pathlib import Path

import warranty_knowledge as wk


def test_loads_qa_and_autocheck_entries():
    wk.load_knowledge_entries.cache_clear()
    entries = wk.load_knowledge_entries()
    assert len(entries) > 10
    sources = {e.source for e in entries}
    assert "qa_csv" in sources


def test_search_power_path(tmp_path, monkeypatch):
    freshdesk = tmp_path / "freshdesk_tickets.json"
    freshdesk.write_text(
        json.dumps([
            {
                "ticket_id": 1,
                "subject": "Chair power issue back switch clicking",
                "question": "I turned on the back switch and heard a sound but remote won't work",
                "answer": "Please verify the power cord is plugged in firmly. Try toggling the back switch off and on. Check the fuse on the chair.",
            }
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(wk, "_FRESHDESK_PATH", freshdesk)
    wk.load_knowledge_entries.cache_clear()

    results = wk.search_knowledge(
        path_text="back switch heard something power remote defect_power_main_pcb",
        defect_category="power",
        limit=3,
    )
    assert results
    assert any(r.source == "freshdesk" for r in results) or any(
        "power" in r.category for r in results
    )


def test_extract_customer_steps_filters_internal():
    steps = wk._extract_customer_steps(
        "Replace main PCB immediately.",
        "Check the fuse and reconnect the power cord.",
    )
    assert steps
    assert all("pcb" not in s.lower() for s in steps)


def test_extract_customer_steps_filters_pii_and_templates():
    steps = wk._extract_customer_steps(
        "The information we need is outlined below: Description of Issue: 4000CS Stuck in zero. "
        "Customer Address: 3222 ARBOR S HOUSTON, TX 77004 Phone Number: 2816860269",
        "With that being said, we always recommend having a qualified technician inspect your chair.",
    )
    assert steps == ()


def test_freshdesk_loader_skips_merged_tickets(tmp_path, monkeypatch):
    freshdesk = tmp_path / "freshdesk_tickets.json"
    freshdesk.write_text(
        json.dumps([
            {
                "ticket_id": 1,
                "subject": "REOPENED: Osaki 4000CS",
                "question": "Chair has no power",
                "answer": "This ticket is closed and merged into ticket 182769",
            },
            {
                "ticket_id": 2,
                "subject": "Power issue",
                "question": "Back switch clicking",
                "answer": "Please verify the power cord is plugged in firmly. Try toggling the back switch off and on.",
            },
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(wk, "_FRESHDESK_PATH", freshdesk)
    wk.load_knowledge_entries.cache_clear()
    entries = [e for e in wk.load_knowledge_entries() if e.source == "freshdesk"]
    assert len(entries) == 1
    assert "4000CS" not in entries[0].title or entries[0].title == "Power issue"
