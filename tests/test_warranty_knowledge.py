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


def test_contextual_search_skips_defect_without_category():
    wk.load_knowledge_entries.cache_clear()
    results = wk.contextual_search_knowledge(
        path_text="chair won't turn on power remote",
        issue_type="defect",
        defect_category=None,
        limit=3,
    )
    assert results == []


def test_contextual_search_filters_repair_entries_on_delivery():
    wk.load_knowledge_entries.cache_clear()
    repair = wk.KnowledgeEntry(
        source="freshdesk",
        category="voice",
        title="Voice crackling speaker",
        diagnostic="Replacing the Voice PCB often fixes crackling speaker issues.",
        customer_steps=("Check the footrest mechanism.",),
    )
    delivery = wk.KnowledgeEntry(
        source="freshdesk",
        category="general",
        title="Carrier tracking delay",
        diagnostic="Delivery tracking may lag until the carrier scans the shipment.",
        customer_steps=("Check your order confirmation email for tracking.",),
    )

    filtered = wk._filter_delivery_entries([repair, delivery], limit=3)
    assert len(filtered) == 1
    assert filtered[0].title == "Carrier tracking delay"


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


def test_extract_customer_steps_filters_logistics_followups():
    steps = wk._extract_customer_steps(
        "I confirmed with our in-house tech that he will contact you to set up a return visit.",
        "Hello, I have contacted our technician to ensure they prioritize your repair.",
    )
    assert steps == ()


def test_is_presentable_match_title_rejects_intake_form_subjects():
    assert wk.is_presentable_match_title("Power issue") is True
    assert wk.is_presentable_match_title(
        "#2885577 You received a message from Angelica via Warranty Inquiry form"
    ) is False


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
