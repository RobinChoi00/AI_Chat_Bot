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
    assert "auto_check" in sources
    assert "fault_judgment" in sources


def test_autocheck_loads_all_numeric_codes():
    wk.load_knowledge_entries.cache_clear()
    entries = wk._load_autocheck_entries()
    assert len(entries) >= 22
    assert all(e.customer_steps for e in entries)


def test_fault_judgment_loads_customer_safe_entries():
    wk.load_knowledge_entries.cache_clear()
    entries = wk._load_fault_judgment_entries()
    assert len(entries) >= 40
    assert all(e.customer_steps for e in entries)
    # Prefer real check steps when the manual includes them.
    power = next(e for e in entries if "POWER button" in e.title)
    assert any("check" in step.lower() for step in power.customer_steps)
    assert all("replace the mechanism" not in step.lower() for step in power.customer_steps)


def test_extract_customer_steps_splits_numbered_manual_line():
    steps = wk._extract_customer_steps(
        "1.check the connector of backrest wire 2.change the backrest wire 3.replace the mechanism"
    )
    assert steps
    assert any("connector" in s.lower() for s in steps)
    assert all("replace the mechanism" not in s.lower() for s in steps)


def test_fallback_manual_steps_for_replace_only_rows():
    steps = wk._customer_steps_from_manual(
        "1.replace remote control wire 2.replace remote control",
        "remote control wire is broken",
    )
    assert steps
    assert any("connector" in s.lower() or "cable" in s.lower() for s in steps)
    assert all("replace remote" not in s.lower() for s in steps)


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
        "Replace main PCB immediately. Refer to: Page 19.",
        "Check the fuse and reconnect the power cord.",
    )
    assert steps
    assert all("pcb" not in s.lower() for s in steps)
    assert all("refer to" not in s.lower() for s in steps)


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


def test_search_rejects_entry_that_explicitly_names_another_model(monkeypatch):
    wrong_model = wk.KnowledgeEntry(
        source="freshdesk_kb",
        category="remote",
        title="4000CS Error Codes",
        diagnostic="4000CS remote diagnostic procedure",
        customer_steps=("Press and hold a remote key for 40 seconds.",),
    )
    generic = wk.KnowledgeEntry(
        source="freshdesk_kb",
        category="remote",
        title="Remote screen troubleshooting",
        diagnostic="Controller screen is not responding",
        customer_steps=("Check that the remote cable is firmly connected.",),
    )
    monkeypatch.setattr(wk, "load_knowledge_entries", lambda: (wrong_model, generic))
    monkeypatch.setattr(wk, "_semantic_enabled", lambda: False)
    monkeypatch.setattr(
        wk,
        "_known_model_signatures",
        lambda: frozenset({"4000cs", "4000xt"}),
    )

    results = wk.search_knowledge(
        path_text="remote controller screen 4000CS",
        defect_category="remote",
        model_name="Osaki OS-4000XT",
        limit=3,
    )

    assert generic in results
    assert wrong_model not in results


def test_entry_allowed_for_category_blocks_other_defect_families():
    air = wk.KnowledgeEntry(
        source="freshdesk",
        category="air",
        title="Air compression suddenly stopping",
        diagnostic="Airbags stop mid-session on Maxim LE.",
        customer_steps=("Check air hose connections under the seat.",),
    )
    remote = wk.KnowledgeEntry(
        source="qa_csv",
        category="remote",
        title="Remote not responding",
        diagnostic="Controller buttons do nothing.",
        customer_steps=("Reseat the remote cable at both ends.",),
    )
    general_airish = wk.KnowledgeEntry(
        source="freshdesk",
        category="general",
        title="Maxim LE common issue",
        diagnostic="Customers report air compression suddenly stopping.",
        customer_steps=("Inspect the air compressor hose for kinks.",),
    )

    assert wk.entry_allowed_for_category(remote, "remote") is True
    assert wk.entry_allowed_for_category(air, "remote") is False
    assert wk.entry_allowed_for_category(general_airish, "remote") is False
    assert wk.entry_allowed_for_category(air, "air") is True


def test_search_knowledge_hard_filters_off_topic_category(monkeypatch):
    air = wk.KnowledgeEntry(
        source="freshdesk",
        category="air",
        title="Air compression suddenly stopping Maxim LE",
        diagnostic="Airbags stop mid-session.",
        customer_steps=("Check all air hose connections carefully under the chair.",),
    )
    remote = wk.KnowledgeEntry(
        source="qa_csv",
        category="remote",
        title="Remote screen blank Maxim LE",
        diagnostic="Remote controller has no display.",
        customer_steps=("Reseat the remote cable and power-cycle the chair.",),
    )
    monkeypatch.setattr(wk, "load_knowledge_entries", lambda: (air, remote))
    monkeypatch.setattr(wk, "_semantic_enabled", lambda: False)

    results = wk.search_knowledge(
        path_text="Maxim LE remote controller not working",
        defect_category="remote",
        model_name="Maxim LE",
        limit=5,
    )
    assert remote in results
    assert air not in results
    assert all(r.category == "remote" for r in results)
