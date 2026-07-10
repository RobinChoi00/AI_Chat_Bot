"""Tests for Fonz warranty ingest and error-code lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import fonz_warranty_data as fonz  # noqa: E402
import error_code_lookup as lookup  # noqa: E402
import warranty_knowledge as wk  # noqa: E402


def test_expand_error_codes_range_and_slash():
    assert fonz.expand_error_codes("C6") == ["C6"]
    assert fonz.expand_error_codes("C1 - C5") == ["C1", "C2", "C3", "C4", "C5"]
    assert fonz.expand_error_codes("CA / CB") == ["CA", "CB"]
    assert fonz.expand_error_codes("E1 - E3") == ["E1", "E2", "E3"]


def test_lookup_error_code_exact_model():
    lookup.clear_error_code_cache()
    hit = lookup.lookup_error_code("3D LTX", "C6")
    assert hit is not None
    assert hit["model"] == "3D LTX"
    assert hit["error_code"] == "C6"
    assert "MOS tube" in hit["meaning"] or "air pump" in hit["meaning"]


def test_lookup_error_code_by_code_only_when_unique_enough():
    lookup.clear_error_code_cache()
    hit = lookup.lookup_error_code("", "C6")
    # C6 may exist on multiple models — accept hit or None, but if hit, must be C6.
    if hit:
        assert hit["error_code"] == "C6"


def test_extract_error_codes_from_text():
    codes = lookup.extract_error_codes_from_text("My chair shows error code C6 on the tablet")
    assert "C6" in codes


def test_warranty_knowledge_loads_fonz_entries():
    wk.clear_knowledge_cache()
    entries = wk.load_knowledge_entries()
    fonz_entries = [e for e in entries if e.source == "fonz_error_code"]
    assert len(fonz_entries) >= 100
    sample = fonz_entries[0]
    assert sample.title
    assert sample.customer_steps or sample.diagnostic


def test_fonz_faiss_documents_metadata():
    docs = fonz.fonz_faiss_documents()
    assert len(docs) >= 100
    meta = docs[0].metadata
    assert meta.get("source") == "fonz"
    assert meta.get("type") == "error_code"
    assert "[Source]: Fonz Warranty List" in docs[0].page_content


def test_ingest_workbook_roundtrip(tmp_path):
    src = fonz.DEFAULT_XLSX_PATH
    if not src.is_file():
        pytest.skip("Fonz workbook not present in raw_data")

    err_out = tmp_path / "fonz_error_codes.json"
    diag_out = tmp_path / "fonz_model_diagnostics.json"
    stats = fonz.ingest_workbook(src, error_out=err_out, diag_out=diag_out)
    assert stats["error_code_entries"] > 1000
    assert stats["model_diagnostics"] > 100

    payload = json.loads(err_out.read_text(encoding="utf-8"))
    assert payload["entry_count"] == stats["error_code_entries"]
    first = payload["entries"][0]
    assert {"model", "model_key", "error_code", "meaning", "troubleshooting", "workflow_category"} <= set(first)
