"""Tests for warranty eligibility and evidence purge helpers."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_eligibility import (  # noqa: E402
    WARRANTY_PLAN_REFERENCE,
    admin_eligibility_note,
    customer_eligibility_note,
    evaluate_purchase_eligibility,
    parse_purchase_date,
)
from warranty_evidence_purge import purge_old_evidence  # noqa: E402


def test_parse_purchase_date_formats():
    assert parse_purchase_date("2025-03-15T18:30:00Z") == date(2025, 3, 15)
    assert parse_purchase_date("March 15, 2025") == date(2025, 3, 15)
    assert parse_purchase_date("") is None


def test_evaluate_in_warranty():
    result = evaluate_purchase_eligibility(
        "March 15, 2025",
        as_of=date(2026, 7, 24),
        years=3,
    )
    assert result.status == "in_warranty"
    assert result.days_remaining is not None and result.days_remaining > 0
    note = customer_eligibility_note(result)
    assert "warranty plan" in note.lower()
    assert "Standard" in result.summary or "plan" in result.summary.lower()
    assert "Standard" in WARRANTY_PLAN_REFERENCE


def test_evaluate_possibly_expired():
    result = evaluate_purchase_eligibility(
        "January 10, 2020",
        as_of=date(2026, 7, 24),
        years=3,
    )
    assert result.status == "possibly_expired"
    note = customer_eligibility_note(result).lower()
    assert "plan" in note
    assert "still review" in note
    assert "Brand extended" in result.summary or "Extended" in result.summary


def test_unknown_eligibility_is_visible_to_admin_not_customer():
    result = evaluate_purchase_eligibility("")
    assert result.status == "unknown"
    assert customer_eligibility_note(result) == ""
    admin = admin_eligibility_note(result).lower()
    assert "does not block" in admin
    assert "netsuite" in admin
    assert "standard" in admin


def test_purge_dry_run_smoke(tmp_path):
    result = purge_old_evidence(days=90, apply=False, upload_root=tmp_path)
    assert result["ok"] is True
    assert result["apply"] is False
    assert "candidate_rows" in result
