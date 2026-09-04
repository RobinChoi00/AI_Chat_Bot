"""Tests for human-readable warranty case references."""

import sys
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_case_ref import (  # noqa: E402
    case_reference_for_ticket,
    format_case_reference,
    normalize_case_reference,
    parse_case_reference,
)


class _FakeTicket:
    ticket_id = "eb51acdf-3d31-466e-9b5a-c318f4a5c37e"
    created_at = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def test_format_case_reference_uses_date_and_uuid_prefix():
    ref = format_case_reference(
        "eb51acdf-3d31-466e-9b5a-c318f4a5c37e",
        created_at=_FakeTicket.created_at,
    )
    assert ref == "WR-20260701-EB51AC"


def test_case_reference_for_ticket():
    assert case_reference_for_ticket(_FakeTicket()) == "WR-20260701-EB51AC"


def test_normalize_and_parse_case_reference():
    assert normalize_case_reference(" wr-20260701-eb51ac ") == "WR-20260701-EB51AC"
    assert normalize_case_reference("20260701-EB51AC") == "WR-20260701-EB51AC"
    assert parse_case_reference("WR-20260701-EB51AC") == ("20260701", "EB51AC")
    assert parse_case_reference("not-a-ref") is None
