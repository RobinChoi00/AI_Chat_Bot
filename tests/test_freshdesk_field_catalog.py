"""Tests for Freshdesk ticket field catalog (status / custom dropdown ID maps)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from freshdesk_field_catalog import (  # noqa: E402
    label_ticket,
    parse_ticket_fields_payload,
    resolve_choice_label,
)


SAMPLE_FIELDS = [
    {
        "id": 1,
        "name": "status",
        "label": "Status",
        "type": "default_status",
        "choices": {
            "2": ["Open", "Open"],
            "3": ["Pending", "Pending"],
            "4": ["Resolved", "Resolved"],
            "5": ["Closed", "Closed"],
            "6": ["Waiting on Dev", "Waiting on Dev"],
            "7": ["Waiting on customer", "Waiting on customer"],
        },
    },
    {
        "id": 99,
        "name": "cf_parts_status",
        "label": "Parts status",
        "type": "custom_dropdown",
        "choices": {
            "1": ["Pending Parts", "Pending Parts"],
            "2": ["Parts Need Info", "Parts Need Info"],
        },
    },
]


def test_parse_ticket_fields_builds_lookup():
    catalog = parse_ticket_fields_payload(SAMPLE_FIELDS)
    assert catalog["status"]["choices"]["3"] == "Pending"
    assert catalog["status"]["choices"]["6"] == "Waiting on Dev"
    assert catalog["lookup"]["status"]["4"] == "Resolved"
    assert len(catalog["custom_fields"]) == 1
    assert catalog["lookup"]["cf_parts_status"]["1"] == "Pending Parts"


def test_parse_list_shaped_choices():
    fields = [
        {
            "id": 10,
            "name": "cf_parts_status",
            "label": "Parts status",
            "type": "custom_dropdown",
            "choices": [
                {"id": 1, "label": "Pending Parts", "value": "Pending Parts"},
                {"id": 2, "label": "Parts Need Info", "value": "Parts Need Info"},
            ],
        }
    ]
    catalog = parse_ticket_fields_payload(fields)
    assert catalog["lookup"]["cf_parts_status"]["1"] == "Pending Parts"


def test_resolve_choice_label_and_label_ticket():
    catalog = parse_ticket_fields_payload(SAMPLE_FIELDS)
    assert resolve_choice_label(catalog, "status", 3) == "Pending"
    assert resolve_choice_label(catalog, "cf_parts_status", 2) == "Parts Need Info"
    assert resolve_choice_label(catalog, "status", 999) == "Unknown(999)"

    labels = label_ticket(
        catalog,
        {
            "status": 6,
            "custom_fields": {"cf_parts_status": 1},
        },
    )
    assert labels["status"] == "Waiting on Dev"
    assert labels["cf_parts_status"] == "Pending Parts"


def test_fetch_ticket_field_catalog_live_shape(monkeypatch):
    import freshdesk_field_catalog as ffc

    monkeypatch.setenv("FRESHDESK_DOMAIN", "titanchair.freshdesk.com")
    monkeypatch.setenv("FRESHDESK_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return SAMPLE_FIELDS

    monkeypatch.setattr(ffc.requests, "get", lambda *args, **kwargs: FakeResponse())

    catalog = ffc.fetch_ticket_field_catalog()
    assert catalog["domain"] == "titanchair.freshdesk.com"
    assert catalog["status"]["choices"]["7"] == "Waiting on customer"


def test_admin_freshdesk_field_catalog_endpoint(monkeypatch):
    import warranty_router as wr

    monkeypatch.setattr(wr, "_ADMIN_API_KEY", "test-admin-key-fd")

    import freshdesk_field_catalog as ffc

    sample = {
        "fetched_at": "2026-07-22T00:00:00+00:00",
        "domain": "titanchair.freshdesk.com",
        **parse_ticket_fields_payload(SAMPLE_FIELDS),
    }
    monkeypatch.setattr(ffc, "get_field_catalog", lambda refresh=False: sample)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(wr.router)
    client = TestClient(app)

    res = client.get(
        "/admin/warranty/freshdesk-field-catalog",
        headers={"X-Admin-Key": "test-admin-key-fd"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"]["choices"]["3"] == "Pending"
    assert body["lookup"]["cf_parts_status"]["2"] == "Parts Need Info"
