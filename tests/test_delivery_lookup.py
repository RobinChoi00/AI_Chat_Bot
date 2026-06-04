"""Unit tests for warranty delivery lookup (Track123 / Shopify)."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from delivery_lookup import (  # noqa: E402
    TrackingSnapshot,
    format_warranty_tracking_message,
    lookup_by_order_or_email,
    lookup_by_tracking_number,
    parse_order_or_email,
)


def test_parse_order_or_email_detects_email():
    assert parse_order_or_email(" buyer@example.com ") == ("", "buyer@example.com")


def test_parse_order_or_email_detects_order():
    assert parse_order_or_email("#12345") == ("12345", "")


def test_format_unavailable_message():
    msg = format_warranty_tracking_message(
        TrackingSnapshot(source="unavailable", available=False, error="not found")
    )
    assert "couldn't verify" in msg.lower()
    assert "not found" in msg


def test_format_success_message_includes_tracking():
    snap = TrackingSnapshot(
        source="track123",
        available=True,
        status="IN_TRANSIT",
        carrier="UPS",
        tracking_number="1Z999AA10123456784",
        eta="Jun 5",
    )
    msg = format_warranty_tracking_message(snap)
    assert "IN_TRANSIT" in msg
    assert "1Z999AA10123456784" in msg


def test_lookup_by_tracking_number_empty():
    snap = lookup_by_tracking_number("  ", "osaki.com")
    assert snap.available is False


def test_lookup_by_tracking_number_uses_track123(monkeypatch):
    def fake_enrich(tn, store):
        assert tn == "1Z999AA10123456784"
        return {
            "status": "IN_TRANSIT",
            "eta": "Tomorrow",
            "last_event": "Departed hub",
            "current_location": "Dallas, TX",
            "events": [],
        }

    class FakeMain:
        @staticmethod
        def get_store_config(domain):
            return {"track123_api_key": "k", "track123_token": "t"}

        enrich_tracking_from_track123 = staticmethod(fake_enrich)

        @staticmethod
        def enrich_tracking_from_aftership(*_a, **_k):
            return {}

        @staticmethod
        def resolve_carrier_name(*_a, **_k):
            return "UPS"

    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: FakeMain())
    snap = lookup_by_tracking_number("1Z999AA10123456784", "osaki.com")
    assert snap.available is True
    assert snap.source == "track123"
    assert snap.tracking_number == "1Z999AA10123456784"


def test_lookup_by_order_or_email_shopify(monkeypatch):
    class FakeMain:
        @staticmethod
        def fetch_shopify_order_status(order, email, domain):
            assert email == "buyer@example.com"
            return {
                "status": "FULFILLED",
                "company": "FedEx",
                "tracking_number": "1234567890",
                "tracking_url": "https://fedex.com/track/123",
                "eta": "Mar 1",
                "last_event": "In transit",
                "current_location": "Memphis",
                "events": [],
            }

    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: FakeMain())
    snap = lookup_by_order_or_email("buyer@example.com", "osaki.com")
    assert snap.available is True
    assert snap.source == "shopify"
    assert snap.tracking_number == "1234567890"
