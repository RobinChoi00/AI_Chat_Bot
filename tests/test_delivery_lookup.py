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


def test_parse_order_or_email_detects_both():
    assert parse_order_or_email("#12345 buyer@example.com") == (
        "12345",
        "buyer@example.com",
    )


def test_lookup_by_order_requires_email(monkeypatch):
    called = {"n": 0}

    class FakeMain:
        @staticmethod
        def fetch_shopify_order_status(*_a, **_k):
            called["n"] += 1
            return {"status": "FULFILLED"}

    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: FakeMain())
    snap = lookup_by_order_or_email("#12345", "osaki.com")
    assert snap.available is False
    assert "privacy" in (snap.error or "").lower()
    assert called["n"] == 0


def test_format_unavailable_message():
    msg = format_warranty_tracking_message(
        TrackingSnapshot(source="unavailable", available=False, error="not found")
    )
    assert "couldn't verify" in msg.lower()
    assert "not found" in msg


def test_format_unavailable_message_includes_self_service_links():
    msg = format_warranty_tracking_message(
        TrackingSnapshot(source="unavailable", available=False, error="not found"),
        domain="osakiusa.com",
        lookup_kind="order",
        raw_input="buyer@example.com",
    )
    assert "Track on our website" in msg
    assert "https://osakiusa.com/apps/track123" in msg


def test_build_carrier_tracking_url_ups():
    from delivery_lookup import build_carrier_tracking_url

    url = build_carrier_tracking_url("1Z999AA10123456784")
    assert "ups.com" in url
    assert "1Z999AA10123456784" in url


def test_build_self_service_lookup_links_for_tracking():
    from delivery_lookup import build_self_service_lookup_links

    links = build_self_service_lookup_links(
        domain="titanchair.com",
        lookup_kind="tracking",
        raw_input="1Z999AA10123456784",
    )
    labels = [label for label, _url in links]
    assert "Track on our website" in labels
    assert "Track with the carrier" in labels


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
    assert "We'll continue with your delivery warranty questions below." in msg


def test_format_success_message_status_path_does_not_promise_more_questions():
    snap = TrackingSnapshot(
        source="track123",
        available=True,
        status="IN_TRANSIT",
        carrier="UPS",
        tracking_number="1Z999AA10123456784",
    )
    msg = format_warranty_tracking_message(snap, continue_with_questions=False)
    assert "IN_TRANSIT" in msg
    assert "We'll continue with your delivery warranty questions below." not in msg
    assert "Our team will follow up shortly" in msg


def test_format_processing_message_status_path_closing():
    snap = TrackingSnapshot(
        source="shopify",
        available=True,
        status="PROCESSING",
        order_number="#1001",
    )
    msg = format_warranty_tracking_message(snap, continue_with_questions=False)
    assert "PROCESSING" in msg
    assert "We'll continue with your delivery warranty questions below." not in msg
    assert "Our team will follow up shortly" in msg


def test_format_success_message_includes_order_details():
    snap = TrackingSnapshot(
        source="shopify",
        available=True,
        status="FULFILLED",
        carrier="FedEx",
        tracking_number="1234567890",
        order_number="#OSKUS11308",
        purchase_date="March 15, 2025",
        product_names=["Osaki OS-4D Pro Maestro LE"],
        total_amount="$5,499.00",
    )
    msg = format_warranty_tracking_message(snap)
    assert "#OSKUS11308" in msg
    assert "March 15, 2025" in msg
    assert "Osaki OS-4D Pro Maestro LE" in msg
    assert "$5,499.00" in msg
    assert "1234567890" in msg


def test_snapshot_from_shopify_order_details():
    from delivery_lookup import _snapshot_from_tracking_data

    snap = _snapshot_from_tracking_data(
        {
            "status": "PROCESSING",
            "purchase_date_raw": "2025-03-15T18:30:00Z",
            "order_number": "#1001",
            "product_names": ["Titan Chair"],
            "total_amount": "2999.00",
            "currency_code": "USD",
            "events": [],
        },
        source="shopify",
    )
    assert snap.order_number == "#1001"
    assert snap.purchase_date == "March 15, 2025"
    assert snap.product_names == ["Titan Chair"]
    assert snap.total_amount == "$2,999.00"


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
                "order_number": "#1001",
                "purchase_date_raw": "2025-01-10T12:00:00Z",
                "product_names": ["Osaki Duo"],
                "total_amount": "4999.00",
                "currency_code": "USD",
            }

    monkeypatch.setattr("delivery_lookup._lazy_logistics", lambda: FakeMain())
    snap = lookup_by_order_or_email("buyer@example.com", "osaki.com")
    assert snap.available is True
    assert snap.source == "shopify"
    assert snap.tracking_number == "1234567890"
    assert snap.order_number == "#1001"
    assert snap.purchase_date == "January 10, 2025"
    assert snap.product_names == ["Osaki Duo"]
    assert snap.total_amount == "$4,999.00"
