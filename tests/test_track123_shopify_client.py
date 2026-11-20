"""Tests for Track123 Shopify App API client."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from track123_shopify_client import (  # noqa: E402
    order_payload_to_enrichment,
    query_by_tracking_number,
    resolve_track123_store_uuid,
    shopify_app_auth_status,
    shopify_store_uuid,
)


def test_shopify_store_uuid_from_myshopify_domain():
    assert shopify_store_uuid("osakiusa.myshopify.com") == "osakiusa"
    assert shopify_store_uuid("titan-chair.myshopify.com") == "titan-chair"


def test_resolve_track123_store_uuid_prefers_explicit_override():
    assert (
        resolve_track123_store_uuid(
            {
                "shop_domain": "osakimassage.myshopify.com",
                "track123_shop_uuid": "ossakimassagechair-draft6",
            }
        )
        == "ossakimassagechair-draft6"
    )
    assert (
        resolve_track123_store_uuid({"shop_domain": "osakiusa.myshopify.com"})
        == "osakiusa"
    )


def test_order_payload_to_enrichment_maps_fulfillment():
    payload = {
        "order": {
            "order_name": "#1001",
            "order_number": 1001,
            "order_create_at": "2026-03-01T10:00:00Z",
            "tracking_link": "https://example.com/track/1Z",
            "status": "FULFILLED",
            "fulfillments": [
                {
                    "tracking_number": "1Z999AA10123456784",
                    "tracking_company": "UPS",
                    "transit_status": "IN_TRANSIT",
                    "last_event": "Departed facility",
                    "courier": {
                        "name": "UPS",
                        "query_link": "https://www.ups.com/track?tracknum=1Z999AA10123456784",
                    },
                    "tracking_details": [
                        {
                            "event_time": "2026-03-01 10:00:00",
                            "event_location": "Dallas, TX, US",
                            "event_detail": "Departed facility",
                        }
                    ],
                    "line_items": [{"title": "Osaki Chair", "quantity": 1}],
                }
            ],
        }
    }
    parsed = order_payload_to_enrichment(payload, tracking_number="1Z999AA10123456784")
    assert parsed["track123_source"] == "shopify_app"
    assert parsed["status"] == "IN_TRANSIT"
    assert parsed["company"] == "UPS"
    assert parsed["tracking_url"].startswith("https://www.ups.com")
    assert parsed["events"][0]["location"] == "Dallas, TX, US"
    assert parsed["product_names"] == ["Osaki Chair"]


def test_query_by_tracking_uses_x_api_key_header():
    calls = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "order": {
                    "fulfillments": [
                        {
                            "tracking_number": "1Z999",
                            "transit_status": "DELIVERED",
                            "courier": {"name": "UPS", "query_link": "https://ups.example/1Z999"},
                            "tracking_details": [],
                        }
                    ]
                }
            }

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("headers") or {}))
        return FakeResponse()

    store = {
        "shop_domain": "osakiusa.myshopify.com",
        "track123_api_key": "shopify-app-key",
    }
    result = query_by_tracking_number("1Z999", store, http_get=fake_get)
    assert result["status"] == "DELIVERED"
    assert result["track123_source"] == "shopify_app"
    assert len(calls) == 1
    url, headers = calls[0]
    assert "shp.track123.com/shopify/api/v1/osakiusa/orders/by-tracking/1Z999.json" in url
    assert headers.get("X-Api-Key") == "shopify-app-key"
    assert "Track123-Api-Secret" not in headers


def test_shopify_app_auth_status_does_not_treat_generic_404_as_verified():
    class FakeResponse:
        status_code = 404

        def json(self):
            return {"error": "Not Found"}

    status = shopify_app_auth_status(
        {"shop_domain": "osakiusa.myshopify.com", "track123_api_key": "k"},
        http_get=lambda *_a, **_k: FakeResponse(),
    )
    assert status["configured"] is True
    assert status["ok"] is False
    assert status["error_code"] == "probe_not_found"
    assert status["store_uuid"] == "osakiusa"


def test_shopify_app_auth_status_unauthorized():
    class FakeResponse:
        status_code = 401

        def json(self):
            return {"error": "Unauthorized"}

    status = shopify_app_auth_status(
        {"shop_domain": "osakimassage.myshopify.com", "track123_api_key": "bad"},
        http_get=lambda *_a, **_k: FakeResponse(),
    )
    assert status["ok"] is False
    assert status["error_code"] == "unauthorized"
