"""Tests for Track123 client and per-store credential routing."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from track123_client import (  # noqa: E402
    infer_courier_code,
    parse_track123_record,
    query_track123_tracking,
)


def test_infer_courier_code_from_ups_number():
    assert infer_courier_code("", "1Z999AA10123456784") == "ups"


def test_infer_courier_code_from_company_name():
    assert infer_courier_code("FedEx Ground", "398891812948") == "fedex"


def test_parse_track123_record_maps_status_and_events():
    record = {
        "trackNo": "1Z999AA10123456784",
        "transitStatus": "IN_TRANSIT",
        "localLogisticsInfo": {
            "courierNameEN": "UPS",
            "courierTrackingLink": "https://www.ups.com/track?tracknum=1Z",
            "trackingDetails": [
                {
                    "address": "Dallas, TX, US",
                    "eventTime": "2026-03-01 10:00:00",
                    "eventDetail": "Departed facility",
                }
            ],
        },
    }
    parsed = parse_track123_record(record)
    assert parsed["status"] == "IN_TRANSIT"
    assert parsed["company"] == "UPS"
    assert parsed["tracking_url"].startswith("https://")
    assert parsed["events"][0]["location"] == "Dallas, TX, US"


def test_query_track123_uses_realtime_then_registered():
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **_kwargs):
        calls.append(url)
        if "query-realtime" in url:
            return FakeResponse({"code": "00000", "data": {}})
        if "/track/query" in url:
            return FakeResponse(
                {
                    "code": "00000",
                    "data": {
                        "accepted": {
                            "content": [
                                {
                                    "trackNo": "123",
                                    "transitStatus": "DELIVERED",
                                    "localLogisticsInfo": {
                                        "courierNameEN": "USPS",
                                        "trackingDetails": [],
                                    },
                                }
                            ]
                        }
                    },
                }
            )
        raise AssertionError(f"unexpected url {url}")

    result = query_track123_tracking(
        "123",
        {"track123_api_key": "secret"},
        http_post=fake_post,
    )
    assert result["status"] == "DELIVERED"
    assert any("query-realtime" in u for u in calls)
    assert any("/track/query" in u for u in calls)


def test_get_store_key_prefix_osaki_titan():
    from store_config import get_store_key_prefix  # noqa: E402

    assert get_store_key_prefix("osaki-titan.com") == "OSAKITITAN"
    assert get_store_key_prefix("www.osakiusa.com") == "OSAKI"


def test_get_store_config_falls_back_to_global_track123(monkeypatch):
    from store_config import get_store_config  # noqa: E402

    monkeypatch.delenv("OSAKITITAN_TRACK123_API_KEY", raising=False)
    monkeypatch.setenv("TRACK123_API_KEY", "global-key")
    monkeypatch.setenv("OSAKITITAN_SHOP_DOMAIN", "osaki-titan.myshopify.com")
    cfg = get_store_config("osaki-titan.com")
    assert cfg["track123_api_key"] == "global-key"
    assert cfg["shop_domain"] == "osaki-titan.myshopify.com"
