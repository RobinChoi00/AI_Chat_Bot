"""Unit tests for per-store configuration helpers."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from store_config import (  # noqa: E402
    get_order_tracking_page_url,
    get_storefront_base_url,
)


def test_get_storefront_base_url_titan():
    assert get_storefront_base_url("titanchair.com") == "https://titanchair.com"


def test_get_order_tracking_page_url_defaults_to_track123_app():
    assert (
        get_order_tracking_page_url("titanchair.com")
        == "https://titanchair.com/apps/track123"
    )
    assert (
        get_order_tracking_page_url("osakiusa.com")
        == "https://osakiusa.com/apps/track123"
    )


def test_get_order_tracking_page_url_env_override(monkeypatch):
    monkeypatch.setenv("TITAN_ORDER_TRACKING_URL", "https://example.com/track")
    assert (
        get_order_tracking_page_url("titanchair.com")
        == "https://example.com/track"
    )
