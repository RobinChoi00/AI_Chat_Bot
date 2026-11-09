"""Unit tests for per-store configuration helpers."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from store_config import (  # noqa: E402
    get_order_tracking_page_url,
    get_storefront_base_url,
    is_supported_storefront_domain,
    normalize_storefront_domain,
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
    monkeypatch.setenv("TITANCHAIR_ORDER_TRACKING_URL", "https://example.com/track")
    assert (
        get_order_tracking_page_url("titanchair.com")
        == "https://example.com/track"
    )


def test_legacy_osaki_and_titan_env_names_still_resolve(monkeypatch):
    from store_config import get_store_config

    monkeypatch.delenv("OSAKIUSA_SHOP_DOMAIN", raising=False)
    monkeypatch.delenv("TITANCHAIR_TRACK123_API_KEY", raising=False)
    monkeypatch.setenv("OSAKI_SHOP_DOMAIN", "legacy-osaki.myshopify.com")
    monkeypatch.setenv("TITAN_TRACK123_API_KEY", "legacy-titan-key")
    osaki = get_store_config("osakiusa.com")
    titan = get_store_config("titanchair.com")
    assert osaki["store_prefix"] == "OSAKIUSA"
    assert osaki["shop_domain"] == "legacy-osaki.myshopify.com"
    assert titan["store_prefix"] == "TITANCHAIR"
    assert titan["track123_api_key"] == "legacy-titan-key"


def test_normalize_and_validate_storefront_domain():
    assert (
        normalize_storefront_domain("https://www.osakiusa.com/products/x")
        == "osakiusa.com"
    )
    assert is_supported_storefront_domain("chat.osakiusa.com")
    assert not is_supported_storefront_domain("osakiusa.com.attacker.invalid")
    assert not is_supported_storefront_domain("example.invalid")
