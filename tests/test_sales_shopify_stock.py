"""Unit tests for live Shopify stock helper (mocked HTTP)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_shopify_stock import fetch_live_stock  # noqa: E402


def test_fetch_live_stock_returns_none_without_credentials(monkeypatch):
    monkeypatch.setattr(
        "sales_shopify_stock.get_store_config",
        lambda _domain: {"shop_domain": "", "shop_access_token": ""},
    )
    assert fetch_live_stock("osaki-vibe-4d") is None


def test_fetch_live_stock_parses_shopify_payload(monkeypatch):
    monkeypatch.setattr(
        "sales_shopify_stock.get_store_config",
        lambda _domain: {
            "shop_domain": "demo.myshopify.com",
            "shop_access_token": "token",
        },
    )
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "data": {
            "productByHandle": {
                "title": "Osaki Vibe 4D",
                "status": "ACTIVE",
                "availableForSale": True,
                "totalInventory": 4,
            }
        }
    }
    with patch("sales_shopify_stock.requests.post", return_value=fake_resp):
        snap = fetch_live_stock("osaki-vibe-4d")
    assert snap is not None
    assert snap.available_for_sale is True
    assert snap.total_inventory == 4
    assert snap.source == "shopify"
