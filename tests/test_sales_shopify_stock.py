"""Unit tests for live Shopify stock helper (mocked HTTP)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_shopify_stock import fetch_live_stock, stock_badge  # noqa: E402
from sales_shopify_stock import LiveStockSnapshot  # noqa: E402


def test_fetch_live_stock_returns_none_without_credentials(monkeypatch):
    monkeypatch.setattr(
        "sales_shopify_stock.get_store_config",
        lambda _domain: {"shop_domain": "", "shop_access_token": ""},
    )
    assert fetch_live_stock("osaki-vibe-4d") is None


def test_fetch_live_stock_passes_domain(monkeypatch):
    seen = {}

    def _cfg(domain):
        seen["domain"] = domain
        return {"shop_domain": "", "shop_access_token": ""}

    monkeypatch.setattr("sales_shopify_stock.get_store_config", _cfg)
    assert fetch_live_stock("x", domain="osakimassagechair.com") is None
    assert seen["domain"] == "osakimassagechair.com"


def test_stock_badge_labels():
    assert stock_badge(None) == "stock unchecked"
    assert (
        stock_badge(
            LiveStockSnapshot("h", "t", "active", True, 8, "shopify")
        )
        == "in stock"
    )
    assert (
        stock_badge(
            LiveStockSnapshot("h", "t", "active", True, 2, "shopify")
        )
        == "low stock (2)"
    )
    assert (
        stock_badge(
            LiveStockSnapshot("h", "t", "active", False, 0, "shopify")
        )
        == "out of stock"
    )


def test_handle_candidates_strip_massage_chair_suffix():
    from sales_shopify_stock import _handle_candidates

    assert _handle_candidates("titan-ecabin-3d-massage-chair") == [
        "titan-ecabin-3d-massage-chair",
        "titan-ecabin-3d",
    ]


def test_fetch_live_stock_falls_back_to_short_handle(monkeypatch):
    monkeypatch.setattr(
        "sales_shopify_stock.get_store_config",
        lambda _domain: {
            "shop_domain": "demo.myshopify.com",
            "shop_access_token": "token",
        },
    )
    calls = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        handle = (json or {}).get("variables", {}).get("handle")
        calls.append(handle)
        fake = MagicMock()
        fake.raise_for_status = MagicMock()
        if handle == "titan-ecabin-3d":
            fake.json.return_value = {
                "data": {
                    "productByHandle": {
                        "handle": "titan-ecabin-3d",
                        "title": "Titan eCabin 3D",
                        "status": "ACTIVE",
                        "totalInventory": 9,
                        "variants": {
                            "edges": [
                                {
                                    "node": {
                                        "availableForSale": True,
                                        "inventoryQuantity": 9,
                                    }
                                }
                            ]
                        },
                    }
                }
            }
        else:
            fake.json.return_value = {"data": {"productByHandle": None}}
        return fake

    with patch("sales_shopify_stock.requests.post", side_effect=_fake_post):
        snap = fetch_live_stock("titan-ecabin-3d-massage-chair", domain="titanchair.com")
    assert snap is not None
    assert snap.handle == "titan-ecabin-3d"
    assert snap.total_inventory == 9
    assert calls[0] == "titan-ecabin-3d-massage-chair"
    assert "titan-ecabin-3d" in calls

