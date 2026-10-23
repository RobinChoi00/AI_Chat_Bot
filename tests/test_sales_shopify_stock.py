"""Unit tests for live Shopify stock helper (mocked HTTP)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import threading

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from sales_shopify_stock import fetch_live_stock, stock_badge  # noqa: E402
from sales_shopify_stock import LiveStockSnapshot  # noqa: E402
import sales_shopify_stock as stock  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_stock_cache():
    stock.clear_stock_cache()
    yield
    stock.clear_stock_cache()


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
        == "low stock"
    )
    assert (
        stock_badge(
            LiveStockSnapshot("h", "t", "active", False, 0, "shopify")
        )
        == "out of stock"
    )


def test_snapshot_uses_shopify_sellability_and_variant_price_range():
    from sales_shopify_stock import _snapshot_from_node

    snap = _snapshot_from_node(
        {
            "handle": "chair",
            "title": "Chair",
            "status": "ACTIVE",
            "totalInventory": 0,
            "variants": {
                "edges": [
                    {
                        "node": {
                            "availableForSale": True,
                            "inventoryQuantity": 0,
                            "price": "2999.00",
                        }
                    },
                    {
                        "node": {
                            "availableForSale": True,
                            "inventoryQuantity": 0,
                            "price": "3499.00",
                        }
                    },
                ]
            },
        },
        fallback_handle="chair",
    )

    assert snap.in_stock  # Shopify may allow continued selling at zero inventory.
    assert snap.price_usd == 2999.0
    assert snap.price_max_usd == 3499.0


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


def _configure_mock_stock(monkeypatch):
    monkeypatch.setattr(stock, "get_store_config", lambda _: {
        "shop_domain": "store.myshopify.com", "shop_access_token": "test-token",
    })


def test_stock_cache_reuses_only_fresh_store_scoped_results(monkeypatch):
    _configure_mock_stock(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(stock, "_cache_clock", lambda: clock[0])
    snapshots = []

    def fetch(handle, **kwargs):
        snap = LiveStockSnapshot(handle, "Chair", "active", True, 8, "shopify", 1000.0, fetched_at="2026-10-23T00:00:00+00:00")
        snapshots.append(snap)
        return snap

    monkeypatch.setattr(stock, "_fetch_stock_uncached", fetch)
    first = stock.fetch_live_stock("chair", domain="osakiusa.com")
    clock[0] += 10
    cached = stock.fetch_live_stock("chair", domain="https://www.osakiusa.com")
    assert len(snapshots) == 1
    assert cached.cache_hit and cached.cache_age_seconds == 10.0
    assert cached.fetched_at == first.fetched_at
    assert first.cache_hit is False  # Cached freshness does not mutate prior replies.
    stock.fetch_live_stock("chair", domain="titanchair.com")
    assert len(snapshots) == 2  # Same handle/shop config cannot cross storefront boundaries.
    clock[0] += 21
    stock.fetch_live_stock("chair", domain="osakiusa.com")
    assert len(snapshots) == 3


def test_expired_stock_never_masks_outage_and_negative_cache_is_short(monkeypatch):
    _configure_mock_stock(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(stock, "_cache_clock", lambda: clock[0])
    calls = []

    def fetch(handle, **kwargs):
        calls.append(handle)
        return LiveStockSnapshot(handle, "Chair", "active", True, 8, "shopify") if len(calls) == 1 else None

    monkeypatch.setattr(stock, "_fetch_stock_uncached", fetch)
    assert stock.fetch_live_stock("chair") is not None
    clock[0] += 31
    assert stock.fetch_live_stock("chair") is None
    assert stock.fetch_live_stock("chair") is None
    assert len(calls) == 2
    clock[0] += 4
    assert stock.fetch_live_stock("chair") is None
    assert len(calls) == 3


def test_cache_limits_invalidation_and_disable(monkeypatch):
    _configure_mock_stock(monkeypatch)
    monkeypatch.setenv("SALES_STOCK_CACHE_MAX_ENTRIES", "2")
    calls = []

    def fetch(handle, **kwargs):
        calls.append(handle)
        return LiveStockSnapshot(handle, "Chair", "active", False, 0, "shopify")

    monkeypatch.setattr(stock, "_fetch_stock_uncached", fetch)
    for handle in ("one", "two", "three", "one"):
        stock.fetch_live_stock(handle)
    assert len(calls) == 4
    assert stock.stock_cache_status()["entries"] == 2
    stock.clear_stock_cache()
    assert stock.stock_cache_status()["entries"] == 0
    monkeypatch.setenv("SALES_STOCK_CACHE_TTL_SECONDS", "0")
    stock.fetch_live_stock("one")
    stock.fetch_live_stock("one")
    assert len(calls) == 6
    assert stock.stock_cache_status()["entries"] == 0


def test_unknown_availability_and_invalid_prices_do_not_become_facts():
    assert stock._snapshot_from_node({"status": "ACTIVE"}, fallback_handle="chair") is None
    assert stock._snapshot_from_node({"status": "ACTIVE", "variants": {"edges": [
        {"node": {"price": "100"}}, {"node": {"availableForSale": False}},
    ]}}, fallback_handle="chair") is None
    snap = stock._snapshot_from_node({"variants": {"edges": [
        {"node": {"availableForSale": True, "price": "NaN"}},
        {"node": {"availableForSale": True, "price": "-1"}},
    ]}}, fallback_handle="chair")
    assert snap.in_stock and snap.price_usd is None


def test_batch_lookups_run_with_bounded_concurrency_and_keep_unknown(monkeypatch):
    monkeypatch.setenv("SALES_STOCK_MAX_WORKERS", "2")
    started = threading.Barrier(2)
    lock = threading.Lock()
    activity = {"current": 0, "max": 0}
    calls = []

    def fetch(handle, **kwargs):
        with lock:
            calls.append(handle)
            activity["current"] += 1
            activity["max"] = max(activity["max"], activity["current"])
        try:
            started.wait(timeout=2)
            if handle == "broken":
                raise RuntimeError("unavailable")
            return LiveStockSnapshot(handle, "Chair", "active", True, 1, "shopify")
        finally:
            with lock:
                activity["current"] -= 1

    results = stock.fetch_stock_many([("chair", "Chair"), ("broken", "Broken"), ("chair", "Chair")], fetcher=fetch)
    assert set(calls) == {"chair", "broken"} and len(calls) == 2
    assert activity["max"] == 2
    assert results["chair"].in_stock and results["broken"] is None
    with pytest.raises(ValueError, match="at most 32"):
        stock.fetch_stock_many([(str(i), None) for i in range(33)], fetcher=fetch)


def test_credentials_remain_required_even_when_snapshot_is_cached(monkeypatch):
    _configure_mock_stock(monkeypatch)
    monkeypatch.setattr(stock, "_fetch_stock_uncached", lambda handle, **_: LiveStockSnapshot(handle, "Chair", "active", True, 8, "shopify"))
    assert stock.fetch_live_stock("chair") is not None
    monkeypatch.setattr(stock, "get_store_config", lambda _: {"shop_domain": "", "shop_access_token": ""})
    assert stock.fetch_live_stock("chair") is None

