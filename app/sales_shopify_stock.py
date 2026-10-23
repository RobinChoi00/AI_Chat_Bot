"""
sales_shopify_stock.py
======================
Live Shopify inventory lookup for Sales AI stock answers.

Falls back gracefully when credentials are missing or Shopify is unreachable
so the chat never invents a quantity.

Cross-store note: OsakiUSA CSV handles (e.g. ``titan-ecabin-3d-massage-chair``)
often differ from Titan Chair handles (``titan-ecabin-3d``). We try the CSV
handle first, then common handle variants, then a title search.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from store_config import get_store_config, normalize_storefront_domain

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()
_stock_cache: OrderedDict[tuple[str, str, str, str], tuple[float, Optional["LiveStockSnapshot"]]] = OrderedDict()
_cache_counters = {"hits": 0, "misses": 0, "fetches": 0, "failures": 0}
_cache_generation = 0
_cache_clock = time.monotonic


def _bounded_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return min(maximum, max(minimum, value)) if math.isfinite(value) else default
    except ValueError:
        return default


def clear_stock_cache() -> None:
    """Invalidate public product snapshots after a catalog refresh."""
    global _cache_generation
    with _cache_lock:
        _cache_generation += 1
        _stock_cache.clear()
        for key in _cache_counters:
            _cache_counters[key] = 0


def stock_cache_status() -> dict[str, Any]:
    """Non-secret freshness/latency counters for internal diagnostics."""
    now = _cache_clock()
    with _cache_lock:
        ages = [max(0.0, now - created) for created, _ in _stock_cache.values()]
        ttl = _bounded_env("SALES_STOCK_CACHE_TTL_SECONDS", 30, 0, 300)
        failure_ttl = min(ttl, _bounded_env("SALES_STOCK_FAILURE_CACHE_TTL_SECONDS", 3, 0, 10))
        fresh = [snap for created, snap in _stock_cache.values()
                 if max(0.0, now - created) < (ttl if snap is not None else failure_ttl)]
        fetched_at = [snap.fetched_at for _, snap in _stock_cache.values()
                      if snap is not None and snap.fetched_at is not None]
        return {
            **_cache_counters,
            "entries": len(_stock_cache),
            "oldest_age_seconds": round(max(ages, default=0.0), 2),
            "fresh_entries": len(fresh),
            "fresh_unavailable_entries": sum(snap is None for snap in fresh),
            "latest_fetched_at": max(fetched_at, default=None),
            "ttl_seconds": ttl,
            "max_entries": int(_bounded_env("SALES_STOCK_CACHE_MAX_ENTRIES", 256, 1, 2048)),
        }

_PRODUCT_FIELDS = """
    handle
    title
    status
    totalInventory
    variants(first: 25) {
      edges {
        node {
          title
          availableForSale
          inventoryQuantity
          price
        }
      }
    }
"""

_PRODUCT_BY_HANDLE = f"""
query ProductByHandle($handle: String!) {{
  productByHandle(handle: $handle) {{
    {_PRODUCT_FIELDS}
  }}
}}
"""

_PRODUCTS_SEARCH = f"""
query ProductsSearch($q: String!) {{
  products(first: 8, query: $q) {{
    edges {{
      node {{
        {_PRODUCT_FIELDS}
      }}
    }}
  }}
}}
"""


@dataclass(frozen=True)
class LiveStockSnapshot:
    handle: str
    title: str
    status: str
    available_for_sale: bool
    total_inventory: Optional[int]
    source: str  # shopify | unavailable
    price_usd: Optional[float] = None
    price_max_usd: Optional[float] = None
    fetched_at: Optional[str] = None
    cache_hit: bool = False
    cache_age_seconds: float = 0.0

    @property
    def in_stock(self) -> bool:
        # Shopify's sellability flag accounts for inventory policy (including
        # "continue selling when out of stock"), so it is more authoritative
        # than the aggregate inventory count.
        return self.available_for_sale

    @property
    def is_low(self) -> bool:
        return (
            self.in_stock
            and self.total_inventory is not None
            and 0 < self.total_inventory <= 3
        )


def stock_badge(snap: Optional[LiveStockSnapshot]) -> str:
    """Short customer-facing stock label."""
    if snap is None:
        return "stock unchecked"
    if snap.in_stock and snap.is_low:
        return "low stock"
    if snap.in_stock:
        return "in stock"
    return "out of stock"


def _default_sales_domain() -> str:
    return (os.getenv("TIDIO_DOMAIN") or "osakiusa.com").strip() or "osakiusa.com"


def _handle_candidates(handle: str) -> list[str]:
    h = (handle or "").strip().strip("/")
    if not h:
        return []
    out = [h]
    # OsakiUSA export often appends "-massage-chair"; Titan store may not.
    if h.endswith("-massage-chair"):
        out.append(h[: -len("-massage-chair")])
    if h.endswith("-massage-chairs"):
        out.append(h[: -len("-massage-chairs")])
    # Dedupe preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _snapshot_from_node(node: dict[str, Any], *, fallback_handle: str) -> Optional[LiveStockSnapshot]:
    total = node.get("totalInventory")
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError, OverflowError):
        total_int = None

    variant_edges = (((node.get("variants") or {}).get("edges")) or [])
    variant_nodes = [(edge or {}).get("node") or {} for edge in variant_edges]
    availability = [v.get("availableForSale") for v in variant_nodes]
    known_availability = [value for value in availability if isinstance(value, bool)]
    any_available = any(known_availability)
    prices: list[float] = []
    if total_int is None and variant_nodes:
        qty_sum = 0
        saw_qty = False
        for v in variant_nodes:
            raw = v.get("inventoryQuantity")
            if raw is None:
                continue
            try:
                qty_sum += int(raw)
                saw_qty = True
            except (TypeError, ValueError, OverflowError):
                continue
        if saw_qty:
            total_int = qty_sum
    for v in variant_nodes:
        raw_price = v.get("price")
        if raw_price is None or raw_price == "":
            continue
        try:
            price = float(raw_price)
            if math.isfinite(price) and price >= 0:
                prices.append(price)
        except (TypeError, ValueError):
            continue
    if not any_available and len(known_availability) != len(variant_nodes):
        return None  # Partial availability data cannot establish an out-of-stock state.
    if not variant_nodes:
        if total_int is None:
            return None
        any_available = str(node.get("status") or "").upper() == "ACTIVE" and total_int > 0

    handle = str(node.get("handle") or fallback_handle).strip()
    return LiveStockSnapshot(
        handle=handle,
        title=str(node.get("title") or handle).strip(),
        status=str(node.get("status") or "").strip().lower(),
        available_for_sale=any_available,
        total_inventory=total_int,
        source="shopify",
        price_usd=min(prices) if prices else None,
        price_max_usd=max(prices) if prices else None,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def _graphql(
    *,
    shop_domain: str,
    token: str,
    query: str,
    variables: dict[str, Any],
    timeout: float,
) -> Optional[dict[str, Any]]:
    url = f"https://{shop_domain}/admin/api/2024-01/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": token,
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            json={"query": query, "variables": variables},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
    except Exception:
        logger.exception("sales_shopify_stock: Shopify request failed")
        return None
    if body.get("errors"):
        logger.warning("sales_shopify_stock: GraphQL lookup failed")
        return None
    return body


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _search_by_title(
    *,
    shop_domain: str,
    token: str,
    title: str,
    timeout: float,
) -> Optional[LiveStockSnapshot]:
    title = (title or "").strip()
    if len(title) < 3:
        return None
    # Prefer quoted title search; also try a distinctive token.
    queries = [f'title:"{title}"', title]
    needle = _norm_title(title)
    title_tokens = re.findall(r"[a-z0-9]{4,}", title.lower())[:3]
    deadline = time.monotonic() + timeout
    for q in queries:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        body = _graphql(
            shop_domain=shop_domain,
            token=token,
            query=_PRODUCTS_SEARCH,
            variables={"q": q},
            timeout=remaining,
        )
        if not body:
            continue
        edges = (((body.get("data") or {}).get("products") or {}).get("edges")) or []
        best = None
        best_score = -1
        for edge in edges:
            node = (edge or {}).get("node") or {}
            cand = _norm_title(str(node.get("title") or ""))
            if not cand:
                continue
            score = 0
            if cand == needle:
                score = 100
            elif needle and (needle in cand or cand in needle):
                score = 80
            elif title_tokens and all(tok in cand for tok in title_tokens):
                score = 60
            if score > best_score:
                best = node
                best_score = score
        if best is not None and best_score >= 60:
            return _snapshot_from_node(best, fallback_handle=str(best.get("handle") or ""))
    return None


def fetch_live_stock(
    handle: str,
    *,
    domain: Optional[str] = None,
    title: Optional[str] = None,
    timeout: float = 8.0,
) -> Optional[LiveStockSnapshot]:
    """Return a fresh store-scoped inventory snapshot, or None on soft failure.

    Snapshots have a short configurable TTL. Expired successful values are never
    served during an outage; an unavailable response remains unknown.
    """
    handle = (handle or "").strip()
    if not handle and not (title or "").strip():
        return None

    store_domain = (domain or "").strip() or _default_sales_domain()
    cfg = get_store_config(store_domain)
    shop_domain = cfg.get("shop_domain") or ""
    token = cfg.get("shop_access_token") or ""
    if not shop_domain or not token:
        logger.info("sales_shopify_stock: credentials missing for %s", store_domain)
        return None

    key = (normalize_storefront_domain(store_domain), shop_domain.lower(), handle.lower(), _norm_title(title or ""))
    ttl = _bounded_env("SALES_STOCK_CACHE_TTL_SECONDS", 30, 0, 300)
    failure_ttl = min(ttl, _bounded_env("SALES_STOCK_FAILURE_CACHE_TTL_SECONDS", 3, 0, 10))
    now = _cache_clock()
    with _cache_lock:
        generation = _cache_generation
        cached = _stock_cache.get(key)
        if cached is not None:
            created, snap = cached
            age = max(0.0, now - created)
            if age < (ttl if snap is not None else failure_ttl):
                _stock_cache.move_to_end(key)
                _cache_counters["hits"] += 1
                return replace(snap, cache_hit=True, cache_age_seconds=round(age, 3)) if snap else None
            del _stock_cache[key]
        _cache_counters["misses"] += 1

    timeout = min(15.0, max(0.1, timeout)) if math.isfinite(timeout) else 8.0
    snap = _fetch_stock_uncached(handle, title=title, shop_domain=shop_domain, token=token, timeout=timeout)
    with _cache_lock:
        _cache_counters["fetches"] += 1
        if snap is None:
            _cache_counters["failures"] += 1
        if generation == _cache_generation and (ttl if snap is not None else failure_ttl) > 0:
            _stock_cache[key] = (_cache_clock(), snap)
            _stock_cache.move_to_end(key)
            max_entries = int(_bounded_env("SALES_STOCK_CACHE_MAX_ENTRIES", 256, 1, 2048))
            while len(_stock_cache) > max_entries:
                _stock_cache.popitem(last=False)
    return snap


def _fetch_stock_uncached(
    handle: str, *, title: Optional[str], shop_domain: str, token: str, timeout: float,
) -> Optional[LiveStockSnapshot]:
    deadline = time.monotonic() + timeout
    for candidate in _handle_candidates(handle):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        body = _graphql(
            shop_domain=shop_domain,
            token=token,
            query=_PRODUCT_BY_HANDLE,
            variables={"handle": candidate},
            timeout=remaining,
        )
        if not body:
            continue
        node = ((body.get("data") or {}).get("productByHandle")) or None
        if node:
            return _snapshot_from_node(node, fallback_handle=candidate)

    # Cross-store handle mismatch → title search (Titan vs OsakiUSA CSV).
    if title:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        snap = _search_by_title(
            shop_domain=shop_domain,
            token=token,
            title=title,
            timeout=remaining,
        )
        if snap is not None:
            logger.info(
                "sales_shopify_stock: resolved via title search domain=%s title=%s handle=%s",
                shop_domain,
                title,
                snap.handle,
            )
            return snap
    return None


def fetch_stock_many(
    products: list[tuple[str, Optional[str]]], *, domain: Optional[str] = None,
    timeout: float = 8.0, fetcher=None,
) -> dict[str, Optional[LiveStockSnapshot]]:
    """Lookup a bounded product batch concurrently, keeping unknown results.

    Duplicate handles are fetched once; callers retain their original ranking.
    ``fetcher`` supports existing test/integration overrides without bypassing
    the production cache. At most eight workers and 32 products are accepted.
    """
    unique = dict(products)
    if len(unique) > 32:
        raise ValueError("Stock batches support at most 32 products")
    lookup = fetcher or fetch_live_stock
    workers = int(_bounded_env("SALES_STOCK_MAX_WORKERS", 4, 1, 8))

    def fetch(item):
        handle, title = item
        try:
            return handle, lookup(handle, domain=domain, title=title, timeout=timeout)
        except Exception:
            logger.warning("sales_shopify_stock: batch lookup unavailable")
            return handle, None

    with ThreadPoolExecutor(max_workers=min(workers, len(unique) or 1)) as pool:
        return dict(pool.map(fetch, unique.items()))
