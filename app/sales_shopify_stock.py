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
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

from store_config import get_store_config

logger = logging.getLogger(__name__)

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


def _snapshot_from_node(node: dict[str, Any], *, fallback_handle: str) -> LiveStockSnapshot:
    total = node.get("totalInventory")
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None

    variant_edges = (((node.get("variants") or {}).get("edges")) or [])
    variant_nodes = [(edge or {}).get("node") or {} for edge in variant_edges]
    any_available = any(bool(v.get("availableForSale")) for v in variant_nodes)
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
            except (TypeError, ValueError):
                continue
        if saw_qty:
            total_int = qty_sum
    for v in variant_nodes:
        raw_price = v.get("price")
        if raw_price is None or raw_price == "":
            continue
        try:
            prices.append(float(raw_price))
        except (TypeError, ValueError):
            continue
    if not variant_nodes:
        any_available = str(node.get("status") or "").upper() == "ACTIVE" and (
            total_int is None or total_int > 0
        )

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
        logger.warning("sales_shopify_stock: GraphQL errors: %s", body.get("errors"))
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
    for q in queries:
        body = _graphql(
            shop_domain=shop_domain,
            token=token,
            query=_PRODUCTS_SEARCH,
            variables={"q": q},
            timeout=timeout,
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
            elif needle and all(tok in cand for tok in re.findall(r"[a-z0-9]{4,}", needle)[:3]):
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
    """Return live inventory for a product handle/title, or None on soft failure."""
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

    for candidate in _handle_candidates(handle):
        body = _graphql(
            shop_domain=shop_domain,
            token=token,
            query=_PRODUCT_BY_HANDLE,
            variables={"handle": candidate},
            timeout=timeout,
        )
        if not body:
            continue
        node = ((body.get("data") or {}).get("productByHandle")) or None
        if node:
            return _snapshot_from_node(node, fallback_handle=candidate)

    # Cross-store handle mismatch → title search (Titan vs OsakiUSA CSV).
    if title:
        snap = _search_by_title(
            shop_domain=shop_domain,
            token=token,
            title=title,
            timeout=timeout,
        )
        if snap is not None:
            logger.info(
                "sales_shopify_stock: resolved via title search domain=%s title=%s handle=%s",
                store_domain,
                title,
                snap.handle,
            )
            return snap
    return None
