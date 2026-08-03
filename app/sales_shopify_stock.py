"""
sales_shopify_stock.py
======================
Live Shopify inventory lookup for Sales AI stock answers.

Falls back gracefully when credentials are missing or Shopify is unreachable
so the chat never invents a quantity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

from store_config import get_store_config

logger = logging.getLogger(__name__)

# Admin GraphQL: `availableForSale` is on ProductVariant, not Product.
_PRODUCT_BY_HANDLE = """
query ProductByHandle($handle: String!) {
  productByHandle(handle: $handle) {
    title
    status
    totalInventory
    variants(first: 25) {
      edges {
        node {
          title
          availableForSale
          inventoryQuantity
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class LiveStockSnapshot:
    handle: str
    title: str
    status: str
    available_for_sale: bool
    total_inventory: Optional[int]
    source: str  # shopify | unavailable

def _sales_store_domain() -> str:
    return (os.getenv("TIDIO_DOMAIN") or "osakiusa.com").strip() or "osakiusa.com"


def fetch_live_stock(handle: str, *, timeout: float = 8.0) -> Optional[LiveStockSnapshot]:
    """Return live inventory for a product handle, or None on soft failure."""
    handle = (handle or "").strip()
    if not handle:
        return None

    cfg = get_store_config(_sales_store_domain())
    shop_domain = cfg.get("shop_domain") or ""
    token = cfg.get("shop_access_token") or ""
    if not shop_domain or not token:
        logger.info("sales_shopify_stock: credentials missing for %s", _sales_store_domain())
        return None

    url = f"https://{shop_domain}/admin/api/2024-01/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": token,
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            json={"query": _PRODUCT_BY_HANDLE, "variables": {"handle": handle}},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
    except Exception:
        logger.exception("sales_shopify_stock: Shopify request failed for %s", handle)
        return None

    if body.get("errors"):
        logger.warning("sales_shopify_stock: GraphQL errors: %s", body.get("errors"))
        return None

    node = ((body.get("data") or {}).get("productByHandle")) or None
    if not node:
        return None

    total = node.get("totalInventory")
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None

    variant_edges = (((node.get("variants") or {}).get("edges")) or [])
    variant_nodes = [(edge or {}).get("node") or {} for edge in variant_edges]
    any_available = any(bool(v.get("availableForSale")) for v in variant_nodes)
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
    if not variant_nodes:
        # No variant payload — treat ACTIVE + positive inventory as sellable.
        any_available = str(node.get("status") or "").upper() == "ACTIVE" and (
            total_int is None or total_int > 0
        )

    return LiveStockSnapshot(
        handle=handle,
        title=str(node.get("title") or handle).strip(),
        status=str(node.get("status") or "").strip().lower(),
        available_for_sale=any_available,
        total_inventory=total_int,
        source="shopify",
    )
