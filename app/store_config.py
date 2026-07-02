"""Per-store Shopify + Track123 credential resolution from environment."""

from __future__ import annotations

import os
from typing import Dict


def get_store_key_prefix(target_domain: str) -> str:
    lowered = (target_domain or "").lower()
    if "titanchair.com" in lowered or "osakichair.com" in lowered:
        return "TITAN"
    if "osakimassagechair.com" in lowered:
        return "OSAKIMASSAGE"
    if "osaki-titan.com" in lowered or "osakititan.com" in lowered:
        return "OSAKITITAN"
    if "osakiusa.com" in lowered:
        return "OSAKI"
    return "OSAKI"


def get_storefront_base_url(target_domain: str) -> str:
    """Public storefront origin for customer self-service links."""
    lowered = (target_domain or "").lower()
    if "titanchair" in lowered or "osakichair" in lowered:
        return "https://titanchair.com"
    if "osakimassage" in lowered:
        return "https://osakimassage.com"
    if "osaki-titan" in lowered or "osakititan" in lowered:
        return "https://osaki-titan.com"
    if "osakiusa" in lowered:
        return "https://osakiusa.com"
    return "https://www.osaki.com"


def get_order_tracking_page_url(target_domain: str) -> str:
    """
    Customer-facing order / shipment status page for this storefront.

    Override per store with ``{PREFIX}_ORDER_TRACKING_URL`` or global
    ``ORDER_TRACKING_PAGE_URL`` in the environment.
    """
    prefix = get_store_key_prefix(target_domain)
    env_url = os.getenv(f"{prefix}_ORDER_TRACKING_URL", "").strip()
    if env_url:
        return env_url
    global_url = os.getenv("ORDER_TRACKING_PAGE_URL", "").strip()
    if global_url:
        return global_url
    base = get_storefront_base_url(target_domain).rstrip("/")
    return f"{base}/apps/track123"


def get_store_config(target_domain: str) -> Dict[str, str]:
    """Resolve per-store Shopify and Track123 credentials from env."""
    prefix = get_store_key_prefix(target_domain)
    track123_key = os.getenv(f"{prefix}_TRACK123_API_KEY", "").strip()
    if not track123_key:
        track123_key = os.getenv("TRACK123_API_KEY", "").strip()
    track123_token = os.getenv(f"{prefix}_TRACK123_TOKEN", "").strip()
    if not track123_token:
        track123_token = os.getenv("TRACK123_TOKEN", "").strip()
    return {
        "shop_domain": os.getenv(f"{prefix}_SHOP_DOMAIN", "").strip(),
        "shop_access_token": os.getenv(f"{prefix}_ACCESS_TOKEN", "").strip(),
        "track123_api_key": track123_key,
        "track123_token": track123_token,
        "store_prefix": prefix,
    }
