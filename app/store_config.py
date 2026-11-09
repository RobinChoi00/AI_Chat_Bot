"""Per-store Shopify + Track123 credential resolution from environment."""

from __future__ import annotations

import os
from typing import Dict
from urllib.parse import urlparse

from warranty_defaults import DEFAULT_STOREFRONT_BASE_URL

SUPPORTED_STOREFRONT_DOMAINS = frozenset(
    {
        "osakiusa.com",
        "titanchair.com",
        "osakichair.com",
        "osakimassagechair.com",
        "osaki-titan.com",
        "osakititan.com",
    }
)

STORE_KEY_PREFIXES = ("OSAKIUSA", "TITANCHAIR", "OSAKIMASSAGE", "OSAKITITAN")
_LEGACY_STORE_PREFIX = {
    "OSAKIUSA": "OSAKI",
    "TITANCHAIR": "TITAN",
}


def normalize_storefront_domain(target_domain: str) -> str:
    """Return a lower-case hostname without scheme, path, port, or ``www``."""
    raw = (target_domain or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}", scheme="https")
    host = (parsed.hostname or "").strip(".")
    return host[4:] if host.startswith("www.") else host


def is_supported_storefront_domain(target_domain: str) -> bool:
    host = normalize_storefront_domain(target_domain)
    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in SUPPORTED_STOREFRONT_DOMAINS
    )


def get_store_key_prefix(target_domain: str) -> str:
    lowered = (target_domain or "").lower()
    if "titanchair.com" in lowered or "osakichair.com" in lowered:
        return "TITANCHAIR"
    if "osakimassagechair.com" in lowered:
        return "OSAKIMASSAGE"
    if "osaki-titan.com" in lowered or "osakititan.com" in lowered:
        return "OSAKITITAN"
    return "OSAKIUSA"


def getenv_store(prefix: str, suffix: str) -> str:
    """Read ``{PREFIX}_{SUFFIX}``, then a renamed legacy prefix if needed."""
    value = os.getenv(f"{prefix}_{suffix}", "").strip()
    if value:
        return value
    legacy = _LEGACY_STORE_PREFIX.get(prefix, "")
    if legacy:
        return os.getenv(f"{legacy}_{suffix}", "").strip()
    return ""


def get_storefront_base_url(target_domain: str) -> str:
    """Public storefront origin for customer self-service links."""
    lowered = (target_domain or "").lower()
    if "titanchair" in lowered or "osakichair" in lowered:
        return "https://titanchair.com"
    if "osakimassage" in lowered:
        return "https://osakimassagechair.com"
    if "osaki-titan" in lowered or "osakititan" in lowered:
        return "https://osaki-titan.com"
    if "osakiusa" in lowered:
        return "https://osakiusa.com"
    return DEFAULT_STOREFRONT_BASE_URL


def get_order_tracking_page_url(target_domain: str) -> str:
    """
    Customer-facing order / shipment status page for this storefront.

    Override per store with ``{PREFIX}_ORDER_TRACKING_URL`` or global
    ``ORDER_TRACKING_PAGE_URL`` in the environment.
    """
    prefix = get_store_key_prefix(target_domain)
    env_url = getenv_store(prefix, "ORDER_TRACKING_URL")
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
    track123_key = getenv_store(prefix, "TRACK123_API_KEY")
    if not track123_key:
        track123_key = os.getenv("TRACK123_API_KEY", "").strip()
    track123_token = getenv_store(prefix, "TRACK123_TOKEN")
    if not track123_token:
        track123_token = os.getenv("TRACK123_TOKEN", "").strip()
    return {
        "shop_domain": getenv_store(prefix, "SHOP_DOMAIN"),
        "shop_access_token": getenv_store(prefix, "ACCESS_TOKEN"),
        "track123_api_key": track123_key,
        "track123_token": track123_token,
        "store_prefix": prefix,
    }
