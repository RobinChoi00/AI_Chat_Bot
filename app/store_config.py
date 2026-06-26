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
