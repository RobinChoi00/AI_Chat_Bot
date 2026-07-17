"""Shared warranty domain / storefront defaults."""

from __future__ import annotations

DEFAULT_WARRANTY_DOMAIN = "osakiusa.com"
DEFAULT_STOREFRONT_BASE_URL = "https://osakiusa.com"

LEGACY_WARRANTY_DOMAINS = frozenset({"osaki.com", "www.osaki.com"})


def normalize_warranty_domain(domain: str | None) -> str:
    """Map invalid legacy domains to the primary Osaki USA storefront."""
    raw = (domain or "").strip().lower()
    raw = raw.replace("https://", "").replace("http://", "").split("/")[0]
    if not raw or raw in LEGACY_WARRANTY_DOMAINS:
        return DEFAULT_WARRANTY_DOMAIN
    return raw
