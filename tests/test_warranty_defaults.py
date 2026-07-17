"""Tests for shared warranty domain defaults."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from store_config import get_storefront_base_url  # noqa: E402
from warranty_defaults import (  # noqa: E402
    DEFAULT_STOREFRONT_BASE_URL,
    DEFAULT_WARRANTY_DOMAIN,
    normalize_warranty_domain,
)
from warranty_resume import _resolve_resume_base_url  # noqa: E402


def test_normalize_warranty_domain_maps_legacy_osaki_com():
    assert normalize_warranty_domain("osaki.com") == DEFAULT_WARRANTY_DOMAIN
    assert normalize_warranty_domain("www.osaki.com") == DEFAULT_WARRANTY_DOMAIN
    assert normalize_warranty_domain(None) == DEFAULT_WARRANTY_DOMAIN


def test_normalize_warranty_domain_keeps_storefronts():
    assert normalize_warranty_domain("titanchair.com") == "titanchair.com"
    assert normalize_warranty_domain("osakimassagechair.com") == "osakimassagechair.com"


def test_storefront_default_is_osakiusa():
    assert get_storefront_base_url("unknown-store.example") == DEFAULT_STOREFRONT_BASE_URL


def test_osakimassagechair_storefront_url():
    assert get_storefront_base_url("osakimassagechair.com") == "https://osakimassagechair.com"


def test_resume_base_url_defaults_to_osakiusa():
    assert _resolve_resume_base_url("") == DEFAULT_STOREFRONT_BASE_URL
