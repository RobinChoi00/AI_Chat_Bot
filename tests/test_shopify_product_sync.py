"""Tests for OsakiUSA Shopify product sync."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import shopify_product_sync as sync  # noqa: E402


@pytest.fixture
def template_csv(tmp_path: Path) -> Path:
    headers = [
        "Handle",
        "Title",
        "Body (HTML)",
        "Vendor",
        "Product Category",
        "Type",
        "Tags",
        "Published",
        "Option1 Name",
        "Option1 Value",
        "Option2 Name",
        "Option2 Value",
        "Option3 Name",
        "Option3 Value",
        "Variant SKU",
        "Variant Price",
        "Track Type (product.metafields.custom.track_type)",
        "Status",
    ]
    path = tmp_path / "products_export.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(
            [
                "solo-flex",
                "Osaki Solo Flex",
                "<p>Old</p>",
                "Osaki",
                "",
                "Massage Chair",
                "",
                "true",
                "Color",
                "Brown",
                "Delivery",
                "Curbside Delivery - Free",
                "Warranty",
                "1 Year(Parts/Labor) 2&3 Year(Parts Only) - Free",
                "SKU-OLD",
                "2499.00",
                "SL",
                "active",
            ]
        )
    return path


def test_product_to_csv_rows_expands_variants_and_metafields():
    headers = [
        "Handle",
        "Title",
        "Vendor",
        "Type",
        "Published",
        "Option1 Name",
        "Option1 Value",
        "Option2 Name",
        "Option2 Value",
        "Option3 Name",
        "Option3 Value",
        "Variant SKU",
        "Variant Price",
        "Track Type (product.metafields.custom.track_type)",
        "Status",
    ]
    metafield_columns = sync._parse_metafield_columns(headers)
    node = {
        "handle": "solo-flex",
        "title": "Osaki Solo Flex",
        "descriptionHtml": "<p>Flex chair</p>",
        "vendor": "Osaki",
        "productType": "Massage Chair",
        "tags": ["featured"],
        "status": "ACTIVE",
        "category": {"fullName": "Furniture > Massage Chairs"},
        "options": [
            {"name": "Color", "values": ["Brown", "Black"]},
            {"name": "Delivery", "values": ["Curbside Delivery - Free"]},
            {"name": "Warranty", "values": ["1 Year(Parts/Labor) 2&3 Year(Parts Only) - Free"]},
        ],
        "variants": {
            "edges": [
                {
                    "node": {
                        "sku": "SF-BROWN",
                        "price": "2499.00",
                        "compareAtPrice": None,
                        "barcode": "",
                        "selectedOptions": [
                            {"name": "Color", "value": "Brown"},
                            {"name": "Delivery", "value": "Curbside Delivery - Free"},
                            {"name": "Warranty", "value": "1 Year(Parts/Labor) 2&3 Year(Parts Only) - Free"},
                        ],
                    }
                },
                {
                    "node": {
                        "sku": "SF-BLACK",
                        "price": "2499.00",
                        "compareAtPrice": None,
                        "barcode": "",
                        "selectedOptions": [
                            {"name": "Color", "value": "Black"},
                            {"name": "Delivery", "value": "Curbside Delivery - Free"},
                            {"name": "Warranty", "value": "1 Year(Parts/Labor) 2&3 Year(Parts Only) - Free"},
                        ],
                    }
                },
            ]
        },
        "metafields": {
            "edges": [
                {"node": {"namespace": "custom", "key": "track_type", "value": "SL-Track"}}
            ]
        },
    }

    rows = sync.product_to_csv_rows(node, headers, metafield_columns)
    assert len(rows) == 2
    assert rows[0]["Title"] == "Osaki Solo Flex"
    assert rows[0]["Option1 Value"] == "Brown"
    assert rows[0]["Track Type (product.metafields.custom.track_type)"] == "SL-Track"
    assert rows[1]["Title"] == ""
    assert rows[1]["Handle"] == "solo-flex"
    assert rows[1]["Option1 Value"] == "Black"


def test_sync_aborts_when_handle_count_too_low(template_csv: Path, monkeypatch):
    monkeypatch.setattr(
        sync,
        "fetch_all_products",
        lambda **kwargs: [{"handle": "only-one", "title": "One", "status": "ACTIVE", "options": [], "variants": {"edges": []}, "metafields": {"edges": []}}],
    )
    monkeypatch.setattr(
        "store_config.get_store_config",
        lambda domain: {"shop_domain": "demo.myshopify.com", "shop_access_token": "token"},
        raising=False,
    )

    import store_config  # noqa: WPS433

    monkeypatch.setattr(
        store_config,
        "get_store_config",
        lambda domain: {"shop_domain": "demo.myshopify.com", "shop_access_token": "token"},
    )

    result = sync.sync_osakiusa_products(csv_path=template_csv, dry_run=False, min_handles=5)
    assert result.ok is False
    assert "Aborting" in result.message

    with template_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Handle"] == "solo-flex"


def test_sync_writes_backup_and_report(template_csv: Path, tmp_path: Path, monkeypatch):
    product = {
        "handle": "solo-flex",
        "title": "Osaki Solo Flex",
        "descriptionHtml": "<p>New</p>",
        "vendor": "Osaki",
        "productType": "Massage Chair",
        "tags": [],
        "status": "ACTIVE",
        "category": {"fullName": "Furniture > Massage Chairs"},
        "options": [{"name": "Color", "values": ["Brown"]}],
        "variants": {
            "edges": [
                {
                    "node": {
                        "sku": "SF-BROWN",
                        "price": "2599.00",
                        "compareAtPrice": None,
                        "barcode": "",
                        "selectedOptions": [{"name": "Color", "value": "Brown"}],
                    }
                }
            ]
        },
        "metafields": {"edges": []},
    }
    monkeypatch.setattr(sync, "fetch_all_products", lambda **kwargs: [product])

    import store_config  # noqa: WPS433

    monkeypatch.setattr(
        store_config,
        "get_store_config",
        lambda domain: {"shop_domain": "demo.myshopify.com", "shop_access_token": "token"},
    )

    result = sync.sync_osakiusa_products(
        csv_path=template_csv,
        report_dir=tmp_path / "reports",
        backup_dir=tmp_path / "backups",
        dry_run=False,
        min_handles=1,
    )
    assert result.ok is True
    assert result.handles == 1

    with template_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Variant Price"] == "2599.00"
    assert any(path.name.startswith("products_export.") for path in (tmp_path / "backups").iterdir())
    report_files = list((tmp_path / "reports").glob("shopify_sync_report_*.json"))
    assert report_files
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert payload["handles"] == 1


def test_classify_handle_renames_by_title():
    previous = {
        "osaki-solo-flex": "Osaki Solo Flex",
        "old-only": "Discontinued Chair",
    }
    current = {
        "osaki-solo-flex-4d": "Osaki Solo Flex",
        "brand-new": "New Chair",
    }
    renamed, truly_added, truly_removed, unchanged = sync._classify_handle_changes(
        previous=previous,
        current=current,
    )
    assert len(renamed) == 1
    assert renamed[0]["old_handle"] == "osaki-solo-flex"
    assert renamed[0]["new_handle"] == "osaki-solo-flex-4d"
    assert truly_added == ["brand-new"]
    assert truly_removed == ["old-only"]
    assert unchanged == 0
