"""
Pull OsakiUSA Shopify catalog → ``raw_data/products_export.csv``.

Safety:
  - Backs up the previous CSV before replace
  - Aborts if the fetch fails or handle count drops below threshold
  - Writes atomically via a temp file + rename
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = _PROJECT_ROOT / "raw_data" / "products_export.csv"
DEFAULT_REPORT_DIR = _PROJECT_ROOT / "data" / "reports"
DEFAULT_BACKUP_DIR = _PROJECT_ROOT / "raw_data" / "backups"

METAFIELD_COL_RE = re.compile(r"\(product\.metafields\.([^.]+)\.([^)]+)\)\s*$", re.I)

PRODUCTS_QUERY = """
query SyncProducts($cursor: String) {
  products(first: 50, after: $cursor) {
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        id
        handle
        title
        descriptionHtml
        vendor
        productType
        tags
        status
        category {
          fullName
        }
        options {
          name
          values
        }
        variants(first: 100) {
          edges {
            node {
              sku
              price
              compareAtPrice
              barcode
              selectedOptions {
                name
                value
              }
            }
          }
        }
        metafields(first: 40) {
          edges {
            node {
              namespace
              key
              value
            }
          }
        }
      }
    }
  }
}
"""


@dataclass
class SyncResult:
    ok: bool
    message: str
    handles: int = 0
    variant_rows: int = 0
    added_handles: list[str] = field(default_factory=list)
    removed_handles: list[str] = field(default_factory=list)
    renamed_products: list[dict[str, str]] = field(default_factory=list)
    truly_added_handles: list[str] = field(default_factory=list)
    truly_removed_handles: list[str] = field(default_factory=list)
    unchanged_handles: int = 0
    csv_path: str = ""
    report_path: str = ""
    dry_run: bool = False


def _parse_metafield_columns(headers: list[str]) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for header in headers:
        match = METAFIELD_COL_RE.search(header)
        if match:
            mapping[(match.group(1).lower(), match.group(2).lower())] = header
    return mapping


def _load_csv_headers(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing template CSV at {csv_path}. "
            "Keep an existing products_export.csv as the column template."
        )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _existing_handles(csv_path: Path) -> set[str]:
    return set(_existing_handle_titles(csv_path).keys())


def _normalize_title(title: str) -> str:
    s = (title or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _existing_handle_titles(csv_path: Path) -> dict[str, str]:
    if not csv_path.is_file():
        return {}
    mapping: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            handle_value = (row.get("Handle") or "").strip()
            title = (row.get("Title") or "").strip()
            if handle_value and title:
                mapping[handle_value] = title
    return mapping


def _classify_handle_changes(
    *,
    previous: dict[str, str],
    current: dict[str, str],
) -> tuple[list[dict[str, str]], list[str], list[str], int]:
    """
    Return renamed pairs, truly added handles, truly removed handles, unchanged count.
    Renames are inferred when an removed handle's title matches a new handle's title.
    """
    prev_handles = set(previous)
    new_handles = set(current)
    added = sorted(new_handles - prev_handles)
    removed = sorted(prev_handles - new_handles)
    unchanged = len(prev_handles & new_handles)

    new_by_title: dict[str, list[str]] = {}
    for handle, title in current.items():
        norm = _normalize_title(title)
        if norm:
            new_by_title.setdefault(norm, []).append(handle)

    renamed: list[dict[str, str]] = []
    renamed_old: set[str] = set()
    renamed_new: set[str] = set()
    for old_handle in removed:
        title = previous.get(old_handle, "")
        norm = _normalize_title(title)
        if not norm:
            continue
        candidates = [h for h in new_by_title.get(norm, []) if h in added]
        if len(candidates) == 1:
            new_handle = candidates[0]
            renamed.append(
                {
                    "title": title,
                    "old_handle": old_handle,
                    "new_handle": new_handle,
                }
            )
            renamed_old.add(old_handle)
            renamed_new.add(new_handle)

    truly_added = sorted(set(added) - renamed_new)
    truly_removed = sorted(set(removed) - renamed_old)
    return renamed, truly_added, truly_removed, unchanged


def _blank_row(headers: list[str]) -> dict[str, str]:
    return {header: "" for header in headers}


def _published_value(status: str) -> str:
    return "true" if (status or "").upper() == "ACTIVE" else "false"


def _option_slot(options: list[dict[str, Any]], index: int) -> tuple[str, str]:
    if index >= len(options):
        return "", ""
    opt = options[index]
    name = str(opt.get("name") or "").strip()
    values = opt.get("values") or []
    default_value = str(values[0]).strip() if values else ""
    return name, default_value


def _variant_option_values(variant: dict[str, Any], options: list[dict[str, Any]]) -> list[str]:
    selected = {
        str(item.get("name") or "").strip(): str(item.get("value") or "").strip()
        for item in (variant.get("selectedOptions") or [])
    }
    values: list[str] = []
    for opt in options[:3]:
        name = str(opt.get("name") or "").strip()
        values.append(selected.get(name, ""))
    while len(values) < 3:
        values.append("")
    return values


def _metafield_map(node: dict[str, Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for edge in ((node.get("metafields") or {}).get("edges") or []):
        mf = edge.get("node") or {}
        namespace = str(mf.get("namespace") or "").strip().lower()
        key = str(mf.get("key") or "").strip().lower()
        value = str(mf.get("value") or "").strip()
        if namespace and key and value:
            out[(namespace, key)] = value
    return out


def product_to_csv_rows(
    node: dict[str, Any],
    headers: list[str],
    metafield_columns: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    """Flatten one Shopify product node into Shopify-export-style variant rows."""
    options = node.get("options") or []
    variants = [edge.get("node") or {} for edge in ((node.get("variants") or {}).get("edges") or [])]
    if not variants:
        variants = [{}]

    tags = node.get("tags") or []
    if isinstance(tags, str):
        tags_text = tags
    else:
        tags_text = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())

    category = ""
    category_node = node.get("category") or {}
    if isinstance(category_node, dict):
        category = str(category_node.get("fullName") or "").strip()

    metafields = _metafield_map(node)
    opt1_name, _ = _option_slot(options, 0)
    opt2_name, _ = _option_slot(options, 1)
    opt3_name, _ = _option_slot(options, 2)

    rows: list[dict[str, str]] = []
    for index, variant in enumerate(variants):
        row = _blank_row(headers)
        if index == 0:
            row["Handle"] = str(node.get("handle") or "").strip()
            row["Title"] = str(node.get("title") or "").strip()
            row["Body (HTML)"] = str(node.get("descriptionHtml") or "").strip()
            row["Vendor"] = str(node.get("vendor") or "").strip()
            row["Product Category"] = category
            row["Type"] = str(node.get("productType") or "").strip()
            row["Tags"] = tags_text
            row["Published"] = _published_value(str(node.get("status") or ""))
            row["Gift Card"] = "false"
            row["Status"] = str(node.get("status") or "").strip().lower()
        else:
            row["Handle"] = str(node.get("handle") or "").strip()

        row["Option1 Name"] = opt1_name
        row["Option2 Name"] = opt2_name
        row["Option3 Name"] = opt3_name
        opt_values = _variant_option_values(variant, options)
        row["Option1 Value"] = opt_values[0]
        row["Option2 Value"] = opt_values[1]
        row["Option3 Value"] = opt_values[2]
        row["Variant SKU"] = str(variant.get("sku") or "").strip()
        row["Variant Price"] = str(variant.get("price") or "").strip()
        compare_at = variant.get("compareAtPrice")
        row["Variant Compare At Price"] = str(compare_at or "").strip()
        row["Variant Barcode"] = str(variant.get("barcode") or "").strip()
        row["Variant Requires Shipping"] = "true"
        row["Variant Taxable"] = "true"
        row["Variant Weight Unit"] = "lb"

        for mf_key, column in metafield_columns.items():
            row[column] = metafields.get(mf_key, "")

        rows.append(row)
    return rows


def fetch_all_products(
    *,
    shop_domain: str,
    access_token: str,
    api_version: str = "2024-01",
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    if not shop_domain or not access_token:
        raise RuntimeError("Shopify shop domain or access token is missing.")

    url = f"https://{shop_domain}/admin/api/{api_version}/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": access_token,
    }

    products: list[dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        payload = {"query": PRODUCTS_QUERY, "variables": {"cursor": cursor}}
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise RuntimeError(f"Shopify GraphQL errors: {body['errors']}")

        data = (((body.get("data") or {}).get("products")) or {})
        for edge in data.get("edges") or []:
            node = edge.get("node")
            if node:
                products.append(node)

        page_info = data.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            break

    return products


def _min_handle_threshold(existing_count: int, configured_min: Optional[int]) -> int:
    if configured_min is not None:
        return max(configured_min, 1)
    if existing_count <= 0:
        return 50
    return max(50, int(existing_count * 0.5))


def sync_osakiusa_products(
    *,
    csv_path: Path = DEFAULT_CSV_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    target_domain: str = "osakiusa.com",
    dry_run: bool = False,
    min_handles: Optional[int] = None,
) -> SyncResult:
    from store_config import get_store_config  # noqa: WPS433

    if not csv_path.is_file():
        return SyncResult(ok=False, message=f"Template CSV not found: {csv_path}")

    headers = _load_csv_headers(csv_path)
    metafield_columns = _parse_metafield_columns(headers)
    previous_catalog = _existing_handle_titles(csv_path)
    previous_handles = set(previous_catalog)

    store = get_store_config(target_domain)
    shop_domain = store.get("shop_domain") or ""
    access_token = store.get("shop_access_token") or ""
    if not shop_domain or not access_token:
        return SyncResult(
            ok=False,
            message=(
                f"Missing Shopify credentials for {target_domain}. "
                "Set OSAKI_SHOP_DOMAIN and OSAKI_ACCESS_TOKEN."
            ),
        )

    try:
        products = fetch_all_products(
            shop_domain=shop_domain,
            access_token=access_token,
        )
    except Exception as exc:
        return SyncResult(ok=False, message=f"Shopify fetch failed: {exc}")

    rows: list[dict[str, str]] = []
    new_handles: set[str] = set()
    new_catalog: dict[str, str] = {}
    for product in products:
        handle = str(product.get("handle") or "").strip()
        if not handle:
            continue
        new_handles.add(handle)
        title = str(product.get("title") or "").strip()
        if title:
            new_catalog[handle] = title
        rows.extend(product_to_csv_rows(product, headers, metafield_columns))

    threshold = _min_handle_threshold(len(previous_handles), min_handles)
    if len(new_handles) < threshold:
        return SyncResult(
            ok=False,
            message=(
                f"Aborting: fetched only {len(new_handles)} handles "
                f"(minimum {threshold}). Existing CSV left unchanged."
            ),
            handles=len(new_handles),
            variant_rows=len(rows),
            dry_run=dry_run,
        )

    added = sorted(new_handles - previous_handles)
    removed = sorted(previous_handles - new_handles)
    renamed, truly_added, truly_removed, unchanged = _classify_handle_changes(
        previous=previous_catalog,
        current=new_catalog,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"shopify_sync_report_{timestamp}.json"
    report_payload = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "target_domain": target_domain,
        "shop_domain": shop_domain,
        "handles": len(new_handles),
        "variant_rows": len(rows),
        "unchanged_handles": unchanged,
        "added_handles": added,
        "removed_handles": removed,
        "renamed_products": renamed,
        "truly_added_handles": truly_added,
        "truly_removed_handles": truly_removed,
        "dry_run": dry_run,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        return SyncResult(
            ok=True,
            message="Dry run complete; CSV not modified.",
            handles=len(new_handles),
            variant_rows=len(rows),
            added_handles=added,
            removed_handles=removed,
            renamed_products=renamed,
            truly_added_handles=truly_added,
            truly_removed_handles=truly_removed,
            unchanged_handles=unchanged,
            csv_path=str(csv_path),
            report_path=str(report_path),
            dry_run=True,
        )

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"products_export.{timestamp}.csv"
    shutil.copy2(csv_path, backup_path)

    temp_path = csv_path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(csv_path)

    return SyncResult(
        ok=True,
        message=f"Synced {len(new_handles)} handles ({len(rows)} variant rows) from {shop_domain}.",
        handles=len(new_handles),
        variant_rows=len(rows),
        added_handles=added,
        removed_handles=removed,
        renamed_products=renamed,
        truly_added_handles=truly_added,
        truly_removed_handles=truly_removed,
        unchanged_handles=unchanged,
        csv_path=str(csv_path),
        report_path=str(report_path),
    )
