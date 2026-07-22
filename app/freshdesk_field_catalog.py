"""
Freshdesk ticket field catalog — official ID ↔ label maps from Admin API.

Fred's Ticket Queue extension (and any export/report) needs numeric status and
custom-field IDs resolved to human labels.  Freshdesk stores only IDs on tickets;
labels live in ``GET /api/v2/admin/ticket_fields`` under each field's
``choices`` object.

This module fetches that catalog, saves a snapshot under ``data/``, and exposes
helpers to resolve ticket payloads.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CATALOG_PATH = _PROJECT_ROOT / "data" / "freshdesk_field_choices.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _choice_label(raw: Any) -> str:
    """Freshdesk choices values are ``[display, internal]`` arrays or plain strings."""
    if isinstance(raw, list) and raw:
        return str(raw[0] or raw[-1] or "").strip()
    return str(raw or "").strip()


def _normalize_choices(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        out: dict[str, str] = {}
        for key, value in raw.items():
            label = _choice_label(value)
            if label:
                out[str(key)] = label
        return out

    if isinstance(raw, list):
        out: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            choice_id = item.get("id")
            if choice_id is None:
                choice_id = item.get("value")
            label = str(
                item.get("label")
                or item.get("value")
                or item.get("label_for_customers")
                or ""
            ).strip()
            if choice_id is not None and label:
                out[str(choice_id)] = label
            for nested in item.get("choices") or []:
                if not isinstance(nested, dict):
                    continue
                nested_id = nested.get("id") or nested.get("value")
                nested_label = str(
                    nested.get("label") or nested.get("value") or ""
                ).strip()
                if nested_id is not None and nested_label:
                    out[str(nested_id)] = nested_label
        return out

    return {}


def _field_needs_detail(field: dict[str, Any]) -> bool:
    name = str(field.get("name") or "")
    field_type = str(field.get("type") or "")
    if name == "status" or field_type == "default_status":
        return True
    if field_type in ("custom_dropdown", "nested_field"):
        return True
    if name.startswith("cf_"):
        return True
    return not _normalize_choices(field.get("choices"))


def _merge_field_choices(base: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ("choices", "label", "label_for_customers", "type"):
        if detail.get(key) is not None:
            merged[key] = detail[key]
    return merged


def parse_ticket_fields_payload(fields: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Turn Freshdesk ``/admin/ticket_fields`` JSON into a lookup-friendly catalog.

    Returns::
        {
          "status": {"name": "status", "label": "Status", "choices": {"2": "Open", ...}},
          "custom_fields": [{"name": "cf_...", "label": "...", "choices": {...}}, ...],
          "lookup": {"status": {...}, "cf_...": {...}},
        }
    """
    status_block: Optional[dict[str, Any]] = None
    custom_fields: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, str]] = {}

    for field in fields or []:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        label = str(field.get("label") or field.get("label_for_customers") or name).strip()
        choices = _normalize_choices(field.get("choices"))
        if not choices:
            continue

        entry = {
            "field_id": field.get("id"),
            "name": name,
            "label": label,
            "type": field.get("type"),
            "choices": choices,
            "by_label": {v.lower(): k for k, v in choices.items()},
        }

        if name == "status":
            status_block = entry
            lookup["status"] = choices
        elif name.startswith("cf_") or field.get("type") in (
            "custom_dropdown",
            "nested_field",
        ):
            custom_fields.append(entry)
            lookup[name] = choices

    custom_fields.sort(key=lambda row: str(row.get("label") or row.get("name") or ""))

    return {
        "status": status_block,
        "custom_fields": custom_fields,
        "lookup": lookup,
    }


def fetch_ticket_field_catalog(*, timeout: int = 20) -> dict[str, Any]:
    """
    Live fetch from Freshdesk Admin API.

    Requires ``FRESHDESK_DOMAIN`` and ``FRESHDESK_API_KEY``.

    Freshdesk often omits ``choices`` on the list endpoint; we fetch each
    dropdown/status field by id (``GET /admin/ticket_fields/{id}``).
    """
    from freshdesk_sync import normalize_freshdesk_domain  # noqa: WPS433

    raw_domain = os.getenv("FRESHDESK_DOMAIN", "").strip()
    api_key = os.getenv("FRESHDESK_API_KEY", "").strip()
    if not raw_domain or not api_key:
        raise EnvironmentError("FRESHDESK_DOMAIN and FRESHDESK_API_KEY are required.")

    domain = normalize_freshdesk_domain(raw_domain)
    auth = (api_key, "X")
    list_url = f"https://{domain}/api/v2/admin/ticket_fields"

    response = requests.get(list_url, auth=auth, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Freshdesk ticket_fields HTTP {response.status_code}: {(response.text or '')[:300]}"
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Freshdesk ticket_fields response is not a list.")

    enriched: list[dict[str, Any]] = []
    for field in payload:
        if not isinstance(field, dict):
            continue
        current = dict(field)
        if _field_needs_detail(current) and current.get("id") is not None:
            field_id = current["id"]
            detail_url = f"https://{domain}/api/v2/admin/ticket_fields/{field_id}"
            try:
                detail_resp = requests.get(detail_url, auth=auth, timeout=timeout)
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    if isinstance(detail, dict):
                        current = _merge_field_choices(current, detail)
            except requests.RequestException as exc:
                logger.warning("Freshdesk field detail fetch failed id=%s: %s", field_id, exc)
        enriched.append(current)

    parsed = parse_ticket_fields_payload(enriched)
    return {
        "fetched_at": _utc_now_iso(),
        "domain": domain,
        **parsed,
    }


def save_field_catalog(catalog: dict[str, Any], path: Path | None = None) -> Path:
    target = path or _CATALOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, ensure_ascii=False, indent=2)
    tmp.replace(target)
    return target


def load_field_catalog(path: Path | None = None) -> dict[str, Any]:
    target = path or _CATALOG_PATH
    if not target.is_file():
        return {}
    try:
        with target.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def get_field_catalog(*, refresh: bool = False) -> dict[str, Any]:
    """Return cached catalog, optionally refreshing from Freshdesk."""
    if refresh:
        catalog = fetch_ticket_field_catalog()
        save_field_catalog(catalog)
        return catalog
    cached = load_field_catalog()
    if cached.get("lookup"):
        return cached
    catalog = fetch_ticket_field_catalog()
    save_field_catalog(catalog)
    return catalog


def resolve_choice_label(
    catalog: dict[str, Any],
    field_name: str,
    choice_id: Any,
) -> str:
    """Map a numeric/string choice id to its display label."""
    if choice_id is None or choice_id == "":
        return ""
    lookup = (catalog.get("lookup") or {}).get(field_name) or {}
    key = str(choice_id)
    if key in lookup:
        return lookup[key]
    return f"Unknown({key})"


def label_ticket(catalog: dict[str, Any], ticket: dict[str, Any]) -> dict[str, str]:
    """
    Resolve ``status`` and ``custom_fields`` on a Freshdesk ticket dict.

    Returns flat ``{field_name: label}`` including ``status``.
    """
    labels: dict[str, str] = {}
    if "status" in ticket:
        labels["status"] = resolve_choice_label(catalog, "status", ticket.get("status"))

    custom = ticket.get("custom_fields") or {}
    if isinstance(custom, dict):
        for field_name, value in custom.items():
            if value is None or value == "":
                continue
            labels[str(field_name)] = resolve_choice_label(catalog, str(field_name), value)
    return labels


def format_status_table(catalog: dict[str, Any]) -> str:
    """Human-readable status ID table for Slack/email."""
    status = catalog.get("status") or {}
    choices = status.get("choices") or {}
    lines = ["Freshdesk Status IDs (official):"]
    for choice_id in sorted(choices, key=lambda k: int(k) if str(k).isdigit() else k):
        lines.append(f"  {choice_id}: {choices[choice_id]}")
    return "\n".join(lines)


def format_parts_related_fields(catalog: dict[str, Any]) -> str:
    """List custom dropdown fields whose label mentions parts (case-insensitive)."""
    rows = catalog.get("custom_fields") or []
    lines: list[str] = []
    for field in rows:
        label = str(field.get("label") or "")
        if "part" not in label.lower() and "part" not in str(field.get("name") or "").lower():
            continue
        lines.append(f"{field.get('name')} — {label}:")
        for choice_id, choice_label in sorted(
            (field.get("choices") or {}).items(),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0],
        ):
            lines.append(f"  {choice_id}: {choice_label}")
    return "\n".join(lines) if lines else "(no parts-related custom fields found)"
