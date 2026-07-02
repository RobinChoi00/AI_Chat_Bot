"""
Freshdesk sync metadata and admin dashboard payload.

Tracks last ticket/KB sync runs in ``data/freshdesk_sync_status.json`` and
exposes a read-only snapshot for the admin UI (connection, file ages, counts).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATUS_PATH = _PROJECT_ROOT / "data" / "freshdesk_sync_status.json"
_TICKETS_PATH = _PROJECT_ROOT / "data" / "freshdesk_tickets.json"
_SOLUTIONS_PATH = _PROJECT_ROOT / "data" / "freshdesk_solutions.json"

_STALE_DAYS = 7


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_status() -> dict[str, Any]:
    if not _STATUS_PATH.is_file():
        return {}
    try:
        with _STATUS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_status(data: dict[str, Any]) -> None:
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATUS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp.replace(_STATUS_PATH)


def record_sync_result(kind: str, result: dict[str, Any]) -> None:
    """Persist the outcome of a ticket or KB sync run."""
    kind = (kind or "").strip().lower()
    if kind not in ("tickets", "kb"):
        return

    status = _load_status()
    entry: dict[str, Any] = {
        "last_sync_at": _utc_now_iso(),
        "ok": bool(result.get("ok")),
        "message": str(result.get("message") or ""),
    }
    if kind == "tickets":
        entry.update(
            {
                "ticket_count": int(result.get("ticket_count") or 0),
                "resolved_scanned": int(result.get("resolved_scanned") or 0),
                "usable_qa_pairs": int(result.get("usable_qa_pairs") or 0),
                "search_pages_fetched": int(result.get("search_pages_fetched") or 0),
                "month_windows_scanned": int(result.get("month_windows_scanned") or 0),
            }
        )
    else:
        entry["article_count"] = int(result.get("article_count") or 0)

    domain = str(result.get("domain") or "").strip()
    if domain:
        entry["domain"] = domain

    status[kind] = entry
    _save_status(status)


def _file_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "count": 0, "modified_at": None, "size_bytes": 0}
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        modified_at = mtime.replace(microsecond=0).isoformat()
    except OSError:
        modified_at = None

    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            count = len(payload)
    except (json.JSONDecodeError, OSError):
        count = 0

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0

    return {
        "exists": True,
        "path": str(path),
        "count": count,
        "modified_at": modified_at,
        "size_bytes": size_bytes,
    }


def _is_stale(iso_ts: str | None) -> bool:
    if not iso_ts:
        return True
    try:
        parsed = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - parsed).total_seconds() / 86400
        return age_days > _STALE_DAYS
    except ValueError:
        return True


def _knowledge_counts() -> dict[str, int]:
    try:
        from warranty_knowledge import load_knowledge_entries  # noqa: WPS433

        entries = load_knowledge_entries()
    except Exception:
        return {"total": 0, "freshdesk": 0, "freshdesk_kb": 0}

    freshdesk = sum(1 for e in entries if getattr(e, "source", "") == "freshdesk")
    freshdesk_kb = sum(1 for e in entries if getattr(e, "source", "") == "freshdesk_kb")
    return {
        "total": len(entries),
        "freshdesk": freshdesk,
        "freshdesk_kb": freshdesk_kb,
    }


def get_freshdesk_dashboard(*, probe_connection: bool = True) -> dict[str, Any]:
    """Build admin dashboard payload for Freshdesk health."""
    from freshdesk_sync import normalize_freshdesk_domain  # noqa: WPS433

    raw_domain = os.getenv("FRESHDESK_DOMAIN", "").strip()
    api_key_set = bool(os.getenv("FRESHDESK_API_KEY", "").strip())
    domain = normalize_freshdesk_domain(raw_domain) if raw_domain else ""

    configured = bool(domain and api_key_set)
    connection_ok: bool | None = None
    if probe_connection and configured:
        try:
            from freshdesk_sync import FreshdeskETL  # noqa: WPS433

            connection_ok = FreshdeskETL().verify_connection()
        except EnvironmentError:
            connection_ok = False

    sync_status = _load_status()
    tickets_sync = sync_status.get("tickets") or {}
    kb_sync = sync_status.get("kb") or {}

    outbound_enabled = os.getenv("WARRANTY_FRESHDESK_CREATE_CASE", "1") == "1"

    return {
        "configured": configured,
        "connection_ok": connection_ok,
        "domain": domain or None,
        "portal_url": f"https://{domain}/a/tickets" if domain else None,
        "credentials": {
            "domain_set": bool(raw_domain),
            "api_key_set": api_key_set,
        },
        "outbound": {
            "create_case_enabled": outbound_enabled and configured,
            "group_id": os.getenv("FRESHDESK_WARRANTY_GROUP_ID", "").strip() or None,
            "product_id": os.getenv("FRESHDESK_WARRANTY_PRODUCT_ID", "").strip() or None,
        },
        "files": {
            "tickets": _file_snapshot(_TICKETS_PATH),
            "kb": _file_snapshot(_SOLUTIONS_PATH),
        },
        "last_sync": {
            "tickets": tickets_sync,
            "kb": kb_sync,
        },
        "stale": {
            "tickets": _is_stale(tickets_sync.get("last_sync_at")),
            "kb": _is_stale(kb_sync.get("last_sync_at")),
        },
        "knowledge": _knowledge_counts(),
    }
