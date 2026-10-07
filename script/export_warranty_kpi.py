#!/usr/bin/env python3
"""Export a KPI snapshot from the warranty ticket database.

Writes a JSON file to ``data/reports/kpi_<date>.json`` that captures the
same funnel numbers shown in ``/admin/warranty/metrics`` plus trace-level
summaries when the ``request_traces`` table exists.

Usage
-----
  # Default: last 30 days
  python script/export_warranty_kpi.py

  # Custom window
  python script/export_warranty_kpi.py --days 7

Designed to run daily via cron or manually before a review.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pytz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

_CST = pytz.timezone("America/Chicago")
_DB_PATH = ROOT / "db_data" / "chat_history.db"
_REPORT_DIR = ROOT / "data" / "reports"


def _now_cst() -> datetime:
    return datetime.now(_CST)


def _percent(part: int, whole: int) -> float:
    return round((part / whole) * 100.0, 1) if whole > 0 else 0.0


def collect_warranty_kpi(days: int) -> Dict[str, Any]:
    """Query the warranty ticket table and build a KPI dict."""
    from warranty_models import WarrantyTicket, warranty_db_session

    end = _now_cst()
    start = (end - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    abandon_hours = 6

    with warranty_db_session() as db:
        tickets = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.created_at >= start)
            .all()
        )

    total = len(tickets)
    terminal = 0
    contact = 0
    admin_decided = 0
    resolved = 0
    abandoned = 0
    self_service_started = 0
    self_service_resolved = 0
    issue_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    abandon_cutoff = end - timedelta(hours=abandon_hours)

    for t in tickets:
        status = (t.status or "in_progress").lower()
        issue = (t.issue_type or "unknown").lower()
        collected = t.get_collected() if hasattr(t, "get_collected") else {}
        has_email = bool(
            str(collected.get("customer_email") or "").strip()
            or str(collected.get("customer_contact_email") or "").strip()
        )
        troubleshooting = str(collected.get("troubleshooting_outcome") or "").strip().lower()

        status_counts[status] += 1
        issue_counts[issue] += 1

        if status != "in_progress":
            terminal += 1
        elif hasattr(t, "updated_at") and t.updated_at:
            updated = t.updated_at
            if updated.tzinfo is None:
                updated = _CST.localize(updated)
            if updated < abandon_cutoff:
                abandoned += 1

        if has_email:
            contact += 1
        if t.admin_decision and str(t.admin_decision) != "self_resolved":
            admin_decided += 1
        if status == "resolved":
            resolved += 1
        if troubleshooting:
            self_service_started += 1
        if troubleshooting == "resolved" or str(t.admin_decision or "") == "self_resolved":
            self_service_resolved += 1

    return {
        "range": {"days": days, "start": start.isoformat(), "end": end.isoformat()},
        "totals": {
            "started": total,
            "reached_terminal": terminal,
            "completion_rate_pct": _percent(terminal, total),
            "contact_captured": contact,
            "contact_rate_pct": _percent(contact, total),
            "admin_decided": admin_decided,
            "resolved": resolved,
            "self_service_started": self_service_started,
            "self_service_resolved": self_service_resolved,
            "self_service_rate_pct": _percent(self_service_resolved, self_service_started),
            "abandoned": abandoned,
            "abandoned_rate_pct": _percent(abandoned, total),
        },
        "by_status": dict(status_counts.most_common()),
        "by_issue_type": dict(issue_counts.most_common()),
    }


def collect_trace_summary() -> Dict[str, Any]:
    """Summarise request_traces if the table exists."""
    import sqlite3
    if not _DB_PATH.exists():
        return {}
    conn = sqlite3.connect(str(_DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='request_traces'")
        if not cur.fetchone():
            return {}

        cur.execute("SELECT COUNT(*) FROM request_traces")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM request_traces WHERE scope_blocked = 1")
        blocked = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM request_traces WHERE cache_hit = 1")
        cached = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM request_traces WHERE guard_sanitized = 1")
        guarded = cur.fetchone()[0]
        cur.execute("SELECT AVG(total_latency_ms) FROM request_traces WHERE total_latency_ms > 0")
        avg_lat = cur.fetchone()[0] or 0
        cur.execute(
            "SELECT AVG(total_latency_ms) FROM request_traces "
            "WHERE total_latency_ms > 0 ORDER BY total_latency_ms "
            "LIMIT 1 OFFSET (SELECT COUNT(*)/100*95 FROM request_traces WHERE total_latency_ms > 0)"
        )
        row = cur.fetchone()
        p95_lat = row[0] if row and row[0] else avg_lat

        return {
            "total_requests": total,
            "scope_blocked": blocked,
            "cache_hit": cached,
            "guard_sanitized": guarded,
            "guard_trip_rate_pct": _percent(guarded, total),
            "avg_latency_ms": round(avg_lat),
            "p95_latency_ms": round(p95_lat),
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export warranty KPI snapshot")
    parser.add_argument("--days", type=int, default=30, help="Lookback window (default 30)")
    args = parser.parse_args()

    kpi = collect_warranty_kpi(args.days)
    trace = collect_trace_summary()
    if trace:
        kpi["trace_summary"] = trace

    kpi["exported_at"] = _now_cst().isoformat()

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = _now_cst().strftime("%Y%m%d")
    out_path = _REPORT_DIR / f"kpi_{date_str}.json"
    out_path.write_text(json.dumps(kpi, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"KPI snapshot written to {out_path}")
    print(json.dumps(kpi, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
