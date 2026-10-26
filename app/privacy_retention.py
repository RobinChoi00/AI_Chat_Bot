"""Conservative, explicit retention of customer conversations.

The default is a dry run and a legal hold is enabled unless explicitly disabled.
No message text, customer identifiers, file paths, or exception text is emitted.
Accounting records are retained; expired sales attribution is de-identified.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
_CLOSED = frozenset({"closed", "resolved"})


def legal_hold_active() -> bool:
    """Fail closed: only an explicit false value releases the global hold."""
    return os.getenv("PRIVACY_RETENTION_LEGAL_HOLD", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _days(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(value, 36500) if value > 0 else default


def safe_evidence_path(value: str, upload_root: Path) -> Path | None:
    """Allow only actual descendants of the configured upload directory."""
    if not value or not value.strip():
        return None
    root = upload_root.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved == root or not resolved.is_relative_to(root):
        return None
    # Do not traverse symlinks, even if their current destination is in-root.
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        return None
    return resolved


def purge_chat_records(
    *, db_path: Path | None = None, upload_root: Path | None = None,
    apply: bool = False, now: datetime | None = None,
) -> dict[str, Any]:
    """Count or remove expired records, preserving active/recent conversations.

    Closed cases must have no recent child activity. Chat/trace deletion also
    excludes any session with a live or retained case. Missing tables are
    skipped; incomplete existing schemas fail closed. A DB is never created.
    """
    chat_days = _days("CHAT_LOG_RETENTION_DAYS", 90)
    trace_days = _days("TRACE_RETENTION_DAYS", 30)
    case_days = _days("CLOSED_CASE_RETENTION_DAYS", 730)
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ZoneInfo("America/Chicago"))
    local = moment.astimezone(ZoneInfo("America/Chicago")).replace(tzinfo=None)
    utc = moment.astimezone(timezone.utc).replace(tzinfo=None)
    chat_cutoff, trace_cutoff, case_cutoff = (
        local - timedelta(days=chat_days), utc - timedelta(days=trace_days),
        local - timedelta(days=case_days),
    )
    result: dict[str, Any] = {
        "ok": True, "apply": bool(apply), "applied": False,
        "legal_hold": legal_hold_active(), "blocked_by_legal_hold": False,
        "retention_days": {"chat_logs": chat_days, "request_traces": trace_days, "closed_cases": case_days},
        "candidate_rows": {}, "deleted_rows": {}, "deidentified_rows": {},
        "candidate_files": 0, "deleted_files": 0, "missing_files": 0,
        "unsafe_file_paths": 0, "shared_file_paths": 0, "file_errors": 0,
    }
    path = (db_path or ROOT / "db_data" / "chat_history.db").resolve()
    if not path.is_file():
        result.update(ok=False, error="database_not_found")
        return result
    conn = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    root = (upload_root or ROOT / "uploaded_evidence").resolve()
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN IMMEDIATE" if apply and not result["legal_hold"] else "BEGIN")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {table: {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
                   for table in tables if table.replace("_", "").isalnum()}

        def rows(table: str, required: tuple[str, ...]):
            if table not in tables:
                return []
            if not set(required).issubset(columns.get(table, set())):
                raise ValueError("Incomplete retention schema")
            selected = ", ".join(f'"{name}"' for name in required)
            return conn.execute(f'SELECT {selected} FROM "{table}"').fetchall()

        def old(value: Any, cutoff: datetime, *, utc_values: bool = False) -> bool:
            if not value:
                return False
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    zone = timezone.utc if utc_values else ZoneInfo("America/Chicago")
                    dt = dt.astimezone(zone).replace(tzinfo=None)
                return dt < cutoff
            except (TypeError, ValueError):
                return False

        warranty = rows("warranty_tickets", ("id", "ticket_id", "session_id", "status", "created_at", "updated_at"))
        sales = rows("sales_sessions", ("id", "session_id", "status", "created_at", "updated_at"))
        protected: set[str] = set()
        # Null/invalid dates never prove expiry.
        for record in (*warranty, *sales):
            if (str(record["status"]).lower() not in _CLOSED or
                    not old(record["created_at"], case_cutoff) or not old(record["updated_at"], case_cutoff)):
                protected.add(str(record["session_id"]))
        for table, date_col, cutoff in (
            ("chat_logs", "created_at", chat_cutoff),
            ("request_traces", "created_at", trace_cutoff),
            ("warranty_chat_consents", "updated_at", chat_cutoff),
            ("warranty_conversation_messages", "created_at", chat_cutoff),
            ("sales_messages", "created_at", chat_cutoff),
            ("sales_leads", "updated_at", chat_cutoff),
            ("sales_feedback", "created_at", chat_cutoff),
            ("sales_events", "created_at", chat_cutoff),
            ("chat_feedback", "updated_at", chat_cutoff),
            ("customer_journey_events", "created_at", chat_cutoff),
        ):
            for record in rows(table, ("session_id", date_col)):
                if not old(record[date_col], cutoff, utc_values=table == "request_traces"):
                    protected.add(str(record["session_id"]))

        ticket_children = {
            table: rows(table, ("id", "ticket_id", date_col))
            for table, date_col in (
                ("warranty_turns", "created_at"), ("warranty_conversation_messages", "created_at"),
                ("warranty_evidences", "created_at"), ("warranty_outcome_observations", "observed_at"),
            )
        }
        recent_tickets = {str(record["ticket_id"]) for records in ticket_children.values()
                          for record in records if not old(record[-1], case_cutoff)}
        expired_warranty = {str(record["ticket_id"]): record for record in warranty
                            if str(record["session_id"]) not in protected and
                            str(record["ticket_id"]) not in recent_tickets}
        evidence = rows("warranty_evidences", ("id", "ticket_id", "file_path"))
        file_plan: dict[str, list[Path]] = {}
        for record in evidence:
            ticket = str(record["ticket_id"])
            if ticket not in expired_warranty or not record["file_path"]:
                continue
            safe = safe_evidence_path(str(record["file_path"]), root)
            if safe is None or (safe.exists() and not safe.is_file()):
                result["unsafe_file_paths"] += 1
                expired_warranty.pop(ticket, None)
                file_plan.pop(ticket, None)
            else:
                file_plan.setdefault(ticket, []).append(safe)

        sales_children = {table: rows(table, ("id", "session_id", date_col))
                          for table, date_col in (
                              ("sales_messages", "created_at"), ("sales_leads", "updated_at"),
                              ("sales_feedback", "created_at"), ("sales_events", "created_at"),
                          )}
        recent_sales = {str(record["session_id"]) for records in sales_children.values()
                        for record in records if not old(record[-1], case_cutoff)}
        conversions = rows("sales_conversions", ("id", "session_id", "created_at", "lifecycle_updated_at"))
        for record in conversions:
            if not old(record["created_at"], case_cutoff) or (
                    record["lifecycle_updated_at"] and not old(record["lifecycle_updated_at"], case_cutoff)):
                recent_sales.add(str(record["session_id"]))
        expired_sales = {str(record["session_id"]): record for record in sales
                         if str(record["session_id"]) not in protected and str(record["session_id"]) not in recent_sales}
        retained = {str(record["session_id"]) for record in warranty if str(record["ticket_id"]) not in expired_warranty}
        retained.update(str(record["session_id"]) for record in sales if str(record["session_id"]) not in expired_sales)
        protected.update(retained)
        # Cross-surface sessions are retained together whenever either surface
        # has a case that is still needed.
        expired_warranty = {key: record for key, record in expired_warranty.items() if str(record["session_id"]) not in protected}
        expired_sales = {key: record for key, record in expired_sales.items() if key not in protected}
        # Evidence may be referenced by more than one ticket. A retained case
        # must never lose its file when another case expires. Propagate any
        # newly retained case across shared sessions until the set is stable.
        while True:
            retained_paths = {safe_evidence_path(str(record["file_path"]), root) for record in evidence
                              if str(record["ticket_id"]) not in expired_warranty and record["file_path"]}
            shared = {ticket for ticket, paths in file_plan.items()
                      if ticket in expired_warranty and any(file in retained_paths for file in paths)}
            result["shared_file_paths"] += len(shared)
            for ticket in shared:
                protected.add(str(expired_warranty[ticket]["session_id"]))
            prior = (len(expired_warranty), len(expired_sales))
            expired_warranty = {key: record for key, record in expired_warranty.items() if str(record["session_id"]) not in protected}
            expired_sales = {key: record for key, record in expired_sales.items() if key not in protected}
            if prior == (len(expired_warranty), len(expired_sales)):
                break
        plan: dict[str, set[int]] = {}

        def add(table: str, ids) -> None:
            selected = {int(value) for value in ids}
            if selected:
                plan.setdefault(table, set()).update(selected)

        for table, cutoff in (("chat_logs", chat_cutoff), ("request_traces", trace_cutoff),
                              ("chat_feedback", chat_cutoff), ("warranty_chat_consents", chat_cutoff)):
            date_col = "updated_at" if table in {"chat_feedback", "warranty_chat_consents"} else "created_at"
            add(table, (record["id"] for record in rows(table, ("id", "session_id", date_col))
                        if str(record["session_id"]) not in protected and
                        old(record[date_col], cutoff, utc_values=table == "request_traces")))
        add("warranty_tickets", (record["id"] for record in expired_warranty.values()))
        for table, records in ticket_children.items():
            add(table, (record["id"] for record in records if str(record["ticket_id"]) in expired_warranty))
        add("sales_sessions", (record["id"] for record in expired_sales.values()))
        for table, records in sales_children.items():
            add(table, (record["id"] for record in records if str(record["session_id"]) in expired_sales))
        for table in ("conversation_releases", "customer_journey_events"):
            for record in rows(table, ("id", "surface", "conversation_id", "session_id")):
                match = (record["surface"] == "warranty" and str(record["conversation_id"]) in expired_warranty) or (
                    record["surface"] == "sales" and str(record["session_id"]) in expired_sales)
                if match and str(record["session_id"]) not in protected:
                    add(table, [record["id"]])
        conversion_ids = {int(record["id"]) for record in conversions if str(record["session_id"]) in expired_sales}
        files = list({path for ticket, paths in file_plan.items() if ticket in expired_warranty for path in paths})
        result["candidate_rows"] = {table: len(ids) for table, ids in sorted(plan.items())}
        if conversion_ids:
            result["candidate_deidentifications"] = {"sales_conversions": len(conversion_ids)}
        result["candidate_files"] = sum(path.is_file() for path in files)
        result["missing_files"] = sum(not path.exists() for path in files)
        if apply and result["legal_hold"]:
            result.update(ok=False, blocked_by_legal_hold=True)
        elif apply:
            # Do not discard evidence metadata if any file cannot be removed.
            # The global transaction rolls back database writes on file error.
            for file in files:
                try:
                    if file.is_file():
                        file.unlink()
                        result["deleted_files"] += 1
                except OSError:
                    result["file_errors"] += 1
            if result["file_errors"]:
                result.update(ok=False, error="evidence_file_removal_failed")
            else:
                for table, ids in plan.items():
                    if table in {"warranty_tickets", "sales_sessions"}:
                        continue
                    _delete_ids(conn, table, ids)
                for table in ("warranty_tickets", "sales_sessions"):
                    _delete_ids(conn, table, plan.get(table, set()))
                if conversion_ids:
                    fields = [name for name in ("session_id", "email", "matched_by") if name in columns["sales_conversions"]]
                    for chunk in _chunks(conversion_ids):
                        conn.execute(f'UPDATE sales_conversions SET {", ".join(name + " = NULL" for name in fields)} '
                                     f'WHERE id IN ({", ".join("?" for _ in chunk)})', chunk)
                    result["deidentified_rows"] = {"sales_conversions": len(conversion_ids)}
                conn.commit()
                result.update(applied=True, deleted_rows=dict(result["candidate_rows"]))
        if not result["applied"]:
            conn.rollback()
    except (sqlite3.Error, OSError, ValueError):
        conn.rollback()
        result.update(ok=False, error="retention_failed")
    finally:
        conn.close()
    logger.info("Privacy retention %s", result)
    return result


def _chunks(ids: set[int]):
    values = sorted(ids)
    for start in range(0, len(values), 400):
        yield values[start:start + 400]


def _delete_ids(conn: sqlite3.Connection, table: str, ids: set[int]) -> None:
    for chunk in _chunks(ids):
        conn.execute(f'DELETE FROM "{table}" WHERE id IN ({", ".join("?" for _ in chunk)})', chunk)
