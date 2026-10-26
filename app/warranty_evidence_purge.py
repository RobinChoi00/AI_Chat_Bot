"""
Purge old warranty evidence files from local disk + DB rows.

Default retention: 90 days (WARRANTY_EVIDENCE_RETENTION_DAYS).
Dry-run by default unless --apply is passed.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz

from privacy_retention import legal_hold_active, safe_evidence_path

logger = logging.getLogger(__name__)


def _now_cst() -> datetime:
    return datetime.now(pytz.timezone("America/Chicago"))


def retention_days(default: int = 90) -> int:
    raw = os.getenv("WARRANTY_EVIDENCE_RETENTION_DAYS", str(default)).strip()
    try:
        days = int(raw)
    except ValueError:
        return default
    return max(14, min(days, 730))


def purge_old_evidence(
    *,
    days: int | None = None,
    apply: bool = False,
    upload_root: Path | None = None,
) -> dict[str, Any]:
    from warranty_models import WarrantyEvidence, WarrantyTicket, warranty_db_session  # noqa: WPS433

    window = max(14, min(days, 730)) if days is not None and days > 0 else retention_days()
    cutoff = _now_cst() - timedelta(days=window)
    root = upload_root or (
        Path(__file__).resolve().parent.parent / "uploaded_evidence"
    )

    deleted_files = 0
    missing_files = 0
    deleted_rows = 0
    bytes_freed = 0
    errors: list[str] = []
    held = legal_hold_active()
    effective_apply = apply and not held

    with warranty_db_session() as db:
        rows = (
            db.query(WarrantyEvidence)
            .join(WarrantyTicket, WarrantyTicket.ticket_id == WarrantyEvidence.ticket_id)
            .filter(WarrantyEvidence.created_at < cutoff)
            .filter(WarrantyTicket.status.in_(("closed", "resolved")))
            .filter(WarrantyTicket.updated_at < cutoff)
            .all()
        )
        candidate_ids = {row.id for row in rows}
        retained_paths = {
            safe_evidence_path(str(path), root)
            for row_id, path in db.query(WarrantyEvidence.id, WarrantyEvidence.file_path).all()
            if row_id not in candidate_ids and path
        }
        for row in rows:
            path_raw = str(row.file_path or "").strip()
            if path_raw:
                path = safe_evidence_path(path_raw, root)
                if path is None:
                    errors.append("unsafe_file_path")
                    continue
                if path in retained_paths:
                    errors.append("shared_file_reference")
                    continue
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        if effective_apply:
                            path.unlink(missing_ok=True)
                        deleted_files += 1
                        bytes_freed += size
                    else:
                        missing_files += 1
                except OSError:
                    errors.append("file_removal_failed")
                    continue
            if effective_apply:
                db.delete(row)
            deleted_rows += 1

    result = {
        "ok": not (apply and held) and not errors,
        "apply": apply,
        "applied": effective_apply,
        "legal_hold": held,
        "blocked_by_legal_hold": bool(apply and held),
        "retention_days": window,
        "cutoff": cutoff.isoformat(),
        "candidate_rows": deleted_rows,
        "deleted_files": deleted_files if effective_apply else 0,
        "would_delete_files": deleted_files,
        "missing_files": missing_files,
        "bytes_freed": bytes_freed if effective_apply else 0,
        "would_free_bytes": bytes_freed,
        "errors": errors[:20],
    }
    logger.info("Evidence purge %s", result)
    return result
