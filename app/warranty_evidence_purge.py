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
    from warranty_models import WarrantyEvidence, warranty_db_session  # noqa: WPS433

    window = days if days is not None else retention_days()
    cutoff = _now_cst() - timedelta(days=window)
    root = upload_root or (
        Path(__file__).resolve().parent.parent / "uploaded_evidence"
    )

    deleted_files = 0
    missing_files = 0
    deleted_rows = 0
    bytes_freed = 0
    errors: list[str] = []

    with warranty_db_session() as db:
        rows = (
            db.query(WarrantyEvidence)
            .filter(WarrantyEvidence.created_at < cutoff)
            .all()
        )
        for row in rows:
            path_raw = str(row.file_path or "").strip()
            if path_raw:
                path = Path(path_raw)
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        if apply:
                            path.unlink(missing_ok=True)
                        deleted_files += 1
                        bytes_freed += size
                    else:
                        missing_files += 1
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
            if apply:
                db.delete(row)
            deleted_rows += 1

        if apply:
            # Remove empty ticket folders under uploaded_evidence/warranty/
            warranty_root = root / "warranty"
            if warranty_root.is_dir():
                for ticket_dir in warranty_root.iterdir():
                    if not ticket_dir.is_dir():
                        continue
                    try:
                        if not any(ticket_dir.iterdir()):
                            ticket_dir.rmdir()
                    except OSError:
                        pass

    result = {
        "ok": True,
        "apply": apply,
        "retention_days": window,
        "cutoff": cutoff.isoformat(),
        "candidate_rows": deleted_rows,
        "deleted_files": deleted_files if apply else 0,
        "would_delete_files": deleted_files,
        "missing_files": missing_files,
        "bytes_freed": bytes_freed if apply else 0,
        "would_free_bytes": bytes_freed,
        "errors": errors[:20],
    }
    logger.info("Evidence purge %s", result)
    return result
