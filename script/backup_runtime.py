#!/usr/bin/env python3
"""Create a consistent SQLite backup, evidence archive, and checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_database(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"database does not exist: {source}")
    with sqlite3.connect(source) as source_db, sqlite3.connect(destination) as backup_db:
        source_db.backup(backup_db)
        result = backup_db.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError(f"backup integrity check failed: {result}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"backup destination is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    database_backup = output / "chat_history.db"
    _backup_database(root / "db_data" / "chat_history.db", database_backup)

    evidence_archive = output / "uploaded_evidence.tar.gz"
    evidence_root = root / "uploaded_evidence"
    with tarfile.open(evidence_archive, "w:gz") as archive:
        if evidence_root.is_dir():
            archive.add(evidence_root, arcname="uploaded_evidence", recursive=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
        "files": {
            database_backup.name: {
                "bytes": database_backup.stat().st_size,
                "sha256": _sha256(database_backup),
            },
            evidence_archive.name: {
                "bytes": evidence_archive.stat().st_size,
                "sha256": _sha256(evidence_archive),
            },
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Created verified backup at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
