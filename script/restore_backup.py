#!/usr/bin/env python3
"""Verify and restore a runtime backup while preserving the replaced data."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _safe_extract_evidence(archive_path: Path, destination: Path) -> Path:
    """Extract only ordinary files/directories below uploaded_evidence/."""
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != "uploaded_evidence"
                or member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
            ):
                raise RuntimeError(f"unsafe evidence archive member: {member.name}")
        archive.extractall(destination, members=members)

    extracted = destination / "uploaded_evidence"
    extracted.mkdir(parents=True, exist_ok=True)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore SQLite and uploaded evidence. Stop the services first."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument(
        "--confirm",
        required=True,
        help="Must be exactly RESTORE to acknowledge replacement of live data.",
    )
    args = parser.parse_args()

    if args.confirm != "RESTORE":
        parser.error("--confirm must be exactly RESTORE")

    root = args.root.resolve()
    backup = args.backup.resolve()
    verifier = Path(__file__).resolve().parent / "verify_backup.py"
    subprocess.run([sys.executable, str(verifier), str(backup)], check=True)

    database_source = backup / "chat_history.db"
    evidence_archive = backup / "uploaded_evidence.tar.gz"
    if not database_source.is_file() or not evidence_archive.is_file():
        raise FileNotFoundError("backup is missing the database or evidence archive")

    restore_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    extract_root = root / f".restore-evidence-{restore_id}"
    extracted_evidence = _safe_extract_evidence(evidence_archive, extract_root)

    db_dir = root / "db_data"
    db_dir.mkdir(parents=True, exist_ok=True)
    live_database = db_dir / "chat_history.db"
    previous_database = db_dir / f"chat_history.db.pre_restore_{restore_id}"
    staged_database = db_dir / f".chat_history.db.restore_{restore_id}"
    shutil.copy2(database_source, staged_database)
    os.chmod(staged_database, 0o600)

    live_evidence = root / "uploaded_evidence"
    previous_evidence = root / f"uploaded_evidence.pre_restore_{restore_id}"

    database_moved = False
    database_installed = False
    evidence_moved = False
    try:
        if live_database.exists():
            os.replace(live_database, previous_database)
            database_moved = True
        os.replace(staged_database, live_database)
        database_installed = True

        if live_evidence.exists():
            os.replace(live_evidence, previous_evidence)
            evidence_moved = True
        os.replace(extracted_evidence, live_evidence)
    except Exception:
        staged_database.unlink(missing_ok=True)
        if live_database.exists() and database_installed:
            live_database.unlink()
        if database_moved:
            os.replace(previous_database, live_database)
        if evidence_moved and not live_evidence.exists():
            os.replace(previous_evidence, live_evidence)
        raise
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)

    print(f"Restore completed from {backup}")
    if database_moved:
        print(f"Previous database preserved at {previous_database}")
    if evidence_moved:
        print(f"Previous evidence preserved at {previous_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
