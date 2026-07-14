from __future__ import annotations

import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(script: str, *args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "script" / script), *(str(arg) for arg in args)],
        check=check,
        capture_output=True,
        text=True,
    )


def test_backup_verify_restore_round_trip(tmp_path: Path):
    runtime = tmp_path / "runtime"
    database_path = runtime / "db_data" / "chat_history.db"
    evidence_path = runtime / "uploaded_evidence" / "warranty" / "case-1" / "photo.jpg"
    database_path.parent.mkdir(parents=True)
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(b"original-evidence")

    with sqlite3.connect(database_path) as database:
        database.execute("CREATE TABLE events (value TEXT NOT NULL)")
        database.execute("INSERT INTO events VALUES ('original')")

    backup = tmp_path / "backup"
    _run("backup_runtime.py", "--root", runtime, "--output", backup)
    _run("verify_backup.py", backup)

    with tarfile.open(backup / "uploaded_evidence.tar.gz", "r:gz") as archive:
        assert "uploaded_evidence/warranty/case-1/photo.jpg" in archive.getnames()

    with sqlite3.connect(database_path) as database:
        database.execute("UPDATE events SET value = 'changed'")
    evidence_path.write_bytes(b"changed-evidence")

    _run(
        "restore_backup.py",
        "--root",
        runtime,
        "--backup",
        backup,
        "--confirm",
        "RESTORE",
    )

    with sqlite3.connect(database_path) as database:
        assert database.execute("SELECT value FROM events").fetchone() == ("original",)
    assert evidence_path.read_bytes() == b"original-evidence"
    assert list((runtime / "db_data").glob("chat_history.db.pre_restore_*"))
    assert list(runtime.glob("uploaded_evidence.pre_restore_*"))


def test_verify_detects_tampered_backup(tmp_path: Path):
    runtime = tmp_path / "runtime"
    database_path = runtime / "db_data" / "chat_history.db"
    database_path.parent.mkdir(parents=True)
    with sqlite3.connect(database_path) as database:
        database.execute("CREATE TABLE events (value TEXT NOT NULL)")

    backup = tmp_path / "backup"
    _run("backup_runtime.py", "--root", runtime, "--output", backup)
    with (backup / "chat_history.db").open("ab") as handle:
        handle.write(b"tampered")

    result = _run("verify_backup.py", backup, check=False)
    assert result.returncode != 0
    assert "checksum verification failed" in result.stderr
