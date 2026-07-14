#!/usr/bin/env python3
"""Verify backup checksums and SQLite integrity without modifying live data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    backup = args.backup.resolve()
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))

    for filename, expected in manifest["files"].items():
        path = backup / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"checksum verification failed: {filename}")

    with sqlite3.connect(backup / "chat_history.db") as database:
        result = database.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {result}")

    print(f"Backup verified successfully: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
