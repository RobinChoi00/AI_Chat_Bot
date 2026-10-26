"""A legal hold also protects historical backup restore points."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("hold, expected_count", [(None, 10), ("1", 10), ("invalid", 10), ("0", 8)])
def test_backup_rotation_preserves_hold_and_rotates_only_when_released(tmp_path, hold, expected_count):
    scripts = tmp_path / "script"
    scripts.mkdir()
    shutil.copyfile(ROOT / "script/backup_db.sh", scripts / "backup_db.sh")
    backups = tmp_path / "backups"
    backups.mkdir()
    for day in range(1, 11):
        (backups / f"202601{day:02d}T000000Z").mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python3"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"], BACKUP_S3_URI="")
    env.pop("PRIVACY_RETENTION_LEGAL_HOLD", None)
    if hold is not None:
        (tmp_path / ".env").write_text(f'PRIVACY_RETENTION_LEGAL_HOLD="{hold}"\n')
    result = subprocess.run(["bash", str(scripts / "backup_db.sh")], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert len(list(backups.iterdir())) == expected_count
