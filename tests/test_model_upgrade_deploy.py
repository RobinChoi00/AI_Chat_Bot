"""Exercise deploy sequencing and rollback with local Git and inert commands."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


@pytest.mark.parametrize("probe_fails", [True, False])
def test_candidate_probe_precedes_service_replacement_and_rolls_back(tmp_path, probe_fails):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "synthetic@example.invalid")
    git(repo, "config", "user.name", "Deployment fixture")
    (repo / "release.txt").write_text("old")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "old")
    previous = git(repo, "rev-parse", "HEAD")
    (repo / "release.txt").write_text("candidate")
    git(repo, "commit", "-am", "candidate")
    candidate = git(repo, "rev-parse", "HEAD")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(origin)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "checkout", "--detach", previous)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text("#!/usr/bin/env bash\nprintf '%s %s\\n' \"$(git rev-parse HEAD)\" \"$*\" >> \"$DEPLOY_TEST_LOG\"\nif [[ \"$1\" == compose && \"${2:-}\" == run && \"$DEPLOY_TEST_PROBE_FAILS\" == 1 ]]; then exit 17; fi\n")
    for executable in (docker, binaries / "python3", binaries / "curl"):
        if not executable.exists():
            executable.write_text("#!/usr/bin/env bash\nexit 0\n")
        executable.chmod(0o755)
    log = tmp_path / "docker.log"
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"], DEPLOY_REPOSITORY_ROOT=str(repo), DEPLOY_TEST_LOG=str(log), DEPLOY_TEST_PROBE_FAILS="1" if probe_fails else "0")
    result = subprocess.run(["bash", str(ROOT / "script/deploy.sh"), candidate], cwd=repo, env=env, capture_output=True, text=True, timeout=15)
    assert log.exists(), result.stdout + result.stderr
    calls = log.read_text().splitlines()
    probe_index = next(i for i, line in enumerate(calls) if "compose run " in line)
    assert "verify_model_upgrade.py --live" in calls[probe_index]
    up = [(i, line) for i, line in enumerate(calls) if "compose up " in line]
    assert len(up) == 1 and probe_index < up[0][0]
    if probe_fails:
        assert result.returncode == 17
        assert up[0][1].startswith(previous)
        assert git(repo, "rev-parse", "HEAD") == previous
    else:
        assert result.returncode == 0, result.stderr
        assert up[0][1].startswith(candidate)
        assert git(repo, "rev-parse", "HEAD") == candidate
