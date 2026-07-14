from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from admin_auth import require_admin_key  # noqa: E402


def test_admin_key_accepts_exact_match():
    require_admin_key("secret-value", "secret-value")


@pytest.mark.parametrize("received", [None, "", "wrong", " secret-value-extra"])
def test_admin_key_rejects_missing_or_wrong_value(received):
    with pytest.raises(HTTPException) as exc:
        require_admin_key(received, "secret-value")
    assert exc.value.status_code == 401


def test_admin_key_fails_closed_when_unconfigured():
    with pytest.raises(HTTPException) as exc:
        require_admin_key("anything", "")
    assert exc.value.status_code == 503
