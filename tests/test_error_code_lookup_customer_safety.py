"""Customer-facing error-code formatting must not expose internal repair data."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from error_code_lookup import format_repair_help


def test_format_repair_help_hides_parts_and_manual_references():
    message = format_repair_help(
        {
            "model": "OS-4000XT",
            "error_code": "01",
            "meaning": "No function when starting.",
            "troubleshooting": (
                "Replace the main PCB. Refer to: Page 19. "
                "Check that the remote cable is firmly connected."
            ),
            "parts_required": "Main PCB and EMC board",
        }
    )

    assert "Error code 01" in message
    assert "remote cable" in message
    assert "Refer to" not in message
    assert "PCB" not in message
    assert "internal reference" not in message
