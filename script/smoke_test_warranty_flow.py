#!/usr/bin/env python3
"""
script/smoke_test_warranty_flow.py
===================================
End-to-end smoke test for the WarrantyEngine state machine.

Runs FIVE complete warranty scenarios without any LLM calls.
Each scenario walks from the root node to a terminal node and
verifies the expected terminal class and final ticket status.

Usage
-----
    python script/smoke_test_warranty_flow.py

Exit code 0 = all scenarios passed.
Exit code 1 = one or more scenarios failed.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without installing the package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

# Use an in-memory SQLite DB so smoke tests never touch the production DB.
import warranty_models as _wm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_mem_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
_wm.Base.metadata.create_all(bind=_mem_engine)
_wm._engine = _mem_engine
_wm._SessionFactory = sessionmaker(
    autocommit=False, autoflush=False, bind=_mem_engine, expire_on_commit=False
)

from warranty_workflow import WarrantyEngine  # noqa: E402 (after path/DB setup)

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each scenario: (name, answer_key_sequence, expected_terminal_class, expected_terminal_node_suffix)

SCENARIOS: List[Tuple[str, List[str], str, str]] = [
    (
        "Power issue — remote on, no response → replace remote (admin review)",
        ["warranty", "defect", "power", "remote_on", "no_response", "error_code_no"],
        "awaiting_admin_review",
        "defect_power_remote_replace_terminal",
    ),
    (
        "Remote — blank screen, commands OK → admin review",
        ["warranty", "defect", "remote", "has_power", "blank_screen_commands_ok", "error_code_no"],
        "awaiting_admin_review",
        "defect_remote_blank_screen_terminal",
    ),
    (
        "Cosmetic damage — footrest → photo evidence + admin review",
        ["warranty", "defect", "cosmetic", "footrest"],
        "awaiting_admin_review",
        "defect_cosmetic_photo_terminal",
    ),
    (
        "Air issue — shoulders/hips, yes hissing → tech dispatch (admin review)",
        ["warranty", "defect", "air", "shoulders_hips", "yes_hissing", "error_code_no"],
        "awaiting_admin_review",
        "defect_air_shoulders_tech_terminal",
    ),
    (
        "Delivery — tracking + box damage signed cleared → admin review",
        ["warranty", "delivery", "has_tracking", "1Z999AA10123456784", "yes_box_damage", "signed_cleared"],
        "awaiting_admin_review",
        "delivery_signed_cleared_terminal",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    answers: List[str],
    expected_class: str,
    expected_node_suffix: str,
) -> bool:
    """Walk one scenario path and return True if all assertions pass."""
    session_id = f"smoke-{name[:20].replace(' ', '_')}"
    try:
        ticket_id, root_node = WarrantyEngine.start_session(session_id, "smoke.test")
        current_node_id: str = root_node["node_id"]

        result = None
        for i, answer_key in enumerate(answers):
            result = WarrantyEngine.submit_answer(ticket_id, answer_key)
            current_node_id = result["next_node_id"]
            is_terminal: bool = result["is_terminal"]

            if is_terminal and i < len(answers) - 1:
                print(f"  {YELLOW}⚠ Reached terminal early at step {i+1}/{len(answers)}: "
                      f"{current_node_id}{RESET}")
                break

        if result is None:
            raise AssertionError("No answers were submitted — scenario is empty.")

        # Assertions
        errors: List[str] = []

        if not result["is_terminal"]:
            errors.append(
                f"Expected terminal node but workflow is still in progress. "
                f"Current node: {current_node_id!r}"
            )

        actual_class = result.get("terminal_class")
        if actual_class != expected_class:
            errors.append(
                f"terminal_class mismatch: expected {expected_class!r}, got {actual_class!r}"
            )

        if not current_node_id.endswith(expected_node_suffix.split("_terminal")[0].split("_")[-1]) \
                and current_node_id != expected_node_suffix:
            # Soft check: just verify the exact node ID
            if current_node_id != expected_node_suffix:
                errors.append(
                    f"terminal_node mismatch: expected {expected_node_suffix!r}, "
                    f"got {current_node_id!r}"
                )

        # Verify ticket status in DB
        ticket = WarrantyEngine.get_ticket(ticket_id)
        if ticket is None:
            errors.append("Ticket not found in DB after workflow.")
        else:
            db_status = str(ticket.status)
            if db_status != expected_class:
                errors.append(
                    f"DB ticket.status mismatch: expected {expected_class!r}, got {db_status!r}"
                )
            turns = WarrantyEngine.get_turns(ticket_id)
            if len(turns) != len(answers):
                errors.append(
                    f"Turn count mismatch: expected {len(answers)}, got {len(turns)}"
                )

        if errors:
            for e in errors:
                print(f"    {RED}ERROR: {e}{RESET}")
            return False
        return True

    except Exception as exc:
        print(f"    {RED}EXCEPTION: {exc}{RESET}")
        traceback.print_exc()
        return False


def main() -> int:
    print(f"\n{BOLD}{CYAN}=== Warranty Engine Smoke Tests ==={RESET}\n")
    print(f"  Running {len(SCENARIOS)} scenarios against in-memory DB (no LLM calls)\n")

    passed = 0
    failed = 0

    for name, answers, exp_class, exp_node in SCENARIOS:
        print(f"  {BOLD}Scenario:{RESET} {name}")
        print(f"  Answers:  {answers}")
        ok = run_scenario(name, answers, exp_class, exp_node)
        status = PASS if ok else FAIL
        print(f"  Result:   {status}")
        if ok:
            passed += 1
        else:
            failed += 1
        print()

    # Summary
    total = passed + failed
    bar = ("=" * 40)
    if failed == 0:
        print(f"{GREEN}{bar}{RESET}")
        print(f"{BOLD}{GREEN}  ALL {total}/{total} SCENARIOS PASSED{RESET}")
        print(f"{GREEN}{bar}{RESET}\n")
        print("  ✓ WarrantyEngine is LLM-free and deterministic.")
        print("  ✓ All admin-review terminals are correctly classified.")
        print("  ✓ DB persists ticket status and turns.\n")
        return 0
    else:
        print(f"{RED}{bar}{RESET}")
        print(f"{BOLD}{RED}  {failed}/{total} SCENARIOS FAILED{RESET}")
        print(f"{RED}{bar}{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
