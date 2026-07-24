#!/usr/bin/env python3
"""
Simulate after-hours RC IVR end-to-end without a live phone call.

Runs the same orchestration path as webhooks:
  call-enter → issue-type menu → digit 3 (defect) → first defect prompt

Usage (from project root / EC2):
  python3 script/run_rc_ivr_e2e_sim.py
  docker compose exec -T backend python script/run_rc_ivr_e2e_sim.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    from ringcentral_ivr import handle_call_enter, handle_command_update
    from ringcentral_voice import get_call_context
    from warranty_workflow import WarrantyEngine

    session_id = f"rc-e2e-sim-{uuid.uuid4().hex[:10]}"
    party_id = f"party-{uuid.uuid4().hex[:8]}"
    payload = {
        "sessionId": session_id,
        "inParty": {
            "id": party_id,
            "from": {"phoneNumber": "+15550001111"},
        },
    }

    with (
        patch("ringcentral_ivr.is_warranty_business_hours", return_value=False),
        patch("ringcentral_ivr.play_prompt"),
        patch("ringcentral_ivr.collect_digits"),
        patch(
            "ringcentral_ivr.resolve_play_uri",
            return_value="https://example.com/sim.wav",
        ),
    ):
        handle_call_enter(payload)
        handle_command_update(
            {
                "sessionId": session_id,
                "status": "Completed",
                "command": "Play",
                "partyId": party_id,
            }
        )
        handle_command_update(
            {
                "sessionId": session_id,
                "status": "Completed",
                "command": "Collect",
                "partyId": party_id,
                "parameters": {"digits": "3"},
            }
        )

    ctx = get_call_context(session_id)
    if ctx is None:
        print("FAIL: no call context after sim")
        return 1

    node = WarrantyEngine.get_current_node(ctx.ticket_id)
    ticket = WarrantyEngine.get_ticket(ctx.ticket_id)
    print("OK RC IVR e2e sim")
    print(f"  session_id={session_id}")
    print(f"  ticket_id={ctx.ticket_id}")
    print(f"  phase={ctx.phase.value}")
    print(f"  issue_type={getattr(ticket, 'issue_type', None)}")
    print(f"  node_id={(node or {}).get('node_id')}")
    print(f"  channel={(ticket.get_collected() if ticket else {}).get('channel')}")

    if ticket is None or str(ticket.issue_type or "") != "defect":
        print("FAIL: expected defect issue_type")
        return 1
    if not node or node.get("node_id") != "defect_problem_type":
        print("FAIL: expected defect_problem_type node after digit 3")
        return 1
    print(
        "NOTE: Live phone E2E still needs RC ApplicationExtension + Roman routing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
