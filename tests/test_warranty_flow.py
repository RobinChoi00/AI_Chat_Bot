"""
tests/test_warranty_flow.py
===========================
Unit tests for the WarrantyEngine state machine.

Each test walks one representative path through the flowchart and verifies:
  - The correct terminal node is reached.
  - The ticket status is set appropriately.
  - Collected data is persisted correctly.
  - Submitted answers are stored as turns.

Scenarios covered (original)
-----------------------------
  1  Installation  → model name input → send_info terminal
  2  Delivery, no tracking → order name input → awaiting_admin_review
  3  Delivery, tracking, box damaged, signed cleared → awaiting_admin_review
  4  Defect air shoulders, hissing → awaiting_admin_review (tech)
  5  Defect remote, no power, fuse broken → awaiting_admin_review (send fuse)
  6  Defect power, remote on, recline not working, moves on power-off → awaiting_admin_review

Additional scenarios
--------------------
  7  Power issue: remote on → no_response → awaiting_admin_review (replace remote)
  8  Remote issue: has_power → blank screen → awaiting_admin_review (replace remote)
  9  Yes/No branching: air feet — yes_worked → inflate_test path
 10  Yes/No branching: air feet — never_worked → pump terminal (different branch)
 11  Evidence requirement lookup via get_evidence_spec()
 12  All admin-gated terminals return terminal_class="awaiting_admin_review"
 13  No LLM call is made inside the workflow engine (module-level assertion)

Edge cases
----------
  E1  Invalid answer raises ValueError with valid keys listed
  E2  Submitting to a terminal node raises ValueError
  E3  Integer index matching
  E4  Case-insensitive label matching
  E5  Admin decision resolves the ticket

Test isolation
--------------
All tests use an in-memory SQLite DB via the `in_memory_db` autouse fixture,
so they never write to the real chat_history.db.
"""

import sys
from pathlib import Path
from typing import cast

# Make sure app/ is on the path so warranty_workflow and warranty_models import cleanly.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import warranty_models as wm
from warranty_workflow import WarrantyEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    """
    Replace the real SQLite DB with a fresh in-memory DB for every test.
    Patches both warranty_models and warranty_workflow module-level references.
    """
    import warranty_workflow as wf

    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    mem_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=mem_engine,
        expire_on_commit=False,
    )
    wm.Base.metadata.create_all(bind=mem_engine)

    monkeypatch.setattr(wm, "_engine", mem_engine)
    monkeypatch.setattr(wm, "_SessionFactory", mem_session_factory)
    monkeypatch.setattr(wf, "_SessionFactory", mem_session_factory)

    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def walk(ticket_id: str, answers: list[str]) -> dict:
    """Submit a sequence of answers and return the last SubmitResult."""
    result: dict = {}
    for ans in answers:
        result = submit(ticket_id, ans)
    return result


def submit(ticket_id: str, answer: str) -> dict:
    """Submit one answer; auto-dismiss error-code gate when tests expect the terminal."""
    import warranty_error_code_gate as gate

    result = WarrantyEngine.submit_answer(ticket_id, answer)
    while result.get("next_node_id") in (
        gate.GATE_VISIBLE_ID,
        gate.GATE_PICK_ID,
        gate.GATE_ENTER_ID,
    ):
        result = WarrantyEngine.submit_answer(ticket_id, "error_code_no")
    return result


def start(session_id: str = "test-session", domain: str = "osakiusa.com"):
    """Convenience wrapper for start_session."""
    return WarrantyEngine.start_session(session_id, domain)


def ticket(ticket_id: str):
    """Fetch ticket and assert it exists. Returns a non-None WarrantyTicket."""
    t = WarrantyEngine.get_ticket(ticket_id)
    assert t is not None, f"No ticket found for {ticket_id!r}"
    return t


# ---------------------------------------------------------------------------
# Scenario 1 — Installation: model name → send_info terminal
# ---------------------------------------------------------------------------

def test_installation_model_to_send_info():
    ticket_id, root_node = start()

    assert root_node["node_id"] == "root"
    assert root_node["type"] == "question"

    result = walk(ticket_id, [
        "warranty",        # root → issue_type
        "installation",    # issue_type → install_model
        "OS-4000T",        # install_model → install_concern
        "general_setup",   # install_concern → install_send_video
    ])

    assert result["next_node_id"] == "install_send_video"
    assert result["is_terminal"] is True

    t = ticket(ticket_id)
    assert str(t.model_name) == "OS-4000T"
    assert str(t.issue_type) == "installation"
    assert str(t.status) == "send_info"  # DIY: send video — no admin needed

    turns = WarrantyEngine.get_turns(ticket_id)
    assert len(turns) == 4
    assert str(turns[2].customer_answer) == "OS-4000T"
    assert str(turns[2].answer_key) == "model_name"


def test_advance_to_issue_type_defect():
    """Phone IVR skips root/issue_type and lands on defect_problem_type."""
    ticket_id, root = start("phone-defect", "phone")
    assert root["node_id"] == "root"

    result = WarrantyEngine.advance_to_issue_type(ticket_id, "defect")
    assert result["next_node_id"] == "defect_problem_type"
    assert result["next_node"]["type"] == "question"

    t = ticket(ticket_id)
    assert str(t.issue_type) == "defect"
    turns = WarrantyEngine.get_turns(ticket_id)
    assert len(turns) == 2
    assert str(turns[0].answer_key) == "warranty"
    assert str(turns[1].answer_key) == "defect"


def test_installation_footrest_air_to_diy_terminal():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "installation",
        "Osaki 4D Maestro LE 2.0",
        "footrest_or_no_air",
    ])

    assert result["next_node_id"] == "install_air_hose_terminal"
    assert result["is_terminal"] is True
    assert str(ticket(ticket_id).status) == "send_info"


def test_defect_voice_not_working_to_diy_terminal():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "defect",
        "voice",
        "voice_no_response",
    ])

    assert result["next_node_id"] == "defect_voice_not_working_terminal"
    assert result["is_terminal"] is True
    assert str(ticket(ticket_id).status) == "send_info"
    assert str(ticket(ticket_id).defect_type) == "voice"


def test_defect_voice_false_triggers_to_diy_terminal():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "defect",
        "voice",
        "false_triggers",
    ])

    assert result["next_node_id"] == "defect_voice_false_triggers_terminal"
    assert result["is_terminal"] is True


def test_defect_heating_not_heating_to_diy_terminal():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "defect",
        "heat",
        "not_heating",
        "still_no_heat",
    ])

    assert result["next_node_id"] == "defect_heating_not_heating_terminal"
    assert result["is_terminal"] is True
    assert str(ticket(ticket_id).defect_type) == "heat"


def test_defect_heating_intermittent_to_diy_terminal():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "defect",
        "heat",
        "intermittent",
    ])

    assert result["next_node_id"] == "defect_heating_intermittent_terminal"
    assert result["is_terminal"] is True


def test_defect_heating_too_hot_to_diy_terminal():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "defect",
        "heat",
        "too_hot",
    ])

    assert result["next_node_id"] == "defect_heating_too_hot_terminal"
    assert result["is_terminal"] is True


# ---------------------------------------------------------------------------
# Scenario 2 — Delivery, no tracking → lookup by name → awaiting_admin
# ---------------------------------------------------------------------------

def test_delivery_no_tracking_continues_to_problem_type():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",        # root → issue_type
        "delivery",        # issue_type → delivery_intent_q
        "damage_issue",    # delivery_intent_q → delivery_tracking_q
        "no_tracking",     # delivery_tracking_q → delivery_get_name
        "customer@example.com",  # delivery_get_name → delivery_problem_type_q
    ])

    assert result["next_node_id"] == "delivery_problem_type_q"
    assert result["is_terminal"] is False

    t = ticket(ticket_id)
    assert str(t.status) == "in_progress"
    assert t.get_collected().get("order_or_email") == "customer@example.com"

    turns = WarrantyEngine.get_turns(ticket_id)
    assert len(turns) == 5


def test_delivery_status_check_ends_at_status_terminal():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "delivery",
        "status_check",
        "no_tracking",
        "customer@example.com",
    ])

    assert result["next_node_id"] == "delivery_status_terminal"
    assert result["is_terminal"] is True
    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert t.get_collected().get("order_or_email") == "customer@example.com"


# ---------------------------------------------------------------------------
# Scenario 3 — Delivery, has tracking, box damaged, signed cleared → awaiting_admin
# ---------------------------------------------------------------------------

def test_delivery_tracking_box_damage_signed_cleared():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",           # root → issue_type
        "delivery",           # issue_type → delivery_intent_q
        "damage_issue",       # → delivery_tracking_q
        "has_tracking",       # delivery_tracking_q → delivery_get_tracking_number
        "1Z999AA10123456784", # tracking number → delivery_problem_type_q
        "damaged_in_transit", # → delivery_visible_damage_q
        "yes_box_damage",     # delivery_visible_damage_q → delivery_signed_q
        "signed_cleared",     # delivery_signed_q → delivery_signed_cleared_terminal
    ])

    assert result["next_node_id"] == "delivery_signed_cleared_terminal"
    assert result["is_terminal"] is True
    assert "damage_photos" in result["evidence_required"]
    assert result["evidence_email"] == "service@osakititan.com"

    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert t.get_collected().get("tracking_number") == "1Z999AA10123456784"


def test_delivery_missing_parts_path():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "delivery",
        "damage_issue",
        "no_tracking",
        "customer@example.com",
        "missing_parts",
        "remote control and side bolts",
    ])

    assert result["next_node_id"] == "delivery_missing_parts_terminal"
    assert result["is_terminal"] is True
    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert t.get_collected().get("missing_parts_description") == "remote control and side bolts"


def test_delivery_never_arrived_path():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "delivery",
        "damage_issue",
        "has_tracking",
        "1Z999AA10123456784",
        "never_arrived",
    ])

    assert result["next_node_id"] == "delivery_never_arrived_terminal"
    assert result["is_terminal"] is True


def test_delivery_box_fine_but_chair_damaged_continues_to_signed():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",
        "delivery",
        "damage_issue",
        "has_tracking",
        "1Z999AA10123456784",
        "damaged_in_transit",
        "no_box_damage",
        "yes_chair_inside_damage",
        "signed_damaged",
        "visible_at_unboxing",
    ])

    assert result["next_node_id"] == "delivery_replace_claim_terminal"
    assert result["is_terminal"] is True


# ---------------------------------------------------------------------------
# Scenario 4 — Defect air, shoulders/hips, hissing → awaiting_admin (tech)
# ---------------------------------------------------------------------------

def test_defect_air_shoulders_hissing_to_tech():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",         # root → issue_type
        "defect",           # issue_type → defect_problem_type
        "air",              # defect_problem_type → defect_air_location
        "shoulders_hips",   # defect_air_location → defect_air_shoulders_hissing_q
        "yes_hissing",      # instruction → defect_air_shoulders_tech_terminal
    ])

    assert result["next_node_id"] == "defect_air_shoulders_tech_terminal"
    assert result["is_terminal"] is True

    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert str(t.defect_type) == "air"

    # Evidence required for this terminal
    assert "video_of_issue" in result["evidence_required"]


# ---------------------------------------------------------------------------
# Scenario 5 — Defect remote, no power, fuse broken → awaiting_admin (fuse)
# ---------------------------------------------------------------------------

def test_defect_remote_no_power_fuse_broken():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",       # root → issue_type
        "defect",         # issue_type → defect_problem_type
        "remote",         # defect_problem_type → defect_remote_power_q
        "no_power",       # defect_remote_power_q → defect_remote_no_power_checks
        "fuse_broken",    # instruction → defect_remote_fuse_terminal
    ])

    assert result["next_node_id"] == "defect_remote_fuse_terminal"
    assert result["is_terminal"] is True

    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert str(t.defect_type) == "remote"
    assert "photo_of_fuse" in result["evidence_required"]


# ---------------------------------------------------------------------------
# Scenario 6 — Defect power, remote on, recline not working, moves on off
#              → awaiting_admin (actuator replacement)
# ---------------------------------------------------------------------------

def test_defect_power_recline_actuator():
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",            # root → issue_type
        "defect",              # issue_type → defect_problem_type
        "power",               # defect_problem_type → defect_power_remote_on_q
        "remote_on",           # defect_power_remote_on_q → defect_power_on_controls_q
        "recline_not_working", # defect_power_on_controls_q → defect_power_recline_move_q
        "moves_on_off",        # instruction → defect_power_actuator_terminal
    ])

    assert result["next_node_id"] == "defect_power_actuator_terminal"
    assert result["is_terminal"] is True

    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert str(t.defect_type) == "power"
    assert "video_of_issue" in result["evidence_required"]

    turns = WarrantyEngine.get_turns(ticket_id)
    assert len(turns) == 7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_invalid_answer_raises():
    ticket_id, _ = start()
    # root has options: warranty / sales
    with pytest.raises(ValueError, match="did not match"):
        WarrantyEngine.submit_answer(ticket_id, "totally_invalid_answer_xyz")


def test_double_answer_on_terminal_raises():
    ticket_id, _ = start()
    # Reach install_send_video terminal
    walk(ticket_id, ["warranty", "installation", "Titan Pro Commander", "general_setup"])

    # Ticket is now in send_info (terminal reached). Further answers should raise.
    with pytest.raises(ValueError, match="no longer in progress"):
        WarrantyEngine.submit_answer(ticket_id, "anything")


def test_integer_index_matching():
    """Answer can be submitted as a 1-based integer string."""
    ticket_id, _ = start()
    # root option 1 = "Warranty"
    result = WarrantyEngine.submit_answer(ticket_id, "1")
    assert result["next_node_id"] == "issue_type"


def test_label_matching_case_insensitive():
    """Answer can be the option label (case-insensitive)."""
    ticket_id, _ = start()
    result = WarrantyEngine.submit_answer(ticket_id, "SALES")
    assert result["next_node_id"] == "sales_routing"
    assert result["is_terminal"] is True


def test_admin_decision():
    """Admin can record a decision that resolves the ticket."""
    ticket_id, _ = start()
    walk(ticket_id, [
        "warranty", "defect", "rolling",
        "noise_up_down",  # → defect_rolling_noise_updown_terminal
    ])

    t_before = ticket(ticket_id)
    assert str(t_before.status) == "awaiting_admin_review"

    # "repair" is a legacy alias → normalized to "approved"
    resolved = WarrantyEngine.admin_decision(
        ticket_id=ticket_id,
        decision="approved",
        note="Track was damaged. Scheduling tech visit.",
        decided_by="admin_user",
        customer_message="We have scheduled a technician to inspect and repair the mechanism.",
    )

    assert str(resolved.status) == "resolved"
    assert str(resolved.admin_decision) == "approved"
    assert str(resolved.decided_by) == "admin_user"
    assert "technician" in str(resolved.customer_message)


# ---------------------------------------------------------------------------
# Scenario 7 — Power issue: remote on → no_response → awaiting_admin_review
# ---------------------------------------------------------------------------

def test_power_issue_remote_on_no_response():
    """
    Power → remote turns on → remote does not respond to any command
    → defect_power_remote_replace_terminal (replace remote) → awaiting_admin_review
    """
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",       # root → issue_type
        "defect",         # issue_type → defect_problem_type
        "power",          # defect_problem_type → defect_power_remote_on_q
        "remote_on",      # → defect_power_on_controls_q
        "no_response",    # → defect_power_remote_replace_terminal
    ])

    assert result["next_node_id"] == "defect_power_remote_replace_terminal"
    assert result["is_terminal"] is True
    assert result["terminal_class"] == "awaiting_admin_review"

    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert str(t.defect_type) == "power"

    assert "photo_of_remote" in result["evidence_required"]
    assert "video_of_issue" in result["evidence_required"]


# ---------------------------------------------------------------------------
# Scenario 8 — Remote issue: has power, blank screen → awaiting_admin_review
# ---------------------------------------------------------------------------

def test_remote_issue_has_power_blank_screen():
    """
    Remote → has power / screen shows something → all commands respond but screen blank
    → defect_remote_blank_screen_terminal (send new remote) → awaiting_admin_review
    """
    ticket_id, _ = start()

    result = walk(ticket_id, [
        "warranty",                   # root → issue_type
        "defect",                     # issue_type → defect_problem_type
        "remote",                     # defect_problem_type → defect_remote_power_q
        "has_power",                  # → defect_remote_screen_q
        "blank_screen_commands_ok",   # → defect_remote_blank_screen_terminal
    ])

    assert result["next_node_id"] == "defect_remote_blank_screen_terminal"
    assert result["is_terminal"] is True
    assert result["terminal_class"] == "awaiting_admin_review"

    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"
    assert str(t.defect_type) == "remote"
    assert "photo_of_remote" in result["evidence_required"]


# ---------------------------------------------------------------------------
# Scenario 9 + 10 — Yes/No branching: air feet (two diverging paths)
# ---------------------------------------------------------------------------

def test_yes_no_branch_air_feet_yes_worked_inflate_test():
    """
    Air feet → YES it worked before → inflate_test instruction node (not terminal yet).
    Verifies the 'yes' branch navigates to defect_air_feet_inflate_test.
    """
    ticket_id, _ = start()

    walk(ticket_id, [
        "warranty",    # root → issue_type
        "defect",      # → defect_problem_type
        "air",         # → defect_air_location
        "feet_calves", # → defect_air_feet_worked_q
    ])

    # Choose YES branch
    result = WarrantyEngine.submit_answer(ticket_id, "yes_worked")
    assert result["next_node_id"] == "defect_air_feet_inflate_test"
    assert result["is_terminal"] is False
    assert result["terminal_class"] is None  # not a terminal yet


def test_yes_no_branch_air_feet_never_worked_pump_terminal():
    """
    Air feet → NO it never worked → pump terminal directly.
    Verifies the 'no' branch shortcircuits to awaiting_admin_review (pump replacement).
    """
    ticket_id, _ = start()

    walk(ticket_id, [
        "warranty",    # root → issue_type
        "defect",      # → defect_problem_type
        "air",         # → defect_air_location
        "feet_calves", # → defect_air_feet_worked_q
    ])

    # Choose NO branch (gate auto-dismissed in submit helper)
    result = submit(ticket_id, "never_worked")
    assert result["next_node_id"] == "defect_air_pump_terminal"
    assert result["is_terminal"] is True
    assert result["terminal_class"] == "awaiting_admin_review"

    t = ticket(ticket_id)
    assert str(t.status) == "awaiting_admin_review"


# ---------------------------------------------------------------------------
# Scenario 11 — Evidence requirement lookup via get_evidence_spec()
# ---------------------------------------------------------------------------

def test_evidence_requirement_lookup():
    """
    get_evidence_spec() returns the correct required/optional evidence
    from warranty_evidence_specs.json for a given terminal node.
    """
    # Terminal that requires a video
    spec = WarrantyEngine.get_evidence_spec("defect_air_tech_terminal")
    assert "video_of_issue" in spec["required"]
    assert "types" in spec
    assert "video_of_issue" in spec["types"]

    # Terminal that requires damage photos (delivery)
    spec2 = WarrantyEngine.get_evidence_spec("delivery_signed_cleared_terminal")
    assert "damage_photos" in spec2["required"]

    # Terminal with no evidence (DIY fix)
    spec3 = WarrantyEngine.get_evidence_spec("defect_air_hose_fix_terminal")
    assert spec3["required"] == []
    assert spec3["optional"] == []

    # Unknown terminal → empty spec (graceful fallback)
    spec4 = WarrantyEngine.get_evidence_spec("nonexistent_terminal_xyz")
    assert spec4["required"] == []
    assert spec4["optional"] == []


# ---------------------------------------------------------------------------
# Scenario 12 — All awaiting_admin terminals must return terminal_class="awaiting_admin_review"
# ---------------------------------------------------------------------------

def test_all_admin_terminals_return_awaiting_admin_review():
    """
    Walk a sample of known awaiting_admin terminals via different paths and
    assert that terminal_class is always "awaiting_admin_review" — never None
    or a different value.  This guards against the engine accidentally
    authorising replacement / compensation / tech dispatch on its own.
    """
    admin_paths = [
        # (answers, expected_terminal_node_id)
        (["warranty", "defect", "rolling", "noise_up_down"],
         "defect_rolling_noise_updown_terminal"),
        (["warranty", "defect", "remote", "no_power", "all_checked_ok"],
         "defect_remote_pcb_check_terminal"),
        (["warranty", "defect", "recline", "backrest", "multiple_not_working", "stays_stuck"],
         "defect_recline_main_pcb_wire_terminal"),
    ]

    for answers, expected_node in admin_paths:
        ticket_id, _ = start(session_id=f"test-{expected_node}")
        result = walk(ticket_id, answers)
        assert result["next_node_id"] == expected_node, \
            f"Expected {expected_node}, got {result['next_node_id']}"
        assert result["terminal_class"] == "awaiting_admin_review", \
            f"Terminal {expected_node} returned class={result['terminal_class']!r}"
        t = ticket(ticket_id)
        assert str(t.status) == "awaiting_admin_review", \
            f"Ticket status for {expected_node} was {t.status!r}"


# ---------------------------------------------------------------------------
# Scenario 13 — No LLM call inside the workflow engine
# ---------------------------------------------------------------------------

def test_no_llm_call_in_workflow_engine(monkeypatch):
    """
    Confirm that WarrantyEngine never imports or calls openai.
    We patch the openai module to raise on any attribute access.
    If any path through submit_answer / start_session touches openai, this fails.
    """
    import sys
    import types

    # Inject a sentinel module that raises on attribute access
    class _NeverCall(types.ModuleType):
        def __getattr__(self, name: str):
            raise AssertionError(
                f"WARRANTY ENGINE VIOLATION: openai.{name} was called inside "
                "warranty_workflow. The engine must be LLM-free."
            )

    original = sys.modules.get("openai")
    sys.modules["openai"] = _NeverCall("openai")  # type: ignore[assignment]
    try:
        ticket_id, root = WarrantyEngine.start_session("llm-test", "test.com")
        result = walk(ticket_id, [
            "warranty", "defect", "air", "shoulders_hips", "yes_hissing",
        ])
        assert result["is_terminal"] is True
        assert result["terminal_class"] == "awaiting_admin_review"
    finally:
        if original is None:
            del sys.modules["openai"]
        else:
            sys.modules["openai"] = original


# ===========================================================================
# Phase C — Agent tool tests
# ===========================================================================

class TestWarrantyTools:
    """Tests for tool_start_warranty_workflow, tool_answer_warranty_question,
    tool_attach_warranty_evidence (calling the Python functions directly,
    not via HTTP)."""

    def test_tool_start_warranty_workflow_returns_ticket_started(self):
        """start_warranty_workflow returns a WARRANTY_TICKET_STARTED prefix."""
        from agent_tools import tool_start_warranty_workflow
        result = tool_start_warranty_workflow(session_id="tool-test-session", domain="osaki.com")
        assert result.startswith("WARRANTY_TICKET_STARTED"), (
            f"Expected WARRANTY_TICKET_STARTED prefix, got: {result[:80]}"
        )
        assert "TICKET_ID:" in result
        assert "PROMPT:" in result

    def test_tool_start_warranty_workflow_empty_session_fails(self):
        """start_warranty_workflow with empty session_id returns INPUT_ERROR."""
        from agent_tools import tool_start_warranty_workflow
        result = tool_start_warranty_workflow(session_id="", domain="osaki.com")
        assert "WARRANTY_INPUT_ERROR" in result

    def test_tool_answer_warranty_question_advances_workflow(self):
        """answer_warranty_question moves to next node and returns WARRANTY_CONTINUE."""
        from agent_tools import tool_start_warranty_workflow, tool_answer_warranty_question
        start_result = tool_start_warranty_workflow(session_id="answer-test", domain="osaki.com")
        # Extract ticket_id from the result string
        ticket_id = None
        for line in start_result.splitlines():
            if line.startswith("TICKET_ID:"):
                ticket_id = line.split(":", 1)[1].strip()
                break
        assert ticket_id is not None, "Could not extract TICKET_ID from start result"

        # Answer the first question (issue_type = "warranty")
        answer_result = tool_answer_warranty_question(
            ticket_id=ticket_id,
            answer_key="warranty",
        )
        assert ("WARRANTY_CONTINUE" in answer_result or "WARRANTY_TERMINAL_REACHED" in answer_result), (
            f"Unexpected result: {answer_result[:120]}"
        )

    def test_tool_answer_warranty_question_bad_ticket_returns_error(self):
        """answer_warranty_question with nonexistent ticket_id returns WARRANTY_ERROR."""
        from agent_tools import tool_answer_warranty_question
        result = tool_answer_warranty_question(
            ticket_id="nonexistent-uuid-xxxx",
            answer_key="warranty",
        )
        assert "WARRANTY_ERROR" in result or "WARRANTY_ANSWER_MISMATCH" in result

    def test_tool_answer_warranty_question_empty_ticket_id_fails(self):
        """answer_warranty_question with empty ticket_id returns INPUT_ERROR."""
        from agent_tools import tool_answer_warranty_question
        result = tool_answer_warranty_question(ticket_id="", answer_key="warranty")
        assert "WARRANTY_INPUT_ERROR" in result

    def test_tool_answer_warranty_question_bad_answer_returns_mismatch(self):
        """answer_warranty_question with wrong answer_key returns MISMATCH with hints."""
        from agent_tools import tool_start_warranty_workflow, tool_answer_warranty_question
        start_result = tool_start_warranty_workflow(session_id="mismatch-test", domain="osaki.com")
        ticket_id = None
        for line in start_result.splitlines():
            if line.startswith("TICKET_ID:"):
                ticket_id = line.split(":", 1)[1].strip()
                break
        assert ticket_id is not None
        result = tool_answer_warranty_question(
            ticket_id=ticket_id,
            answer_key="totally_invalid_key_xyz",
        )
        assert "WARRANTY_ANSWER_MISMATCH" in result
        assert "VALID OPTIONS" in result

    def test_tool_answer_does_not_nlp_advance_menu_from_free_text(self):
        """Option menus stay put when the model guesses a nearby answer_key."""
        from agent_tools import tool_start_warranty_workflow, tool_answer_warranty_question
        from warranty_workflow import WarrantyEngine

        start_result = tool_start_warranty_workflow(
            session_id="no-nlp-advance",
            domain="osaki.com",
        )
        ticket_id = None
        for line in start_result.splitlines():
            if line.startswith("TICKET_ID:"):
                ticket_id = line.split(":", 1)[1].strip()
                break
        assert ticket_id is not None
        before = WarrantyEngine.get_current_node(ticket_id)["node_id"]
        result = tool_answer_warranty_question(
            ticket_id=ticket_id,
            answer_key="not_a_real_key",
            customer_text="asdf qwerty not an option",
        )
        after = WarrantyEngine.get_current_node(ticket_id)["node_id"]
        assert after == before
        assert "WARRANTY_ANSWER_MISMATCH" in result

    def test_tool_attach_warranty_evidence_records_metadata(self):
        """attach_warranty_evidence records metadata for a valid ticket."""
        from agent_tools import tool_start_warranty_workflow, tool_attach_warranty_evidence
        # Start a session so we have a real ticket
        start_result = tool_start_warranty_workflow(session_id="evidence-test", domain="osaki.com")
        ticket_id = None
        for line in start_result.splitlines():
            if line.startswith("TICKET_ID:"):
                ticket_id = line.split(":", 1)[1].strip()
                break
        assert ticket_id is not None

        result = tool_attach_warranty_evidence(
            ticket_id=ticket_id,
            evidence_type="damage_photos",
            original_filename="front_damage.jpg",
        )
        assert "EVIDENCE_NOTED" in result
        assert "damage_photos" in result
        assert "front_damage.jpg" in result

    def test_tool_attach_warranty_evidence_invalid_type_fails(self):
        """attach_warranty_evidence with invalid evidence_type returns INPUT_ERROR."""
        from agent_tools import tool_start_warranty_workflow, tool_attach_warranty_evidence
        start_result = tool_start_warranty_workflow(session_id="ev-bad-type", domain="osaki.com")
        ticket_id = None
        for line in start_result.splitlines():
            if line.startswith("TICKET_ID:"):
                ticket_id = line.split(":", 1)[1].strip()
                break
        assert ticket_id is not None

        result = tool_attach_warranty_evidence(
            ticket_id=ticket_id,
            evidence_type="invalid_type_xyz",
        )
        assert "WARRANTY_INPUT_ERROR" in result

    def test_tool_attach_warranty_evidence_empty_ticket_fails(self):
        """attach_warranty_evidence with empty ticket_id returns INPUT_ERROR."""
        from agent_tools import tool_attach_warranty_evidence
        result = tool_attach_warranty_evidence(
            ticket_id="",
            evidence_type="damage_photos",
        )
        assert "WARRANTY_INPUT_ERROR" in result

    def test_warranty_tool_schemas_contains_new_names(self):
        """TOOL_SCHEMAS contains all three new warrant tool names."""
        from agent_tools import TOOL_SCHEMAS
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert "start_warranty_workflow" in names
        assert "answer_warranty_question" in names
        assert "attach_warranty_evidence" in names

    def test_warranty_tool_schemas_no_legacy_names(self):
        """TOOL_SCHEMAS does NOT expose old warranty_start / warranty_answer names."""
        from agent_tools import TOOL_SCHEMAS
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert "warranty_start" not in names, "Legacy name warranty_start still in TOOL_SCHEMAS"
        assert "warranty_answer" not in names, "Legacy name warranty_answer still in TOOL_SCHEMAS"

    def test_warranty_tool_schemas_subset(self):
        """WARRANTY_TOOL_SCHEMAS is a strict subset containing only workflow-related tools."""
        from agent_tools import TOOL_SCHEMAS, WARRANTY_TOOL_SCHEMAS
        all_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        subset_names = {s["function"]["name"] for s in WARRANTY_TOOL_SCHEMAS}
        assert subset_names.issubset(all_names), "WARRANTY_TOOL_SCHEMAS has names not in TOOL_SCHEMAS"
        # Must include answer_warranty_question but NOT start_warranty_workflow
        assert "answer_warranty_question" in subset_names
        assert "start_warranty_workflow" not in subset_names, (
            "start_warranty_workflow should NOT be in WARRANTY_TOOL_SCHEMAS "
            "(workflow is already started; only answering is valid mid-flow)"
        )


# ===========================================================================
# Phase D-lite — Evidence recording
# ===========================================================================

class TestEvidenceRecording:
    """Unit-level tests for WarrantyEngine.record_evidence() (no HTTP)."""

    def test_record_evidence_creates_row(self):
        """record_evidence saves a WarrantyEvidence row and returns it."""
        ticket_id, _ = start()
        ev = WarrantyEngine.record_evidence(
            ticket_id=ticket_id,
            evidence_type="damage_photos",
            file_path="/tmp/test.jpg",
            original_filename="test.jpg",
            mime_type="image/jpeg",
            file_size_bytes=12345,
        )
        assert str(ev.ticket_id) == ticket_id
        assert str(ev.evidence_type) == "damage_photos"
        assert str(ev.original_filename) == "test.jpg"
        assert cast(int, ev.file_size_bytes) == 12345
        assert cast(int, ev.emailed) == 0   # not yet emailed

    def test_record_evidence_persisted_in_get_evidences(self):
        """Evidence row is retrievable via get_evidences()."""
        ticket_id, _ = start()
        WarrantyEngine.record_evidence(
            ticket_id=ticket_id,
            evidence_type="video_of_issue",
            original_filename="clip.mp4",
        )
        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 1
        assert str(evidences[0].evidence_type) == "video_of_issue"

    def test_record_evidence_multiple_files(self):
        """Multiple evidence rows can be attached to a single ticket."""
        ticket_id, _ = start()
        for i in range(3):
            WarrantyEngine.record_evidence(
                ticket_id=ticket_id,
                evidence_type="damage_photos",
                original_filename=f"photo_{i}.jpg",
            )
        evidences = WarrantyEngine.get_evidences(ticket_id)
        assert len(evidences) == 3

    def test_record_evidence_nonexistent_ticket_raises(self):
        """record_evidence raises ValueError for a nonexistent ticket_id."""
        with pytest.raises(ValueError, match="not found"):
            WarrantyEngine.record_evidence(
                ticket_id="ghost-ticket-id",
                evidence_type="damage_photos",
            )


# ===========================================================================
# Phase E-lite — Admin decision tests
# ===========================================================================

class TestAdminDecision:
    """Tests ensuring only admin_decision() can set approved/rejected."""

    def _reach_terminal(self) -> str:
        """Walk to a terminal node and return the ticket_id."""
        ticket_id, _ = start()
        walk(ticket_id, [
            "warranty", "defect", "rolling",
            "noise_up_down",  # → defect_rolling_noise_updown_terminal
        ])
        return ticket_id

    def test_admin_decision_approved(self):
        """admin_decision sets status=resolved and admin_decision=approved."""
        ticket_id = self._reach_terminal()
        resolved = WarrantyEngine.admin_decision(
            ticket_id=ticket_id,
            decision="approved",
            note="Approved for tech visit.",
            decided_by="ops_team",
            customer_message="A technician will contact you within 2 business days.",
        )
        assert str(resolved.status) == "resolved"
        assert str(resolved.admin_decision) == "approved"
        assert str(resolved.decided_by) == "ops_team"
        assert "technician" in str(resolved.customer_message)

    def test_admin_decision_rejected(self):
        """admin_decision sets status=resolved and admin_decision=rejected."""
        ticket_id = self._reach_terminal()
        resolved = WarrantyEngine.admin_decision(
            ticket_id=ticket_id,
            decision="rejected",
            note="Out of warranty period.",
            decided_by="admin",
        )
        assert str(resolved.status) == "resolved"
        assert str(resolved.admin_decision) == "rejected"

    def test_admin_decision_need_more_information(self):
        """need_more_information does NOT set resolved — ticket stays open."""
        ticket_id = self._reach_terminal()
        updated = WarrantyEngine.admin_decision(
            ticket_id=ticket_id,
            decision="need_more_information",
            note="Please send photos of the damage.",
            decided_by="admin",
        )
        assert str(updated.status) == "need_more_information"
        assert str(updated.admin_decision) == "need_more_information"

    def test_admin_decision_admin_reviewing(self):
        """admin_reviewing does NOT set resolved — ticket stays open."""
        ticket_id = self._reach_terminal()
        updated = WarrantyEngine.admin_decision(
            ticket_id=ticket_id,
            decision="admin_reviewing",
            decided_by="admin",
        )
        assert str(updated.status) == "admin_reviewing"

    def test_admin_decision_closed(self):
        """closed sets status=resolved."""
        ticket_id = self._reach_terminal()
        closed = WarrantyEngine.admin_decision(
            ticket_id=ticket_id,
            decision="closed",
            decided_by="admin",
        )
        assert str(closed.status) == "resolved"
        assert str(closed.admin_decision) == "closed"

    def test_admin_decision_invalid_raises(self):
        """Invalid decision string raises ValueError."""
        ticket_id = self._reach_terminal()
        with pytest.raises(ValueError, match="Invalid decision"):
            WarrantyEngine.admin_decision(
                ticket_id=ticket_id,
                decision="free_money",
            )

    def test_admin_decision_only_approved_path(self):
        """
        The workflow engine never sets status='approved' or status='rejected'
        during normal customer traversal — only admin_decision() can do that.
        """
        ticket_id, _ = start()
        # Walk all the way to a terminal
        result = walk(ticket_id, [
            "warranty", "defect", "air", "shoulders_hips", "yes_hissing",
        ])
        assert result["is_terminal"] is True
        t = ticket(ticket_id)
        # Status must be awaiting_admin_review, NEVER approved or rejected
        status = str(t.status)
        assert status not in ("approved", "rejected"), (
            f"Workflow engine set status={status!r} — only admin_decision() may do that!"
        )
        assert status == "awaiting_admin_review"

    def test_add_admin_note(self):
        """add_admin_note appends a note without changing the ticket status."""
        ticket_id = self._reach_terminal()
        t_before = ticket(ticket_id)
        status_before = str(t_before.status)

        updated = WarrantyEngine.add_admin_note(
            ticket_id=ticket_id,
            note="Customer called back; waiting for photos.",
            added_by="agent_sarah",
        )
        assert str(updated.status) == status_before, "add_admin_note must not change status"
        assert "Customer called back" in str(updated.admin_note)
        assert "agent_sarah" in str(updated.admin_note)

    def test_add_admin_note_nonexistent_raises(self):
        """add_admin_note raises ValueError for a nonexistent ticket."""
        with pytest.raises(ValueError, match="not found"):
            WarrantyEngine.add_admin_note(
                ticket_id="ghost-ticket-999",
                note="This should fail.",
            )


# ---------------------------------------------------------------------------
# Go back (rewind one turn)
# ---------------------------------------------------------------------------

def test_go_back_restores_previous_question():
    ticket_id, _ = start()
    walk(ticket_id, ["warranty", "defect", "voice"])

    t = ticket(ticket_id)
    assert str(t.defect_type) == "voice"
    assert len(WarrantyEngine.get_turns(ticket_id)) == 3

    result = WarrantyEngine.go_back(ticket_id)
    assert result["restored_node_id"] == "defect_problem_type"
    assert result["turn_count"] == 2

    t = ticket(ticket_id)
    assert str(t.issue_type) == "defect"
    assert t.defect_type is None
    assert str(t.current_node_id) == "defect_problem_type"
    assert str(t.status) == "in_progress"
    assert WarrantyEngine.can_go_back(ticket_id) is True


def test_go_back_from_issue_type_keeps_registered_model():
    ticket_id, _ = start()
    submit(ticket_id, "warranty")
    WarrantyEngine.set_model_name(ticket_id, "OS-4000T")
    submit(ticket_id, "defect")

    assert str(ticket(ticket_id).current_node_id) == "defect_problem_type"

    WarrantyEngine.go_back(ticket_id)

    t = ticket(ticket_id)
    assert str(t.model_name) == "OS-4000T"
    assert str(t.current_node_id) == "issue_type"
    assert t.issue_type is None
    assert len(WarrantyEngine.get_turns(ticket_id)) == 1


def test_go_back_blocked_at_root_without_turns():
    ticket_id, _ = start()
    assert WarrantyEngine.can_go_back(ticket_id) is False
    with pytest.raises(ValueError, match="Nothing to go back"):
        WarrantyEngine.go_back(ticket_id)


def test_go_back_blocked_after_terminal():
    ticket_id, _ = start()
    walk(ticket_id, [
        "warranty",
        "installation",
        "OS-4000T",
        "general_setup",
    ])
    t = ticket(ticket_id)
    assert str(t.status) == "send_info"
    with pytest.raises(ValueError, match="Cannot go back"):
        WarrantyEngine.go_back(ticket_id)
