"""
warranty_workflow.py
====================
Deterministic state-machine engine for the Warranty AI Workflow.

Design contract
---------------
- ZERO LLM calls in this module.
- The flowchart JSON (data/warranty_flowchart.json) is the SOLE source of truth
  for branching logic.
- warranty_evidence_specs.json is loaded alongside for evidence lookup.
- Every branching decision is made by matching a customer answer to an
  `options[].answer_key` in the current node — never by free-form inference.
- LLM role (handled upstream in main.py): paraphrase the `prompt` field and
  present the `options` list to the customer in a friendly tone.
- The engine NEVER approves warranty actions (replacement, compensation,
  tech dispatch, refund). Those require an admin decision recorded via
  admin_decision() — the ticket lands in status="awaiting_admin_review" first.

Public API
----------
  WarrantyEngine.start_session(session_id, domain)
      → (ticket_id: str, node: dict)

  WarrantyEngine.get_current_node(ticket_id)
      → node: dict | None

  WarrantyEngine.submit_answer(ticket_id, raw_answer)
      → SubmitResult dict

  WarrantyEngine.get_ticket(ticket_id)
      → WarrantyTicket | None

  WarrantyEngine.get_turns(ticket_id)
      → list[WarrantyTurn]

  WarrantyEngine.get_evidences(ticket_id)
      → list[WarrantyEvidence]

  WarrantyEngine.get_tickets(status, domain, limit, offset)
      → list[WarrantyTicket]

  WarrantyEngine.admin_decision(ticket_id, decision, note, decided_by, customer_message)
      → WarrantyTicket

  WarrantyEngine.get_evidence_spec(terminal_node_id)
      → dict  {"required": [...], "optional": [...]} from warranty_evidence_specs.json

  WarrantyEngine.get_evidence_specs()
      → dict  full evidence specs (evidence_types + terminal_evidence_map)

SubmitResult (dict) keys
------------------------
  next_node_id       str        – ID of the node we transitioned to
  next_node          dict       – full node object (with node_id injected)
  answer_key         str        – normalised answer_key that was matched
  is_terminal        bool       – True if the next node is a terminal
  terminal_class     str | None – terminal category when is_terminal is True:
                                    "awaiting_admin_review"  – replacement / tech /
                                                               compensation / refund /
                                                               approval required
                                    "send_info"              – self-service info sent
                                    "awaiting_evidence"      – evidence upload requested
                                    "sales_handoff"          – routed to sales
  evidence_required  list[str]  – evidence types required  (terminal nodes only)
  evidence_email     str | None – email to send evidence to (terminal nodes only)

Ticket status values
--------------------
  "in_progress"           – workflow is ongoing, waiting for next customer answer
  "awaiting_admin_review" – reached a terminal that requires admin action
  "awaiting_evidence"     – terminal that only asks for evidence upload (no admin)
  "send_info"             – self-service terminal (no admin needed)
  "sales_handoff"         – customer routed to sales
  "resolved"              – admin has recorded a decision via admin_decision()
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from warranty_error_code_gate import (
    COL_ERROR_CODE,
    COL_GATE_COMPLETED,
    COL_PENDING_TERMINAL,
    GATE_ENTER_ID,
    GATE_PICK_ID,
    GATE_VISIBLE_ID,
    finalize_error_code_submission,
    intercept_terminal_node_id,
    is_gate_node,
    resolve_gate_node,
)
from warranty_models import (
    _SessionFactory,
    WarrantyTicket,
    WarrantyTurn,
    WarrantyEvidence,
    warranty_db_session,
)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

_DATA_DIR          = Path(__file__).resolve().parent.parent / "data"
_FLOWCHART_PATH    = _DATA_DIR / "warranty_flowchart.json"
_EVIDENCE_SPECS_PATH = _DATA_DIR / "warranty_evidence_specs.json"


def _load_flowchart() -> dict:
    with open(_FLOWCHART_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_evidence_specs() -> dict:
    if not _EVIDENCE_SPECS_PATH.exists():
        return {"evidence_types": {}, "terminal_evidence_map": {}}
    with open(_EVIDENCE_SPECS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Module-level cache — reloaded via reload_flowchart() when files are edited
_FLOWCHART: dict = _load_flowchart()
_NODES: dict     = _FLOWCHART["nodes"]
_ROOT: str       = _FLOWCHART["root"]

_EVIDENCE_SPECS: dict              = _load_evidence_specs()
_TERMINAL_EVIDENCE_MAP: dict[str, dict] = _EVIDENCE_SPECS.get("terminal_evidence_map", {})


def reload_flowchart() -> None:
    """Re-read both JSON files from disk. Call after editing data/."""
    global _FLOWCHART, _NODES, _ROOT, _EVIDENCE_SPECS, _TERMINAL_EVIDENCE_MAP
    _FLOWCHART = _load_flowchart()
    _NODES     = _FLOWCHART["nodes"]
    _ROOT      = _FLOWCHART["root"]
    _EVIDENCE_SPECS        = _load_evidence_specs()
    _TERMINAL_EVIDENCE_MAP = _EVIDENCE_SPECS.get("terminal_evidence_map", {})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_view(node_id: str) -> dict:
    """Return a node dict with 'node_id' injected."""
    node = _NODES[node_id]
    return {"node_id": node_id, **node}


def _match_option(options: list[dict], raw_answer: str) -> Optional[dict]:
    """
    Try to match raw_answer against options in priority order:
      1. Exact answer_key match (case-sensitive)
      2. Case-insensitive label match
      3. 1-based integer index ("1", "2", …)

    Returns the matched option dict or None.
    """
    # 1. Exact answer_key
    for opt in options:
        if opt.get("answer_key") == raw_answer:
            return opt

    # 2. Case-insensitive label
    lower = raw_answer.strip().lower()
    for opt in options:
        if opt.get("label", "").strip().lower() == lower:
            return opt

    # 3. 1-based integer index
    try:
        idx = int(raw_answer.strip()) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except (ValueError, AttributeError):
        pass

    return None


def _resolve_node(node_id: str, ticket: Optional[WarrantyTicket] = None) -> Optional[dict]:
    if is_gate_node(node_id):
        if ticket is None:
            return None
        return resolve_gate_node(node_id, ticket)
    node = _NODES.get(node_id)
    if node is None:
        return None
    return {"node_id": node_id, **node}


def _apply_pending_terminal(ticket: WarrantyTicket, pending_id: str) -> dict:
    pending_node = _NODES[pending_id]
    ticket.current_node_id = pending_id
    ticket.status = _terminal_status(pending_node.get("action", "awaiting_admin"))
    return pending_node


def _build_submit_result(
    *,
    next_node_id: str,
    previous_node_id: str,
    next_node: dict,
    answer_key: str,
) -> dict:
    is_terminal = next_node.get("type") == "terminal"
    evidence_required: list[str] = []
    evidence_email: Optional[str] = None
    terminal_class: Optional[str] = None
    if is_terminal:
        action = next_node.get("action", "awaiting_admin")
        terminal_class = _terminal_status(action)
        evidence_required = next_node.get("evidence_required", [])
        evidence_email = next_node.get("evidence_email")
    return {
        "next_node_id":      next_node_id,
        "previous_node_id":  previous_node_id,
        "next_node":         {"node_id": next_node_id, **next_node},
        "answer_key":        answer_key,
        "is_terminal":       is_terminal,
        "terminal_class":    terminal_class,
        "evidence_required": evidence_required,
        "evidence_email":    evidence_email,
    }


def _terminal_status(action: str) -> str:
    """Map a terminal node's action to the corresponding ticket status.

    awaiting_admin / awaiting_admin_review terminals are ones that require a
    human admin to approve replacement, compensation, tech dispatch, or refund.
    The engine NEVER performs those actions autonomously.
    """
    return {
        "send_info":        "send_info",
        "sales_handoff":    "sales_handoff",
        "request_evidence": "awaiting_evidence",
    }.get(action, "awaiting_admin_review")


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------

class WarrantyEngine:
    """Stateless class — all state lives in the DB and the flowchart JSON."""

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @staticmethod
    def start_session(session_id: str, domain: str) -> tuple[str, dict]:
        """
        Create a new warranty ticket and return (ticket_id, root_node).

        The caller should display root_node["prompt"] and root_node["options"]
        to the customer.
        """
        ticket_id = str(uuid.uuid4())
        with warranty_db_session() as db:
            ticket = WarrantyTicket(
                ticket_id=ticket_id,
                session_id=session_id,
                domain=domain,
                status="in_progress",
                current_node_id=_ROOT,
                collected_data="{}",
            )
            db.add(ticket)
        return ticket_id, _node_view(_ROOT)

    _QUICK_START_ISSUES = frozenset({"installation", "delivery", "defect"})

    @staticmethod
    def advance_to_issue_type(ticket_id: str, issue_type: str) -> dict:
        """
        Skip the root and issue_type menus and land on the issue entry node.

        Used by the phone IVR (defect-only after hours) and mirrors web
        quick-start: warranty → installation | delivery | defect.
        """
        issue_type = (issue_type or "").strip().lower()
        if issue_type not in WarrantyEngine._QUICK_START_ISSUES:
            raise ValueError(
                f"issue_type must be one of: {sorted(WarrantyEngine._QUICK_START_ISSUES)}"
            )

        node = WarrantyEngine.get_current_node(ticket_id)
        if node is None:
            raise ValueError(f"Ticket {ticket_id!r} not found.")

        node_id = node.get("node_id")
        if node_id == "root":
            WarrantyEngine.submit_answer(ticket_id, "warranty")
            return WarrantyEngine.submit_answer(ticket_id, issue_type)
        if node_id == "issue_type":
            return WarrantyEngine.submit_answer(ticket_id, issue_type)

        raise ValueError(
            f"Cannot advance to {issue_type!r} from node {node_id!r}."
        )

    # ------------------------------------------------------------------
    # Read current state
    # ------------------------------------------------------------------

    @staticmethod
    def get_current_node(ticket_id: str) -> Optional[dict]:
        """Return the current flowchart node for this ticket, or None."""
        with warranty_db_session() as db:
            ticket = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if not ticket:
                return None
            node_id = ticket.current_node_id
        return _resolve_node(node_id, ticket)

    @staticmethod
    def get_ticket(ticket_id: str) -> Optional[WarrantyTicket]:
        """Return the WarrantyTicket ORM object (detached) or None."""
        with warranty_db_session() as db:
            return (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )

    @staticmethod
    def get_turns(ticket_id: str) -> list[WarrantyTurn]:
        """Return all turns for a ticket, ordered chronologically."""
        with warranty_db_session() as db:
            return (
                db.query(WarrantyTurn)
                .filter(WarrantyTurn.ticket_id == ticket_id)
                .order_by(WarrantyTurn.id.asc())
                .all()
            )

    @staticmethod
    def get_evidences(ticket_id: str) -> list[WarrantyEvidence]:
        """Return all evidence rows for a ticket."""
        with warranty_db_session() as db:
            return (
                db.query(WarrantyEvidence)
                .filter(WarrantyEvidence.ticket_id == ticket_id)
                .order_by(WarrantyEvidence.id.asc())
                .all()
            )

    @staticmethod
    def get_evidence_by_id(evidence_id: int) -> Optional[WarrantyEvidence]:
        """Return a single evidence record by its primary key, or None."""
        with warranty_db_session() as db:
            return (
                db.query(WarrantyEvidence)
                .filter(WarrantyEvidence.id == evidence_id)
                .first()
            )

    @staticmethod
    def set_model_name(ticket_id: str, model_name: str) -> None:
        """Persist normalized chair model before or during workflow intake."""
        display = (model_name or "").strip()
        if not display:
            raise ValueError("model_name must not be empty")
        with warranty_db_session() as db:
            ticket = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if ticket is None:
                raise ValueError(f"Ticket {ticket_id!r} not found.")
            ticket.model_name = display
            ticket.set_collected("model_name", display)

    @staticmethod
    def get_active_session_ticket(session_id: str) -> Optional[WarrantyTicket]:
        """Return the most recent in-progress ticket for a chat session, or None."""
        with warranty_db_session() as db:
            return (
                db.query(WarrantyTicket)
                .filter(
                    WarrantyTicket.session_id == session_id,
                    WarrantyTicket.status == "in_progress",
                )
                .order_by(WarrantyTicket.created_at.desc())
                .first()
            )

    @staticmethod
    def abandon_session_tickets(session_id: str) -> int:
        """
        Close any open (non-resolved) tickets for a session so the customer can
        start fresh with a "Start over" action.

        Tickets that have already been reviewed/resolved by the admin are left
        untouched. Tickets currently in_progress, awaiting_admin_review,
        awaiting_evidence, or send_info are marked resolved with
        admin_decision='abandoned' so they drop out of the active queue but
        stay queryable for audit.

        Returns the number of tickets that were closed.
        """
        closed_statuses = {
            "in_progress",
            "awaiting_admin_review",
            "awaiting_evidence",
            "send_info",
            "sales_handoff",
            "admin_reviewing",
            "need_more_information",
        }
        closed_count = 0
        with warranty_db_session() as db:
            tickets = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.session_id == session_id)
                .all()
            )
            for ticket in tickets:
                if str(ticket.status or "") in closed_statuses:
                    ticket.status = "resolved"
                    ticket.admin_decision = "abandoned"
                    if not ticket.decided_by:
                        ticket.decided_by = "customer_restart"
                    closed_count += 1
        return closed_count

    @staticmethod
    def get_tickets(
        status: Optional[str] = None,
        domain: Optional[str] = None,
        channel: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WarrantyTicket]:
        """List tickets filtered by status, domain, and/or intake channel."""
        with warranty_db_session() as db:
            q = db.query(WarrantyTicket)
            if status:
                q = q.filter(WarrantyTicket.status == status)
            if domain:
                q = q.filter(WarrantyTicket.domain.contains(domain))
            if channel == "phone":
                q = q.filter(WarrantyTicket.collected_data.like('%"channel": "phone"%'))
            elif channel == "web":
                q = q.filter(~WarrantyTicket.collected_data.like('%"channel": "phone"%'))
            return (
                q.order_by(WarrantyTicket.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    @staticmethod
    def submit_answer(
        ticket_id: str,
        raw_answer: str,
        *,
        customer_display: Optional[str] = None,
    ) -> dict:
        """
        Process the customer's answer for the current node and transition state.

        Parameters
        ----------
        ticket_id  : UUID string for the open ticket
        raw_answer : Customer's raw input — either an answer_key, label text,
                     or 1-based integer index string.
        customer_display : Optional verbatim text to store on the turn (e.g. when
                     NLP mapped natural language to an answer_key).

        Returns
        -------
        SubmitResult dict (see module docstring).

        Raises
        ------
        ValueError  if ticket not found, already terminal, or answer unrecognised.
        """
        with warranty_db_session() as db:
            ticket = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if not ticket:
                raise ValueError(f"Ticket {ticket_id!r} not found.")

            if ticket.status != "in_progress":
                raise ValueError(
                    f"Ticket {ticket_id!r} is no longer in progress "
                    f"(status={ticket.status!r}). Cannot accept further answers."
                )

            node_id = ticket.current_node_id
            node = _resolve_node(node_id, ticket)
            if node is None:
                raise ValueError(
                    f"Flowchart node {node_id!r} not found. "
                    "The flowchart may have been updated — please reload."
                )

            node_type = node.get("type")

            if node_type == "terminal":
                raise ValueError(
                    f"Node {node_id!r} is a terminal — no answer expected."
                )

            # -----------------------------------------------------------
            # Virtual error-code gate (engine intercept)
            # -----------------------------------------------------------
            if is_gate_node(node_id):
                pending_id = str(ticket.get_collected().get(COL_PENDING_TERMINAL) or "")
                if not pending_id or pending_id not in _NODES:
                    raise ValueError(
                        "Error-code gate state is invalid — please contact support."
                    )

                if node_id == GATE_VISIBLE_ID:
                    options = node.get("options", [])
                    matched = _match_option(options, raw_answer)
                    if matched is None:
                        valid_keys = [
                            o.get("answer_key", o.get("label")) for o in options
                        ]
                        raise ValueError(
                            f"Answer {raw_answer!r} did not match any option at node "
                            f"{node_id!r}. Valid answer_keys: {valid_keys}"
                        )
                    answer_key = str(matched.get("answer_key") or raw_answer)
                    turn = WarrantyTurn(
                        ticket_id=ticket_id,
                        node_id=node_id,
                        node_type=node_type,
                        node_prompt=node.get("prompt", ""),
                        customer_answer=(
                            customer_display if customer_display is not None else raw_answer
                        ),
                        answer_key=answer_key,
                    )
                    db.add(turn)

                    if answer_key == "error_code_yes":
                        ticket.current_node_id = GATE_PICK_ID
                        next_node = resolve_gate_node(GATE_PICK_ID, ticket) or {}
                        return _build_submit_result(
                            next_node_id=GATE_PICK_ID,
                            previous_node_id=node_id,
                            next_node=next_node,
                            answer_key=answer_key,
                        )

                    ticket.set_collected(COL_GATE_COMPLETED, "skipped")
                    pending_node = _apply_pending_terminal(ticket, pending_id)
                    ticket.set_collected(COL_PENDING_TERMINAL, "")
                    return _build_submit_result(
                        next_node_id=pending_id,
                        previous_node_id=node_id,
                        next_node=pending_node,
                        answer_key=answer_key,
                    )

                if node_id == GATE_PICK_ID:
                    from error_code_lookup import parse_pick_answer_key  # noqa: WPS433

                    options = node.get("options", [])
                    matched = _match_option(options, raw_answer)
                    if matched is None:
                        valid_keys = [
                            o.get("answer_key", o.get("label")) for o in options
                        ]
                        raise ValueError(
                            f"Answer {raw_answer!r} did not match any option at node "
                            f"{node_id!r}. Valid answer_keys: {valid_keys}"
                        )
                    answer_key = str(matched.get("answer_key") or raw_answer)
                    turn = WarrantyTurn(
                        ticket_id=ticket_id,
                        node_id=node_id,
                        node_type=node_type,
                        node_prompt=node.get("prompt", ""),
                        customer_answer=(
                            customer_display if customer_display is not None else raw_answer
                        ),
                        answer_key=answer_key,
                    )
                    db.add(turn)

                    if answer_key == "error_code_other":
                        ticket.current_node_id = GATE_ENTER_ID
                        next_node = resolve_gate_node(GATE_ENTER_ID, ticket) or {}
                        return _build_submit_result(
                            next_node_id=GATE_ENTER_ID,
                            previous_node_id=node_id,
                            next_node=next_node,
                            answer_key=answer_key,
                        )

                    picked_code = parse_pick_answer_key(answer_key)
                    if not picked_code:
                        raise ValueError(
                            f"Could not parse error code from answer_key {answer_key!r}."
                        )
                    finalize_error_code_submission(ticket, picked_code)
                    ticket.set_collected(COL_GATE_COMPLETED, "picked")
                    pending_node = _apply_pending_terminal(ticket, pending_id)
                    ticket.set_collected(COL_PENDING_TERMINAL, "")
                    return _build_submit_result(
                        next_node_id=pending_id,
                        previous_node_id=node_id,
                        next_node=pending_node,
                        answer_key=answer_key,
                    )

                if node_id == GATE_ENTER_ID:
                    text = (
                        customer_display if customer_display is not None else raw_answer
                    ).strip()
                    finalize_error_code_submission(ticket, text)

                    turn = WarrantyTurn(
                        ticket_id=ticket_id,
                        node_id=node_id,
                        node_type=node_type,
                        node_prompt=node.get("prompt", ""),
                        customer_answer=text,
                        answer_key=COL_ERROR_CODE,
                    )
                    db.add(turn)
                    ticket.set_collected(COL_GATE_COMPLETED, "entered")
                    pending_node = _apply_pending_terminal(ticket, pending_id)
                    ticket.set_collected(COL_PENDING_TERMINAL, "")
                    return _build_submit_result(
                        next_node_id=pending_id,
                        previous_node_id=node_id,
                        next_node=pending_node,
                        answer_key=COL_ERROR_CODE,
                    )

            # -----------------------------------------------------------
            # Determine the next node
            # -----------------------------------------------------------
            if node_type == "question_text":
                next_node_id = node["next"]
                answer_key = node.get("answer_key", "text_input")
                # Store free-text input in collected_data
                ticket.set_collected(answer_key, raw_answer)
                from warranty_email import extract_email  # noqa: WPS433

                detected = extract_email(raw_answer)
                if detected:
                    ticket.set_collected("customer_contact_email", detected)
            else:
                # question or instruction: match to an option
                options = node.get("options", [])
                matched = _match_option(options, raw_answer)
                if matched is None:
                    valid_keys = [o.get("answer_key", o.get("label")) for o in options]
                    raise ValueError(
                        f"Answer {raw_answer!r} did not match any option at node "
                        f"{node_id!r}. Valid answer_keys: {valid_keys}"
                    )
                next_node_id = matched["next"]
                answer_key = matched.get("answer_key", raw_answer)

            # -----------------------------------------------------------
            # Persist this turn
            # -----------------------------------------------------------
            turn = WarrantyTurn(
                ticket_id=ticket_id,
                node_id=node_id,
                node_type=node_type,
                node_prompt=node.get("prompt", ""),
                customer_answer=customer_display if customer_display is not None else raw_answer,
                answer_key=answer_key,
            )
            db.add(turn)

            # -----------------------------------------------------------
            # Update ticket fields from well-known nodes
            # -----------------------------------------------------------
            if node_id == "issue_type":
                ticket.issue_type = answer_key
            elif node_id == "defect_problem_type":
                ticket.defect_type = answer_key
            elif node_id == "install_model":
                from product_catalog import resolve_model_name  # noqa: WPS433

                display = customer_display if customer_display is not None else raw_answer
                ticket.model_name = resolve_model_name(display) or display

            # -----------------------------------------------------------
            # Transition to next node (optional error-code gate intercept)
            # -----------------------------------------------------------
            next_node = _NODES[next_node_id]
            gate_id = intercept_terminal_node_id(ticket, next_node_id)
            if gate_id and next_node.get("type") == "terminal":
                ticket.set_collected(COL_PENDING_TERMINAL, next_node_id)
                next_node_id = gate_id
                next_node = resolve_gate_node(gate_id, ticket) or {}
                is_terminal = False
                ticket.current_node_id = next_node_id
            else:
                is_terminal = next_node.get("type") == "terminal"
                ticket.current_node_id = next_node_id
                if is_terminal:
                    ticket.status = _terminal_status(next_node.get("action", "awaiting_admin"))

        # Build result (outside session — ticket is detached but expire_on_commit=False)
        if is_terminal:
            return _build_submit_result(
                next_node_id=next_node_id,
                previous_node_id=node_id,
                next_node=next_node,
                answer_key=answer_key,
            )
        return _build_submit_result(
            next_node_id=next_node_id,
            previous_node_id=node_id,
            next_node=next_node,
            answer_key=answer_key,
        )

    # ------------------------------------------------------------------
    # Admin decision (Phase D/E)
    # ------------------------------------------------------------------

    @staticmethod
    def admin_decision(
        ticket_id: str,
        decision: str,
        note: str = "",
        decided_by: str = "admin",
        customer_message: str = "",
    ) -> WarrantyTicket:
        """
        Record an admin decision on a ticket.

        Valid decisions
        ---------------
        admin_reviewing      – admin has picked up the ticket and is reviewing
        need_more_information – admin needs more info from the customer
        approved             – admin approves the warranty action (replacement/tech/etc.)
        rejected             – admin rejects the warranty claim
        closed               – case closed without further action

        Legacy aliases still accepted (mapped internally):
        replacement / tech_dispatch / compensation / repair → approved
        reject / request_more_info → rejected / need_more_information

        IMPORTANT: Only this method may set status='approved' or status='rejected'.
        Customer-facing chat and the workflow engine must never set these directly.

        After calling this, the ticket status becomes 'resolved'.
        """
        # Normalise legacy decision strings to canonical values
        _legacy_map = {
            "replacement":       "approved",
            "tech_dispatch":     "approved",
            "compensation":      "approved",
            "repair":            "approved",
            "reject":            "rejected",
            "request_more_info": "need_more_information",
        }
        decision = _legacy_map.get(decision, decision)

        valid_decisions = {
            "admin_reviewing", "need_more_information",
            "approved", "rejected", "closed",
        }
        if decision not in valid_decisions:
            raise ValueError(
                f"Invalid decision {decision!r}. Must be one of: {sorted(valid_decisions)}"
            )

        with warranty_db_session() as db:
            ticket = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if not ticket:
                raise ValueError(f"Ticket {ticket_id!r} not found.")

            ticket.admin_decision = decision
            ticket.admin_note = note or str(ticket.admin_note or "")
            ticket.decided_by = decided_by
            ticket.customer_message = customer_message or str(ticket.customer_message or "")
            # admin_reviewing / need_more_information do not close the ticket;
            # approved / rejected / closed mark it as resolved.
            if decision in ("approved", "rejected", "closed"):
                ticket.status = "resolved"
            else:
                ticket.status = decision  # "admin_reviewing" | "need_more_information"

            return ticket

    # ------------------------------------------------------------------
    # Evidence recording
    # ------------------------------------------------------------------

    @staticmethod
    def record_evidence(
        ticket_id: str,
        evidence_type: str,
        file_path: str = "",
        original_filename: str = "",
        mime_type: str = "",
        file_size_bytes: int = 0,
        customer_email: str = "",
    ) -> WarrantyEvidence:
        """
        Persist an evidence record for a ticket.

        Does NOT upload files or send email — it only writes the metadata row.
        The file upload and (future) email notification are handled by the API layer.

        Raises ValueError if ticket does not exist.
        """
        with warranty_db_session() as db:
            ticket = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if not ticket:
                raise ValueError(f"Ticket {ticket_id!r} not found.")

            normalized_email = ""
            if customer_email:
                from warranty_email import extract_email  # noqa: WPS433

                normalized = extract_email(customer_email)
                if normalized:
                    normalized_email = normalized
                    ticket.set_collected("customer_contact_email", normalized)

            ev = WarrantyEvidence(
                ticket_id=ticket_id,
                evidence_type=evidence_type,
                file_path=file_path,
                original_filename=original_filename,
                mime_type=mime_type,
                file_size_bytes=file_size_bytes,
                customer_email=normalized_email or None,
                emailed=0,
            )
            db.add(ev)
            return ev

    @staticmethod
    def add_admin_note(ticket_id: str, note: str, added_by: str = "admin") -> WarrantyTicket:
        """
        Append a note to the ticket's admin_note field without changing the status.
        Suitable for lightweight admin commentary that doesn't constitute a final decision.
        """
        with warranty_db_session() as db:
            ticket = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.ticket_id == ticket_id)
                .first()
            )
            if not ticket:
                raise ValueError(f"Ticket {ticket_id!r} not found.")

            existing = str(ticket.admin_note or "").strip()
            new_note = f"{existing}\n[{added_by}] {note}".strip()
            ticket.admin_note = new_note
            if not ticket.decided_by:
                ticket.decided_by = added_by
            return ticket

    # ------------------------------------------------------------------
    # Evidence spec helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_evidence_spec(terminal_node_id: str) -> dict:
        """
        Return the evidence requirements for a terminal node from
        warranty_evidence_specs.json.

        Returns a dict with keys:
            "required": list[str]  – evidence_type keys that must be submitted
            "optional": list[str]  – evidence_type keys that are helpful but not mandatory
            "types":    dict       – full type definitions for each required/optional key
                                     (from evidence_types section of the specs file)

        Returns {"required": [], "optional": [], "types": {}} if the node has no
        entry in the terminal_evidence_map.
        """
        spec = _TERMINAL_EVIDENCE_MAP.get(terminal_node_id, {"required": [], "optional": []})
        all_keys = spec.get("required", []) + spec.get("optional", [])
        all_types = _EVIDENCE_SPECS.get("evidence_types", {})
        return {
            "required": spec.get("required", []),
            "optional": spec.get("optional", []),
            "types":    {k: all_types[k] for k in all_keys if k in all_types},
        }

    @staticmethod
    def get_evidence_specs() -> dict:
        """Return the full loaded warranty_evidence_specs.json dict."""
        return _EVIDENCE_SPECS

    # ------------------------------------------------------------------
    # Flowchart introspection helpers (useful for the chat layer)
    # ------------------------------------------------------------------

    @staticmethod
    def node_options(node_id: str) -> list[dict]:
        """Return the options list for a given node (empty for terminals)."""
        node = _NODES.get(node_id, {})
        return node.get("options", [])

    @staticmethod
    def is_active(ticket_id: str) -> bool:
        """Return True if the ticket exists and is still in_progress."""
        ticket = WarrantyEngine.get_ticket(ticket_id)
        if ticket is None:
            return False
        return str(ticket.status) == "in_progress"

    @staticmethod
    def get_flowchart_nodes() -> dict:
        """Expose the loaded node map (read-only reference)."""
        return _NODES
