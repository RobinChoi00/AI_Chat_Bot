#!/usr/bin/env python3
"""
validate_flowchart.py  (v2)
===========================
Validates data/warranty_flowchart.json for structural integrity.

Checks
------
 1. All active nodes are reachable from the root (BFS).
 2. No active node is isolated (must have ≥1 incoming OR be the root).
 3. All question / instruction / question_text nodes have at least one valid next-node reference.
 4. Every yes/no (option) branch points to an existing node.
 5. Every terminal node declares an 'action' field (serves as terminal_type).
 6. Every terminal that implies replacement / compensation / tech dispatch / refund /
    warranty approval is marked action='awaiting_admin'.
 7. Every evidence_required item on a terminal references a defined evidence spec.
    Every terminal_evidence_map key in warranty_evidence_specs.json maps to a real terminal.

Report
------
  - Total nodes
  - Total active (non-draft) nodes
  - Draft / needs_review nodes  (status field not present in current flowchart → 0)
  - Terminal nodes
  - Validation errors (hard failures)
  - Validation warnings (soft, review recommended)
  - Branches needing manual review

Usage
-----
    python script/validate_flowchart.py                      # default paths
    python script/validate_flowchart.py data/my_chart.json  # override flowchart
"""

from __future__ import annotations

import json
import re
import sys
from collections import deque, defaultdict
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
GREEN  = lambda t: _c("32;1", t)
CYAN   = lambda t: _c("36;1", t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
FLOWCHART_PATH     = BASE_DIR / "data" / "warranty_flowchart.json"
EVIDENCE_SPECS_PATH = BASE_DIR / "data" / "warranty_evidence_specs.json"

# Keywords in internal_note that imply an admin decision is required.
# If any appear in a terminal whose action is NOT 'awaiting_admin', we flag it.
_ADMIN_KEYWORDS: tuple[str, ...] = (
    "replacement",
    "replace ",
    "send tech",
    "technician",
    "tech needed",
    "send a tech",
    "tech to ",
    "refund",
    "compensation",
    "carrier claim",
    "file claim",
    "admin to ",
    "admin action",
    "warranty approval",
    "send new ",
    "send replacement",
    "replace remote",
    "replace pcb",
    "replace main",
    "replace actuator",
)

# Actions that are permitted to NOT be awaiting_admin even if note contains admin keywords.
# (These explicitly send info/instructions and are safe as non-admin gates.)
_SAFE_NON_ADMIN_ACTIONS: frozenset[str] = frozenset({"send_info", "sales_handoff"})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_children(node: dict) -> list[str]:
    """Return all next-node IDs referenced by a node (options or direct 'next')."""
    children: list[str] = []
    for opt in node.get("options", []):
        if "next" in opt:
            children.append(opt["next"])
    if "next" in node:
        children.append(node["next"])
    return children


def build_incoming_map(nodes: dict) -> dict[str, list[str]]:
    """For each node, list the node IDs that reference it as a child."""
    incoming: dict[str, list[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        for child_id in get_children(node):
            incoming[child_id].append(node_id)
    return incoming


def is_draft(node: dict) -> bool:
    """Nodes with status='draft' or 'needs_review' are excluded from active count."""
    return node.get("status", "active") in ("draft", "needs_review")


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 + 2: Reachability and Isolation
# ─────────────────────────────────────────────────────────────────────────────

def check_reachability(nodes: dict, root: str) -> tuple[list[str], list[str]]:
    """
    Returns:
        errors   – nodes that exist but are unreachable from root (hard failure)
        warnings – nodes that have no incoming edges and are not the root
    """
    errors: list[str] = []
    warnings: list[str] = []

    # BFS
    visited: set[str] = set()
    queue: deque[str] = deque([root])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for child in get_children(nodes.get(current, {})):
            if child not in visited:
                queue.append(child)

    unreachable = set(nodes.keys()) - visited
    if unreachable:
        for nid in sorted(unreachable):
            errors.append(f"[CHECK 1] Unreachable node: '{nid}'")

    # Check 2: isolation (unreachable already covers this, but also flag sink-only nodes)
    incoming = build_incoming_map(nodes)
    for node_id in nodes:
        if node_id == root:
            continue
        if not incoming.get(node_id) and node_id not in unreachable:
            warnings.append(f"[CHECK 2] No incoming edges for node '{node_id}' (possible island)")

    return errors, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 + 4: Question next-nodes and branch validity
# ─────────────────────────────────────────────────────────────────────────────

def check_edges(nodes: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    question_types = {"question", "instruction", "question_text"}

    for node_id, node in nodes.items():
        node_type = node.get("type", "unknown")
        children = get_children(node)

        # Check 3: non-terminal question nodes must have ≥1 outgoing edge
        if node_type in question_types and not children:
            errors.append(
                f"[CHECK 3] {node_type} node '{node_id}' has NO outgoing edges"
            )

        # Check 4: every branch target must exist
        for opt in node.get("options", []):
            target = opt.get("next")
            if target and target not in nodes:
                errors.append(
                    f"[CHECK 4] Node '{node_id}' option '{opt.get('answer_key', '?')}' "
                    f"→ unknown node '{target}'"
                )
        if "next" in node and node["next"] not in nodes:
            errors.append(
                f"[CHECK 4] Node '{node_id}' direct next → unknown node '{node['next']}'"
            )

        # Terminal nodes must not have outgoing edges
        if node_type == "terminal" and children:
            errors.append(
                f"[CHECK 4] Terminal node '{node_id}' has outgoing edges: {children}"
            )

    return errors, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: Terminal nodes must declare an action (terminal_type)
# ─────────────────────────────────────────────────────────────────────────────

def check_terminal_action(nodes: dict) -> list[str]:
    errors: list[str] = []
    known_actions = {"awaiting_admin", "send_info", "request_evidence", "sales_handoff", "awaiting_evidence"}
    for node_id, node in nodes.items():
        if node.get("type") != "terminal":
            continue
        action = node.get("action")
        if not action:
            errors.append(
                f"[CHECK 5] Terminal '{node_id}' is missing 'action' (terminal_type)"
            )
        elif action not in known_actions:
            errors.append(
                f"[CHECK 5] Terminal '{node_id}' has unknown action='{action}' "
                f"(expected one of: {sorted(known_actions)})"
            )
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: Admin-gate enforcement
# ─────────────────────────────────────────────────────────────────────────────

def check_admin_gate(nodes: dict) -> tuple[list[str], list[str]]:
    """
    Flag terminals whose internal_note implies an admin-only action (replacement,
    tech dispatch, refund, compensation, etc.) but whose action is NOT awaiting_admin.
    """
    errors: list[str] = []
    manual_review: list[str] = []

    for node_id, node in nodes.items():
        if node.get("type") != "terminal":
            continue
        action = node.get("action", "")
        if action == "awaiting_admin":
            continue  # correctly gated

        internal_note = (node.get("internal_note") or "").lower()
        matched = [kw for kw in _ADMIN_KEYWORDS if kw.lower() in internal_note]

        if not matched:
            continue

        if action in _SAFE_NON_ADMIN_ACTIONS:
            # send_info + admin keywords is a soft flag (DIY instructions sometimes
            # mention the admin backup path)
            manual_review.append(
                f"[CHECK 6] Terminal '{node_id}' (action={action}) has admin keywords "
                f"in internal_note {matched} — verify this is truly self-service"
            )
        else:
            # request_evidence without awaiting_admin but mentioning admin actions
            errors.append(
                f"[CHECK 6] Terminal '{node_id}' (action={action}) implies admin action "
                f"({matched}) but is NOT marked awaiting_admin"
            )

    return errors, manual_review


# ─────────────────────────────────────────────────────────────────────────────
# Check 7: Evidence references
# ─────────────────────────────────────────────────────────────────────────────

def check_evidence(nodes: dict, specs: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    defined_types = set(specs.get("evidence_types", {}).keys())
    terminal_map: dict = specs.get("terminal_evidence_map", {})
    flowchart_terminals = {nid for nid, n in nodes.items() if n.get("type") == "terminal"}

    # 7a. Every evidence_required on a terminal node must reference a defined type
    for node_id, node in nodes.items():
        if node.get("type") != "terminal":
            continue
        for ev_key in node.get("evidence_required", []):
            if ev_key not in defined_types:
                errors.append(
                    f"[CHECK 7] Terminal '{node_id}' evidence_required='{ev_key}' "
                    f"is not defined in warranty_evidence_specs.json"
                )

    # 7b. Every entry in terminal_evidence_map must reference a real terminal node
    for map_id, ev_cfg in terminal_map.items():
        if map_id not in nodes:
            errors.append(
                f"[CHECK 7] terminal_evidence_map key '{map_id}' not found in flowchart nodes"
            )
        elif nodes[map_id].get("type") != "terminal":
            errors.append(
                f"[CHECK 7] terminal_evidence_map key '{map_id}' is not a terminal "
                f"(type={nodes[map_id].get('type')})"
            )
        for ev_key in ev_cfg.get("required", []) + ev_cfg.get("optional", []):
            if ev_key not in defined_types:
                errors.append(
                    f"[CHECK 7] terminal_evidence_map['{map_id}'] references "
                    f"undefined evidence type '{ev_key}'"
                )

    # 7c. Every flowchart terminal should appear in terminal_evidence_map (WARNING only)
    unmapped = flowchart_terminals - set(terminal_map.keys())
    for nid in sorted(unmapped):
        action = nodes[nid].get("action", "?")
        if action not in ("sales_handoff",):  # sales_handoff legitimately needs no evidence
            warnings.append(
                f"[CHECK 7] Terminal '{nid}' (action={action}) has no entry in "
                f"terminal_evidence_map — add an explicit mapping even if empty"
            )

    return errors, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def build_stats(nodes: dict, specs: dict | None) -> dict:
    terminal_map = (specs or {}).get("terminal_evidence_map", {}) if specs else {}

    type_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    draft_nodes: list[str] = []

    for nid, node in nodes.items():
        t = node.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        if node.get("type") == "terminal":
            a = node.get("action", "unknown")
            action_counts[a] = action_counts.get(a, 0) + 1
        if is_draft(node):
            draft_nodes.append(nid)

    terminals = [(nid, n) for nid, n in nodes.items() if n.get("type") == "terminal"]
    active_count = sum(1 for n in nodes.values() if not is_draft(n))

    return {
        "total_nodes":     len(nodes),
        "active_nodes":    active_count,
        "draft_nodes":     draft_nodes,
        "terminal_count":  len(terminals),
        "type_counts":     type_counts,
        "action_counts":   action_counts,
        "terminals":       terminals,
        "evidence_types":  len((specs or {}).get("evidence_types", {})),
        "terminal_mapped": len(terminal_map),
    }


def print_report(
    flowchart: dict,
    specs: dict | None,
    errors: list[str],
    warnings: list[str],
    manual_review: list[str],
) -> None:
    nodes = flowchart.get("nodes", {})
    stats = build_stats(nodes, specs)

    sep = "─" * 60

    print()
    print(BOLD("━" * 60))
    print(BOLD("  WARRANTY FLOWCHART VALIDATION REPORT"))
    print(BOLD("━" * 60))
    print(f"  Source  : {flowchart.get('source', '?')}")
    print(f"  Version : {flowchart.get('version', '?')}")
    print(f"  Root    : {flowchart.get('root', '?')}")
    print()

    # ── Node counts ──────────────────────────────────────────────────────────
    print(CYAN(BOLD("  NODE SUMMARY")))
    print(f"  {sep}")
    print(f"  Total nodes             : {BOLD(str(stats['total_nodes']))}")
    print(f"  Active nodes            : {BOLD(str(stats['active_nodes']))}")
    print(f"  Draft / needs_review    : {BOLD(str(len(stats['draft_nodes'])))}", end="")
    if stats["draft_nodes"]:
        print(f"  → {DIM(', '.join(stats['draft_nodes']))}", end="")
    print()
    print(f"  Terminal nodes          : {BOLD(str(stats['terminal_count']))}")
    print()

    print(f"  By type:")
    for t, cnt in sorted(stats["type_counts"].items()):
        print(f"    {t:<22}: {cnt}")
    print()

    print(f"  Terminal actions:")
    for a, cnt in sorted(stats["action_counts"].items()):
        marker = "⚠" if a not in ("awaiting_admin", "send_info", "request_evidence", "sales_handoff") else " "
        print(f"    {marker} {a:<22}: {cnt}")

    if specs:
        print()
        print(f"  Evidence spec types     : {stats['evidence_types']}")
        print(f"  Terminal evidence map   : {stats['terminal_mapped']} entries")

    # ── Validation results ────────────────────────────────────────────────────
    print()
    print(CYAN(BOLD("  VALIDATION RESULTS")))
    print(f"  {sep}")

    if not errors and not warnings and not manual_review:
        print(f"  {GREEN('✅  All checks passed — flowchart is valid!')}")
    else:
        if errors:
            print(f"  {RED(f'❌  {len(errors)} error(s) found (must fix before deploy):')}")
            for e in errors:
                print(f"    • {RED(e)}")
            print()

        if warnings:
            print(f"  {YELLOW(f'⚠️   {len(warnings)} warning(s):')}")
            for w in warnings:
                print(f"    • {YELLOW(w)}")
            print()

        if not errors and not warnings:
            print(f"  {GREEN('✅  No hard errors or warnings.')}")

    # ── Manual review ─────────────────────────────────────────────────────────
    if manual_review:
        print()
        print(CYAN(BOLD("  BRANCHES NEEDING MANUAL REVIEW")))
        print(f"  {sep}")
        for item in manual_review:
            print(f"    • {item}")

    # ── Terminal node table ───────────────────────────────────────────────────
    print()
    print(CYAN(BOLD("  TERMINAL NODES")))
    print(f"  {sep}")
    for nid, node in sorted(stats["terminals"]):
        action = node.get("action", "?")
        ev = node.get("evidence_required", [])
        note_preview = (node.get("internal_note") or "")[:60].rstrip()
        if len(node.get("internal_note") or "") > 60:
            note_preview += "…"
        colour = RED if action not in ("awaiting_admin", "send_info", "request_evidence",
                                       "sales_handoff", "awaiting_evidence") else DIM
        print(f"    [{colour(action):<16}] {nid}")
        if ev:
            print(f"                       evidence: {ev}")
    print()
    print(BOLD("━" * 60))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    flowchart_path = Path(sys.argv[1]) if len(sys.argv) > 1 else FLOWCHART_PATH
    evidence_path  = EVIDENCE_SPECS_PATH

    print(f"Flowchart : {flowchart_path}")
    print(f"Evidence  : {evidence_path}")

    flowchart = load_json(flowchart_path)
    nodes: dict = flowchart.get("nodes", {})
    root: str   = flowchart.get("root", "root")

    if root not in nodes:
        print(RED(f"\n❌  FATAL: root node '{root}' not found in nodes!"))
        sys.exit(1)

    specs: dict | None = None
    if evidence_path.exists():
        specs = load_json(evidence_path)
    else:
        print(f"\n  [skip] Evidence specs not found at {evidence_path}")

    # Run all checks
    errors:        list[str] = []
    warnings:      list[str] = []
    manual_review: list[str] = []

    # 1 + 2: reachability / isolation
    e, w = check_reachability(nodes, root)
    errors.extend(e); warnings.extend(w)

    # 3 + 4: edge validity
    e, w = check_edges(nodes)
    errors.extend(e); warnings.extend(w)

    # 5: terminal action field
    errors.extend(check_terminal_action(nodes))

    # 6: admin gate
    e, mr = check_admin_gate(nodes)
    errors.extend(e); manual_review.extend(mr)

    # 7: evidence references
    if specs is not None:
        e, w = check_evidence(nodes, specs)
        errors.extend(e); warnings.extend(w)

    print_report(flowchart, specs, errors, warnings, manual_review)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
