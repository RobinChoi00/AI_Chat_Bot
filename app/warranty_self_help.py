"""
Customer-safe diagnosis and self-help for warranty workflow terminals.

Pulls from warranty_knowledge (Q&A CSV + Freshdesk tickets + Auto-Check).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from warranty_knowledge import KnowledgeEntry, search_knowledge

_DEFECT_CATEGORY_KEYS = frozenset({
    "power", "remote", "air", "rolling", "recline", "footrest", "cosmetic", "heat",
})

_CATEGORY_LABELS: dict[str, str] = {
    "power": "power",
    "remote": "remote / controller",
    "air": "air inflation",
    "rolling": "massage mechanism",
    "recline": "recline / position",
    "footrest": "footrest",
    "cosmetic": "cosmetic",
    "heat": "heating",
}

_NODE_HINTS: dict[str, tuple[str, ...]] = {
    "defect_power_main_pcb_terminal": (
        "Confirm the power cord is firmly plugged into the chair and wall outlet.",
        "Try a different wall outlet or test the outlet with another device.",
        "Check whether the chair's fuse is intact (refer to your manual for fuse location).",
        "Toggle the back power switch OFF, wait 10 seconds, then ON — note any clicking sound.",
    ),
    "defect_power_clicking_terminal": (
        "Toggle the back power switch OFF and ON — listen for a clicking sound.",
        "Try the side panel buttons to see if the chair responds without the remote.",
    ),
    "defect_power_no_click_terminal": (
        "Verify the power cord and outlet, then check the chair fuse.",
        "Toggle the back power switch OFF and ON again.",
    ),
    "defect_power_pcb_fuse_terminal": (
        "Double-check the power cord at both the wall and the chair.",
        "Inspect the fuse on the chair if you can access it safely.",
    ),
    "defect_air_hose_fix_terminal": (
        "Check air hoses in the affected area for kinks or loose connections.",
        "Reconnect any disconnected hoses securely.",
    ),
    "defect_air_side_reconnect_terminal": (
        "Reconnect the air hose from the base to the side panel fitting securely.",
    ),
    "defect_air_base_hose_terminal": (
        "Inspect base air hose connections for kinks or loose fittings.",
    ),
}

_ANSWER_KEY_HINTS: dict[str, tuple[str, ...]] = {
    "back_switch_sound": (
        "Note exactly what you hear when toggling the back switch (click, hum, or other).",
        "Confirm the power cord and fuse are OK before contacting support.",
    ),
    "clicking_sound": (
        "A clicking sound when toggling the back switch often points to a remote connection issue — try reseating the remote cable if accessible.",
    ),
    "fuse_blown": (
        "If the fuse appears blown, do not force the chair on — note the fuse condition for our team.",
    ),
}

HELP_OFFER_OPTIONS: tuple[dict[str, str], ...] = (
    {"answer_key": "yes_team_help", "label": "Yes, please help me"},
    {"answer_key": "no_self_help", "label": "I'll try these steps on my own"},
)


def infer_defect_category_from_turns(turns) -> Optional[str]:
    for turn in turns:
        key = str(getattr(turn, "answer_key", "") or "")
        if key in _DEFECT_CATEGORY_KEYS:
            return key
    return None


def build_path_text(turns) -> str:
    parts: list[str] = []
    for turn in turns:
        for attr in ("customer_answer", "node_prompt", "node_id"):
            val = str(getattr(turn, attr, "") or "").strip()
            if val:
                parts.append(val)
    return " ".join(parts)


def _collect_fallback_hints(turns, node_id: str) -> tuple[str, ...]:
    hints: list[str] = []
    if node_id in _NODE_HINTS:
        hints.extend(_NODE_HINTS[node_id])
    for turn in turns:
        key = str(getattr(turn, "answer_key", "") or "")
        if key in _ANSWER_KEY_HINTS:
            hints.extend(_ANSWER_KEY_HINTS[key])
    seen: set[str] = set()
    unique: list[str] = []
    for hint in hints:
        norm = hint.lower()
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(hint)
    return tuple(unique[:4])


def _dedupe_steps(steps: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for step in steps:
        key = step.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(step)
    return out[:5]


def build_workflow_diagnosis(
    *,
    defect_category: Optional[str],
    path_text: str,
    model_name: str = "",
    node_id: str = "",
    turns=None,
    issue_type: str = "",
) -> dict[str, Any]:
    """
    Build structured diagnosis from Freshdesk/Q&A/Auto-Check + workflow context.
    """
    matches: list[KnowledgeEntry] = search_knowledge(
        path_text=path_text,
        defect_category=defect_category,
        model_name=model_name,
        limit=3,
    )
    fallback = _collect_fallback_hints(turns or (), node_id) if turns is not None else ()
    if node_id in _NODE_HINTS and not fallback:
        fallback = _NODE_HINTS[node_id]

    steps: list[str] = list(fallback)
    for entry in matches:
        steps.extend(entry.customer_steps[:2])
    steps = _dedupe_steps(steps)

    model_display = (model_name or "your chair").strip()
    if matches:
        summary = (
            f"Based on your answers and similar support cases, this looks related to "
            f"**{matches[0].title}** on your {model_display}."
        )
    elif defect_category:
        label = _CATEGORY_LABELS.get(defect_category, defect_category)
        summary = (
            f"Based on your answers, this appears to be a **{label}** issue "
            f"with your {model_display}."
        )
    elif issue_type == "delivery":
        summary = (
            "Based on your delivery answers, here is what we typically see "
            "in similar warranty cases."
        )
    elif issue_type == "installation":
        summary = (
            f"For your {model_display}, here are setup tips from similar installation cases."
        )
    else:
        summary = (
            "Based on what you told us and similar support history, here is our assessment."
        )

    return {
        "summary": summary,
        "steps": steps,
        "sources": [entry.source for entry in matches[:3]],
        "top_match": matches[0].title if matches else None,
    }


def format_diagnosis_message(diagnosis: dict[str, Any]) -> str:
    """Format diagnosis dict into customer-facing chat text (before help offer)."""
    parts: list[str] = [str(diagnosis.get("summary") or "").strip()]
    steps: list[str] = list(diagnosis.get("steps") or [])
    if steps:
        parts.append("\n\n**What you can try:**")
        for idx, step in enumerate(steps, start=1):
            parts.append(f"{idx}. {step}")
    else:
        parts.append(
            "\n\nWe couldn't find a specific DIY fix in our records, but our team can still review your case."
        )
    parts.append(
        "\n\n**Would you like our warranty team to follow up and help you resolve this?**"
    )
    return "\n".join(parts)


def soften_terminal_prompt(prompt: str) -> str:
    lower = (prompt or "").lower()
    if re.search(
        r"\b(replace|repair or replacement|pcb|actuator|compensation|refund|technician|"
        r"send a tech|dispatch|arrange a replacement)\b",
        lower,
    ):
        return (
            "We've noted everything from your workflow answers. If you'd like our team involved, "
            "they will review your case and follow up with next steps."
        )
    return ""
