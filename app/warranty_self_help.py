"""
Format customer-safe self-help for warranty defect terminals.

Pulls from warranty_knowledge (Q&A CSV + Freshdesk tickets + Auto-Check).
"""

from __future__ import annotations

import re
from typing import Optional

from warranty_knowledge import search_knowledge

_DEFECT_CATEGORY_KEYS = frozenset({
    "power", "remote", "air", "rolling", "recline", "footrest", "cosmetic", "heat",
})

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


def find_defect_self_help(
    *,
    defect_category: Optional[str],
    path_text: str,
    model_name: str = "",
    node_id: str = "",
    turns=None,
) -> Optional[str]:
    matches = search_knowledge(
        path_text=path_text,
        defect_category=defect_category,
        model_name=model_name,
        limit=3,
    )

    fallback = _collect_fallback_hints(turns or (), node_id) if turns is not None else ()
    if node_id in _NODE_HINTS and not fallback:
        fallback = _NODE_HINTS[node_id]

    if not matches and not fallback:
        return None

    lines: list[str] = [
        "Based on similar cases, here are steps that have helped other customers:"
    ]

    step_num = 1
    for hint in fallback:
        lines.append(f"{step_num}. {hint}")
        step_num += 1

    for entry in matches:
        lines.append(f"\n**{entry.title}**")
        if entry.diagnostic and len(entry.diagnostic) < 200:
            lines.append(f"Check: {entry.diagnostic}")
        for step in entry.customer_steps[:2]:
            lines.append(f"{step_num}. {step}")
            step_num += 1

    if step_num <= 1:
        return None

    return "\n".join(lines)


def soften_terminal_prompt(prompt: str) -> str:
    lower = (prompt or "").lower()
    if re.search(
        r"\b(replace|repair or replacement|pcb|actuator|compensation|refund|technician|"
        r"send a tech|dispatch|arrange a replacement)\b",
        lower,
    ):
        return (
            "Thank you for sharing those details. Our warranty team will review your case "
            "and follow up with the next steps — we won't ask you to commit to any repair "
            "until we've reviewed it."
        )
    if "review" in lower or "follow up" in lower:
        return prompt
    return (
        "Thank you for the information. Our warranty team will review your case "
        "and follow up with you."
    )
