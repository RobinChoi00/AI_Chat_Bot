"""
Customer-safe diagnosis and self-help for warranty workflow terminals.

Pulls from warranty_knowledge (Q&A CSV + Freshdesk tickets + Auto-Check).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from warranty_knowledge import KnowledgeEntry, map_workflow_defect_category, search_knowledge

_DEFECT_CATEGORY_KEYS = frozenset({
    "power", "remote", "air", "rolling", "recline", "footrest", "cosmetic", "heat", "voice",
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
    "voice": "voice control",
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

_INSTALL_AIR_HOSE_STEPS: tuple[str, ...] = (
    "With the chair powered off, find the air hose that connects the footrest to the base of the chair.",
    "Disconnect and firmly reconnect both ends of that hose — this fixes many cases where air does not work anywhere on the chair.",
    "Make sure the hose is not pinched or trapped between the footrest and base from assembly.",
    "Power the chair back on, raise the footrest, and test leg air from the remote.",
)


def infer_install_concern_from_turns(turns) -> Optional[str]:
    for turn in reversed(list(turns or [])):
        key = str(getattr(turn, "answer_key", "") or "")
        if key in ("footrest_or_no_air", "general_setup", "other"):
            return key
    return None


def infer_voice_symptom_from_turns(turns) -> str:
    for turn in reversed(list(turns or [])):
        key = str(getattr(turn, "answer_key", "") or "")
        if key == "false_triggers":
            return "false_triggers"
        if key in ("voice_no_response", "voice_not_sure"):
            return "voice_no_response"
    return "voice_no_response"


def infer_rolling_noise_type_from_turns(turns) -> str:
    for turn in reversed(list(turns or [])):
        key = str(getattr(turn, "answer_key", "") or "")
        if key in _ROLLING_NOISE_STEPS:
            return key
    return "noise_massaging"


_VOICE_NOT_WORKING_STEPS: tuple[str, ...] = (
    "Use only the voice commands listed in your chair's manual or on-screen command list — custom phrases may not work.",
    "Speak clearly toward the built-in microphone and try from about an arm's length away.",
    "Locate the microphone on your model (often near the side panel or headrest — check your user manual).",
    "Check that side panel connections are fully seated, especially if the chair was recently installed.",
    "Power cycle the chair: turn the back switch OFF, wait 10 seconds, then turn it ON and try again.",
)

_VOICE_FALSE_TRIGGER_STEPS: tuple[str, ...] = (
    "Move the chair away from TVs, speakers, or busy conversation areas if possible.",
    "Turn off voice control in the chair settings if you do not want voice features active.",
    "Unplug the chair from the wall when you are not using it to prevent idle listening.",
    "Try lowering room noise — voice systems can react to nearby speech or TV audio.",
)

_ROLLING_NOISE_STEPS: dict[str, tuple[str, ...]] = {
    "noise_up_down": (
        "Check that the massage strap is not tangled around the mechanism head.",
        "Look for anything blocking the track path if you can safely see the mechanism area.",
        "Try one up/down cycle in manual mode and note exactly when the loud noise occurs.",
        "Record a short video of the noise while the mechanism moves — rollers visible if possible.",
    ),
    "noise_massaging": (
        "Open or remove the back pad and check that the strap is not tangled with the massage head.",
        "Inspect the back pad and backrest lining for holes, bunching, or loose material.",
        "Try manual mode and note whether the noise happens on every massage stroke or only certain areas.",
        "Record a short video during massage with the back area visible if you need team follow-up.",
    ),
    "pops": (
        "Make sure the back pad is not bunched up — zip or velcro it smoothly in place.",
        "Check the backrest lining and back pad for holes or loose material that could catch the mechanism.",
        "Try manual mode and note when the pop or click happens during the massage cycle.",
    ),
}

_FRESHDESK_PRIORITY_CATEGORIES = frozenset({"power", "remote", "mech"})


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


def _friendly_match_summary(
    matches: list[KnowledgeEntry],
    *,
    defect_category: Optional[str],
    model_name: str,
    issue_type: str,
) -> str:
    model_display = (model_name or "your chair").strip()
    preferred = next((m for m in matches if m.source in {"qa_csv", "auto_check"}), None)
    top = preferred or (matches[0] if matches else None)

    if top and top.source == "freshdesk":
        if defect_category:
            label = _CATEGORY_LABELS.get(defect_category, defect_category)
            return (
                f"Based on your answers, this looks like a **{label}** issue "
                f"with your {model_display}."
            )
        if issue_type == "delivery":
            return (
                "Based on your delivery answers, here is what we typically see "
                "in similar warranty cases."
            )
        if issue_type == "installation":
            return (
                f"For your {model_display}, here are setup tips from similar installation cases."
            )
        return (
            f"Based on your answers, here are troubleshooting steps that often help "
            f"with your {model_display}."
        )

    if top and top.title:
        return (
            f"Based on your answers and similar support cases, this looks related to "
            f"**{top.title}** on your {model_display}."
        )
    if defect_category:
        label = _CATEGORY_LABELS.get(defect_category, defect_category)
        return (
            f"Based on your answers, this appears to be a **{label}** issue "
            f"with your {model_display}."
        )
    if issue_type == "delivery":
        return (
            "Based on your delivery answers, here is what we typically see "
            "in similar warranty cases."
        )
    if issue_type == "installation":
        return (
            f"For your {model_display}, here are setup tips from similar installation cases."
        )
    return (
        "Based on what you told us and similar support history, here is our assessment."
    )


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


def build_install_air_hose_diagnosis(
    *,
    path_text: str,
    model_name: str = "",
    turns=None,
) -> dict[str, Any]:
    """DIY steps for post-install footrest / whole-chair air issues."""
    enriched_path = f"{path_text} installation footrest no air hose whole chair"
    matches: list[KnowledgeEntry] = search_knowledge(
        path_text=enriched_path,
        defect_category="air",
        model_name=model_name,
        limit=3,
    )
    steps: list[str] = list(_INSTALL_AIR_HOSE_STEPS)
    for entry in matches:
        if entry.source in ("qa_csv", "auto_check"):
            steps.extend(entry.customer_steps[:2])
    if len(steps) <= len(_INSTALL_AIR_HOSE_STEPS):
        for entry in matches:
            if entry.source == "freshdesk":
                steps.extend(entry.customer_steps[:2])
    steps = _dedupe_steps(steps)

    model_display = (model_name or "your chair").strip()
    summary = (
        f"For your **{model_display}**, footrest or whole-chair air problems after installation "
        "are often caused by the **footrest-to-base air hose** not being fully connected."
    )
    if matches:
        summary = (
            f"{summary} Similar cases in our support history suggest trying the steps below first."
        )

    return {
        "summary": summary,
        "steps": steps,
        "sources": [entry.source for entry in matches[:3]],
        "top_match": matches[0].title if matches else None,
    }


def format_install_air_hose_message(
    *,
    diagnosis: dict[str, Any],
    video_link_lines: str,
    repair_manual_url: str,
) -> str:
    parts: list[str] = [str(diagnosis.get("summary") or "").strip()]
    steps: list[str] = list(diagnosis.get("steps") or [])
    if steps:
        parts.append("\n\n**What you can try:**")
        for idx, step in enumerate(steps, start=1):
            parts.append(f"{idx}. {step}")
    if video_link_lines.strip():
        parts.append("\n\n**Installation video for your model:**")
        parts.append(video_link_lines.strip())
    parts.append(f"\n\nMore guides: [{repair_manual_url}]({repair_manual_url}).")
    parts.append(
        "\n\n**Would you like our warranty team to follow up if air still does not work after these steps?**"
    )
    return "\n".join(parts)


def build_voice_diagnosis(
    *,
    symptom: str,
    path_text: str,
    model_name: str = "",
) -> dict[str, Any]:
    """DIY steps for voice control issues (no replacement/dispatch language)."""
    base_steps = (
        _VOICE_FALSE_TRIGGER_STEPS
        if symptom == "false_triggers"
        else _VOICE_NOT_WORKING_STEPS
    )
    query = (
        f"{path_text} voice control false trigger ghost random"
        if symptom == "false_triggers"
        else f"{path_text} voice control command microphone not working"
    )
    matches: list[KnowledgeEntry] = search_knowledge(
        path_text=query,
        defect_category="voice",
        model_name=model_name,
        limit=3,
    )
    steps: list[str] = list(base_steps)
    for entry in matches:
        if entry.source in ("qa_csv", "auto_check"):
            steps.extend(entry.customer_steps[:2])
    if len(steps) <= len(base_steps):
        for entry in matches:
            if entry.source == "freshdesk":
                steps.extend(entry.customer_steps[:2])
    steps = _dedupe_steps(steps)

    model_display = (model_name or "your chair").strip()
    if symptom == "false_triggers":
        summary = (
            f"For your **{model_display}**, voice control can react to **background speech or TV audio**. "
            "Try the steps below before requesting service."
        )
    else:
        summary = (
            f"For your **{model_display}**, voice control issues are often fixed by using the "
            "**correct commands**, speaking clearly to the mic, and checking connections."
        )
    if matches:
        summary = f"{summary} Similar support cases suggest starting with the steps below."

    return {
        "summary": summary,
        "steps": steps,
        "sources": [entry.source for entry in matches[:3]],
        "top_match": matches[0].title if matches else None,
    }


def format_voice_self_help_message(*, diagnosis: dict[str, Any], repair_manual_url: str) -> str:
    parts: list[str] = [str(diagnosis.get("summary") or "").strip()]
    steps: list[str] = list(diagnosis.get("steps") or [])
    if steps:
        parts.append("\n\n**What you can try:**")
        for idx, step in enumerate(steps, start=1):
            parts.append(f"{idx}. {step}")
    parts.append(f"\n\nMore guides: [{repair_manual_url}]({repair_manual_url}).")
    parts.append(
        "\n\n**Would you like our warranty team to follow up if voice control still does not work?**"
    )
    return "\n".join(parts)


def _merge_knowledge_steps(
    *,
    steps: list[str],
    matches: list[KnowledgeEntry],
    fallback_len: int,
    defect_category: Optional[str],
) -> list[str]:
    """Merge Q&A/Freshdesk steps; prefer Freshdesk for power, remote, and mech."""
    mapped = map_workflow_defect_category(defect_category)
    prefer_freshdesk = mapped in _FRESHDESK_PRIORITY_CATEGORIES

    if prefer_freshdesk:
        for entry in matches:
            if entry.source == "freshdesk":
                steps.extend(entry.customer_steps[:2])
    for entry in matches:
        if entry.source != "freshdesk":
            steps.extend(entry.customer_steps[:2])
    if not prefer_freshdesk and len(steps) <= fallback_len:
        for entry in matches:
            if entry.source == "freshdesk":
                steps.extend(entry.customer_steps[:1])
    return steps


def build_rolling_noise_diagnosis(
    *,
    noise_type: str,
    path_text: str,
    model_name: str = "",
) -> dict[str, Any]:
    """DIY steps for massage mechanism noise before team review."""
    base_steps = _ROLLING_NOISE_STEPS.get(noise_type, _ROLLING_NOISE_STEPS["noise_massaging"])
    query = f"{path_text} massage mechanism noise rolling mech"
    matches: list[KnowledgeEntry] = search_knowledge(
        path_text=query,
        defect_category="rolling",
        model_name=model_name,
        limit=3,
    )
    steps: list[str] = list(base_steps)
    steps = _merge_knowledge_steps(
        steps=steps,
        matches=matches,
        fallback_len=len(base_steps),
        defect_category="rolling",
    )
    steps = _dedupe_steps(steps)

    model_display = (model_name or "your chair").strip()
    labels = {
        "noise_up_down": "when the mechanism moves up or down",
        "noise_massaging": "during massage",
        "pops": "popping or clicking during massage",
    }
    when = labels.get(noise_type, "with the massage mechanism")
    summary = (
        f"For your **{model_display}**, noise **{when}** is often related to the "
        "**strap, back pad, or track area**. Try the steps below first."
    )
    if matches:
        summary = f"{summary} Similar support cases suggest these checks before service."

    return {
        "summary": summary,
        "steps": steps,
        "sources": [entry.source for entry in matches[:3]],
        "top_match": matches[0].title if matches else None,
    }


def format_rolling_noise_self_help_message(*, diagnosis: dict[str, Any], repair_manual_url: str) -> str:
    parts: list[str] = [str(diagnosis.get("summary") or "").strip()]
    steps: list[str] = list(diagnosis.get("steps") or [])
    if steps:
        parts.append("\n\n**What you can try:**")
        for idx, step in enumerate(steps, start=1):
            parts.append(f"{idx}. {step}")
    parts.append(f"\n\nMore guides: [{repair_manual_url}]({repair_manual_url}).")
    parts.append(
        "\n\n**Would you like our warranty team to follow up if the noise continues after these steps?**"
    )
    return "\n".join(parts)


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
    steps = _merge_knowledge_steps(
        steps=steps,
        matches=matches,
        fallback_len=len(fallback),
        defect_category=defect_category,
    )
    steps = _dedupe_steps(steps)

    if matches:
        summary = _friendly_match_summary(
            matches,
            defect_category=defect_category,
            model_name=model_name,
            issue_type=issue_type,
        )
    else:
        model_display = (model_name or "your chair").strip()
        if defect_category:
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
