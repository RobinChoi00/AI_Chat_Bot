"""
Build customer-facing assistant messages for warranty terminal nodes.
"""

from __future__ import annotations

from typing import Any, Optional

from config import REPAIR_MANUAL_URL, WARRANTY_BUSINESS_HOURS, WARRANTY_PHONE, WARRANTY_TEAM_EMAIL
from install_videos import lookup_install_video
from warranty_self_help import (
    build_path_text,
    find_defect_self_help,
    infer_defect_category_from_turns,
)


def _contact_footer() -> str:
    return (
        f"Warranty team: {WARRANTY_PHONE} · {WARRANTY_TEAM_EMAIL}\n"
        f"Hours: {WARRANTY_BUSINESS_HOURS}"
    )


def _install_message(model_name: str, base_prompt: str) -> dict[str, Any]:
    video = lookup_install_video(model_name)
    url = video["url"]
    label = video["label"]
    model_display = (model_name or "your chair").strip()

    message = (
        f"{base_prompt}\n\n"
        f"Installation video for **{model_display}**:\n"
        f"[Watch installation video — {label}]({url})\n\n"
        f"You can also browse guides at [{REPAIR_MANUAL_URL}]({REPAIR_MANUAL_URL}).\n\n"
        f"If you still need help after watching, tap **I still need help** below.\n\n"
        f"{_contact_footer()}"
    )
    return {
        "message": message,
        "install_video": {"url": url, "label": label},
        "show_contact_form": False,
        "defer_email": True,
    }


def _defect_message(
    engine,
    ticket_id: str,
    ticket,
    node: dict,
    base_prompt: str,
    evidence_required: list[str],
) -> dict[str, Any]:
    turns = engine.get_turns(ticket_id)
    category = infer_defect_category_from_turns(turns)
    path_text = build_path_text(turns)
    self_help = find_defect_self_help(
        defect_category=category,
        path_text=path_text,
        model_name=str(getattr(ticket, "model_name", "") or ""),
    )

    parts = [base_prompt]
    if self_help:
        parts.append(f"\n\n{self_help}")
    elif evidence_required:
        parts.append(
            "\n\nIf you still need our team's help, leave your email below — "
            "photos and videos are optional."
        )
    else:
        parts.append(
            "\n\nIf you still need our team's help, leave your email below."
        )

    parts.append(f"\n\n{_contact_footer()}")

    return {
        "message": "".join(parts),
        "self_help": self_help,
        "show_contact_form": True,
        "defer_email": False,
    }


def build_terminal_enrichment(
    engine,
    ticket,
    node: Optional[dict],
) -> Optional[dict[str, Any]]:
    if not node or node.get("type") != "terminal":
        return None

    node_id = str(node.get("node_id") or "")
    base_prompt = str(node.get("prompt") or "").strip()
    if not base_prompt:
        return None

    ticket_id = str(getattr(ticket, "ticket_id", "") or "")
    issue_type = str(getattr(ticket, "issue_type", "") or "").lower()
    evidence_required = list(node.get("evidence_required") or [])
    action = str(node.get("action") or "")

    if node_id == "install_send_video" or (
        issue_type == "installation" and action == "send_info"
    ):
        model_name = str(getattr(ticket, "model_name", "") or "")
        return _install_message(model_name, base_prompt)

    if issue_type == "defect":
        result = _defect_message(
            engine,
            ticket_id,
            ticket,
            node,
            base_prompt,
            evidence_required,
        )
        if action == "send_info":
            result["show_contact_form"] = False
            result["defer_email"] = True
            if "still need help" not in result["message"].lower():
                result["message"] = (
                    f"{result['message']}\n\n"
                    "If the issue continues, tap **I still need help** below."
                )
        return result

    if action == "send_info":
        return {
            "message": f"{base_prompt}\n\n{_contact_footer()}",
            "show_contact_form": False,
            "defer_email": True,
        }

    if evidence_required:
        message = (
            f"{base_prompt}\n\n"
            "If you still need our team's help, leave your email below — "
            "photos and videos are optional.\n\n"
            f"{_contact_footer()}"
        )
    else:
        message = f"{base_prompt}\n\n{_contact_footer()}"

    return {
        "message": message,
        "show_contact_form": bool(evidence_required),
        "defer_email": False,
    }
