"""
Build customer-facing assistant messages for warranty terminal nodes.

Flow after workflow ends:
  1) Diagnosis + DIY steps (Freshdesk / Q&A / Auto-Check)
  2) Ask whether the customer wants team follow-up
  3) Only after Yes → email / optional photos (handled in frontend)
"""

from __future__ import annotations

from typing import Any, Optional

from config import REPAIR_MANUAL_URL, WARRANTY_BUSINESS_HOURS, WARRANTY_PHONE, WARRANTY_TEAM_EMAIL
from install_videos import lookup_install_video
from warranty_self_help import (
    HELP_OFFER_OPTIONS,
    build_install_air_hose_diagnosis,
    build_path_text,
    build_voice_diagnosis,
    build_workflow_diagnosis,
    format_diagnosis_message,
    format_install_air_hose_message,
    format_voice_self_help_message,
    infer_defect_category_from_turns,
    infer_voice_symptom_from_turns,
)


def _contact_footer() -> str:
    return (
        f"Warranty team: {WARRANTY_PHONE} · {WARRANTY_TEAM_EMAIL}\n"
        f"Hours: {WARRANTY_BUSINESS_HOURS}"
    )


def _help_offer_enrichment(message: str, diagnosis: Optional[dict] = None) -> dict[str, Any]:
    return {
        "message": f"{message}\n\n{_contact_footer()}",
        "diagnosis": diagnosis,
        "phase": "awaiting_help_consent",
        "help_offer_options": list(HELP_OFFER_OPTIONS),
        "show_contact_form": False,
        "defer_email": True,
    }


def _install_message(model_name: str, base_prompt: str) -> dict[str, Any]:
    video = lookup_install_video(model_name)
    model_display = (model_name or "your chair").strip()
    clips = video.get("videos") or [{"url": video["url"], "label": video["label"]}]
    link_lines = "\n".join(
        f"[Watch — {clip['label']}]({clip['url']})" for clip in clips
    )

    body = (
        f"Here is the installation guide for your **{model_display}**:\n"
        f"{link_lines}\n\n"
        f"If **air does not work anywhere on the chair** after setup, check that the "
        f"**air hose between the footrest and base** is firmly connected.\n\n"
        f"More guides: [{REPAIR_MANUAL_URL}]({REPAIR_MANUAL_URL}).\n\n"
        f"**Would you like our warranty team to follow up if you still need help after watching?**"
    )
    return _help_offer_enrichment(body)


def _install_air_hose_message(engine, ticket_id: str, ticket) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    diagnosis = build_install_air_hose_diagnosis(
        path_text=path_text,
        model_name=model_name,
        turns=turns,
    )
    video = lookup_install_video(model_name)
    clips = video.get("videos") or [{"url": video["url"], "label": video["label"]}]
    link_lines = "\n".join(
        f"[Watch — {clip['label']}]({clip['url']})" for clip in clips
    )
    body = format_install_air_hose_message(
        diagnosis=diagnosis,
        video_link_lines=link_lines,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _voice_self_help_message(engine, ticket_id: str, ticket, *, false_triggers: bool) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = "false_triggers" if false_triggers else "voice_no_response"
    if not false_triggers:
        symptom = infer_voice_symptom_from_turns(turns)
    diagnosis = build_voice_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_voice_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _workflow_end_message(
    engine,
    ticket_id: str,
    ticket,
    node: dict,
) -> dict[str, Any]:
    node_id = str(node.get("node_id") or "")
    turns = engine.get_turns(ticket_id)
    issue_type = str(getattr(ticket, "issue_type", "") or "").lower()
    category = infer_defect_category_from_turns(turns)
    path_text = build_path_text(turns)
    model_name = str(getattr(ticket, "model_name", "") or "")

    diagnosis = build_workflow_diagnosis(
        defect_category=category,
        path_text=path_text,
        model_name=model_name,
        node_id=node_id,
        turns=turns,
        issue_type=issue_type,
    )
    message = format_diagnosis_message(diagnosis)
    return _help_offer_enrichment(message, diagnosis=diagnosis)


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
    action = str(node.get("action") or "")

    if node_id == "install_air_hose_terminal":
        return _install_air_hose_message(engine, ticket_id, ticket)

    if node_id == "defect_voice_not_working_terminal":
        return _voice_self_help_message(engine, ticket_id, ticket, false_triggers=False)

    if node_id == "defect_voice_false_triggers_terminal":
        return _voice_self_help_message(engine, ticket_id, ticket, false_triggers=True)

    if node_id == "install_send_video" or (
        issue_type == "installation" and action == "send_info"
    ):
        model_name = str(getattr(ticket, "model_name", "") or "")
        return _install_message(model_name, base_prompt)

    return _workflow_end_message(engine, ticket_id, ticket, node)
