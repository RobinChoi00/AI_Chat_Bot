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
    build_admin_review_diagnosis,
    build_air_diagnosis,
    build_cosmetic_diagnosis,
    build_delivery_diagnosis,
    build_footrest_diagnosis,
    build_heating_diagnosis,
    build_install_air_hose_diagnosis,
    build_path_text,
    build_power_diagnosis,
    build_recline_diagnosis,
    build_remote_diagnosis,
    build_rolling_noise_diagnosis,
    build_voice_diagnosis,
    build_workflow_diagnosis,
    format_admin_review_message,
    format_air_self_help_message,
    format_cosmetic_self_help_message,
    format_delivery_self_help_message,
    format_diagnosis_message,
    format_footrest_self_help_message,
    format_heating_self_help_message,
    format_install_air_hose_message,
    format_power_self_help_message,
    format_recline_self_help_message,
    format_remote_self_help_message,
    format_rolling_noise_self_help_message,
    format_voice_self_help_message,
    infer_air_symptom_from_turns,
    infer_cosmetic_symptom_from_turns,
    infer_defect_category_from_turns,
    infer_delivery_symptom_from_turns,
    infer_footrest_symptom_from_turns,
    infer_heating_symptom_from_turns,
    infer_power_symptom_from_turns,
    infer_recline_symptom_from_turns,
    infer_remote_symptom_from_turns,
    infer_rolling_noise_type_from_turns,
    infer_voice_symptom_from_turns,
)

_ROLLING_NOISE_TERMINALS = frozenset({
    "defect_rolling_noise_updown_terminal",
    "defect_rolling_noise_massage_terminal",
    "defect_rolling_pops_terminal",
})

_REMOTE_TERMINALS = frozenset({
    "defect_remote_blank_screen_terminal",
    "defect_remote_cable_terminal",
    "defect_remote_partial_terminal",
    "defect_remote_fuse_terminal",
    "defect_remote_connection_terminal",
    "defect_remote_intermittent_terminal",
    "defect_remote_pcb_check_terminal",
})

_POWER_TERMINALS = frozenset({
    "defect_power_remote_replace_terminal",
    "defect_power_main_pcb_terminal",
    "defect_power_actuator_terminal",
    "defect_power_main_pcb_wire_terminal",
    "defect_power_pcb_fuse_terminal",
    "defect_power_clicking_terminal",
    "defect_power_no_click_terminal",
})

_AIR_TERMINALS = frozenset({
    "defect_air_hose_fix_terminal",
    "defect_air_tech_terminal",
    "defect_air_pump_terminal",
    "defect_air_arms_tech_terminal",
    "defect_air_shoulders_tech_terminal",
    "defect_air_footrest_wg_tech_terminal",
    "defect_air_side_wg_tech_terminal",
    "defect_air_side_reconnect_terminal",
    "defect_air_base_wg_tech_terminal",
    "defect_air_base_hose_terminal",
})

_FOOTREST_TERMINALS = frozenset({
    "defect_footrest_extend_terminal",
    "defect_footrest_foot_rollers_terminal",
    "defect_footrest_calf_roller_terminal",
})

_COSMETIC_TERMINALS = frozenset({
    "defect_cosmetic_photo_terminal",
    "defect_cosmetic_side_fixed_terminal",
    "defect_cosmetic_wg_terminal",
    "defect_cosmetic_signed_cleared_terminal",
    "defect_cosmetic_box_photos_terminal",
    "defect_cosmetic_replace_terminal",
})

_RECLINE_TERMINALS = frozenset({
    "defect_recline_actuator_terminal",
    "defect_recline_main_pcb_wire_terminal",
    "defect_recline_main_pcb_terminal",
})

_HEATING_TERMINALS = frozenset({
    "defect_heating_not_heating_terminal",
    "defect_heating_intermittent_terminal",
    "defect_heating_too_hot_terminal",
})

_DELIVERY_TERMINALS = frozenset({
    "delivery_missing_parts_terminal",
    "delivery_wrong_item_terminal",
    "delivery_never_arrived_terminal",
    "delivery_late_terminal",
    "delivery_other_problem_terminal",
    "delivery_signed_cleared_terminal",
    "delivery_replace_claim_terminal",
    "delivery_minor_comp_terminal",
})


def _contact_footer() -> str:
    return (
        f"Warranty team: {WARRANTY_PHONE} · {WARRANTY_TEAM_EMAIL}\n"
        f"Hours: {WARRANTY_BUSINESS_HOURS}"
    )


def _help_offer_enrichment(
    message: str,
    diagnosis: Optional[dict] = None,
    *,
    interaction_mode: Optional[str] = None,
) -> dict[str, Any]:
    if interaction_mode not in {"troubleshooting", "preparation"}:
        interaction_mode = (
            "preparation" if "What to prepare" in message else "troubleshooting"
        )
    return {
        "message": f"{message}\n\n{_contact_footer()}",
        "diagnosis": diagnosis,
        "phase": "awaiting_help_consent",
        "interaction_mode": interaction_mode,
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
        f"**Please watch the guide and try the setup checks first. Then use the check below to tell us whether the issue is resolved.**"
    )
    return _help_offer_enrichment(body, interaction_mode="troubleshooting")


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


def _rolling_noise_message(engine, ticket_id: str, ticket) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    noise_type = infer_rolling_noise_type_from_turns(turns)
    diagnosis = build_rolling_noise_diagnosis(
        noise_type=noise_type,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_rolling_noise_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _remote_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_remote_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_remote_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_remote_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _power_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_power_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_power_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_power_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _air_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_air_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_air_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_air_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _footrest_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_footrest_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_footrest_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_footrest_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _cosmetic_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_cosmetic_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_cosmetic_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_cosmetic_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _recline_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_recline_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_recline_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_recline_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _heating_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_heating_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_heating_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_heating_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
    return _help_offer_enrichment(body, diagnosis=diagnosis)


def _delivery_self_help_message(engine, ticket_id: str, ticket, node_id: str) -> dict[str, Any]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    turns = engine.get_turns(ticket_id)
    path_text = build_path_text(turns)
    symptom = infer_delivery_symptom_from_turns(turns, node_id=node_id)
    diagnosis = build_delivery_diagnosis(
        symptom=symptom,
        path_text=path_text,
        model_name=model_name,
    )
    body = format_delivery_self_help_message(
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


def _admin_review_terminal_message(
    engine,
    ticket_id: str,
    ticket,
    node: dict,
) -> dict[str, Any]:
    node_id = str(node.get("node_id") or "")
    base_prompt = str(node.get("prompt") or "").strip()
    evidence_required = list(node.get("evidence_required") or [])
    turns = engine.get_turns(ticket_id)
    category = infer_defect_category_from_turns(turns)
    model_name = str(getattr(ticket, "model_name", "") or "")

    diagnosis = build_admin_review_diagnosis(
        base_prompt=base_prompt,
        evidence_required=evidence_required,
        node_id=node_id,
        turns=turns,
        defect_category=category,
        model_name=model_name,
    )
    message = format_admin_review_message(
        diagnosis=diagnosis,
        repair_manual_url=REPAIR_MANUAL_URL,
    )
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
        result = _install_air_hose_message(engine, ticket_id, ticket)
    elif node_id == "defect_voice_not_working_terminal":
        result = _voice_self_help_message(engine, ticket_id, ticket, false_triggers=False)
    elif node_id == "defect_voice_false_triggers_terminal":
        result = _voice_self_help_message(engine, ticket_id, ticket, false_triggers=True)
    elif node_id in _ROLLING_NOISE_TERMINALS:
        result = _rolling_noise_message(engine, ticket_id, ticket)
    elif node_id in _REMOTE_TERMINALS:
        result = _remote_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id in _POWER_TERMINALS:
        result = _power_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id in _AIR_TERMINALS:
        result = _air_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id in _FOOTREST_TERMINALS:
        result = _footrest_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id in _COSMETIC_TERMINALS:
        result = _cosmetic_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id in _RECLINE_TERMINALS:
        result = _recline_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id in _HEATING_TERMINALS:
        result = _heating_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id in _DELIVERY_TERMINALS:
        result = _delivery_self_help_message(engine, ticket_id, ticket, node_id)
    elif node_id == "install_send_video" or (
        issue_type == "installation" and action == "send_info"
    ):
        model_name = str(getattr(ticket, "model_name", "") or "")
        result = _install_message(model_name, base_prompt)
    elif action == "awaiting_admin":
        result = _admin_review_terminal_message(engine, ticket_id, ticket, node)
    else:
        result = _workflow_end_message(engine, ticket_id, ticket, node)

    from warranty_error_code_gate import append_fonz_to_terminal_enrichment  # noqa: WPS433

    return append_fonz_to_terminal_enrichment(result, ticket)
