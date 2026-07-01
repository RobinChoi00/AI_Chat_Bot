"""
ringcentral_ivr.py
==================
Orchestrate RingCentral IVR callbacks with WarrantyEngine.

Call flow (DTMF-only MVP):
  on-call-enter  → start ticket → play menu
  Play complete  → collect DTMF
  Collect digit  → submit_answer OR forward/hangup at terminal
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ringcentral_client import (
    RC_WARRANTY_TRANSFER_TO,
    collect_digits,
    forward_call,
    hangup,
    play_prompt,
)
from ringcentral_voice import (
    AGENT_DTMF,
    IvrPhase,
    VoiceCallContext,
    build_menu_script,
    build_terminal_script,
    get_call_context,
    menu_dtmf_patterns,
    pop_call_context,
    post_diy_dtmf_patterns,
    resolve_play_uri,
    set_call_context,
)

logger = logging.getLogger(__name__)


def _lazy_engine():
    from warranty_workflow import WarrantyEngine  # noqa: WPS433

    return WarrantyEngine


def _lazy_enrichment():
    from warranty_terminal_enrichment import build_terminal_enrichment  # noqa: WPS433

    return build_terminal_enrichment


def _caller_phone(payload: dict[str, Any]) -> str:
    in_party = payload.get("inParty") or {}
    from_block = in_party.get("from") or {}
    return str(from_block.get("phoneNumber") or "")


def _party_id(payload: dict[str, Any]) -> str:
    in_party = payload.get("inParty") or {}
    return str(in_party.get("id") or payload.get("partyId") or "")


def _session_id(payload: dict[str, Any]) -> str:
    return str(payload.get("sessionId") or "")


def _store_caller_metadata(ticket_id: str, caller: str) -> None:
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    with warranty_db_session() as db:
        ticket = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket is None:
            return
        ticket.set_collected("channel", "phone")
        if caller:
            ticket.set_collected("caller_phone", caller)


def _play_script(ctx: VoiceCallContext, script: str, *, phase: IvrPhase) -> None:
    uri = resolve_play_uri(script)
    play_prompt(session_id=ctx.session_id, party_id=ctx.party_id, audio_uri=uri)
    ctx.phase = phase
    ctx.awaiting_command = "Play"


def _start_collect(ctx: VoiceCallContext, patterns: list[str]) -> None:
    collect_digits(session_id=ctx.session_id, party_id=ctx.party_id, patterns=patterns)
    ctx.awaiting_command = "Collect"


def _transfer(ctx: VoiceCallContext, reason: str) -> None:
    logger.info(
        "RC IVR transfer session=%s ticket=%s reason=%s",
        ctx.session_id,
        ctx.ticket_id,
        reason,
    )
    forward_call(
        session_id=ctx.session_id,
        party_id=ctx.party_id,
        phone_number=RC_WARRANTY_TRANSFER_TO,
    )
    ctx.phase = IvrPhase.DONE
    ctx.awaiting_command = None


def _present_node(ctx: VoiceCallContext, node: dict) -> None:
    node_type = node.get("type")
    if node_type == "terminal":
        _present_terminal(ctx, node)
        return
    if node_type == "question_text":
        script = (
            "We need a few details that are easier with a specialist. "
            f"Press {AGENT_DTMF} to speak with a warranty agent now."
        )
        _play_script(ctx, script, phase=IvrPhase.MENU)
        return
    if node_type in ("question", "instruction"):
        script = build_menu_script(node)
        _play_script(ctx, script, phase=IvrPhase.MENU)
        return
    logger.warning("Unsupported node type %s — transferring", node_type)
    _transfer(ctx, "unsupported_node")


def _present_terminal(ctx: VoiceCallContext, node: dict) -> None:
    action = str(node.get("action") or "awaiting_admin")
    if action == "sales_handoff":
        _transfer(ctx, "sales_handoff")
        return
    if action in ("awaiting_admin", "awaiting_admin_review", "awaiting_evidence"):
        script = (
            "Thank you. I am connecting you with a warranty specialist "
            "who can review your case."
        )
        _play_script(ctx, script, phase=IvrPhase.DONE)
        ctx.awaiting_command = "PlayThenTransfer"
        return

    engine = _lazy_engine()
    ticket = engine.get_ticket(ctx.ticket_id)
    enrichment = None
    if ticket is not None:
        enrichment = _lazy_enrichment()(engine, ticket, node)
    script = build_terminal_script(node, enrichment)
    _play_script(ctx, script, phase=IvrPhase.POST_DIY)


def handle_call_enter(payload: dict[str, Any]) -> None:
    session_id = _session_id(payload)
    party_id = _party_id(payload)
    if not session_id or not party_id:
        logger.error("RC on-call-enter missing sessionId/partyId: %s", payload)
        return

    engine = _lazy_engine()
    caller = _caller_phone(payload)
    ticket_id, root_node = engine.start_session(session_id, "phone")
    _store_caller_metadata(ticket_id, caller)

    ctx = VoiceCallContext(
        session_id=session_id,
        party_id=party_id,
        ticket_id=ticket_id,
        caller_phone=caller,
    )
    set_call_context(ctx)
    logger.info("RC IVR started session=%s ticket=%s caller=%s", session_id, ticket_id, caller)
    _present_node(ctx, root_node)


def _handle_menu_digit(ctx: VoiceCallContext, digit: str) -> None:
    if digit == AGENT_DTMF:
        _transfer(ctx, "caller_requested_agent")
        return

    engine = _lazy_engine()
    node = engine.get_current_node(ctx.ticket_id)
    if node is None:
        _transfer(ctx, "missing_node")
        return

    if node.get("type") == "question_text":
        _transfer(ctx, "question_text")
        return

    try:
        result = engine.submit_answer(ctx.ticket_id, digit)
    except ValueError as exc:
        logger.warning("RC IVR invalid digit %s at %s: %s", digit, node.get("node_id"), exc)
        script = "Sorry, that was not a valid option. Please try again."
        _play_script(ctx, script, phase=IvrPhase.MENU)
        return

    next_node = result.get("next_node") or {}
    _present_node(ctx, next_node)


def _handle_post_diy_digit(ctx: VoiceCallContext, digit: str) -> None:
    if digit in (AGENT_DTMF, "2"):
        _transfer(ctx, "post_diy_need_agent")
        return
    if digit == "1":
        logger.info("RC IVR resolved on call session=%s ticket=%s", ctx.session_id, ctx.ticket_id)
        hangup(session_id=ctx.session_id, party_id=ctx.party_id)
        ctx.phase = IvrPhase.DONE
        ctx.awaiting_command = None
        return
    script = "Sorry, press 1 if the issue is fixed, or 2 for an agent."
    _play_script(ctx, script, phase=IvrPhase.POST_DIY)


def handle_command_update(payload: dict[str, Any]) -> None:
    session_id = _session_id(payload)
    ctx = get_call_context(session_id)
    if ctx is None:
        logger.warning("RC on-command-update with unknown session=%s", session_id)
        return

    status = str(payload.get("status") or "")
    if status != "Completed":
        return

    command = str(payload.get("command") or "")
    party_id = str(payload.get("partyId") or ctx.party_id)
    ctx.party_id = party_id

    if command == "Play":
        if ctx.awaiting_command == "PlayThenTransfer":
            _transfer(ctx, "terminal_admin")
            return
        if ctx.phase == IvrPhase.MENU:
            node = _lazy_engine().get_current_node(ctx.ticket_id) or {}
            if node.get("type") == "question_text":
                _start_collect(ctx, [AGENT_DTMF])
            else:
                _start_collect(ctx, menu_dtmf_patterns(node))
            return
        if ctx.phase == IvrPhase.POST_DIY:
            _start_collect(ctx, post_diy_dtmf_patterns())
            return

    if command == "Collect":
        params = payload.get("parameters") or {}
        digit = str(params.get("digits") or "").strip()
        if not digit:
            script = "We did not receive a selection. Please try again."
            _play_script(ctx, script, phase=ctx.phase)
            return
        if ctx.phase == IvrPhase.MENU:
            _handle_menu_digit(ctx, digit)
            return
        if ctx.phase == IvrPhase.POST_DIY:
            _handle_post_diy_digit(ctx, digit)
            return


def handle_call_exit(payload: dict[str, Any]) -> None:
    session_id = _session_id(payload)
    ctx = pop_call_context(session_id)
    if ctx:
        logger.info("RC IVR call exit session=%s ticket=%s", session_id, ctx.ticket_id)
