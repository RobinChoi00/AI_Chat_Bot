"""
ringcentral_ivr.py
==================
Orchestrate RingCentral IVR callbacks with WarrantyEngine.

Call flow:
  on-call-enter (open)   → closed-hours message N/A; play connect script → forward to warranty queue
  on-call-enter (closed) → after-hours welcome (closed + hours + docs) → issue type menu → …
  Play complete          → collect DTMF, connect forward, or sales transfer
  on-call-exit           → SMS + team email (after-hours tickets only)

After-hours: no silent transfer to sales — sales_handoff plays closed message instead.
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
from ringcentral_followup import send_phone_call_followups
from ringcentral_hours import is_warranty_business_hours
from ringcentral_voice import (
    IvrPhase,
    REPEAT_DTMF,
    VoiceCallContext,
    build_after_hours_sales_closed_script,
    build_after_hours_welcome_script,
    build_business_hours_connect_script,
    build_menu_script,
    build_question_text_handoff_script,
    build_sales_transfer_script,
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


def _log_business_hours_connect(session_id: str, caller: str) -> str:
    """Record open-hours calls forwarded to the live warranty queue."""
    import uuid

    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    ticket_id = str(uuid.uuid4())
    with warranty_db_session() as db:
        ticket = WarrantyTicket(
            ticket_id=ticket_id,
            session_id=session_id,
            domain="phone",
            status="closed",
            current_node_id="phone_live_forward",
            collected_data="{}",
        )
        db.add(ticket)
        ticket.set_collected("channel", "phone")
        if caller:
            ticket.set_collected("caller_phone", caller)
        ticket.set_collected("ivr_path", "business_hours_live_forward")
    logger.info(
        "RC IVR logged live forward ticket=%s session=%s caller=%s",
        ticket_id,
        session_id,
        caller,
    )
    return ticket_id


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


def _forward_to_warranty_queue(ctx: VoiceCallContext) -> None:
    logger.info(
        "RC IVR connecting to warranty queue session=%s caller=%s",
        ctx.session_id,
        ctx.caller_phone,
    )
    forward_call(session_id=ctx.session_id, party_id=ctx.party_id)
    ctx.phase = IvrPhase.DONE
    ctx.awaiting_command = None


def _present_node(
    ctx: VoiceCallContext,
    node: dict,
    *,
    intro_prefix: str = "",
) -> None:
    node_type = node.get("type")
    if node_type == "terminal":
        _present_terminal(ctx, node)
        return
    if node_type == "question_text":
        _play_script(ctx, build_question_text_handoff_script(), phase=IvrPhase.MENU)
        return
    if node_type in ("question", "instruction"):
        script = f"{intro_prefix}{build_menu_script(node)}"
        _play_script(ctx, script, phase=IvrPhase.MENU)
        return
    logger.warning("Unsupported node type %s — transferring", node_type)
    if is_warranty_business_hours():
        _transfer(ctx, "unsupported_node")
    else:
        _play_script(
            ctx,
            build_after_hours_sales_closed_script(),
            phase=IvrPhase.POST_DIY,
        )


def _present_terminal(ctx: VoiceCallContext, node: dict) -> None:
    action = str(node.get("action") or "awaiting_admin")
    if action == "sales_handoff":
        if is_warranty_business_hours():
            _play_script(ctx, build_sales_transfer_script(), phase=IvrPhase.SALES_TRANSFER)
        else:
            _play_script(ctx, build_after_hours_sales_closed_script(), phase=IvrPhase.POST_DIY)
        return
    if action in ("awaiting_admin", "awaiting_admin_review", "awaiting_evidence"):
        from ringcentral_voice import build_after_hours_closure_script  # noqa: WPS433

        _play_script(ctx, build_after_hours_closure_script(), phase=IvrPhase.POST_DIY)
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

    caller = _caller_phone(payload)
    if is_warranty_business_hours():
        ticket_id = _log_business_hours_connect(session_id, caller)
        logger.info(
            "RC IVR business hours — connect message then forward session=%s caller=%s ticket=%s",
            session_id,
            caller,
            ticket_id,
        )
        ctx = VoiceCallContext(
            session_id=session_id,
            party_id=party_id,
            ticket_id=ticket_id,
            caller_phone=caller,
            phase=IvrPhase.CONNECTING,
        )
        set_call_context(ctx)
        _play_script(ctx, build_business_hours_connect_script(), phase=IvrPhase.CONNECTING)
        return

    engine = _lazy_engine()
    ticket_id, _root = engine.start_session(session_id, "phone")
    _store_caller_metadata(ticket_id, caller)
    engine.submit_answer(ticket_id, "warranty")
    entry_node = engine.get_current_node(ticket_id)
    if not entry_node:
        logger.error("RC IVR failed to advance to issue_type for ticket=%s", ticket_id)
        return

    ctx = VoiceCallContext(
        session_id=session_id,
        party_id=party_id,
        ticket_id=ticket_id,
        caller_phone=caller,
    )
    set_call_context(ctx)
    logger.info(
        "RC IVR started session=%s ticket=%s caller=%s node=%s",
        session_id,
        ticket_id,
        caller,
        entry_node.get("node_id"),
    )
    intro = f"{build_after_hours_welcome_script()} "
    _present_node(ctx, entry_node, intro_prefix=intro)


def _replay_current_node(ctx: VoiceCallContext) -> None:
    engine = _lazy_engine()
    node = engine.get_current_node(ctx.ticket_id)
    if node is None:
        logger.warning("RC IVR repeat with missing node ticket=%s", ctx.ticket_id)
        return
    if node.get("type") == "terminal":
        _present_terminal(ctx, node)
        return
    _present_node(ctx, node)


def _handle_menu_digit(ctx: VoiceCallContext, digit: str) -> None:
    if digit == REPEAT_DTMF:
        _replay_current_node(ctx)
        return

    engine = _lazy_engine()
    node = engine.get_current_node(ctx.ticket_id)
    if node is None:
        if is_warranty_business_hours():
            _transfer(ctx, "missing_node")
        return

    if node.get("type") == "question_text":
        _replay_current_node(ctx)
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
    if digit == REPEAT_DTMF:
        _replay_current_node(ctx)
        return
    if digit == "1":
        logger.info("RC IVR resolved on call session=%s ticket=%s", ctx.session_id, ctx.ticket_id)
        hangup(session_id=ctx.session_id, party_id=ctx.party_id)
        ctx.phase = IvrPhase.DONE
        ctx.awaiting_command = None
        return
    script = "Sorry, press 1 if the issue is fixed, or press 0 to hear the message again."
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
        if ctx.phase == IvrPhase.CONNECTING:
            _forward_to_warranty_queue(ctx)
            pop_call_context(session_id)
            return
        if ctx.phase == IvrPhase.SALES_TRANSFER:
            _transfer(ctx, "sales_handoff")
            return
        if ctx.phase == IvrPhase.MENU:
            node = _lazy_engine().get_current_node(ctx.ticket_id) or {}
            if node.get("type") == "question_text":
                _start_collect(ctx, [REPEAT_DTMF])
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
    if ctx and ctx.ticket_id:
        logger.info("RC IVR call exit session=%s ticket=%s", session_id, ctx.ticket_id)
        send_phone_call_followups(
            caller_phone=ctx.caller_phone,
            ticket_id=ctx.ticket_id,
            session_id=ctx.session_id,
        )
