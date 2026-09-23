"""
ringcentral_ivr.py
==================
Orchestrate RingCentral IVR callbacks with WarrantyEngine.

Call flow:
  on-call-enter (open)   → play connect script → forward to warranty queue
  on-call-enter (closed) → department menu (2=sales, 3=warranty)
                         → press 3: after-hours welcome + issue menu
                           (1=setup, 2=sales/delivery, 3=defect)
                         → press 2: announce, then forward to sales (ext.2)
  Play complete          → collect DTMF, connect forward, or sales transfer
  on-call-exit           → SMS + team email (after-hours warranty tickets only)

After-hours flowchart sales_handoff: no silent transfer — plays closed message instead.
Department-menu press 2 is an intentional sales transfer (announced).
"""

from __future__ import annotations

import logging
from typing import Any

from ringcentral_client import (
    RC_SALES_TRANSFER_EXTENSION,
    RC_SALES_TRANSFER_TO,
    RC_WARRANTY_TRANSFER_TO,
    collect_digits,
    forward_call,
    hangup,
    play_prompt,
)
from ringcentral_followup import send_phone_call_followups
from ringcentral_hours import is_warranty_business_hours
from ringcentral_voice import (
    DEPT_SALES_DTMF,
    DEPT_WARRANTY_DTMF,
    ISSUE_DEFECT_DTMF,
    ISSUE_INSTALL_DTMF,
    ISSUE_SALES_DTMF,
    IvrPhase,
    REPEAT_DTMF,
    VoiceCallContext,
    build_after_hours_sales_closed_script,
    build_after_hours_welcome_script,
    build_business_hours_connect_script,
    build_department_menu_script,
    build_menu_script,
    build_phone_issue_menu_script,
    build_question_text_handoff_script,
    build_sales_transfer_script,
    build_terminal_script,
    department_dtmf_patterns,
    get_call_context,
    menu_dtmf_patterns,
    phone_issue_dtmf_patterns,
    pop_call_context,
    post_diy_dtmf_patterns,
    resolve_play_uri,
    set_call_context,
)

# Live forwards leave the Voice App; do not SMS a warranty resume link.
_SKIP_PHONE_FOLLOWUP_PATHS = frozenset(
    {
        "business_hours_live_forward",
        "department_sales_forward",
    }
)

logger = logging.getLogger(__name__)


class RetryableRingCentralEvent(RuntimeError):
    """The callback is valid but depends on an earlier event/state write."""


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


def _set_collected(ticket_id: str, key: str, value: str) -> None:
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    if not ticket_id:
        return
    with warranty_db_session() as db:
        ticket = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket is None:
            return
        ticket.set_collected(key, value)


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
        "RC IVR logged live forward ticket=%s session=%s",
        ticket_id,
        session_id,
    )
    return ticket_id


def _play_script(ctx: VoiceCallContext, script: str, *, phase: IvrPhase) -> None:
    uri = resolve_play_uri(script)
    play_prompt(session_id=ctx.session_id, party_id=ctx.party_id, audio_uri=uri)
    ctx.phase = phase
    ctx.awaiting_command = "Play"
    ctx.last_audio_key = uri.rsplit("/", 1)[-1].removesuffix(".wav")
    set_call_context(ctx)


def _start_collect(ctx: VoiceCallContext, patterns: list[str]) -> None:
    collect_digits(session_id=ctx.session_id, party_id=ctx.party_id, patterns=patterns)
    ctx.awaiting_command = "Collect"
    set_call_context(ctx)


def _transfer(ctx: VoiceCallContext, reason: str) -> None:
    logger.info(
        "RC IVR transfer session=%s ticket=%s reason=%s",
        ctx.session_id,
        ctx.ticket_id,
        reason,
    )
    if reason == "sales_handoff":
        _forward_to_sales_queue(ctx)
        return
    forward_call(
        session_id=ctx.session_id,
        party_id=ctx.party_id,
        phone_number=RC_WARRANTY_TRANSFER_TO,
    )
    ctx.phase = IvrPhase.DONE
    ctx.awaiting_command = None
    set_call_context(ctx)


def _forward_to_warranty_queue(ctx: VoiceCallContext) -> None:
    logger.info(
        "RC IVR connecting to warranty queue session=%s",
        ctx.session_id,
    )
    forward_call(session_id=ctx.session_id, party_id=ctx.party_id)
    ctx.phase = IvrPhase.DONE
    ctx.awaiting_command = None
    set_call_context(ctx)


def _forward_to_sales_queue(ctx: VoiceCallContext) -> None:
    logger.info(
        "RC IVR connecting to sales queue session=%s ext=%s",
        ctx.session_id,
        RC_SALES_TRANSFER_EXTENSION,
    )
    forward_call(
        session_id=ctx.session_id,
        party_id=ctx.party_id,
        phone_number=RC_SALES_TRANSFER_TO,
        extension=RC_SALES_TRANSFER_EXTENSION,
    )
    ctx.phase = IvrPhase.DONE
    ctx.awaiting_command = None
    set_call_context(ctx)


def _play_department_menu(ctx: VoiceCallContext, *, intro_prefix: str = "") -> None:
    script = f"{intro_prefix}{build_department_menu_script()}"
    _play_script(ctx, script, phase=IvrPhase.DEPT_MENU)


def _start_after_hours_warranty_flow(ctx: VoiceCallContext) -> None:
    engine = _lazy_engine()
    node = engine.get_current_node(ctx.ticket_id)
    if node and node.get("node_id") == "root":
        engine.submit_answer(ctx.ticket_id, "warranty")
        node = engine.get_current_node(ctx.ticket_id)
    if not node:
        logger.error(
            "RC IVR failed to advance to issue_type for ticket=%s",
            ctx.ticket_id,
        )
        return
    intro = (
        "You selected warranty. "
        f"{build_after_hours_welcome_script()} "
    )
    _present_node(ctx, node, intro_prefix=intro)


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
        if node.get("node_id") == "issue_type":
            script = f"{intro_prefix}{build_phone_issue_menu_script()}"
        else:
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
        raise ValueError("RC on-call-enter missing sessionId/partyId")

    if get_call_context(session_id) is not None:
        logger.info("RC IVR ignored duplicate call-enter session=%s", session_id)
        return

    caller = _caller_phone(payload)
    if is_warranty_business_hours():
        ticket_id = _log_business_hours_connect(session_id, caller)
        logger.info(
            "RC IVR business hours — connect message then forward session=%s ticket=%s",
            session_id,
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
    _set_collected(ticket_id, "ivr_path", "after_hours_department_menu")

    ctx = VoiceCallContext(
        session_id=session_id,
        party_id=party_id,
        ticket_id=ticket_id,
        caller_phone=caller,
        phase=IvrPhase.DEPT_MENU,
    )
    set_call_context(ctx)
    logger.info(
        "RC IVR after-hours department menu session=%s ticket=%s",
        session_id,
        ticket_id,
    )
    _play_department_menu(ctx)


def _replay_current_node(ctx: VoiceCallContext) -> None:
    if ctx.phase == IvrPhase.DEPT_MENU:
        _play_department_menu(ctx)
        return
    engine = _lazy_engine()
    node = engine.get_current_node(ctx.ticket_id)
    if node is None:
        logger.warning("RC IVR repeat with missing node ticket=%s", ctx.ticket_id)
        return
    if node.get("type") == "terminal":
        _present_terminal(ctx, node)
        return
    _present_node(ctx, node)


def _handle_department_digit(ctx: VoiceCallContext, digit: str) -> None:
    if digit == REPEAT_DTMF:
        _play_department_menu(ctx)
        return
    if digit == DEPT_SALES_DTMF:
        _begin_sales_transfer(ctx)
        return
    if digit == DEPT_WARRANTY_DTMF:
        _set_collected(ctx.ticket_id, "ivr_path", "after_hours_warranty")
        _start_after_hours_warranty_flow(ctx)
        return
    _play_department_menu(
        ctx,
        intro_prefix="Sorry, that was not a valid option. ",
    )


def _begin_sales_transfer(ctx: VoiceCallContext) -> None:
    _set_collected(ctx.ticket_id, "ivr_path", "department_sales_forward")
    _play_script(ctx, build_sales_transfer_script(), phase=IvrPhase.SALES_TRANSFER)


def _handle_issue_type_digit(ctx: VoiceCallContext, digit: str) -> None:
    if digit == REPEAT_DTMF:
        _replay_current_node(ctx)
        return
    if digit == ISSUE_SALES_DTMF:
        _begin_sales_transfer(ctx)
        return
    if digit == ISSUE_INSTALL_DTMF:
        answer = "installation"
    elif digit == ISSUE_DEFECT_DTMF:
        answer = "defect"
    else:
        script = (
            "Sorry, that was not a valid option. "
            f"{build_phone_issue_menu_script()}"
        )
        _play_script(ctx, script, phase=IvrPhase.MENU)
        return

    engine = _lazy_engine()
    try:
        result = engine.submit_answer(ctx.ticket_id, answer)
    except ValueError as exc:
        logger.warning("RC IVR invalid issue-type digit %s: %s", digit, exc)
        script = (
            "Sorry, that was not a valid option. "
            f"{build_phone_issue_menu_script()}"
        )
        _play_script(ctx, script, phase=IvrPhase.MENU)
        return
    _present_node(ctx, result.get("next_node") or {})


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

    if node.get("node_id") == "issue_type":
        _handle_issue_type_digit(ctx, digit)
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
        set_call_context(ctx)
        return
    script = "Sorry, press 1 if the issue is fixed, or press 0 to hear the message again."
    _play_script(ctx, script, phase=IvrPhase.POST_DIY)


def handle_command_update(payload: dict[str, Any]) -> None:
    session_id = _session_id(payload)
    ctx = get_call_context(session_id)
    if ctx is None:
        raise RetryableRingCentralEvent(
            f"RC command arrived before call state for session={session_id}"
        )

    status = str(payload.get("status") or "")
    if status != "Completed":
        return

    command = str(payload.get("command") or "")
    party_id = str(payload.get("partyId") or ctx.party_id)
    ctx.party_id = party_id
    set_call_context(ctx)

    if ctx.phase == IvrPhase.DONE:
        return
    if ctx.awaiting_command and command != ctx.awaiting_command:
        logger.info(
            "RC IVR ignored stale command session=%s expected=%s received=%s",
            session_id,
            ctx.awaiting_command,
            command,
        )
        return

    if command == "Play":
        if ctx.phase == IvrPhase.CONNECTING:
            _forward_to_warranty_queue(ctx)
            pop_call_context(session_id)
            return
        if ctx.phase == IvrPhase.SALES_TRANSFER:
            _forward_to_sales_queue(ctx)
            pop_call_context(session_id)
            return
        if ctx.phase == IvrPhase.DEPT_MENU:
            _start_collect(ctx, department_dtmf_patterns())
            return
        if ctx.phase == IvrPhase.MENU:
            node = _lazy_engine().get_current_node(ctx.ticket_id) or {}
            if node.get("type") == "question_text":
                _start_collect(ctx, [REPEAT_DTMF])
            elif node.get("node_id") == "issue_type":
                _start_collect(ctx, phone_issue_dtmf_patterns())
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
        if ctx.phase == IvrPhase.DEPT_MENU:
            _handle_department_digit(ctx, digit)
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
    if ctx is None:
        from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            ticket = (
                db.query(WarrantyTicket)
                .filter(WarrantyTicket.session_id == session_id)
                .order_by(WarrantyTicket.id.desc())
                .first()
            )
            if ticket is None:
                raise RetryableRingCentralEvent(
                    f"RC exit arrived before call state for session={session_id}"
                )
            collected = ticket.get_collected()
            if collected.get("ivr_path") in _SKIP_PHONE_FOLLOWUP_PATHS:
                return
            ctx = VoiceCallContext(
                session_id=session_id,
                party_id="",
                ticket_id=str(ticket.ticket_id),
                caller_phone=str(collected.get("caller_phone") or ""),
                phase=IvrPhase.DONE,
            )
    if ctx.ticket_id:
        ticket = _lazy_engine().get_ticket(ctx.ticket_id)
        collected = ticket.get_collected() if ticket is not None else {}
        if collected.get("ivr_path") in _SKIP_PHONE_FOLLOWUP_PATHS:
            return
        logger.info("RC IVR call exit session=%s ticket=%s", session_id, ctx.ticket_id)
        send_phone_call_followups(
            caller_phone=ctx.caller_phone,
            ticket_id=ctx.ticket_id,
            session_id=ctx.session_id,
        )
