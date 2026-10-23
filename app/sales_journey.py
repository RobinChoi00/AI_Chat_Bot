"""Allowlisted server observations for Sales; never store customer text here."""
from __future__ import annotations

from sales_models import SalesSession
from warranty_models import warranty_db_session


def record_reply(session_id: str, domain: str, reply) -> None:
    from customer_journey import register_conversation, record_server_event

    with warranty_db_session() as db:
        row = db.query(SalesSession).filter(SalesSession.session_id == session_id).one_or_none()
        if row is None:
            return
        common = dict(surface='sales', conversation_id=str(row.id), session_id=session_id, domain=domain)
        register_conversation(db, **common, is_new=False)
        record_server_event(db, **common, stage='response', event='completed')
        stage = str(reply.flow_stage or '')
        tools = set(reply.tools_used or [])
        if reply.intent == 'recommend' or stage.startswith('ask_'):
            record_server_event(db, **common, stage='recommendation', event='requested')
            if stage.startswith('ask_'):
                record_server_event(db, **common, stage='recommendation', event='shown')
        if reply.products and reply.intent == 'recommend':
            record_server_event(db, **common, stage='recommendation', event='completed')
        if 'cases.nofit' in tools:
            record_server_event(db, **common, stage='outcome', event='completed')
        if reply.handoff:
            record_server_event(db, **common, stage='handoff', event='requested')
        if reply.intent in {'specs', 'price', 'stock', 'prepurchase_policy', 'compare'} and (
            reply.products or any(tool.startswith('policy.') for tool in tools)
        ):
            record_server_event(db, **common, stage='help', event='shown')
        if reply.feedback and reply.feedback.get('rating') == 'helpful':
            record_server_event(db, **common, stage='help', event='completed')
        elif reply.feedback and reply.feedback.get('rating') == 'not_helpful':
            record_server_event(db, **common, stage='help', event='failed')
        patch = reply.prefs_patch or {}
        if patch.get('awaiting_email_for_pick') or patch.get('awaiting_showroom_email'):
            record_server_event(db, **common, stage='email', event='shown')
            record_server_event(db, **common, stage='lead', event='shown')
        elif patch.get('awaiting_showroom_window'):
            record_server_event(db, **common, stage='lead', event='shown')
        if 'visitor.resume_continue' in tools or 'visitor.resume_picks' in tools:
            record_server_event(db, **common, stage='resume', event='resumed')


def record_human_confirmation(session_id: str, domain: str, *, event_id: str | None = None) -> None:
    """Only call after the authenticated Tidio operator webhook is verified."""
    from customer_journey import register_conversation, record_server_event

    with warranty_db_session() as db:
        row = db.query(SalesSession).filter(SalesSession.session_id == session_id).one_or_none()
        if row is None:
            return
        common = dict(surface='sales', conversation_id=str(row.id), session_id=session_id, domain=domain)
        register_conversation(db, **common, is_new=False)
        record_server_event(db, **common, stage='handoff', event='completed', event_id=event_id)
