"""Record consented, final Warranty API exchanges without changing replies.

Only customer conversation POSTs are captured. Reads, uploads, consent/contact
forms, administrative actions and unsuccessful responses never create messages.
Delivery identity fields are not copied into the transcript.
"""
from __future__ import annotations

import json
import logging
import re
from uuid import uuid4

from fastapi.routing import APIRoute

from warranty_models import (
    WarrantyChatConsent, WarrantyConversationMessage, WarrantyTicket,
    _now_cst, warranty_db_session,
)

logger = logging.getLogger(__name__)
_CUSTOMER_POST = re.compile(
    r"^/api/v1/warranty/(?:session/[^/]+/(?:register-model|confirm-model|"
    r"quick-start|natural-start|smart-start|delivery-tracking|handoff)|"
    r"[^/]+/(?:answer|back|customer-note|troubleshooting-outcome))/?$"
)


def _customer_input(path: str, body: dict) -> tuple[str, str]:
    if path.rstrip('/').endswith('/delivery-tracking'):
        return 'Requested latest delivery tracking.', 'user_action'
    if path.rstrip('/').endswith('/back'):
        return 'Returned to the previous question.', 'user_action'
    if path.rstrip('/').endswith('/handoff') and not body.get('message'):
        return 'Requested a support specialist.', 'user_action'
    for name in ('answer', 'message', 'model', 'note', 'outcome'):
        value = body.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip(), 'customer_input'
    if isinstance(body.get('issue_type'), str):
        return 'Selected issue: ' + body['issue_type'].strip(), 'user_action'
    if isinstance(body.get('confirmed'), bool):
        return ('Confirmed the suggested model.' if body['confirmed'] else 'Rejected the suggested model.'), 'user_action'
    return '', 'user_action'


def _reply(payload: dict) -> str:
    for value in (
        payload.get('assistant_message'),
        (payload.get('terminal_enrichment') or {}).get('message'),
        ((payload.get('ticket') or {}).get('current_node') or {}).get('prompt'),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def record_warranty_exchange(payload: dict, *, path: str, body: dict, path_params: dict | None = None) -> bool:
    """Store one successful final response, retaining full text and input order.

    This intentionally records repeated answers: identical text on two distinct
    customer requests is evidence that an administrator needs to see.
    """
    if not isinstance(payload, dict) or not _CUSTOMER_POST.fullmatch(path):
        return False
    ticket_payload = payload.get('ticket') or {}
    if not isinstance(ticket_payload, dict):
        return False
    ticket_id = ticket_payload.get('ticket_id')
    if not isinstance(ticket_id, str) or not ticket_id:
        return False
    response_text = _reply(payload)
    if not response_text:
        return False
    customer_text, input_origin = _customer_input(path, body)
    with warranty_db_session() as db:
        ticket = db.query(WarrantyTicket).filter(WarrantyTicket.ticket_id == ticket_id).first()
        if ticket is None:
            return False
        session_id = str(ticket.session_id)
        path_params = path_params or {}
        if path_params.get('session_id') not in (None, session_id):
            return False
        if path_params.get('ticket_id') not in (None, ticket_id):
            return False
        if payload.get('session_id') not in (None, session_id):
            return False
        consent = db.query(WarrantyChatConsent).filter(
            WarrantyChatConsent.session_id == session_id,
            WarrantyChatConsent.accepted_at.is_not(None),
        ).first()
        if consent is None:
            return False
        exchange_id = str(uuid4())
        created_at = _now_cst()
        if customer_text:
            db.add(WarrantyConversationMessage(
                ticket_id=ticket_id, session_id=session_id, exchange_id=exchange_id,
                role='user', origin=input_origin, content=customer_text, created_at=created_at,
            ))
        db.add(WarrantyConversationMessage(
            ticket_id=ticket_id, session_id=session_id, exchange_id=exchange_id,
            role='assistant', origin='api_response', content=response_text, created_at=created_at,
        ))
    return True


class WarrantyConversationRoute(APIRoute):
    """Capture the final serialized payload, after all route-level overrides."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def record_final_response(request):
            capture = request.method == 'POST' and bool(_CUSTOMER_POST.fullmatch(request.url.path))
            body = {}
            if capture:
                try:
                    incoming = await request.json()
                    if isinstance(incoming, dict):
                        body = incoming
                except (ValueError, UnicodeError):
                    pass
            response = await handler(request)
            if capture and 200 <= response.status_code < 300:
                try:
                    raw = getattr(response, 'body', b'')
                    payload = json.loads(raw)
                    record_warranty_exchange(payload, path=request.url.path, body=body, path_params=request.path_params)
                except Exception as exc:
                    # Recording must not break the customer reply or log its PII.
                    logger.warning('Warranty response recording failed (%s)', type(exc).__name__)
            return response

        return record_final_response
