"""Read only recorded customer exchanges, scoped to one consented case.

Workflow references, staff notes and legacy reconstruction are intentionally
excluded: they are not records of what was sent to the customer.
"""

from __future__ import annotations

from sqlalchemy import and_, or_

from warranty_models import WarrantyConversationMessage, warranty_db_session


def customer_conversation_page(
    *, ticket_id: str, session_id: str, before_id: int | None = None, limit: int = 100,
) -> dict:
    """Return the newest bounded page in display order, with an older cursor."""
    if not 1 <= limit <= 100:
        raise ValueError("Message limit must be between 1 and 100.")
    if before_id is not None and before_id < 1:
        raise ValueError("Message cursor must be positive.")
    with warranty_db_session() as db:
        query = db.query(WarrantyConversationMessage).filter(
            WarrantyConversationMessage.ticket_id == ticket_id,
            WarrantyConversationMessage.session_id == session_id,
            or_(
                and_(WarrantyConversationMessage.role == "user",
                     WarrantyConversationMessage.origin.in_(("customer_input", "user_action"))),
                and_(WarrantyConversationMessage.role == "assistant",
                     WarrantyConversationMessage.origin == "api_response"),
            ),
        )
        if before_id is not None:
            query = query.filter(WarrantyConversationMessage.id < before_id)
        rows = query.order_by(WarrantyConversationMessage.id.desc()).limit(limit + 1).all()
        has_more = len(rows) > limit
        messages = [
            {"id": int(row.id), "role": str(row.role), "content": str(row.content)}
            for row in reversed(rows[:limit])
        ]
    return {
        "session_id": session_id,
        "ticket_id": ticket_id,
        "messages": messages,
        "has_more": has_more,
        "before_id": messages[0]["id"] if messages else None,
    }
