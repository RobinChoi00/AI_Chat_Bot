"""Best-effort Shopify purchase verification for warranty intake."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _model_tokens(value: str) -> set[str]:
    ignored = {"osaki", "titan", "massage", "chair", "pro", "the"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (value or "").lower())
        if len(token) >= 3 and token not in ignored
    }


def _matches_model(expected: str, purchased: str) -> bool:
    expected_tokens = _model_tokens(expected)
    purchased_tokens = _model_tokens(purchased)
    if not expected_tokens or not purchased_tokens:
        return False
    overlap = expected_tokens & purchased_tokens
    return len(overlap) >= max(1, min(len(expected_tokens), len(purchased_tokens)) - 1)


def _patch_ticket(ticket_id: str, updates: dict[str, Any]) -> None:
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    with warranty_db_session() as db:
        ticket = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .one_or_none()
        )
        if ticket is None:
            return
        for key, value in updates.items():
            ticket.set_collected(key, value)


def verify_ticket_purchase(
    *,
    ticket_id: str,
    session_id: str,
    domain: str,
    expected_model: str,
) -> dict[str, Any]:
    """Verify the latest checkout-email order without blocking warranty intake.

    Eligibility is persisted only when a Shopify product matches the model the
    customer confirmed. A mismatch is flagged for admin review, never treated
    as a denial.
    """
    from delivery_lookup import safe_lookup_by_order_or_email  # noqa: WPS433
    from warranty_consent import get_chat_consent  # noqa: WPS433

    consent = get_chat_consent(session_id)
    email = str(getattr(consent, "contact_email", "") or "").strip()
    if not email:
        result = {"status": "skipped_no_email"}
        _patch_ticket(ticket_id, {"shopify_purchase_verification": json.dumps(result)})
        return result

    snapshot = safe_lookup_by_order_or_email(email, domain)
    if not snapshot.available:
        result = {
            "status": "lookup_failed",
            "source": snapshot.source,
            "error": str(snapshot.error or "unavailable")[:200],
        }
        _patch_ticket(ticket_id, {"shopify_purchase_verification": json.dumps(result)})
        return result

    matched_product = next(
        (
            product
            for product in snapshot.product_names
            if _matches_model(expected_model, product)
        ),
        "",
    )
    if not matched_product:
        result = {
            "status": "model_mismatch",
            "source": snapshot.source,
            "order_number": snapshot.order_number,
            "expected_model": expected_model,
            "product_names": snapshot.product_names,
            "looked_up_at": snapshot.looked_up_at,
        }
        _patch_ticket(ticket_id, {"shopify_purchase_verification": json.dumps(result)})
        return result

    from delivery_lookup import persist_snapshot  # noqa: WPS433

    persist_snapshot(
        ticket_id,
        snapshot,
        raw_input=email,
        lookup_kind="warranty_purchase",
    )
    result = {
        "status": "verified_model_match",
        "source": snapshot.source,
        "order_number": snapshot.order_number,
        "matched_product": matched_product,
        "purchase_date": snapshot.purchase_date,
        "looked_up_at": snapshot.looked_up_at,
    }
    _patch_ticket(ticket_id, {"shopify_purchase_verification": json.dumps(result)})
    logger.info("Warranty purchase verified ticket=%s source=%s", ticket_id, snapshot.source)
    return result
