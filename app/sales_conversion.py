"""
sales_conversion.py
===================
Attribute Shopify orders back to the chat session that influenced them.

The sales funnel previously ended at "lead captured", which meant the chat
could never be shown to have produced revenue — the one number that decides
whether this system is worth running.

How attribution works
---------------------
Shopify calls ``/api/v1/sales/shopify/webhook`` on ``orders/create`` or
``orders/paid``. We take the checkout email and look for the most recent chat
session that captured the same address, either on the session itself or on a
lead row, within an attribution window (30 days by default).

Deliberate limits
-----------------
- **Email is the only join key.** A shopper who chats anonymously and then
  checks out with an address we never saw stays unattributed. We record the
  order anyway with ``session_id = NULL`` so the attributed share is honest
  rather than flattering.
- **Last touch, not multi touch.** The most recent qualifying session gets
  the credit.
- **Idempotent.** ``shopify_order_id`` is unique, so webhook retries and the
  ``orders/create`` + ``orders/paid`` overlap cannot double-count revenue.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

from sales_models import SalesConversion, SalesLead, SalesSession
from warranty_models import warranty_db_session

logger = logging.getLogger(__name__)

MATCH_SESSION_EMAIL = "session_email"
MATCH_LEAD_EMAIL = "lead_email"


def attribution_window_days() -> int:
    try:
        return max(1, min(int(os.getenv("SALES_ATTRIBUTION_WINDOW_DAYS", "30")), 365))
    except ValueError:
        return 30


# ---------------------------------------------------------------------------
# Webhook authenticity
# ---------------------------------------------------------------------------


def _webhook_secret(domain: str = "") -> str:
    from store_config import get_store_key_prefix

    prefix = get_store_key_prefix(domain)
    for key in (f"{prefix}_SHOPIFY_WEBHOOK_SECRET", "SHOPIFY_WEBHOOK_SECRET"):
        secret = (os.getenv(key) or "").strip()
        if secret:
            return secret
    return ""


def verify_shopify_hmac(body: bytes, header: str, secret: str) -> bool:
    """Shopify signs the raw body with base64 HMAC-SHA256."""
    if not secret or not header:
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, header.strip())


def require_shopify_signature(body: bytes, header: str, domain: str = "") -> None:
    """Raise ``ValueError`` when the request is not provably from Shopify."""
    secret = _webhook_secret(domain)
    if not secret:
        # Mirrors the Tidio adapter: fail closed in production, warn in dev so
        # the endpoint stays testable locally.
        if os.getenv("APP_ENV", "development").strip().lower() == "production":
            raise ValueError("SHOPIFY_WEBHOOK_SECRET not configured")
        logger.warning("shopify webhook signature skipped — no secret configured")
        return
    if not verify_shopify_hmac(body, header, secret):
        raise ValueError("invalid shopify webhook signature")


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


def _order_email(payload: dict[str, Any]) -> str:
    for key in ("email", "contact_email"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value
    customer = payload.get("customer")
    if isinstance(customer, dict):
        return str(customer.get("email") or "").strip().lower()
    return ""


def _order_total(payload: dict[str, Any]) -> Optional[float]:
    for key in ("total_price", "current_total_price", "subtotal_price"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _ordered_at(payload: dict[str, Any]) -> Optional[datetime]:
    raw = str(payload.get("created_at") or payload.get("processed_at") or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def find_attributed_session(
    email: str,
    *,
    ordered_at: Optional[datetime] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(session_id, matched_by)`` for the chat that earned this order."""
    email = (email or "").strip().lower()
    if not email:
        return None, None

    cutoff = (ordered_at or datetime.utcnow()) - timedelta(days=attribution_window_days())

    with warranty_db_session() as db:
        session = (
            db.query(SalesSession)
            .filter(SalesSession.contact_email.isnot(None))
            .filter(SalesSession.created_at >= cutoff)
            .order_by(SalesSession.created_at.desc())
            .all()
        )
        for row in session:
            if str(row.contact_email or "").strip().lower() == email:
                return str(row.session_id), MATCH_SESSION_EMAIL

        leads = (
            db.query(SalesLead)
            .filter(SalesLead.email.isnot(None))
            .filter(SalesLead.created_at >= cutoff)
            .order_by(SalesLead.created_at.desc())
            .all()
        )
        for lead in leads:
            if str(lead.email or "").strip().lower() == email:
                return str(lead.session_id), MATCH_LEAD_EMAIL

    return None, None


def record_order(payload: dict[str, Any], *, domain: str = "unknown") -> Optional[dict]:
    """Persist an order and attribute it. Returns ``None`` on a duplicate."""
    order_id = str(payload.get("id") or payload.get("admin_graphql_api_id") or "").strip()
    if not order_id:
        return None

    with warranty_db_session() as db:
        exists = (
            db.query(SalesConversion)
            .filter(SalesConversion.shopify_order_id == order_id)
            .one_or_none()
        )
        if exists is not None:
            return None

    email = _order_email(payload)
    ordered_at = _ordered_at(payload)
    session_id, matched_by = find_attributed_session(email, ordered_at=ordered_at)

    with warranty_db_session() as db:
        row = SalesConversion(
            shopify_order_id=order_id,
            order_number=str(payload.get("order_number") or payload.get("name") or "") or None,
            session_id=session_id,
            email=email or None,
            domain=domain or "unknown",
            total_usd=_order_total(payload),
            currency=str(payload.get("currency") or "") or None,
            matched_by=matched_by,
            ordered_at=ordered_at,
        )
        db.add(row)
        db.flush()
        result = row.to_dict()

    logger.info(
        "sales conversion recorded order=%s attributed=%s matched_by=%s",
        order_id,
        bool(session_id),
        matched_by or "none",
    )
    return result
