"""
sales_conversion.py
===================
Attribute Shopify orders back to the chat session that influenced them.

This records purchase association, not causal proof that a chat produced a
sale. Revenue requires authenticated payment evidence and is net of known
successful refunds, grouped by the order's actual currency.

How attribution works
---------------------
Shopify calls ``/api/v1/sales/shopify/webhook`` for order, payment,
cancellation and refund events. The same-store checkout email/phone is matched
to the most recent qualifying stored session or lead before the order, within
an attribution window (30 days by default).

Deliberate limits
-----------------
- **Email is the preferred join key.** Phone is a second pass when checkout
  and a chat lead share the same 10-digit number.
- **Last touch, not multi touch.** The most recent qualifying session gets
  the credit.
- **Idempotent.** Order updates and refund IDs cannot count a purchase or
  successful refund twice. Unpaid and unknown legacy orders remain distinct.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import json
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import pytz

from sales_models import SalesConversion, SalesLead, SalesSession
from warranty_models import warranty_db_session

logger = logging.getLogger(__name__)

MATCH_SESSION_EMAIL = "session_email"
MATCH_LEAD_EMAIL = "lead_email"
MATCH_SESSION_PHONE = "session_phone"
MATCH_LEAD_PHONE = "lead_phone"


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


def _phone_digits(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    # Never match unrelated international numbers by truncating to ten digits.
    return digits if len(digits) == 10 else ""


def _order_phone(payload: dict[str, Any]) -> str:
    for key in ("phone", "contact_phone"):
        digits = _phone_digits(str(payload.get(key) or ""))
        if len(digits) == 10:
            return digits
    for nested_key in ("customer", "billing_address", "shipping_address"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            digits = _phone_digits(str(nested.get("phone") or ""))
            if len(digits) == 10:
                return digits
    return ""


def _order_total(payload: dict[str, Any]) -> Optional[float]:
    for key in ("total_price", "current_total_price"):
        amount = _money(payload.get(key))
        if amount is not None:
            return amount
    return None


def _ordered_at(payload: dict[str, Any]) -> Optional[datetime]:
    return _timestamp(payload.get("created_at") or payload.get("processed_at"))


def _timestamp(raw: Any) -> Optional[datetime]:
    """Shopify timestamps are normalized to naive UTC, never just stripped."""
    try:
        parsed = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _database_timestamp(raw: Any) -> Optional[datetime]:
    """Legacy session/lead SQLite timestamps are Chicago wall time."""
    if not isinstance(raw, datetime):
        return None
    if raw.tzinfo is None:
        try:
            raw = pytz.timezone("America/Chicago").localize(raw, is_dst=None)
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
            return None  # do not invent an attribution during ambiguous DST
    return raw.astimezone(timezone.utc).replace(tzinfo=None)


def _money(raw: Any) -> Optional[float]:
    try:
        value = Decimal(str(raw))
        if not value.is_finite() or value < 0:
            return None
        return float(value.quantize(Decimal("0.01")))
    except (InvalidOperation, TypeError, ValueError):
        return None


def resolve_order_store(domain: str) -> Optional[str]:
    """Resolve known storefront aliases or an exact configured Shopify host.

    Unknown hosts never inherit the default store. Ambiguous configured shop
    hosts are rejected rather than crediting a different storefront.
    """
    from store_config import get_store_key_prefix, is_supported_storefront_domain, normalize_storefront_domain

    host = normalize_storefront_domain(domain)
    if not host:
        return None
    if is_supported_storefront_domain(host):
        return get_store_key_prefix(host)
    matches = {
        prefix for prefix in ("OSAKI", "TITAN", "OSAKIMASSAGE", "OSAKITITAN")
        if normalize_storefront_domain(os.getenv(f"{prefix}_SHOP_DOMAIN", "")) == host
    }
    return next(iter(matches)) if len(matches) == 1 else None


def conversion_storefront_domain(domain: str) -> Optional[str]:
    return {
        "OSAKI": "osakiusa.com", "TITAN": "titanchair.com",
        "OSAKIMASSAGE": "osakimassagechair.com", "OSAKITITAN": "osaki-titan.com",
    }.get(resolve_order_store(domain) or "")


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def find_attributed_session(
    email: str,
    *,
    phone: str = "",
    ordered_at: Optional[datetime] = None,
    domain: str = "unknown",
) -> tuple[Optional[str], Optional[str]]:
    """Last matching contact touch in the same store before the order.

    Email remains the preferred key; within a key, sessions and leads compete
    by timestamp. Missing order time or store cannot safely establish a match.
    """
    email = (email or "").strip().lower()
    phone_digits = _phone_digits(phone)
    order_time = _timestamp(ordered_at) if ordered_at else None
    store = resolve_order_store(domain)
    if order_time is None or store is None:
        return None, None
    cutoff = order_time - timedelta(days=attribution_window_days())

    with warranty_db_session() as db:
        # SQL bounds use Chicago wall time because that is how the existing
        # tables persist naive datetimes; final checks normalize to UTC.
        local_zone = pytz.timezone("America/Chicago")
        lower = cutoff.replace(tzinfo=timezone.utc).astimezone(local_zone).replace(tzinfo=None)
        upper = order_time.replace(tzinfo=timezone.utc).astimezone(local_zone).replace(tzinfo=None)
        candidates: dict[str, list[tuple[datetime, str, str]]] = {"email": [], "phone": []}
        for model, email_field, phone_field, email_match, phone_match in (
            (SalesSession, "contact_email", "contact_phone", MATCH_SESSION_EMAIL, MATCH_SESSION_PHONE),
            (SalesLead, "email", "phone", MATCH_LEAD_EMAIL, MATCH_LEAD_PHONE),
        ):
            for row in db.query(model).filter(model.created_at >= lower, model.created_at <= upper).all():
                touched_at = _database_timestamp(row.created_at)
                if resolve_order_store(str(row.domain or "")) != store or touched_at is None or not cutoff <= touched_at <= order_time:
                    continue
                session_id = str(row.session_id or "").strip()
                if not session_id:
                    continue
                if email and str(getattr(row, email_field) or "").strip().lower() == email:
                    candidates["email"].append((touched_at, session_id, email_match))
                if len(phone_digits) == 10 and _phone_digits(str(getattr(row, phone_field) or "")) == phone_digits:
                    candidates["phone"].append((touched_at, session_id, phone_match))
        for key in ("email", "phone"):
            if candidates[key]:
                _, session_id, matched_by = max(candidates[key])
                return session_id, matched_by

    return None, None


def _refund_amount(refund: dict[str, Any], currency: str) -> Optional[float]:
    transactions = refund.get("transactions")
    if not isinstance(transactions, list):
        return None
    amounts = []
    for transaction in transactions:
        if not isinstance(transaction, dict) or str(transaction.get("kind") or "").lower() != "refund":
            continue
        if str(transaction.get("status") or "").lower() != "success":
            continue
        transaction_currency = str(transaction.get("currency") or currency).upper()
        amount = _money(transaction.get("amount"))
        if amount is None or not currency or transaction_currency != currency:
            return None
        amounts.append(amount)
    return round(sum(amounts), 2) if amounts else None


def _refunds(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def is_paid_conversion(row: SalesConversion) -> bool:
    """Paid active orders only; legacy unknown and cancelled/refunded excluded."""
    return str(row.financial_status or "unknown") in {"paid", "partially_refunded"} and row.cancelled_at is None


def conversion_revenue_summary(conversions) -> dict[str, Any]:
    """Revenue is net, payment-confirmed and grouped by actual currency."""
    rows = list(conversions)
    paid = [row for row in rows if is_paid_conversion(row)]
    attributed = [row for row in paid if row.session_id]
    currency_totals: dict[str, float] = {}
    for row in attributed:
        amount = _money(row.net_paid_amount)
        currency = str(row.currency or "").upper()
        if amount is not None and currency:
            currency_totals[currency] = round(currency_totals.get(currency, 0) + amount, 2)
    return {
        "paid_orders_seen": len(paid),
        "paid_orders_attributed": len(attributed),
        "payment_unknown_orders": sum(str(row.financial_status or "unknown") == "unknown" for row in rows),
        "attributed_revenue_usd": currency_totals.get("USD", 0.0),
        "attributed_revenue_by_currency": currency_totals,
    }


def record_order(
    payload: dict[str, Any], *, domain: str = "unknown", topic: str = "orders/create"
) -> Optional[dict]:
    """Upsert authenticated lifecycle evidence, returning None for no change.

    Refund IDs make redelivery idempotent. Payment evidence, refunds and
    cancellation cannot be erased by a late pending/create event. Unknown
    payment state and currencies are never reported as confirmed USD revenue.
    """
    topic = str(topic or "orders/create").strip().lower()
    if topic not in {"orders/create", "orders/paid", "orders/updated", "orders/cancelled", "refunds/create"}:
        return None
    refund_topic = topic == "refunds/create"
    order_id = str(payload.get("order_id") if refund_topic else payload.get("id") or payload.get("admin_graphql_api_id") or "").strip()
    canonical_domain = conversion_storefront_domain(domain)
    if not order_id or order_id == "None" or canonical_domain is None:
        return None
    email = "" if refund_topic else _order_email(payload)
    phone = "" if refund_topic else _order_phone(payload)
    ordered_at = None if refund_topic else _ordered_at(payload)
    session_id, matched_by = find_attributed_session(email, phone=phone, ordered_at=ordered_at, domain=domain)
    event_time = _timestamp(payload.get("updated_at") or payload.get("created_at"))

    with warranty_db_session() as db:
        if db.get_bind().dialect.name == "sqlite":
            # Serialize lifecycle read/modify/write across webhook workers.
            db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        row = db.query(SalesConversion).filter(SalesConversion.shopify_order_id == order_id).one_or_none()
        if row is not None and resolve_order_store(str(row.domain or "")) != resolve_order_store(domain):
            return None  # an ID collision must not update a different store
        is_new = row is None
        if row is None:
            row = SalesConversion(shopify_order_id=order_id, domain=canonical_domain,
                                  financial_status="unknown", refund_events="{}")
            db.add(row)
        before = (row.to_dict(), row.refund_events, row.lifecycle_updated_at)
        stale = event_time is not None and row.lifecycle_updated_at is not None and event_time < row.lifecycle_updated_at
        if ordered_at is not None and row.ordered_at is None:
            row.ordered_at = ordered_at
        if session_id and row.session_id is None:
            row.session_id, row.matched_by = session_id, matched_by
        if email and not row.email:
            row.email = email
        if not refund_topic:
            row.order_number = row.order_number or str(payload.get("order_number") or payload.get("name") or "") or None
            currency = str(payload.get("currency") or "").strip().upper()
            if currency and row.currency and currency != row.currency:
                return None  # an order's original currency cannot change
            if currency and not row.currency:
                row.currency = currency
            total = _order_total(payload)
            if total is not None and (row.total_amount is None or not stale):
                row.total_amount = total
            cancelled = _timestamp(payload.get("cancelled_at"))
            if cancelled is not None or topic == "orders/cancelled":
                row.cancelled_at = row.cancelled_at or cancelled or event_time or datetime.now(timezone.utc).replace(tzinfo=None)
            financial = str(payload.get("financial_status") or "").lower()
            if topic == "orders/paid":
                financial = "paid"
            rank = {"unknown": 0, "pending": 1, "authorized": 1, "partially_paid": 2,
                    "paid": 3, "partially_refunded": 4, "refunded": 5, "voided": 6}
            existing = str(row.financial_status or "unknown")
            stale_unpaid_state = stale and financial in {"pending", "authorized", "partially_paid", "voided"}
            if financial in rank and not stale_unpaid_state and rank[financial] >= rank.get(existing, 0):
                row.financial_status = financial
        refunds = _refunds(row.refund_events)
        incoming = [payload] if refund_topic else payload.get("refunds") or []
        if isinstance(incoming, list):
            for refund in incoming:
                if not isinstance(refund, dict) or refund.get("id") is None:
                    continue
                refund_id = str(refund["id"])
                refund_currency = str(row.currency or "").upper()
                if not refund_currency:
                    currencies = {
                        str(transaction.get("currency") or "").upper()
                        for transaction in refund.get("transactions", [])
                        if isinstance(transaction, dict) and transaction.get("currency")
                    }
                    refund_currency = next(iter(currencies)) if len(currencies) == 1 else ""
                amount = _refund_amount(refund, refund_currency)
                previous = refunds.get(refund_id)
                # Unknown/pending redelivery cannot erase an already verified refund.
                if previous is None or amount is not None:
                    if isinstance(previous, dict) and previous.get("currency") == refund_currency and previous.get("amount") is not None and amount is not None:
                        amount = max(float(previous["amount"]), amount)
                    refunds[refund_id] = {"amount": amount, "currency": refund_currency}
        row.refund_events = json.dumps(refunds, sort_keys=True)
        amounts = [
            _money(value.get("amount"))
            if isinstance(value, dict) and value.get("currency") == row.currency else None
            for value in refunds.values()
        ]
        row.refunded_amount = None if any(value is None for value in amounts) else round(sum(amounts), 2)
        if refunds and row.financial_status in {"paid", "partially_refunded"}:
            if row.refunded_amount is not None and row.total_amount is not None and row.refunded_amount >= row.total_amount:
                row.financial_status = "refunded"
            else:
                row.financial_status = "partially_refunded"
        if row.cancelled_at is not None or row.financial_status in {"refunded", "voided", "pending", "authorized", "partially_paid"}:
            row.net_paid_amount = 0.0
        elif row.financial_status == "paid" and row.total_amount is not None:
            row.net_paid_amount = max(0.0, round(row.total_amount - (row.refunded_amount or 0), 2))
        elif row.financial_status == "partially_refunded" and refunds and row.refunded_amount is not None and row.total_amount is not None:
            row.net_paid_amount = max(0.0, round(row.total_amount - row.refunded_amount, 2))
        else:
            row.net_paid_amount = None
        row.total_usd = row.net_paid_amount if row.currency == "USD" else None
        if event_time is not None and (row.lifecycle_updated_at is None or event_time > row.lifecycle_updated_at):
            row.lifecycle_updated_at = event_time
        after = (row.to_dict(), row.refund_events, row.lifecycle_updated_at)
        if not is_new and before == after:
            return None
        db.flush()
        result = row.to_dict()
    logger.info("sales order lifecycle recorded order=%s attributed=%s state=%s", order_id, bool(result.get("session_id")), result.get("financial_status"))
    return result
