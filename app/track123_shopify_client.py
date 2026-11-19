"""
Track123 Shopify App API client.

Uses the credentials from Track123 App → Settings → General → API & Webhook:
  Header: X-Api-Key
  Host:   https://shp.track123.com/shopify/api/v1/{myshopify-subdomain}/...

Docs:
https://intercom.help/track123/en/articles/11782016-track123-shopify-app-api-documentation

This is NOT the Open API (api.track123.com + Track123-Api-Secret).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

_BASE = "https://shp.track123.com/shopify/api/v1"


def shopify_store_uuid(shop_domain: str) -> str:
    """``osakiusa.myshopify.com`` → ``osakiusa``."""
    host = (shop_domain or "").strip().lower()
    if host.endswith(".myshopify.com"):
        return host[: -len(".myshopify.com")]
    return host.split(".")[0] if host else ""


def resolve_track123_store_uuid(store_config: Dict[str, str]) -> str:
    """Prefer explicit ``track123_shop_uuid`` when Admin host ≠ Track123 uuid."""
    explicit = str(store_config.get("track123_shop_uuid") or "").strip()
    if explicit:
        return explicit.replace(".myshopify.com", "").strip().lower()
    return shopify_store_uuid(str(store_config.get("shop_domain") or ""))


def _api_key(store_config: Dict[str, str]) -> str:
    return str(store_config.get("track123_api_key") or "").strip()


def _get_json(
    path: str,
    api_key: str,
    *,
    http_get: Callable[..., Any],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"{_BASE}/{path.lstrip('/')}"
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json", "accept": "application/json"}
    try:
        response = http_get(url, headers=headers, timeout=12)
    except Exception as exc:
        logger.warning("Track123 Shopify API network error: %s", type(exc).__name__)
        return None, "network_error"
    status = int(getattr(response, "status_code", 0) or 0)
    try:
        payload = response.json()
    except Exception:
        payload = None
    if status == 401:
        return None, "unauthorized"
    if status == 404:
        return None, "not_found"
    if status == 429:
        return None, "rate_limited"
    if status >= 400:
        logger.warning("Track123 Shopify API HTTP %s path=%s", status, path)
        return None, f"http_{status}"
    if not isinstance(payload, dict):
        return None, "invalid_payload"
    return payload, None


def _normalize_events(details: Any) -> List[Dict[str, str]]:
    if not isinstance(details, list):
        return []
    events: List[Dict[str, str]] = []
    for item in details[-5:]:
        if not isinstance(item, dict):
            continue
        events.append(
            {
                "time": str(item.get("event_time") or item.get("event_time_utc") or "").strip()
                or "Unknown time",
                "location": str(item.get("event_location") or "").strip() or "Carrier network",
                "event": str(item.get("event_detail") or "").strip() or "Carrier update",
                "hub": "",
            }
        )
    return events


def _pick_fulfillment(order: Dict[str, Any], tracking_number: str = "") -> Dict[str, Any]:
    fulfillments = order.get("fulfillments") or []
    if not isinstance(fulfillments, list) or not fulfillments:
        return {}
    want = re.sub(r"\s+", "", (tracking_number or "").strip()).upper()
    if want:
        for row in fulfillments:
            if not isinstance(row, dict):
                continue
            tn = re.sub(r"\s+", "", str(row.get("tracking_number") or "")).upper()
            if tn == want:
                return row
    first = fulfillments[0]
    return first if isinstance(first, dict) else {}


def order_payload_to_enrichment(
    payload: Dict[str, Any],
    *,
    tracking_number: str = "",
) -> Dict[str, Any]:
    """Map Shopify App API order JSON into delivery_lookup enrichment shape."""
    order = payload.get("order") if isinstance(payload.get("order"), dict) else payload
    if not isinstance(order, dict):
        return {}

    fulfillment = _pick_fulfillment(order, tracking_number)
    courier = fulfillment.get("courier") if isinstance(fulfillment.get("courier"), dict) else {}
    details = fulfillment.get("tracking_details") or []
    events = _normalize_events(details)
    latest = events[-1] if events else {}

    tn = str(fulfillment.get("tracking_number") or tracking_number or "").strip()
    company = (
        str(courier.get("name") or "").strip()
        or str(fulfillment.get("tracking_company") or "").strip()
        or str(fulfillment.get("carrier_code") or "").strip()
    )
    tracking_url = (
        str(courier.get("query_link") or "").strip()
        or str(order.get("tracking_link") or "").strip()
    )
    status = (
        str(fulfillment.get("transit_status") or "").strip()
        or str(order.get("status") or "").strip()
        or "UNKNOWN"
    )
    last_event = (
        str(fulfillment.get("last_event") or "").strip()
        or latest.get("event", "")
        or "Latest carrier update"
    )

    products: List[str] = []
    for item in fulfillment.get("line_items") or []:
        if isinstance(item, dict) and item.get("title"):
            qty = item.get("quantity") or 1
            title = str(item["title"]).strip()
            products.append(f"{title} (×{qty})" if int(qty or 1) > 1 else title)

    result: Dict[str, Any] = {
        "track123_source": "shopify_app",
        "status": status,
        "company": company,
        "tracking_number": tn,
        "tracking_url": tracking_url,
        "current_location": latest.get("location", "") or "Carrier network",
        "current_hub": "",
        "eta": "Pending carrier update",
        "last_event": last_event,
        "events": events,
        "order_number": str(order.get("order_name") or order.get("order_number") or "").strip(),
        "purchase_date_raw": str(order.get("order_create_at") or "").strip(),
    }
    if products:
        result["product_names"] = products
    return result


def query_by_tracking_number(
    tracking_number: str,
    store_config: Dict[str, str],
    *,
    http_get: Callable[..., Any],
) -> Dict[str, Any]:
    tn = (tracking_number or "").strip()
    api_key = _api_key(store_config)
    uuid = resolve_track123_store_uuid(store_config)
    if not tn or not api_key or not uuid:
        return {}
    payload, error = _get_json(
        f"{quote(uuid, safe='')}/orders/by-tracking/{quote(tn, safe='')}.json",
        api_key,
        http_get=http_get,
    )
    if error or not payload:
        if error and error != "not_found":
            logger.warning("Track123 Shopify by-tracking failed: %s store=%s", error, uuid)
        return {"track123_shopify_error": error or "empty"} if error else {}
    return order_payload_to_enrichment(payload, tracking_number=tn)


def query_by_order_number(
    order_number: str,
    store_config: Dict[str, str],
    *,
    http_get: Callable[..., Any],
    checkout_email: str = "",
) -> Dict[str, Any]:
    """Lookup by Shopify order number; optionally require checkout email match."""
    raw = (order_number or "").strip().lstrip("#")
    digits = "".join(re.findall(r"\d+", raw)) or raw
    api_key = _api_key(store_config)
    uuid = resolve_track123_store_uuid(store_config)
    if not digits or not api_key or not uuid:
        return {}
    payload, error = _get_json(
        f"{quote(uuid, safe='')}/orders/by-number/{quote(digits, safe='')}.json",
        api_key,
        http_get=http_get,
    )
    if error or not payload:
        if error and error != "not_found":
            logger.warning("Track123 Shopify by-number failed: %s store=%s", error, uuid)
        return {"track123_shopify_error": error or "empty"} if error else {}

    order = payload.get("order") if isinstance(payload.get("order"), dict) else payload
    if checkout_email and isinstance(order, dict):
        customer = order.get("customer") if isinstance(order.get("customer"), dict) else {}
        found = str(customer.get("email") or "").strip().lower()
        want = checkout_email.strip().lower()
        if found and want and found != want:
            return {"track123_shopify_error": "email_mismatch"}
    return order_payload_to_enrichment(payload)


def shopify_app_auth_status(
    store_config: Dict[str, str],
    *,
    http_get: Callable[..., Any],
) -> Dict[str, Any]:
    """Probe auth with a definitely-missing order number."""
    api_key = _api_key(store_config)
    uuid = resolve_track123_store_uuid(store_config)
    if not api_key or not uuid:
        return {"configured": False, "ok": False, "error_code": "missing_credentials"}
    _payload, error = _get_json(
        f"{quote(uuid, safe='')}/orders/by-number/99999999.json",
        api_key,
        http_get=http_get,
    )
    if error in {None, "not_found"}:
        # 404 means the key was accepted.
        return {"configured": True, "ok": True, "store_uuid": uuid, "error_code": None}
    return {"configured": True, "ok": False, "store_uuid": uuid, "error_code": error or "unknown"}
