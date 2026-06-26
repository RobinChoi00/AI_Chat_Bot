"""
Track123 API client for warranty delivery lookups.

Uses Track123 v2.1:
  - POST /tk/v2.1/track/query-realtime  (ad-hoc tracking numbers)
  - POST /tk/v2.1/track/query           (numbers already in Track123 account)

Docs: https://docs.track123.com/reference/query-realtime
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_COURIER_PATTERNS = [
    (r"^1Z[A-Z0-9]{16}$", "ups"),
    (r"^9[2-5]\d{20,}$", "usps"),
    (r"^(94|93|92|95)\d{18,}$", "usps"),
    (r"^\d{20,22}$", "usps"),
    (r"^\d{12,15}$", "fedex"),
    (r"^C\d{8,}$", "ontrac"),
    (r"^1LS\d+$", "lasership"),
    (r"^TBA\d+$", "amazon-logistics-us"),
]

_COMPANY_TO_CODE = {
    "ups": "ups",
    "fedex": "fedex",
    "usps": "usps",
    "dhl": "dhl",
    "ontrac": "ontrac",
    "lasership": "lasership",
    "amazon": "amazon-logistics-us",
}


def infer_courier_code(
    company: str = "",
    tracking_number: str = "",
    tracking_url: str = "",
) -> Optional[str]:
    """Best-effort Track123 courierCode (None = auto-detect)."""
    tn = (tracking_number or "").strip()
    for pattern, slug in _COURIER_PATTERNS:
        if re.match(pattern, tn, re.IGNORECASE):
            return slug

    url_lower = (tracking_url or "").lower()
    for keyword, slug in (
        ("fedex.com", "fedex"),
        ("ups.com", "ups"),
        ("usps.com", "usps"),
        ("dhl.com", "dhl"),
        ("ontrac.com", "ontrac"),
        ("lasership", "lasership"),
        ("amazon", "amazon-logistics-us"),
    ):
        if keyword in url_lower:
            return slug

    company_lower = (company or "").lower()
    for key, slug in _COMPANY_TO_CODE.items():
        if key in company_lower:
            return slug
    return None


def _pick_first(data: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_events(details: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for event in details[-3:]:
        if not isinstance(event, dict):
            continue
        normalized.append(
            {
                "time": _pick_first(event, ["eventTime", "time", "checkpoint_time"]) or "Unknown time",
                "location": _pick_first(event, ["address", "location", "city"]) or "Carrier network",
                "event": _pick_first(event, ["eventDetail", "message", "description", "tag"])
                or "Carrier update",
                "hub": _pick_first(event, ["facility", "hub", "center"]),
            }
        )
    return normalized


def parse_track123_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Track123 tracking record into enrich_tracking_from_track123 shape."""
    logistics = record.get("localLogisticsInfo") or {}
    if not isinstance(logistics, dict):
        logistics = {}

    details = logistics.get("trackingDetails") or record.get("trackingDetails") or []
    if not isinstance(details, list):
        details = []

    normalized_events = _normalize_events(details)
    latest = normalized_events[-1] if normalized_events else {}

    eta = _pick_first(
        record,
        ["expectedDeliveryTime", "deliveredTime", "nextUpdateTime", "lastTrackingTime"],
    ) or "Pending carrier update"

    courier_name = _pick_first(logistics, ["courierNameEN", "courierNameCN", "courierCode"])
    tracking_url = _pick_first(logistics, ["courierTrackingLink"])

    return {
        "track123_source": "enabled",
        "status": _pick_first(record, ["transitStatus", "transitSubStatus", "trackingStatus"])
        or "IN_TRANSIT",
        "company": courier_name,
        "tracking_url": tracking_url,
        "current_location": latest.get("location", "") or "Carrier network",
        "current_hub": latest.get("hub", "") or "Carrier transit hub",
        "eta": eta,
        "last_event": latest.get("event", "") or "Latest carrier update pending.",
        "events": normalized_events,
    }


def _extract_tracking_record(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the first usable tracking object from Track123 response envelopes."""
    data = payload.get("data")
    if isinstance(data, dict):
        accepted = data.get("accepted")
        if isinstance(accepted, dict):
            content = accepted.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                return first if isinstance(first, dict) else None
            if isinstance(accepted, dict) and accepted.get("trackNo"):
                return accepted
        if isinstance(accepted, list) and accepted:
            first = accepted[0]
            return first if isinstance(first, dict) else None
        if data.get("trackNo"):
            return data

    root = payload.get("tracking", payload)
    if isinstance(root, dict) and root.get("trackNo"):
        return root
    return None


def _track123_post(
    path: str,
    body: Dict[str, Any],
    api_key: str,
    *,
    http_post,
) -> Optional[Dict[str, Any]]:
    base_url = os.getenv("TRACK123_API_BASE_URL", "https://api.track123.com").rstrip("/")
    url = f"{base_url}{path}"
    headers = {
        "Track123-Api-Secret": api_key,
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        response = http_post(url, headers=headers, json=body, timeout=8)
        if response.status_code >= 400:
            logger.warning("Track123 %s failed: HTTP %s", path, response.status_code)
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        code = str(payload.get("code", ""))
        if code and code not in ("00000", "0", "200"):
            logger.warning("Track123 %s returned code=%s msg=%s", path, code, payload.get("msg"))
            return None
        return payload
    except Exception as exc:
        logger.warning("Track123 %s error: %s", path, exc)
        return None


def query_track123_tracking(
    tracking_number: str,
    store_config: Dict[str, str],
    *,
    company: str = "",
    tracking_url: str = "",
    http_post,
) -> Dict[str, Any]:
    """
    Query Track123 for *tracking_number* using store credentials.

    Tries realtime lookup first (no pre-registration), then registered query.
    """
    tn = (tracking_number or "").strip()
    api_key = (store_config.get("track123_api_key") or "").strip()
    if not api_key or not tn:
        return {}

    courier_code = infer_courier_code(company, tn, tracking_url)
    realtime_body: Dict[str, Any] = {"trackNo": tn, "lang": "en"}
    if courier_code:
        realtime_body["courierCode"] = courier_code

    payload = _track123_post(
        "/gateway/open-api/tk/v2.1/track/query-realtime",
        realtime_body,
        api_key,
        http_post=http_post,
    )
    record = _extract_tracking_record(payload) if payload else None

    if record is None:
        track_info: Dict[str, Any] = {"trackNo": tn}
        if courier_code:
            track_info["courierCode"] = courier_code
        payload = _track123_post(
            "/gateway/open-api/tk/v2.1/track/query",
            {"trackNoInfos": [track_info], "queryPageSize": 1},
            api_key,
            http_post=http_post,
        )
        record = _extract_tracking_record(payload) if payload else None

    if not record:
        return {}

    return parse_track123_record(record)
