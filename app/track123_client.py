"""
Track123 API client for warranty delivery lookups.

Uses Track123 v2.1:
  - POST /tk/v2.1/track/query-realtime  (ad-hoc tracking numbers)
  - POST /tk/v2.1/track/query           (numbers already in Track123 account)
  - POST /tk/v2/track/query             (legacy trackNos body)

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
        ["expectedDeliveryTime"],
    ) or "Pending carrier update"

    courier_name = _pick_first(logistics, ["courierNameEN", "courierNameCN", "courierCode"])
    tracking_url = _pick_first(logistics, ["courierTrackingLink"])

    return {
        "track123_source": "enabled",
        "status": _pick_first(record, ["transitStatus", "transitSubStatus", "trackingStatus"])
        or "UNKNOWN",
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
            if accepted.get("trackNo"):
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


def _credential_candidates(store_config: Dict[str, str]) -> list[tuple[str, str]]:
    """Prefer API key, then token — some storefronts store the Open API secret as TOKEN."""
    candidates: list[tuple[str, str]] = []
    for label, raw in (
        ("api_key", store_config.get("track123_api_key")),
        ("token", store_config.get("track123_token")),
    ):
        value = str(raw or "").strip()
        if value and value not in {item[1] for item in candidates}:
            candidates.append((label, value))
    return candidates


def _track123_post(
    path: str,
    body: Dict[str, Any],
    api_secret: str,
    *,
    http_post,
) -> Optional[Dict[str, Any]]:
    base_url = os.getenv("TRACK123_API_BASE_URL", "https://api.track123.com").rstrip("/")
    url = f"{base_url}{path}"
    headers = {
        "Track123-Api-Secret": api_secret,
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
            logger.warning(
                "Track123 %s returned code=%s msg=%s",
                path,
                code,
                str(payload.get("msg") or "")[:80],
            )
            return {"__track123_error_code": code, "__track123_error_msg": str(payload.get("msg") or "")}
        return payload
    except Exception as exc:
        logger.warning("Track123 %s error: %s", path, type(exc).__name__)
        return None


def _payload_ok(payload: Optional[Dict[str, Any]]) -> bool:
    return bool(payload) and "__track123_error_code" not in (payload or {})


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

    Tries each configured secret (API key, then token) against realtime,
    registered v2.1, and legacy v2 query endpoints.
    """
    tn = (tracking_number or "").strip()
    credentials = _credential_candidates(store_config)
    if not credentials or not tn:
        return {}

    courier_code = infer_courier_code(company, tn, tracking_url)
    realtime_body: Dict[str, Any] = {"trackNo": tn, "lang": "en"}
    if courier_code:
        realtime_body["courierCode"] = courier_code
    track_info: Dict[str, Any] = {"trackNo": tn}
    if courier_code:
        track_info["courierCode"] = courier_code

    attempts = (
        ("/gateway/open-api/tk/v2.1/track/query-realtime", realtime_body),
        (
            "/gateway/open-api/tk/v2.1/track/query",
            {"trackNoInfos": [track_info], "queryPageSize": 1},
        ),
        ("/gateway/open-api/tk/v2/track/query", {"trackNos": [tn]}),
    )

    last_auth_error = ""
    for _label, secret in credentials:
        for path, body in attempts:
            payload = _track123_post(path, body, secret, http_post=http_post)
            if payload and payload.get("__track123_error_code") in {"B0001", "A0230"}:
                last_auth_error = str(payload.get("__track123_error_code") or "")
                continue
            if not _payload_ok(payload):
                continue
            record = _extract_tracking_record(payload or {})
            if record:
                return parse_track123_record(record)

    if last_auth_error:
        return {"track123_auth_failed": True, "track123_error_code": last_auth_error}
    return {}


def track123_auth_status(store_config: Dict[str, str], *, http_get) -> Dict[str, Any]:
    """Non-secret health check against the account quota endpoint."""
    credentials = _credential_candidates(store_config)
    if not credentials:
        return {"configured": False, "ok": False, "error_code": "missing_credentials"}

    base_url = os.getenv("TRACK123_API_BASE_URL", "https://api.track123.com").rstrip("/")
    url = f"{base_url}/gateway/open-api/tk/v2.1/account/quota"
    for label, secret in credentials:
        try:
            response = http_get(
                url,
                headers={
                    "Track123-Api-Secret": secret,
                    "accept": "application/json",
                },
                timeout=8,
            )
            payload = response.json() if response.status_code < 500 else {}
            code = str((payload or {}).get("code") or "")
            if code in ("00000", "0", "200"):
                return {"configured": True, "ok": True, "credential": label, "error_code": None}
            if code:
                return {
                    "configured": True,
                    "ok": False,
                    "credential": label,
                    "error_code": code,
                }
        except Exception:
            continue
    return {"configured": True, "ok": False, "error_code": "unreachable"}
