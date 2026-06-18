import os
import logging
import hmac
import hashlib
import base64
import re
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import SALES_EMAIL_BY_DOMAIN, EMAIL_SENDER, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, cast
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import pytz
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.documents import Document
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential,
)
import json as _json
try:
    from app.agent_tools import (
        HybridRetriever,
        TOOL_SCHEMAS,
        WARRANTY_TOOL_SCHEMAS,
        tool_search_chair_specs,
        tool_recommend_chairs,
        tool_get_repair_help,
        tool_get_warranty_or_policy,
        tool_lookup_order_status,
        tool_capture_sales_lead,
        tool_escalate_to_human,
        tool_get_showroom_info,
        tool_start_warranty_workflow,
        tool_answer_warranty_question,
        tool_attach_warranty_evidence,
    )
    from app.cost_guard import (
        UsageRecorder,
        cache_get,
        cache_set,
        cache_stats,
        faiss_rwlock,
        limiter,
        make_cache_key,
    )
    from app.warranty_workflow import WarrantyEngine
except ImportError:
    from agent_tools import (  # type: ignore
        HybridRetriever,
        TOOL_SCHEMAS,
        WARRANTY_TOOL_SCHEMAS,
        tool_search_chair_specs,
        tool_recommend_chairs,
        tool_get_repair_help,
        tool_get_warranty_or_policy,
        tool_lookup_order_status,
        tool_capture_sales_lead,
        tool_escalate_to_human,
        tool_get_showroom_info,
        tool_start_warranty_workflow,
        tool_answer_warranty_question,
        tool_attach_warranty_evidence,
    )
    from cost_guard import (  # type: ignore
        UsageRecorder,
        cache_get,
        cache_set,
        cache_stats,
        faiss_rwlock,
        limiter,
        make_cache_key,
    )
    from warranty_workflow import WarrantyEngine  # type: ignore

# 💡 [비즈니스 & 시스템 설정 임포트]
from config import (
    SUPPORT_BUSINESS_HOURS,
    COMPANY_ADDRESS,
    DEFAULT_TARGET_DOMAIN,
    AGENT_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    OPENAI_REQUEST_TIMEOUT,
    OPENAI_MAX_RETRIES,
    CORS_ALLOWED_ORIGINS,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_PER_HOUR,
    get_contact_msg,
)



def _resolve_sales_email(domain: str) -> str:
    """Return the correct sales inbox for a given target_domain URL string."""
    domain_lower = (domain or "").lower()
    for key, email in SALES_EMAIL_BY_DOMAIN.items():
        if key in domain_lower:
            return email
    return list(SALES_EMAIL_BY_DOMAIN.values())[0]  # fallback: first brand

def send_sales_lead_email(customer_email: str, query_content: str, product_info: str, domain: str) -> bool:
    """세일즈 팀에게 고객 리드 정보를 이메일로 전송합니다."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.error("🚨 [Email Error] EMAIL_SENDER or EMAIL_PASSWORD is not set in .env")
        return False

    target_email = _resolve_sales_email(domain)
    logger.info(f"📧 [Email Prep] domain='{domain}' → target='{target_email}'")

    subject = f"[AI Lead] New Sales Inquiry from {customer_email}"
    body = (
        f"New sales lead from AI chatbot.\n\n"
        f"Customer Email : {customer_email}\n"
        f"Site           : {domain}\n"
        f"Customer Query : {query_content}\n"
        f"Bot Response   : {product_info}\n\n"
        f"-- Sent automatically by AI Agent --"
    )

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = target_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(f"✅ [Email Sent] Sales lead → {target_email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"🚨 [Email Auth Failed] Check EMAIL_SENDER/PASSWORD in .env: {e}")
    except smtplib.SMTPException as e:
        logger.error(f"🚨 [Email SMTP Error] {e}")
    except Exception as e:
        logger.error(f"🚨 [Email Unknown Error] {e}")
    return False

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 💡 [보안] Fail-Fast 원칙: 필수 환경변수는 시작 단계에서 검증.
# 누락된 경우 의미 있는 메시지와 함께 즉시 종료하여 런타임에서 조용히
# 실패하는 일을 막습니다. SMTP 같은 옵션 변수는 경고만 출력합니다.
# ---------------------------------------------------------------------------
REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "SHOPIFY_WEBHOOK_SECRET")
_missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
if _missing:
    raise RuntimeError(
        f"🚨 CRITICAL: Missing required env vars: {', '.join(_missing)}. "
        "Refusing to start. See .env.example for the full list."
    )

SHOPIFY_WEBHOOK_SECRET = os.getenv("SHOPIFY_WEBHOOK_SECRET") or ""  # validated above

# Soft-check email transport — log a warning if disabled so it's obvious.
if not (os.getenv("EMAIL_SENDER") and os.getenv("EMAIL_PASSWORD")):
    logger.warning(
        "⚠️ EMAIL_SENDER / EMAIL_PASSWORD not set — sales-lead capture will log only, "
        "no outbound email will be sent."
    )

# --- [0] Helper constants & Functions ---
# NOTE: Legacy keyword-based intent routing (TECHNICIAN_KEYWORDS, PRODUCT_QUERY_KEYWORDS,
# TRACKING_KEYWORDS, etc.) was removed when the agentic tool-calling endpoint became
# canonical — the LLM + tool schemas now perform routing. The patterns used by the
# active endpoint live in `intent_router.py` (repair / tracking / price / recommend).
# `_TRACKING_INTENT_PATTERNS`, and `_SHOWROOM_INTENT_PATTERNS` further below.

def get_store_key_prefix(target_domain: str) -> str:
    lowered = (target_domain or "").lower()
    if "titanchair.com" in lowered:
        return "TITAN"
    if "osakimassagechair.com" in lowered:
        return "OSAKIMASSAGE"
    return "OSAKI"

def get_store_config(target_domain: str) -> Dict[str, str]:
    """Resolve per-store Shopify and Track123 credentials from env."""
    prefix = get_store_key_prefix(target_domain)
    return {
        "shop_domain": os.getenv(f"{prefix}_SHOP_DOMAIN", "").strip(),
        "shop_access_token": os.getenv(f"{prefix}_ACCESS_TOKEN", "").strip(),
        "track123_api_key": os.getenv(f"{prefix}_TRACK123_API_KEY", "").strip(),
        "track123_token": os.getenv(f"{prefix}_TRACK123_TOKEN", "").strip(),
    }

def _pick_first_non_empty(data: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""

# ---------------------------------------------------------------------------
# HTTP helpers with automatic retry/backoff
# ---------------------------------------------------------------------------
# Network blips between us and Shopify/Track123/AfterShip cause spurious
# "Order not found" errors today. tenacity's exponential backoff retries
# transient failures (connection errors, timeouts, 5xx) up to 3 times
# without blowing up the LLM-facing tool result.

_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class _Retryable5xxError(requests.exceptions.HTTPError):
    """Sentinel so tenacity retries on transient 5xx but NOT on 4xx."""


def _check_status_for_retry(resp: requests.Response) -> requests.Response:
    if 500 <= resp.status_code < 600:
        raise _Retryable5xxError(f"{resp.status_code}: {resp.text[:200]}")
    return resp


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=1, max=6),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS + (_Retryable5xxError,)),
    reraise=True,
)
def http_get_with_retry(url: str, **kwargs) -> requests.Response:
    return _check_status_for_retry(requests.get(url, **kwargs))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=1, max=6),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS + (_Retryable5xxError,)),
    reraise=True,
)
def http_post_with_retry(url: str, **kwargs) -> requests.Response:
    return _check_status_for_retry(requests.post(url, **kwargs))


def _normalize_track123_events(events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for event in events[-3:]:
        city = _pick_first_non_empty(event, ["city", "location_city"])
        state = _pick_first_non_empty(event, ["state", "province", "location_state"])
        country = _pick_first_non_empty(event, ["country", "country_name", "location_country"])
        location_parts = [x for x in [city, state, country] if x]
        location = ", ".join(location_parts) if location_parts else _pick_first_non_empty(event, ["location", "facility", "hub"])

        normalized.append({
            "time": _pick_first_non_empty(event, ["time", "checkpoint_time", "event_time", "updated_at"]) or "Unknown time",
            "location": location or "Carrier network",
            "event": _pick_first_non_empty(event, ["message", "description", "status", "tag"]) or "Carrier update",
            "hub": _pick_first_non_empty(event, ["facility", "hub", "center"]) or "",
        })
    return normalized

def enrich_tracking_from_track123(tracking_number: str, store_config: Dict[str, str]) -> Dict[str, Any]:
    """Fetch richer location/hub/ETA data from Track123 if configured."""
    api_key = store_config.get("track123_api_key", "")
    token = store_config.get("track123_token", "")
    if not api_key or not tracking_number:
        return {}

    base_url = os.getenv("TRACK123_API_BASE_URL", "https://api.track123.com").rstrip("/")
    endpoint_template = os.getenv(
        "TRACK123_TRACKING_ENDPOINT_TEMPLATE",
        "/api/v1/trackings/{tracking_number}"
    )
    endpoint = endpoint_template.format(tracking_number=tracking_number)
    url = f"{base_url}{endpoint}"

    headers = {
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if token:
        headers["X-Track123-Token"] = token

    try:
        response = http_get_with_retry(url, headers=headers, timeout=6)
        if response.status_code >= 400:
            logger.warning(f"⚠️ Track123 lookup failed: {response.status_code}")
            return {}

        payload = response.json()
        root = payload.get("data", payload)
        if isinstance(root, list):
            root = root[0] if root else {}
        tracking = root.get("tracking", root) if isinstance(root, dict) else {}
        if not isinstance(tracking, dict):
            return {}

        events = tracking.get("events") or tracking.get("checkpoints") or tracking.get("history") or []
        if not isinstance(events, list):
            events = []

        normalized_events = _normalize_track123_events(events)
        latest_event = normalized_events[-1] if normalized_events else {}

        eta = _pick_first_non_empty(tracking, ["eta", "estimated_delivery", "expected_delivery", "delivery_date"]) or "Pending carrier update"
        current_hub = _pick_first_non_empty(tracking, ["current_hub", "hub", "facility", "distribution_center"])
        if not current_hub and latest_event:
            current_hub = latest_event.get("hub", "")

        return {
            "track123_source": "enabled",
            "status": _pick_first_non_empty(tracking, ["status", "delivery_status", "tag"]),
            "current_location": _pick_first_non_empty(tracking, ["current_location", "location"]) or latest_event.get("location", "Carrier network"),
            "current_hub": current_hub or "Carrier transit hub",
            "eta": eta,
            "last_event": _pick_first_non_empty(tracking, ["last_event"]) or latest_event.get("event", "Latest carrier update pending."),
            "events": normalized_events,
        }
    except Exception as e:
        logger.warning(f"⚠️ Track123 enrich error: {e}")
        return {}

def _extract_shopify_order_details(node: Dict[str, Any]) -> Dict[str, Any]:
    """Pull warranty-relevant order fields from a Shopify GraphQL order node."""
    details: Dict[str, Any] = {}

    name = (node.get("name") or "").strip()
    if name:
        details["order_number"] = name

    created_at = (node.get("createdAt") or "").strip()
    if created_at:
        details["purchase_date_raw"] = created_at

    products: List[str] = []
    for edge in (node.get("lineItems") or {}).get("edges") or []:
        item = (edge or {}).get("node") or {}
        title = (item.get("title") or "").strip()
        if not title:
            continue
        qty = item.get("quantity") or 1
        try:
            qty_int = int(qty)
        except (TypeError, ValueError):
            qty_int = 1
        if qty_int > 1:
            products.append(f"{title} (×{qty_int})")
        else:
            products.append(title)
    if products:
        details["product_names"] = products

    money = (node.get("totalPriceSet") or {}).get("shopMoney") or {}
    amount = money.get("amount")
    currency = (money.get("currencyCode") or "USD").strip()
    if amount is not None and str(amount).strip():
        details["total_amount"] = str(amount).strip()
        details["currency_code"] = currency

    return details


def _merge_shopify_order_details(base: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(_extract_shopify_order_details(node))
    return merged


def _shopify_orders_search(
    url: str,
    headers: Dict[str, str],
    query: str,
    search_query: str,
) -> List[Dict[str, Any]]:
    """Run a Shopify orders GraphQL search and return edge list."""
    response = http_post_with_retry(
        url,
        json={"query": query, "variables": {"query": search_query}},
        headers=headers,
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("data", {}).get("orders", {}).get("edges", [])


def fetch_shopify_order_status(order_number: str, email: str, target_domain: str) -> Dict[str, Any]:
    """접속 도메인에 맞춰 3개의 스토어 토큰 중 하나를 선택해 쇼피파이 API를 직접 호출합니다."""
    store_config = get_store_config(target_domain)
    SHOP_DOMAIN = store_config["shop_domain"]
    ACCESS_TOKEN = store_config["shop_access_token"]

    if not SHOP_DOMAIN or not ACCESS_TOKEN:
        logger.error(f"🚨 API Token missing for domain: {target_domain}")
        return {"error": f"시스템 설정 오류: {target_domain}의 API 인증 정보가 없습니다."}

    url = f"https://{SHOP_DOMAIN}/admin/api/2024-01/graphql.json"
    headers = {"Content-Type": "application/json", "X-Shopify-Access-Token": ACCESS_TOKEN}

    query = """
    query getOrderTracking($query: String!) {
      orders(first: 1, query: $query, sortKey: CREATED_AT, reverse: true) {
        edges {
          node {
            name
            createdAt
            displayFulfillmentStatus
            totalPriceSet {
              shopMoney {
                amount
                currencyCode
              }
            }
            lineItems(first: 10) {
              edges {
                node {
                  title
                  quantity
                }
              }
            }
            fulfillments {
              trackingInfo {
                company
                number
                url
              }
            }
          }
        }
      }
    }
    """
    try:
        clean_order = order_number.replace("#", "").strip()
        clean_email = (email or "").strip()
        order_candidates = [clean_order] if clean_order else []
        if clean_order:
            digits_only = "".join(re.findall(r"\d+", clean_order))
            if digits_only and digits_only != clean_order:
                order_candidates.append(digits_only)

        edges: List[Dict[str, Any]] = []

        if clean_email and clean_order:
            for candidate in order_candidates:
                edges = _shopify_orders_search(
                    url,
                    headers,
                    query,
                    f"name:'{candidate}' AND email:'{clean_email}'",
                )
                if edges:
                    logger.info(f"✅ Order found by name:'{candidate}' + email")
                    break

        if not edges and clean_order:
            logger.info(f"🔍 Trying order-number-only search for: {clean_order}")
            for candidate in order_candidates:
                edges = _shopify_orders_search(
                    url,
                    headers,
                    query,
                    f"name:'{candidate}'",
                )
                if edges:
                    logger.info(f"✅ Order found by name:'{candidate}'")
                    break

        if not edges and clean_email:
            logger.info(f"🔍 Trying email-only fallback for: {clean_email}")
            edges = _shopify_orders_search(
                url,
                headers,
                query,
                f"email:'{clean_email}'",
            )
            if edges:
                logger.info(f"✅ Order found by email-only fallback: {clean_email}")

        if not edges:
            return {"error": "Order not found, or the email does not match our records."}

        node = edges[0]["node"]
        status = node.get("displayFulfillmentStatus", "UNFULFILLED")

        if status == "UNFULFILLED" or not node.get("fulfillments"):
            return _merge_shopify_order_details(
                {
                    "status": "PROCESSING",
                    "message": "Your order is confirmed and being prepared at the warehouse.",
                    "current_location": "Origin warehouse",
                    "current_hub": "Fulfillment center (pre-shipment)",
                    "eta": "Pending carrier pickup",
                    "last_event": "Order confirmed and waiting for carrier handoff.",
                    "events": [],
                },
                node,
            )

        fulfillments = node.get("fulfillments") or []
        tracking_info_list = (fulfillments[0] or {}).get("trackingInfo") or []
        if not tracking_info_list:
            return _merge_shopify_order_details(
                {
                    "status": "PROCESSING",
                    "message": "Your order is confirmed and being prepared at the warehouse.",
                    "current_location": "Origin warehouse",
                    "current_hub": "Fulfillment center (pre-shipment)",
                    "eta": "Pending carrier pickup",
                    "last_event": "Order confirmed and waiting for carrier handoff.",
                    "events": [],
                },
                node,
            )

        tracking_info = tracking_info_list[0]
        raw_company = tracking_info.get("company", "")
        raw_number = tracking_info.get("number", "")
        raw_url = tracking_info.get("url", "")
        resolved_company = resolve_carrier_name(raw_company, raw_number, raw_url)
        tracking_data = _merge_shopify_order_details(
            {
                "status": status,
                "company": resolved_company,
                "tracking_number": raw_number,
                "tracking_url": tracking_info.get("url", ""),
                "current_location": "Carrier network",
                "current_hub": "In transit hub (latest carrier scan)",
                "eta": "Pending carrier update",
                "last_event": "Carrier label created or initial scan received.",
                "events": [],
            },
            node,
        )
        enriched = enrich_tracking_from_track123(
            tracking_data.get("tracking_number", ""),
            store_config
        )
        if not enriched:
            enriched = enrich_tracking_from_aftership(
                tracking_data.get("company", ""),
                tracking_data.get("tracking_number", "")
            )
        if enriched:
            tracking_data.update(enriched)
        return tracking_data
    except Exception as e:
        logger.error(f"🚨 Shopify API Error: {e}")
        return {"error": "A temporary logistics server communication error occurred."}

CARRIER_PATTERNS = [
    (r"^1Z[A-Z0-9]{16}$", "UPS", "ups"),
    (r"^9[2-5]\d{20,}$", "USPS", "usps"),
    (r"^(94|93|92|95)\d{18,}$", "USPS", "usps"),
    (r"^\d{20,22}$", "USPS", "usps"),
    (r"^\d{12,15}$", "FedEx", "fedex"),
    (r"^C\d{8,}$", "OnTrac", "ontrac"),
    (r"^1LS\d+$", "LaserShip", "lasership"),
    (r"^TBA\d+$", "Amazon Logistics", "amazon-logistics-us"),
]

CARRIER_URL_KEYWORDS = {
    "fedex.com": "FedEx",
    "ups.com": "UPS",
    "usps.com": "USPS",
    "dhl.com": "DHL",
    "aitworldwide": "AIT Logistics",
    "abf.com": "ABF Freight",
    "arcb.com": "ABF Freight",
    "metro-logistics": "Metropolitan Logistics",
    "metropolitan": "Metropolitan Logistics",
    "rrts.com": "Roadrunner Freight",
    "roadrunner": "Roadrunner Freight",
    "estes-express": "Estes Express",
    "ontrac.com": "OnTrac",
    "lasership": "LaserShip",
    "amazon": "Amazon Logistics",
}

def infer_carrier_from_url(tracking_url: str) -> str:
    """Infer carrier from tracking URL domain (most reliable for freight)."""
    url_lower = (tracking_url or "").lower()
    for keyword, carrier_name in CARRIER_URL_KEYWORDS.items():
        if keyword in url_lower:
            return carrier_name
    return ""

def infer_carrier_from_tracking_number(tracking_number: str) -> str:
    """Infer carrier name from tracking number pattern (fallback)."""
    tn = (tracking_number or "").strip().upper()
    for pattern, name, _ in CARRIER_PATTERNS:
        if re.match(pattern, tn, re.IGNORECASE):
            return name
    return ""

def resolve_carrier_name(company: str, tracking_number: str, tracking_url: str = "") -> str:
    """Return a meaningful carrier name using multiple signals."""
    c = (company or "").strip()
    if c and c.lower() not in ("other", "unknown", "알 수 없는 택배사", ""):
        return c
    from_url = infer_carrier_from_url(tracking_url)
    if from_url:
        return from_url
    from_number = infer_carrier_from_tracking_number(tracking_number)
    if from_number:
        return from_number
    return company or "Unknown carrier"

def infer_aftership_slug(company: str) -> str:
    """Map common carrier names to AfterShip slugs."""
    c = (company or "").lower()
    mapping = {
        "ups": "ups",
        "fedex": "fedex",
        "usps": "usps",
        "dhl": "dhl",
        "ontrac": "ontrac",
        "lasership": "lasership",
        "amazon logistics": "amazon-logistics-us",
    }
    for key, slug in mapping.items():
        if key in c:
            return slug
    return "ups"

def enrich_tracking_from_aftership(company: str, tracking_number: str) -> Dict[str, Any]:
    """Best-effort tracking enrichment: latest hub/location/ETA/checkpoints."""
    api_key = os.getenv("AFTERSHIP_API_KEY", "").strip()
    if not api_key or not tracking_number:
        return {}

    slug = infer_aftership_slug(company)
    url = f"https://api.aftership.com/v4/trackings/{slug}/{tracking_number}"
    headers = {
        "aftership-api-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = http_get_with_retry(url, headers=headers, timeout=6)
        if response.status_code >= 400:
            logger.warning(f"⚠️ AfterShip lookup failed: {response.status_code}")
            return {}

        payload = response.json()
        tracking = payload.get("data", {}).get("tracking", {})
        checkpoints = tracking.get("checkpoints", []) or []
        expected_delivery = tracking.get("expected_delivery") or ""
        tag = tracking.get("tag", "")

        latest_checkpoint = checkpoints[-1] if checkpoints else {}
        city = latest_checkpoint.get("city", "")
        state = latest_checkpoint.get("state", "")
        country = latest_checkpoint.get("country_name", "")
        location_parts = [x for x in [city, state, country] if x]
        location = ", ".join(location_parts) if location_parts else "Carrier network"
        latest_event = latest_checkpoint.get("message", "") or latest_checkpoint.get("tag", "") or "Latest carrier scan received."
        hub = latest_checkpoint.get("facility", "") or "Carrier transit hub"

        recent_events = []
        for checkpoint in checkpoints[-3:]:
            cp_city = checkpoint.get("city", "")
            cp_state = checkpoint.get("state", "")
            cp_country = checkpoint.get("country_name", "")
            cp_location_parts = [x for x in [cp_city, cp_state, cp_country] if x]
            cp_location = ", ".join(cp_location_parts) if cp_location_parts else "Carrier network"
            recent_events.append({
                "time": checkpoint.get("checkpoint_time", ""),
                "location": cp_location,
                "event": checkpoint.get("message", "") or checkpoint.get("tag", "Carrier update"),
            })

        return {
            "aftership_tag": tag,
            "current_location": location,
            "current_hub": hub,
            "eta": expected_delivery or "Pending carrier update",
            "last_event": latest_event,
            "events": recent_events,
        }
    except Exception as e:
        logger.warning(f"⚠️ AfterShip enrich error: {e}")
        return {}

def build_deterministic_tracking_response(tracking_data: Dict[str, Any], target_domain: str) -> str:
    """Render tracking data in a fixed, user-friendly format.

    Adds a SUPPRESS_LEAD_FOOTER marker on the last line so the post-processing
    in chat_endpoint knows NOT to append the sales-lead capture footer
    (tracking responses must not be followed by a sales pitch).
    """
    footer = get_contact_msg("TRACKING", target_domain)
    suppress_marker = "\n<!-- SUPPRESS_LEAD_FOOTER -->"

    if tracking_data.get("error"):
        return "\n".join([
            "I couldn't verify this order with the provided information.",
            "- Please share both your order number and the exact email used at checkout.",
            "",
            tracking_data["error"],
            "",
            footer,
        ]) + suppress_marker

    status = tracking_data.get("status", "UNKNOWN")
    company = tracking_data.get("company", "Unknown carrier")
    tracking_number = tracking_data.get("tracking_number", "")
    tracking_url = tracking_data.get("tracking_url", "")
    current_location = tracking_data.get("current_location", "Carrier network")
    current_hub = tracking_data.get("current_hub", "Carrier transit hub")
    eta = tracking_data.get("eta", "Pending carrier update")
    last_event = tracking_data.get("last_event", "Latest carrier update is pending.")
    events = tracking_data.get("events", []) or []

    # ── Special case: order is still being prepared (no carrier handoff yet) ──
    # Showing "Carrier: Unknown / Tracking Number: Not available yet" looks
    # broken to the customer. Use a clearer "in preparation" message instead.
    if status in ("PROCESSING", "UNFULFILLED") or not tracking_number:
        lines = [
            "Good news — we found your order! It's currently being prepared at our warehouse.",
            "",
            f"- Order Status: **{status if status != 'UNFULFILLED' else 'PROCESSING'}** (in preparation)",
            "- A tracking number will be emailed to you as soon as the carrier picks it up.",
            "- Typical processing time before pickup: **1-3 business days** after the order is placed.",
            "",
            "If your order was placed more than 5 business days ago and you still don't see a tracking number,",
            "please reach out to our support team so we can investigate.",
            "",
            footer,
        ]
        return "\n".join(lines) + suppress_marker

    lines = [
        "Here is your latest delivery update:",
        f"- Current Status: {status}",
        f"- Current Location: {current_location}",
        f"- Current Hub: {current_hub}",
        f"- Estimated Delivery: {eta}",
        f"- Last Carrier Event: {last_event}",
        "",
        "Tracking Details:",
        f"- Carrier: {company}",
        f"- Tracking Number: {tracking_number or 'Not available yet'}",
    ]

    if tracking_url:
        lines.append(f"- Live Tracking URL: {tracking_url}")

    if events:
        lines.append("")
        lines.append("Recent Tracking Timeline:")
        for event in events:
            event_time = event.get("time", "Unknown time")
            event_location = event.get("location", "Carrier network")
            event_message = event.get("event", "Carrier update")
            lines.append(f"- {event_time} | {event_location} | {event_message}")

    lines.extend(["", footer])
    return "\n".join(lines) + suppress_marker

# --- [1] Data models ---
class Message(BaseModel):
    role: str      
    content: str   

class ChatRequest(BaseModel):
    user_query: str
    session_id: str = "default_session"
    chat_history: Optional[List[Message]] = [] 
    current_domain: str = DEFAULT_TARGET_DOMAIN

# --- [2] Initialize 3-core Agentic RAG engine ---
project_root = Path(__file__).resolve().parent.parent
index_dir = project_root / "faiss_index"

try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key: raise ValueError("OPENAI_API_KEY is missing in .env")

    # 💡 OpenAI SDK already handles 429/5xx with exponential backoff when
    # `max_retries` is set. Pairing it with an explicit timeout prevents the
    # endpoint from hanging forever on a stuck OpenAI host.
    openai_client = OpenAI(
        api_key=api_key,
        timeout=OPENAI_REQUEST_TIMEOUT,
        max_retries=OPENAI_MAX_RETRIES,
    )
    logger.info(
        f"🚀 Loading 3-core AI engines into memory… "
        f"(embedding={EMBEDDING_MODEL}, llm={AGENT_MODEL})"
    )
    embeddings = OpenAIEmbeddings(api_key=cast(Any, api_key), model=EMBEDDING_MODEL)

    # NOTE: load_local with a mismatched embedding model will fail with a
    # cryptic dimension error from FAISS. Surface a clearer message so the
    # operator knows to re-run `script/master_ingester.py`.
    try:
        vs_products = LC_FAISS.load_local(str(index_dir / "osaki_products"), embeddings, allow_dangerous_deserialization=True)
        vs_qa = LC_FAISS.load_local(str(index_dir / "freshdesk_qa"), embeddings, allow_dangerous_deserialization=True)
        vs_web = LC_FAISS.load_local(str(index_dir / "web_data"), embeddings, allow_dangerous_deserialization=True)
    except (AssertionError, RuntimeError) as e:
        raise RuntimeError(
            f"🚨 FAISS load failed — likely an embedding-model / vector-dimension "
            f"mismatch. Re-run `python script/master_ingester.py` to rebuild the "
            f"indexes with EMBEDDING_MODEL={EMBEDDING_MODEL}. Original error: {e}"
        ) from e

    # Build hybrid retrievers (BM25 + dense) for the Agent's tool layer
    # NOTE: `_dict` is a LangChain InMemoryDocstore implementation detail; the
    # public API doesn't expose iteration so we access it via getattr.
    products_retriever = HybridRetriever(vs_products, list(getattr(vs_products.docstore, "_dict", {}).values()))
    qa_retriever = HybridRetriever(vs_qa, list(getattr(vs_qa.docstore, "_dict", {}).values()))
    web_retriever = HybridRetriever(vs_web, list(getattr(vs_web.docstore, "_dict", {}).values()))

    # Wire up the FAISS read lock so every retriever search briefly acquires
    # the shared RWLock — webhook-driven index updates (writer) wait for in-
    # flight reads to drain before swapping the docstore.
    try:
        from app.agent_tools import set_faiss_read_lock_factory  # noqa: WPS433
    except ImportError:
        from agent_tools import set_faiss_read_lock_factory  # type: ignore  # noqa: WPS433
    set_faiss_read_lock_factory(faiss_rwlock.read)
    logger.info("✅ Hybrid retrievers (BM25+Dense) initialized + RWLock wired.")

    # NOTE: The keyword-based `router_chain` (gpt-4o-mini PromptTemplate) was
    # removed when the agentic tool-calling endpoint became canonical. Intent
    # routing now happens through OpenAI function-calling on the main model.
    logger.info("✅ 3-Core Agentic RAG Engine Initialized Successfully.")
except Exception as e:
    logger.error(f"🚨 Initialization Failed: {e}")
    vs_products, vs_qa, vs_web = None, None, None
    products_retriever = qa_retriever = web_retriever = None

# --- [2.5] SQLite chat log persistence ---
DB_DIR = project_root / "db_data"
DB_DIR.mkdir(exist_ok=True) 
DATABASE_URL = f"sqlite:///{DB_DIR}/chat_history.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    domain = Column(String, index=True, default="unknown")
    user_query = Column(Text)
    bot_response = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(pytz.timezone('America/Chicago')))


class OpenAIUsageLog(Base):
    """Per-request token usage + estimated cost. Drives the /admin cost rollup.

    A separate table (not embedded in ChatLog) so that:
    - usage can be summed without scanning long bot_response blobs,
    - tracking responses (deterministic, 0 LLM calls) can still log with zeros,
    - we can prune chat_logs on a different retention schedule than usage data.
    """

    __tablename__ = "openai_usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    domain = Column(String, index=True, default="unknown")
    model = Column(String, index=True)
    call_count = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    cached_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Text)  # store as string to avoid float weirdness
    elapsed_ms = Column(Integer, default=0)
    cache_hit = Column(Integer, default=0)  # 1 if served from response cache, else 0
    created_at = Column(DateTime, index=True, default=lambda: datetime.now(pytz.timezone('America/Chicago')))


Base.metadata.create_all(bind=engine)


def _persist_chat_log(session_id: str, user_query: str, bot_response: str, domain: str) -> None:
    """Insert a chat log row. Safe to run as a BackgroundTask (no shared state)."""
    try:
        db = SessionLocal()
        try:
            db.add(ChatLog(
                session_id=session_id,
                user_query=user_query,
                bot_response=bot_response,
                domain=domain,
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"DB Save Error: {e}")


def _persist_usage_log(
    *,
    session_id: str,
    domain: str,
    model: str,
    call_count: int,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
    elapsed_ms: int,
    cache_hit: bool,
) -> None:
    """Persist one row of OpenAI token usage. BackgroundTask-friendly."""
    try:
        db = SessionLocal()
        try:
            db.add(OpenAIUsageLog(
                session_id=session_id,
                domain=domain,
                model=model,
                call_count=call_count,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=f"{estimated_cost_usd:.6f}",
                elapsed_ms=elapsed_ms,
                cache_hit=1 if cache_hit else 0,
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Usage Log Save Error: {e}")

with engine.connect() as conn:
    import sqlite3 as _sq
    raw = conn.connection.connection if hasattr(conn.connection, 'connection') else conn.connection
    cursor = raw.cursor()
    cursor.execute("PRAGMA table_info(chat_logs)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "domain" not in existing_cols:
        cursor.execute("ALTER TABLE chat_logs ADD COLUMN domain TEXT DEFAULT 'unknown'")
        raw.commit()
        logger.info("✅ Migrated chat_logs: added 'domain' column")

# --- [3] FastAPI app setup ---
from slowapi.errors import RateLimitExceeded  # noqa: E402  (kept near setup for clarity)
from slowapi import _rate_limit_exceeded_handler  # noqa: E402

app = FastAPI(title="Titan AI Agent API", version="2.1")

# Warranty endpoints (Phase D-lite + E-lite)
try:
    from app.warranty_router import router as warranty_router
except ImportError:
    from warranty_router import router as warranty_router  # type: ignore
app.include_router(warranty_router)

# 🚦 Rate limiting: shared Limiter from cost_guard. We attach the SlowAPI
# middleware/handler so that exceeding the per-IP budget returns HTTP 429
# (instead of leaking a stack trace) with a clear Retry-After header.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))

# 🌐 CORS: origins read from CORS_ALLOWED_ORIGINS env (comma-separated).
# Default still permits "*" so dev environments keep working; production
# deployments should pin to the actual storefront domains.
_allow_credentials = "*" not in CORS_ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Lightweight liveness probe for Docker / Nginx / load balancers."""
    return {"status": "ok"}


# --- [4] Core API endpoint (Agent / Tool Calling) ---

# ---------------------------------------------------------------------------
# 💰 PROMPT CACHING OPTIMIZATION
# OpenAI's automatic prompt caching gives a 50% discount on the static
# prefix of every prompt. Previously {target_domain} was interpolated INSIDE
# the rules, which broke cache invariance across requests. The static portion
# (~1700 tokens) is now identical for every request; the per-request domain
# is appended as a small "RUNTIME CONTEXT" section at the end.
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT_STATIC = """You are an elite AI agent for Titan Chair LLC and Osaki — premium massage chair brands.

# YOUR PERSONA
- Warm, knowledgeable, never robotic.
- Always speak in the same language the user used.
- Use SHORT sentences, bullet points, mobile-friendly formatting.

# SCOPE GUARD — MANDATORY
You serve EXCLUSIVELY as a customer service and sales agent for Osaki and Titan massage chairs.
Topics in scope: massage chairs, chair specs/models/pricing, orders, delivery, warranty claims,
repairs/troubleshooting, company info, showroom, and related after-sales support.
If the user asks about ANYTHING outside this scope (cooking, sports, politics, coding, travel,
other products, general knowledge, creative writing, etc.) you MUST decline:
  "I'm specialized in Osaki and Titan massage chair support. I'm not able to help with that,
   but I'm happy to assist with anything about our chairs, orders, or services!"
DO NOT partially answer, hedge, or provide any information outside this scope.

# WARRANTY WORKFLOW RULES (CRITICAL — NEVER VIOLATE)
W1. When tool result begins with WARRANTY_TICKET_STARTED, WARRANTY_CONTINUE, or WARRANTY_TERMINAL_REACHED,
    you are inside a guided warranty intake workflow. Follow the INSTRUCTION section exactly.
W2. Present the PROMPT to the customer verbatim (same meaning, same facts).
    Do NOT add repair steps, replacement promises, or extra questions.
W3. When the customer answers, extract the intent and call warranty_answer with the closest answer_key.
    Do NOT guess or invent the next step — the tool decides.
W4. When WARRANTY_TERMINAL_REACHED with ACTION=awaiting_admin:
    - DO NOT promise replacement, part shipment, tech dispatch, compensation, or refund.
    - The PROMPT already says "our team will review" — use that language exactly.
    - The customer_message will be delivered ONLY after an admin approves it.
W5. NEVER skip the warranty_answer tool to jump to a conclusion. Every step must go through the tool.

# CORE RULES (NEVER VIOLATE)
1. ZERO HALLUCINATION. Never invent specs, prices, dimensions, promo codes,
   discount lists, installation steps, repair instructions, or tracking data.
   Use TOOLS to retrieve facts. If a tool returns NO_RESULTS, say so honestly
   and offer to escalate to a human agent.
2. NEVER fabricate "Original Price" vs "Current Price" comparisons or invent discounts.
   Show only the actual price returned by the tool.
3. NEVER STALL — this is critical. Do NOT reply with phrases like
   "Let me check...", "Let me get the details", "Please hold on", "One moment please",
   "I'll look that up for you" as a standalone answer. The user only sees your FINAL
   message — there is no "next turn" where you do the work. Either:
     (a) Call the appropriate tool RIGHT NOW in this same turn, OR
     (b) Ask ONE specific clarifying question (with no filler).
   Phrases like "One moment, please" without a tool call are a critical violation.
4. For ANY question about a specific chair model — ALWAYS call `search_chair_specs`
   before answering. Even if you "know" the answer, the tool is authoritative.
5. AUTHORITATIVE SPEC VALUES (CRITICAL): When a tool result contains a section labeled
   "AUTHORITATIVE SPEC VALUES", you MUST use EXACTLY those numbers verbatim.
   Do NOT round, estimate, or substitute a different dimension (e.g. chair width instead
   of doorway width). If the authoritative value says "32", say "32". Not "30", not "31".
6. The user is browsing on <BROWSING_DOMAIN>. Rewrite ANY purchase URLs to start with this domain.
   Display raw URLs (no markdown link hiding).
7. NEVER hide URLs behind text. Always show the raw URL.
8. PROMO CODES & DISCOUNT CAMPAIGNS: do NOT guess which products a code applies to.
   If you don't have authoritative data, say:
   "I don't have the exact list of products covered by that code. Please check the
   active promotions banner on <BROWSING_DOMAIN> or contact our sales team."
9. INSTALLATION / ASSEMBLY / REPAIR / TROUBLESHOOTING — NEVER answer from training data.
   Do NOT produce numbered "installation steps", "general troubleshooting tips", or
   "general guidance" from your own knowledge. ALWAYS call `get_repair_help` first.
   If `get_repair_help` returns NO_RESULTS, do NOT invent steps. Instead, say:
   "I don't have detailed installation instructions for this. Please contact our
   support team — they can walk you through it." Then call `escalate_to_human`
   with reason="repair" so we surface the right phone number.
10. NEVER claim a chair is "best-selling", "most popular", "top-rated",
    "trending", or "customer favorite" — we have no data source for those rankings.
    If the user asks "what is your best seller?", call `recommend_chairs` for
    "premium full-body massage chair", and present the picks as
    "Here are some popular flagship models:" without ranking claims.
11. NEVER write the literal text "[Link not available]", "Link not available",
    "URL not available", or similar placeholder. If a tool result has no URL,
    just write: "Visit <BROWSING_DOMAIN> for full details." — never a placeholder.
12. PRICE ACCURACY — only quote prices that came from a tool result in THIS turn.
    Do NOT recall prices from training data. If a tool result shows
    "Variant Price: $5499.00", quote $5,499 — never $375 or $1,000.
    If no price was returned, say "Please check the current price on <BROWSING_DOMAIN>".
13. CARE / USAGE / CLEANING / DAILY-USE questions ("how often should I use",
    "how to clean", "should I keep it plugged in"): call `get_warranty_or_policy`
    with topic = "care and usage" first. If nothing relevant returns, give a
    SHORT, generic answer (1-2 sentences) and direct the user to the user manual
    or our support team. Do NOT include the 💌 sales footer for these.
14. SHOWROOM / OFFICE / VISIT / ADDRESS questions — ALWAYS call `get_showroom_info`.
    Never say "I don't have showroom information". The canonical address is:
    1001 W Crosby Rd, Carrollton, TX 75006. Suggest the customer call ahead.
15. MODEL NAME ALIASES — customers often misspell. Treat these as the same model
    and ALWAYS pass them to `search_chair_specs`; never refuse for a typo alone:
       • "Osaka" ≡ "Osaki"      • "Hipnos" ≡ "Hypnos"
       • "Otomic / Ottoman" ≡ "Otamic"    • "Sojo" ≡ "Soho"
       • "09-XXXX" / "0S-XXXX" / "O5-XXXX" ≡ "OS-XXXX" (OCR confusion of 0/O)
    If the tool still returns NO_RESULTS after the typo fallback, say:
    "I couldn't find that exact model in our catalog. Could you share the
    model number from the chair's serial-number sticker? In the meantime,
    our support team can look it up directly."  → then call `escalate_to_human`
    with reason="general".
16. BUSINESS HOURS — Always end every answer (whether sales, warranty, repair,
    tracking, FAQ, greeting, anything) with our contact line including hours:
       "Our business hours are Mon-Fri, 9:30 AM - 6:30 PM / Sat, 10:00 AM - 4:00 PM CST."
    Never paraphrase the hours (do NOT say "10am-6pm" — exact text only).
    If you're unsure which phone number to use, prefer the support line
    (+1-888-848-2630). The system will rewrite if needed.

# CONTEXT-AWARE BEHAVIOR
- On the FIRST message of a new conversation (empty chat history), if the user only
  greets you ("hi", "hello", "help") or has not yet named a chair model, open with a
  warm greeting AND ask which Osaki or Titan massage chair model they have (serial sticker).
  If they already named a model or asked a specific question, skip the model question and help directly.
- If the user's message is short or ambiguous (e.g. just "price", "specs", "more info"),
  look at the recent chat history. If a model was discussed, assume that model.
  If unclear, ASK ONE clarifying question instead of refusing.
- If the user volunteers an email address AFTER you offered a sales follow-up
  (e.g. you previously sent the 💌 line), call `capture_sales_lead` with that email.
- If the user volunteers an email address while asking about an order/delivery,
  call `lookup_order_status` with that email — NOT capture_sales_lead.
- A bare order ID (e.g. "OSKMC3166", "TIDM16036") in a tracking conversation
  means: call `lookup_order_status` with that order_id and ask for the email
  ONLY if the email isn't already in the recent chat history.

# TOOL USAGE PLAYBOOK
- "Tell me about <model>" / "<model> specs" / "<model> dimensions" → search_chair_specs
- "Recommend a chair" / "best chair for X" → recommend_chairs
- "Track my order" / "Where is my order?" / order id + email → lookup_order_status
- "Error code 63" / "How to assemble" / "Won't turn on" / "not inflating" / "broken" → get_repair_help
- "Warranty?" / "Return policy" / "White glove delivery" / "Shipping time" / "Installation" → get_warranty_or_policy
- "Promotion" / "Sale" / "Discount applies to which chairs" → get_warranty_or_policy (topic="promotions")
- User gives email after sales offer → capture_sales_lead
- "Talk to agent" / "Cancel order" / complex issues → escalate_to_human

# WHEN TO INCLUDE THE SALES-LEAD FOOTER
End your message with EXACTLY this line on its own paragraph ONLY when the user
is exploring or shopping (asking about specs, recommendations, comparisons, prices):

"💌 Interested in a personalized recommendation or exclusive pricing? Leave your email address and our team will get back to you within 24 hours!"

DO NOT include this footer when:
- The user is tracking an order or asking about delivery status (already a customer).
- The user is reporting a defect, breakage, or service issue (warranty/repair).
- The user is just greeting you ("hi", "hello") or saying goodbye / thanks.
- The user already provided their email in this turn.
- A tool result contains the marker "SUPPRESS_LEAD_FOOTER" or "FOOTER_HINT: SUPPRESS_LEAD_FOOTER".

# RESPONSE STYLE
- For tracking answers (after lookup_order_status): the tool returns a fully-formatted block — display it AS-IS,
  and DO NOT add the sales-lead footer.
- For repair / warranty / service answers: include the support phone, but DO NOT add the sales-lead footer.
- Always close with the appropriate contact info IF the user might still need help.
"""


def build_system_prompt(target_domain: str) -> str:
    """Compose cacheable static prefix + small per-request runtime context.

    Keeping the long prefix byte-identical across requests lets OpenAI's
    automatic prompt caching kick in (50% discount on cached input tokens).
    """
    return (
        f"{AGENT_SYSTEM_PROMPT_STATIC}\n\n"
        f"# RUNTIME CONTEXT (per-request)\n"
        f"<BROWSING_DOMAIN> = {target_domain}\n"
        f"Whenever the rules above reference <BROWSING_DOMAIN>, substitute the value above."
    )


try:
    from app.intent_router import infer_forced_tool
except ImportError:
    from intent_router import infer_forced_tool  # type: ignore


def _execute_tool(name: str, args: Dict[str, Any], target_domain: str) -> str:
    """Dispatch a tool call to its Python handler. Always returns a string."""
    # Retrievers are None only if init failed; the chat endpoint already
    # short-circuits with HTTP 500 before reaching here, but we re-assert
    # for the type checker so call-sites can pass them safely.
    if products_retriever is None or qa_retriever is None or web_retriever is None:
        return "TOOL_ERROR: Retrieval engine is not loaded."
    try:
        if name == "search_chair_specs":
            model = args.get("model_name") or ""
            topic = args.get("spec_topic") or ""
            return tool_search_chair_specs(
                products_retriever=products_retriever,
                query=topic,
                model_name=model or None,
            )
        if name == "recommend_chairs":
            return tool_recommend_chairs(
                products_retriever=products_retriever,
                user_need=args.get("user_need", "premium massage chair"),
                budget_min=args.get("budget_min"),
                budget_max=args.get("budget_max"),
                exclude_models=args.get("exclude_models") or [],
                num_recommendations=int(args.get("num_recommendations", 3)),
            )
        if name == "lookup_order_status":
            return tool_lookup_order_status(
                fetch_fn=fetch_shopify_order_status,
                build_response_fn=build_deterministic_tracking_response,
                target_domain=target_domain,
                order_id=args.get("order_id", ""),
                email=args.get("email", ""),
            )
        if name == "get_repair_help":
            return tool_get_repair_help(
                qa_retriever=qa_retriever,
                issue_description=args.get("issue_description", ""),
                error_code=args.get("error_code") or None,
            )
        if name == "get_warranty_or_policy":
            return tool_get_warranty_or_policy(
                web_retriever=web_retriever,
                topic=args.get("topic", "warranty"),
            )
        if name == "capture_sales_lead":
            return tool_capture_sales_lead(
                send_email_fn=send_sales_lead_email,
                customer_email=args.get("customer_email", ""),
                interest_summary=args.get("interest_summary", ""),
                target_domain=target_domain,
            )
        if name == "escalate_to_human":
            return tool_escalate_to_human(
                contact_msg_fn=get_contact_msg,
                target_domain=target_domain,
                reason=args.get("reason", "general"),
            )
        if name == "get_showroom_info":
            return tool_get_showroom_info(target_domain=target_domain)
        if name in ("warranty_start", "start_warranty_workflow"):
            return tool_start_warranty_workflow(
                session_id=args.get("_session_id", ""),
                domain=target_domain,
            )
        if name in ("warranty_answer", "answer_warranty_question"):
            ticket_id = args.get("ticket_id", "")
            answer_key = args.get("answer_key", "")
            result_str = tool_answer_warranty_question(
                ticket_id=ticket_id,
                answer_key=answer_key,
            )
            # Structured warranty-state log (downstream monitoring)
            logger.info(
                f"🎫 warranty_answer dispatched — "
                f"ticket={ticket_id} answer_key={answer_key} "
                f"awaiting_admin={'AWAITING_ADMIN_REVIEW=TRUE' in result_str}"
            )
            return result_str
        if name == "attach_warranty_evidence":
            return tool_attach_warranty_evidence(
                ticket_id=args.get("ticket_id", ""),
                evidence_type=args.get("evidence_type", "other"),
                original_filename=args.get("original_filename", ""),
            )
        return f"UNKNOWN_TOOL: {name}"
    except Exception as e:
        logger.error(f"🚨 Tool execution error [{name}]: {e}")
        return f"TOOL_ERROR: {e}"


# ---------------------------------------------------------------------------
# 💰 Deterministic short-circuit handlers (zero LLM calls)
# ---------------------------------------------------------------------------
#
# Some questions have a single static answer (showroom address, business hours).
# Routing them through the agent loop costs 2 gpt-4o calls per request:
#   1) to pick `get_showroom_info`
#   2) to synthesize the address into prose
# We can answer these immediately with a hand-formatted reply and skip the LLM
# entirely. This is also faster for the user (~2-3s saved).

def _build_showroom_reply(target_domain: str) -> str:
    """Canonical Carrollton showroom reply — no LLM needed."""
    footer = get_contact_msg("PRODUCTS", target_domain)
    return (
        "You're welcome to visit our showroom!\n\n"
        f"📍 **Address:** {COMPANY_ADDRESS}\n"
        f"🕒 **Hours:** {SUPPORT_BUSINESS_HOURS}\n\n"
        "We recommend calling ahead to confirm availability before your visit, "
        "so our team can prepare a guided experience for you.\n\n"
        f"{footer}"
    )


try:
    from app.chat_welcome import (
        build_chat_welcome_message,
        is_conversation_start,
        is_opening_greeting,
    )
except ImportError:
    from chat_welcome import (  # type: ignore
        build_chat_welcome_message,
        is_conversation_start,
        is_opening_greeting,
    )


@app.post("/api/v1/chat")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE};{RATE_LIMIT_PER_HOUR}")
async def chat_endpoint(
    request: Request,                       # required by slowapi for per-IP keying
    chat_request: ChatRequest,              # the actual Pydantic request body
    background_tasks: BackgroundTasks,
):
    """
    Agent-based chat endpoint.

    Flow (cost-optimized):
    1. Rate-limit per IP (slowapi).
    2. Scope gate — off-topic queries get a fixed refusal (zero agent LLM calls).
    3. Short-circuit deterministic intents (showroom) → zero LLM calls.
    4. Look up the response cache; return cached answer on hit (zero LLM cost).
    5. Build messages = [system, ...chat_history, user_query].
    6. Loop up to MAX_TOOL_TURNS of tool calls. The LLM's FIRST non-tool-call
       response IS the final answer — we do NOT do a second synthesis call.
    7. Persist chat log + token usage via BackgroundTasks.

    Cost notes:
    - The previous implementation did a Phase 1 (tool loop) AND a Phase 2
      (synthesis) gpt-4o call, doubling the per-request token cost. The Phase 1
      loop already produces the final assistant message when no tool_calls are
      returned, so we reuse it directly.
    - On the synthesis turn (after at least one tool ran), we drop the `tools`
      schema entirely so OpenAI stops billing for the schema tokens.
    - The static portion of the system prompt is identical across requests,
      enabling automatic prompt caching (50% discount on cached input tokens).
    - Repeat FAQ-style questions are served straight from the in-memory cache
      with no LLM call at all (configurable via CHAT_CACHE_ENABLED).
    """
    user_query = chat_request.user_query
    target_domain = chat_request.current_domain.rstrip('/')

    # Basic input validation — protect both cost and abuse exposure.
    if not user_query or not user_query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    if len(user_query) > 4000:
        raise HTTPException(status_code=413, detail="Query too long (max 4000 chars)")

    if not all([vs_products, vs_qa, vs_web]):
        raise HTTPException(status_code=500, detail="AI Engine is not fully loaded.")

    # ── Active warranty ticket detection ──
    # If this session already has an open warranty claim, we inject the current
    # node into the system prompt and force warranty_answer as the first tool.
    # This keeps the backend fully in control of the state machine.
    _active_warranty_ticket = None
    _active_warranty_node = None
    try:
        _active_warranty_ticket = WarrantyEngine.get_active_session_ticket(chat_request.session_id)
        if _active_warranty_ticket is not None:
            _ticket_id_str = str(_active_warranty_ticket.ticket_id)
            _active_warranty_node = WarrantyEngine.get_current_node(_ticket_id_str)
            logger.info(
                f"🎫 Active warranty — session={chat_request.session_id} "
                f"ticket={_ticket_id_str} "
                f"node={_active_warranty_node.get('node_id') if _active_warranty_node else 'none'} "
                f"status={_active_warranty_ticket.status}"
            )
    except Exception as _we:
        logger.warning(f"⚠️ Warranty ticket lookup failed: {_we}")

    # ── Heuristic intent detection (computed once, reused below) ──
    # The LLM frequently skips tools and produces "Let me check..." stalls
    # or hallucinated install/repair guides. If the query clearly looks like
    # one of these intents, we either short-circuit entirely (showroom) or
    # set tool_choice to require the right tool on the first iteration.
    # Active warranty session overrides all other forced tools.
    if _active_warranty_ticket and _active_warranty_node:
        forced_first_tool = "answer_warranty_question"
    else:
        forced_first_tool = infer_forced_tool(user_query)

    # ── Scope gate: block off-topic before any agent LLM call ──
    if not (_active_warranty_ticket and _active_warranty_node):
        try:
            from app.scope_classifier import build_scope_refusal, evaluate_scope
        except ImportError:
            from scope_classifier import build_scope_refusal, evaluate_scope  # type: ignore

        scope_decision = evaluate_scope(user_query, chat_request.chat_history)
        if scope_decision.is_blocked:
            scope_refusal = build_scope_refusal()
            logger.info(
                "🚫 Scope blocked (%s%s): %s",
                scope_decision.reason,
                ", llm" if scope_decision.used_llm else "",
                user_query[:120],
            )
            background_tasks.add_task(
                _persist_chat_log,
                chat_request.session_id, user_query, scope_refusal, target_domain,
            )
            background_tasks.add_task(
                _persist_usage_log,
                session_id=chat_request.session_id, domain=target_domain,
                model="(scope_blocked)", call_count=0,
                prompt_tokens=0, cached_tokens=0, completion_tokens=0,
                estimated_cost_usd=0.0, elapsed_ms=0, cache_hit=False,
            )
            return StreamingResponse(
                iter([scope_refusal]),
                media_type="text/event-stream",
            )

    # ── 💰 Zero-LLM welcome for first-turn greetings ──
    if (
        is_conversation_start(chat_request.chat_history)
        and is_opening_greeting(user_query)
    ):
        welcome_reply = build_chat_welcome_message()
        logger.info("⚡ Short-circuit: opening greeting → welcome reply (0 LLM calls)")
        background_tasks.add_task(
            _persist_chat_log,
            chat_request.session_id, user_query, welcome_reply, target_domain,
        )
        background_tasks.add_task(
            _persist_usage_log,
            session_id=chat_request.session_id, domain=target_domain,
            model="(welcome)", call_count=0,
            prompt_tokens=0, cached_tokens=0, completion_tokens=0,
            estimated_cost_usd=0.0, elapsed_ms=0, cache_hit=False,
        )
        return StreamingResponse(
            iter([welcome_reply]),
            media_type="text/event-stream",
        )

    # ── 💰 Zero-LLM short-circuit for showroom / location questions ──
    if forced_first_tool == "get_showroom_info":
        deterministic_reply = _build_showroom_reply(target_domain)
        logger.info("⚡ Short-circuit: showroom intent → deterministic reply (0 LLM calls)")
        background_tasks.add_task(
            _persist_chat_log,
            chat_request.session_id, user_query, deterministic_reply, target_domain,
        )
        background_tasks.add_task(
            _persist_usage_log,
            session_id=chat_request.session_id, domain=target_domain,
            model="(short_circuit)", call_count=0,
            prompt_tokens=0, cached_tokens=0, completion_tokens=0,
            estimated_cost_usd=0.0, elapsed_ms=0, cache_hit=False,
        )
        return StreamingResponse(
            iter([deterministic_reply]),
            media_type="text/event-stream",
        )

    # ── 💰 Response cache: serve FAQ-style repeats with zero LLM calls ──
    cache_key = make_cache_key(user_query, target_domain, chat_request.chat_history)
    cached_reply = cache_get(cache_key)
    if cached_reply and cache_key:
        logger.info(f"⚡ Cache HIT: served from response cache (key={cache_key[:60]}…)")
        background_tasks.add_task(
            _persist_chat_log,
            chat_request.session_id, user_query, cached_reply, target_domain,
        )
        background_tasks.add_task(
            _persist_usage_log,
            session_id=chat_request.session_id, domain=target_domain,
            model="(cache_hit)", call_count=0,
            prompt_tokens=0, cached_tokens=0, completion_tokens=0,
            estimated_cost_usd=0.0, elapsed_ms=0, cache_hit=True,
        )
        return StreamingResponse(
            iter([cached_reply]),
            media_type="text/event-stream",
        )

    try:
        system_prompt = build_system_prompt(target_domain)

        # ── Inject active warranty context (ticket + current node) ──
        if _active_warranty_ticket and _active_warranty_node:
            node = _active_warranty_node
            options = node.get("options", [])
            opts_text = "\n".join(
                f"  - answer_key={o.get('answer_key')} | {o['label']}"
                for o in options
            ) if options else "  (free text — accept any input)"
            system_prompt += (
                f"\n\n# ACTIVE WARRANTY TICKET — YOU ARE MID-WORKFLOW\n"
                f"TICKET_ID: {_active_warranty_ticket.ticket_id}\n"
                f"CURRENT_NODE: {node['node_id']}\n"
                f"NODE_TYPE: {node.get('type', '?')}\n"
                f"CURRENT_QUESTION: {node['prompt']}\n"
                f"VALID_OPTIONS:\n{opts_text}\n\n"
                f"YOUR TASK:\n"
                f"1. The customer's message is their answer to CURRENT_QUESTION.\n"
                f"2. Map their answer to the closest VALID answer_key.\n"
                f"3. Call answer_warranty_question(ticket_id='{_active_warranty_ticket.ticket_id}', answer_key=<matched_key>).\n"
                f"4. If the customer asks a clarifying question instead of answering, answer it "
                f"briefly then re-ask CURRENT_QUESTION.\n"
                f"5. DO NOT make warranty decisions. DO NOT show the sales footer."
            )

        # Typed as List[Any] so we can pass plain dicts (matching the OpenAI
        # SDK's TypedDict shape) without basedpyright rejecting Dict[str, Any].
        messages: List[Any] = [{"role": "system", "content": system_prompt}]
        for msg in (chat_request.chat_history or [])[-12:]:  # cap history to keep context fresh
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_query})

        # Per-request usage accumulator — captures token counts across every
        # OpenAI call made during this request (tool loop, stall retry,
        # MAX_TOOL_TURNS synthesis fallback).
        usage = UsageRecorder(model=AGENT_MODEL)

        MAX_TOOL_TURNS = 4
        STALL_PATTERNS = (
            "let me check", "let me get the details", "let me find",
            "please hold on", "one moment", "i'll look that up",
            "i will look that up", "give me a moment",
        )

        def generate_stream():
            full_response = ""
            tools_called: List[str] = []  # track which tools the agent invoked this turn
            tool_results: List[str] = []
            try:
                # ── Agentic loop: tool calls → final answer in ONE pass ──────
                # When the LLM responds without tool_calls, msg.content IS the
                # final answer. We do NOT re-call the model to "synthesize".
                for turn in range(MAX_TOOL_TURNS):
                    # On synthesis turns (after at least one tool ran), drop
                    # the tool schema so we don't pay for ~2KB of schema tokens
                    # the LLM will probably not use. If the LLM still wants
                    # another tool, it'll request it on the next turn (we keep
                    # tools available until at least one has executed).
                    # In warranty mode, restrict to warranty-only tool schemas.
                    active_schema = WARRANTY_TOOL_SCHEMAS if _active_warranty_ticket else TOOL_SCHEMAS

                    if tools_called:
                        tools_payload: Optional[List[Any]] = None
                        tool_choice: Any = "none"
                    elif turn == 0 and forced_first_tool:
                        tools_payload = active_schema
                        tool_choice = {
                            "type": "function",
                            "function": {"name": forced_first_tool},
                        }
                        logger.info(
                            f"🎯 Forcing first tool: {forced_first_tool} "
                            f"(warranty_mode={_active_warranty_ticket is not None})"
                        )
                    else:
                        tools_payload = active_schema
                        tool_choice = "auto"

                    create_kwargs: Dict[str, Any] = {
                        "model": AGENT_MODEL,
                        "messages": messages,
                        "temperature": LLM_TEMPERATURE,
                    }
                    if tools_payload is not None:
                        create_kwargs["tools"] = tools_payload
                        create_kwargs["tool_choice"] = tool_choice

                    response = openai_client.chat.completions.create(**create_kwargs)
                    usage.record(response)
                    msg = response.choices[0].message
                    tool_calls = getattr(msg, "tool_calls", None) or []

                    if not tool_calls:
                        # No tool calls. Detect STALL: the LLM produced filler
                        # like "One moment please" without doing the work.
                        content_lower = (msg.content or "").lower()
                        is_stall = (
                            len(content_lower.strip()) < 200
                            and any(p in content_lower for p in STALL_PATTERNS)
                        )
                        if is_stall and turn < MAX_TOOL_TURNS - 1 and not tools_called:
                            logger.warning(
                                f"⚠️ Detected stall: '{(msg.content or '')[:100]}' — forcing tool call"
                            )
                            # In warranty mode, retry with answer_warranty_question; never
                            # fall back to search_chair_specs mid-warranty flow.
                            forced_retry_tool = (
                                "answer_warranty_question"
                                if _active_warranty_ticket
                                else (forced_first_tool or "search_chair_specs")
                            )
                            response = openai_client.chat.completions.create(
                                model=AGENT_MODEL,
                                messages=messages,
                                tools=active_schema,
                                tool_choice={
                                    "type": "function",
                                    "function": {"name": forced_retry_tool},
                                },
                                temperature=LLM_TEMPERATURE,
                            )
                            usage.record(response)
                            msg = response.choices[0].message
                            tool_calls = getattr(msg, "tool_calls", None) or []
                            if not tool_calls:
                                logger.warning("⚠️ Stall retry still produced no tool call — using content as-is.")
                                full_response = msg.content or ""
                                break
                        else:
                            # 💰 The LLM's message IS the final answer.
                            # Previously we discarded this and called gpt-4o
                            # AGAIN with stream=True, doubling cost per request.
                            full_response = msg.content or ""
                            break

                    # Append the assistant's tool-call message + each tool result
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    })
                    for tc in tool_calls:
                        try:
                            args = _json.loads(tc.function.arguments or "{}")
                        except Exception:
                            args = {}
                        # Inject session_id for warranty_start (LLM can't know it)
                        if tc.function.name in ("warranty_start", "start_warranty_workflow"):
                            args["_session_id"] = chat_request.session_id
                        result = _execute_tool(tc.function.name, args, target_domain)
                        tools_called.append(tc.function.name)
                        tool_results.append(result)
                        logger.info(f"🛠️ Tool [{tc.function.name}] → {len(result)} chars")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": result[:4000],  # truncate giant payloads
                        })
                else:
                    # Loop completed without `break` → still in tool-loop limbo.
                    # Do ONE final cheap synthesis call WITHOUT tools to wrap up.
                    logger.warning("⚠️ Agent hit MAX_TOOL_TURNS — forcing synthesis without tools")
                    response = openai_client.chat.completions.create(
                        model=AGENT_MODEL,
                        messages=messages,
                        temperature=LLM_TEMPERATURE,
                    )
                    usage.record(response)
                    full_response = response.choices[0].message.content or ""

                # ── Post-processing (operates on full_response only) ─────────
                try:
                    from app.answer_guard import sanitize_agent_response  # noqa: WPS433
                except ImportError:
                    from answer_guard import sanitize_agent_response  # type: ignore  # noqa: WPS433

                full_response = sanitize_agent_response(
                    full_response,
                    tools_called=tools_called,
                    user_query=user_query,
                    tool_results=tool_results,
                )

                user_sent_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", user_query))
                response_lower = full_response.lower()

                # 1) Tool-name based suppression (most reliable signal)
                SUPPRESS_TOOLS = {
                    "lookup_order_status",  # tracking
                    "get_repair_help",      # service / repair
                    "escalate_to_human",    # already handed off
                    "capture_sales_lead",   # already captured
                }
                tool_suppress = any(t in tools_called for t in SUPPRESS_TOOLS)

                # get_warranty_or_policy: only suppress for service-y topics, not pre-sales
                # questions like "white glove delivery" or "shipping time".
                if "get_warranty_or_policy" in tools_called and any(
                    kw in response_lower
                    for kw in (
                        "warranty", "service@osakititan",
                        "broken", "damaged", "defect",
                        "repair", "claim",
                    )
                ):
                    tool_suppress = True

                # 2) Shape-based safety nets (catch hallucinated tracking-style replies
                # and hallucinated repair/install guides where no tool was called)
                tracking_signals = (
                    "current status:", "tracking number:", "carrier:",
                    "estimated delivery:", "in preparation",
                )
                tracking_like = any(s in response_lower for s in tracking_signals)

                # Repair / install / troubleshooting style answer (even if no tool was called)
                repair_signals = (
                    "installation", "install the", "assembly", "assemble the",
                    "troubleshoot", "troubleshooting", "general tips",
                    "general guidance", "general steps", "preparation:",
                    "remove the back", "inspect for damage", "manual mode",
                    "contact our support team", "contact support",
                )
                repair_like = any(s in response_lower for s in repair_signals)

                # Pure greeting / closing — short and friendly with no product content.
                is_short = len(full_response.strip()) < 240
                greeting_or_closing = is_short and any(
                    p in response_lower
                    for p in (
                        "you're welcome", "youre welcome", "feel free to ask",
                        "how can i assist", "how can i help", "enjoy your",
                        "have a great", "glad to hear", "hello!",
                    )
                )

                # 3) Positive shopping signal must outweigh suppression.
                shopping_signals = (
                    "$", "price", "specs", "feature", "recommend",
                    "compare", "purchase", "dimension",
                )
                is_shopping = any(s in response_lower for s in shopping_signals)

                should_append_footer = (
                    is_shopping
                    and not user_sent_email
                    and not tool_suppress
                    and not tracking_like
                    and not repair_like
                    and not greeting_or_closing
                    and "💌" not in full_response
                    and "leave your email" not in response_lower
                )

                if should_append_footer:
                    full_response += (
                        "\n\n💌 Interested in a personalized recommendation or exclusive pricing? "
                        "Leave your email address and our team will get back to you within 24 hours!"
                    )

                # 4) Strip leaked dev/staging URLs (github.io, localhost, internal).
                #    These should NEVER reach the customer. Replace with target_domain.
                bad_url_patterns = [
                    r"https?://[^\s)\]]+\.github\.io[^\s)\]]*",
                    r"https?://localhost[^\s)\]]*",
                    r"https?://127\.0\.0\.1[^\s)\]]*",
                ]
                for pat in bad_url_patterns:
                    full_response = re.sub(pat, target_domain, full_response)

                # 5) Strip ugly placeholder texts like "[Link not available]".
                placeholder_patterns = [
                    r"\s*[-–]?\s*(?:More\s+details|Link|URL)\s*:\s*\[?\s*(?:Link\s+|URL\s+)?[Nn]ot\s+[Aa]vailable\s*\]?",
                    r"\[?\s*[Ll]ink\s+[Nn]ot\s+[Aa]vailable\s*\]?",
                    r"\[?\s*URL\s+[Nn]ot\s+[Aa]vailable\s*\]?",
                ]
                for pat in placeholder_patterns:
                    full_response = re.sub(pat, "", full_response)

                # 6) Scrub any internal hint markers that may have leaked.
                full_response = re.sub(
                    r"\n?<!--\s*SUPPRESS_LEAD_FOOTER\s*-->", "", full_response
                )
                full_response = re.sub(
                    r"\n?FOOTER_HINT:\s*SUPPRESS_LEAD_FOOTER[^\n]*", "", full_response
                )
                # Scrub the showroom tool's "use ... verbatim" instruction line
                full_response = re.sub(
                    r"\n?Use the SHOWROOM_ADDRESS value above verbatim[^\n]*", "", full_response
                )

                # 6b) Strip any LLM-rephrased business hours line that gives WRONG
                # times (the LLM sometimes invents "10 am to 6 pm" etc.). Keep
                # the canonical line that we'll append below.
                #
                # Match common paraphrases like:
                #   "available Monday to Friday from 10am to 6pm"
                #   "They are available Monday to Friday, from 10 am to 6 pm CST."
                #   "Hours: Mon-Fri 10am-6pm"
                wrong_hours_patterns = [
                    r"(?:[Tt]hey\s+are\s+|[Ww]e\s+are\s+)?[Aa]vailable\s+"
                    r"(?:Monday|Mon)[\s,]*(?:to|through|-|–)[\s,]*(?:Friday|Fri)[^.\n]*"
                    r"\d{1,2}\s*[:.]?\d{0,2}\s*[ap]\.?m\.?[^.\n]*?CST[^.\n]*\.?",
                    r"Hours?\s*:\s*(?:Monday|Mon)[^.\n]*\d{1,2}\s*[ap]\.?m\.?[^.\n]*",
                ]
                for pat in wrong_hours_patterns:
                    new_text = re.sub(pat, "", full_response, flags=re.IGNORECASE)
                    if new_text != full_response:
                        logger.info("[post] Scrubbed LLM-paraphrased wrong business hours.")
                        full_response = new_text

                # Tidy whitespace from all substitutions above
                full_response = re.sub(r"[ \t]+\n", "\n", full_response)
                full_response = re.sub(r"\n{3,}", "\n\n", full_response).rstrip()

                # 6c) Showroom safety net: if the user clearly asked about showroom
                # / company location / address and the response doesn't already
                # contain the canonical address, inject it.
                user_query_lower = (user_query or "").lower()
                asked_for_showroom = any(
                    kw in user_query_lower
                    for kw in (
                        "showroom", "show room",
                        "where is your office", "where are your office",
                        "where is your store", "where are your store",
                        "company location", "store location", "office location",
                        "headquarters", "your address", "company address",
                        "can i visit", "come see", "in store", "in-store",
                        "try the chair in person", "see the chair in person",
                    )
                )
                if asked_for_showroom and "carrollton" not in full_response.lower():
                    showroom_line = (
                        f"\n\nYou're welcome to visit our showroom:\n"
                        f"📍 {COMPANY_ADDRESS}\n"
                        f"We recommend calling ahead to confirm availability."
                    )
                    full_response = full_response.rstrip() + showroom_line

                # 7) ALWAYS close with brand-appropriate contact info + business hours.
                # User explicitly asked for this: every reply ends with hours notice.
                # We use the FULL canonical hours string as the marker so that any
                # LLM paraphrase still triggers an append of the official line.
                canonical_hours_check = SUPPORT_BUSINESS_HOURS.lower()
                if canonical_hours_check not in full_response.lower():
                    if (
                        "get_repair_help" in tools_called
                        or "escalate_to_human" in tools_called
                        or "lookup_order_status" in tools_called
                        or repair_like
                        or tracking_like
                    ):
                        contact_routing = "QA"
                    elif (
                        "get_warranty_or_policy" in tools_called
                        or any(kw in response_lower for kw in ("warranty", "service@osakititan"))
                    ):
                        contact_routing = "QA"
                    else:
                        contact_routing = "PRODUCTS"
                    contact_line = get_contact_msg(contact_routing, target_domain)
                    full_response = full_response.rstrip() + f"\n\n{contact_line}"

                # Now stream out the cleaned response in pseudo-chunks so the UI
                # still feels responsive. Word boundaries keep markdown intact.
                CHUNK_SIZE = 40  # chars per chunk; tuned for snappy feel
                cursor = 0
                while cursor < len(full_response):
                    end = min(cursor + CHUNK_SIZE, len(full_response))
                    yield full_response[cursor:end]
                    cursor = end

                # 💰 Cache the final cleaned response for repeat FAQ queries.
                # `make_cache_key` already returned None for PII / customer-
                # specific queries upstream, so caching here is safe.
                if cache_key:
                    cache_set(cache_key, full_response)

                # 💰 DB writes are moved off the streaming hot path via
                # BackgroundTasks. The user's response is already fully
                # delivered by this point; we just need to persist after
                # the request settles.
                background_tasks.add_task(
                    _persist_chat_log,
                    chat_request.session_id, user_query, full_response, target_domain,
                )
                logger.info(
                    f"📊 Usage: model={usage.model} calls={usage.call_count} "
                    f"in={usage.prompt_tokens} (cached={usage.cached_tokens}) "
                    f"out={usage.completion_tokens} "
                    f"cost≈${usage.estimated_cost_usd:.5f} "
                    f"elapsed={usage.elapsed_ms}ms"
                )
                background_tasks.add_task(
                    _persist_usage_log,
                    session_id=chat_request.session_id, domain=target_domain,
                    model=usage.model, call_count=usage.call_count,
                    prompt_tokens=usage.prompt_tokens,
                    cached_tokens=usage.cached_tokens,
                    completion_tokens=usage.completion_tokens,
                    estimated_cost_usd=usage.estimated_cost_usd,
                    elapsed_ms=usage.elapsed_ms,
                    cache_hit=False,
                )

            except Exception as e:
                logger.error(f"🚨 Agent loop error: {e}")
                yield "🚨 An unexpected error occurred. Please try again."

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"API Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI Server Error")


# ---------------------------------------------------------------------------
# 📊 Cost / cache observability endpoints
# ---------------------------------------------------------------------------

@app.get("/admin/cost_summary")
async def cost_summary(days: int = 7) -> Dict[str, Any]:
    """Aggregate OpenAI spend over the last N days.

    Returns total tokens, total estimated USD, average per-request cost,
    and cache-hit ratio. Intended for an internal dashboard — production
    deployments should put this behind auth.
    """
    if days <= 0 or days > 365:
        raise HTTPException(status_code=400, detail="days must be 1..365")

    from datetime import timedelta
    cutoff = datetime.now(pytz.timezone('America/Chicago')) - timedelta(days=days)

    db = SessionLocal()
    try:
        rows = (
            db.query(OpenAIUsageLog)
            .filter(OpenAIUsageLog.created_at >= cutoff)
            .all()
        )
        if not rows:
            return {"days": days, "requests": 0, "summary": "no usage in window"}

        # Cast SQLAlchemy column attribute reads into plain Python types so
        # basedpyright stops treating each `r.field` as a Column descriptor.
        total_cost = sum(float(cast(str, r.estimated_cost_usd) or 0) for r in rows)
        total_calls = sum(cast(int, r.call_count) or 0 for r in rows)
        total_prompt = sum(cast(int, r.prompt_tokens) or 0 for r in rows)
        total_cached = sum(cast(int, r.cached_tokens) or 0 for r in rows)
        total_completion = sum(cast(int, r.completion_tokens) or 0 for r in rows)
        cache_hits = sum(1 for r in rows if cast(int, r.cache_hit) > 0)

        by_model: Dict[str, Dict[str, float]] = {}
        for r in rows:
            bucket = by_model.setdefault(cast(str, r.model) or "(unknown)", {
                "requests": 0, "calls": 0, "prompt": 0, "cached": 0,
                "completion": 0, "cost_usd": 0.0,
            })
            bucket["requests"] += 1
            bucket["calls"] += cast(int, r.call_count) or 0
            bucket["prompt"] += cast(int, r.prompt_tokens) or 0
            bucket["cached"] += cast(int, r.cached_tokens) or 0
            bucket["completion"] += cast(int, r.completion_tokens) or 0
            bucket["cost_usd"] += float(cast(str, r.estimated_cost_usd) or 0)

        return {
            "days": days,
            "requests": len(rows),
            "openai_calls": total_calls,
            "prompt_tokens": total_prompt,
            "cached_tokens": total_cached,
            "completion_tokens": total_completion,
            "cache_hits": cache_hits,
            "cache_hit_ratio": round(cache_hits / max(len(rows), 1), 3),
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_request_usd": round(total_cost / max(len(rows), 1), 6),
            "by_model": {
                k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in by_model.items()
            },
            "cache": cache_stats(),
        }
    finally:
        db.close()


# ==========================================
# 💡 [아키텍처 확장] 웹후크 하드코딩 제거 및 동적 스토어 매핑
# ==========================================
def update_faiss_index_background(payload: dict, shop_domain: Optional[str] = None):
    product_status = payload.get('status', 'unknown').lower()
    item_title = payload.get('title', 'Unknown Item')
    item_id = str(payload.get('id', ''))
    
    if product_status != 'active':
        logger.info(f"⏸️ [Skip Update] Product '{item_title}' is inactive.")
        return
        
    logger.info(f"🔄 [Background Task] Updating RAG database for product: {item_title}")
    
    try:
        body_html = payload.get('body_html', '') or ''
        clean_body = re.sub('<[^<]+>', '', body_html) 
        variants = payload.get('variants', [])
        price = variants[0].get('price', 'N/A') if variants else 'N/A'
        variant_id = variants[0].get('id', '') if variants else ''
        handle = payload.get('handle', '')
        
        # 하드코딩 제거: 웹후크를 보낸 쇼피파이 스토어의 실제 도메인을 주입
        base_domain = shop_domain if shop_domain else "titanchair.com"
        product_url = f"https://{base_domain}/products/{handle}"
        checkout_url = f"https://{base_domain}/cart/{variant_id}:1" 
        
        page_content = f"Product Name: {item_title}\nPrice: ${price}\nDescription: {clean_body}\nDirect Purchase Link: {product_url}\nInstant Checkout Link: {checkout_url}"
        
        metadata = {"source": product_url, "title": item_title, "shopify_id": item_id}
        new_doc = Document(page_content=page_content, metadata=metadata)
        
        global vs_products
        # ✏️ Writer lock: blocks ALL readers (chat hybrid retrievers) until
        # the index is fully rebuilt+saved. Without this, a chat request that
        # called `similarity_search` mid-update could hit a half-rebuilt
        # docstore and silently return wrong results.
        with faiss_rwlock.write():
            if vs_products is not None:
                _docs_dict = getattr(vs_products.docstore, "_dict", {})
                ids_to_delete = [doc_id for doc_id, doc in _docs_dict.items() if doc.metadata.get('title') == item_title or item_title in doc.page_content]
                if ids_to_delete:
                    vs_products.delete(ids_to_delete)
                vs_products.add_documents([new_doc])
                vs_products.save_local(str(index_dir / "osaki_products"))
                logger.info(f"💾 [FAISS] Successfully updated index for {item_title} from {base_domain}.")
            else:
                logger.error("🚨 [FAISS] Update failed: 'vs_products' is not loaded.")
                
    except Exception as e:
        logger.error(f"🚨 [Background Task] Fatal error during FAISS update: {e}")

@app.post("/webhook/shopify/product-update")
async def shopify_webhook(
    request: Request, 
    background_tasks: BackgroundTasks,
    x_shopify_hmac_sha256: Optional[str] = Header(None),
    x_shopify_shop_domain: Optional[str] = Header(None)
):
    if not x_shopify_hmac_sha256:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing HMAC header")

    body = await request.body()
    secret = SHOPIFY_WEBHOOK_SECRET.encode('utf-8')
    hash_calc = hmac.new(secret, body, hashlib.sha256)
    calculated_hmac = base64.b64encode(hash_calc.digest()).decode('utf-8')

    if not hmac.compare_digest(calculated_hmac, x_shopify_hmac_sha256):
        logger.warning("🚨 [Security Alert] 유효하지 않은 웹후크 서명 감지!")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid HMAC signature")

    payload = await request.json()
    logger.info(f"📦 [Webhook Received] 쇼피파이 검증 통과. 출처 도메인: {x_shopify_shop_domain}")

    background_tasks.add_task(update_faiss_index_background, payload, x_shopify_shop_domain)

    return {"message": "Webhook received successfully."}