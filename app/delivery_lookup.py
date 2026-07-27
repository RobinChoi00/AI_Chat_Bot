"""
delivery_lookup.py
==================
Warranty Delivery flow — live tracking lookup (no LLM).

Supports either:
  - Tracking number  → Track123 (+ AfterShip fallback)
  - Order # OR email → Shopify (+ Track123 enrich inside fetch_shopify_order_status)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from store_config import get_order_tracking_page_url

_CARRIER_TRACKING_URLS = {
    "ups": "https://www.ups.com/track?tracknum={tn}",
    "fedex": "https://www.fedex.com/fedextrack/?trknbr={tn}",
    "usps": "https://tools.usps.com/go/TrackConfirmAction?tLabels={tn}",
    "dhl": "https://www.dhl.com/global-en/home/tracking/tracking-express.html?submit=1&tracking-id={tn}",
    "ontrac": "https://www.ontrac.com/tracking/?number={tn}",
    "lasership": "https://www.lasership.com/track/{tn}",
    "amazon": "https://track.amazon.com/tracking/{tn}",
}


@dataclass
class TrackingSnapshot:
    source: str  # track123 | shopify | unavailable
    available: bool
    status: str = ""
    carrier: str = ""
    tracking_number: str = ""
    tracking_url: str = ""
    eta: str = ""
    last_event: str = ""
    current_location: str = ""
    current_hub: str = ""
    events: List[Dict[str, str]] = field(default_factory=list)
    order_number: str = ""
    purchase_date: str = ""
    product_names: List[str] = field(default_factory=list)
    total_amount: str = ""
    error: Optional[str] = None
    looked_up_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _lazy_logistics():
    """Import main lazily — main already loads warranty_router at startup."""
    import main as logistics  # noqa: WPS433

    return logistics


def _format_purchase_date(iso_raw: str) -> str:
    if not iso_raw:
        return ""
    try:
        dt = datetime.fromisoformat(iso_raw.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except ValueError:
        return iso_raw[:10] if len(iso_raw) >= 10 else iso_raw


def _format_order_total(amount: str, currency: str = "USD") -> str:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return amount or ""
    if currency.upper() == "USD":
        return f"${value:,.2f}"
    return f"{currency.upper()} {value:,.2f}"


def _order_details_lines(snapshot: TrackingSnapshot) -> List[str]:
    if not (snapshot.order_number or snapshot.product_names or snapshot.total_amount):
        return []

    lines: List[str] = []
    if snapshot.order_number:
        lines.append(f"We found your order **{snapshot.order_number}**:")
    else:
        lines.append("We found your order:")

    if snapshot.purchase_date:
        lines.append(f"- Purchase Date: {snapshot.purchase_date}")
    if snapshot.product_names:
        if len(snapshot.product_names) == 1:
            lines.append(f"- Product: {snapshot.product_names[0]}")
        else:
            lines.append("- Products:")
            for product in snapshot.product_names:
                lines.append(f"  • {product}")
    if snapshot.total_amount:
        lines.append(f"- Order Total: {snapshot.total_amount}")
    return lines


def _snapshot_from_tracking_data(
    data: Dict[str, Any],
    *,
    source: str,
    tracking_number: str = "",
) -> TrackingSnapshot:
    if data.get("error"):
        return TrackingSnapshot(
            source="unavailable",
            available=False,
            tracking_number=tracking_number or str(data.get("tracking_number", "")),
            error=str(data["error"]),
            looked_up_at=_now_iso(),
        )

    tn = str(data.get("tracking_number", "") or tracking_number)
    status = str(data.get("status", "UNKNOWN"))
    processing = status in ("PROCESSING", "UNFULFILLED") or not tn

    purchase_date = str(data.get("purchase_date", "") or "")
    if not purchase_date and data.get("purchase_date_raw"):
        purchase_date = _format_purchase_date(str(data["purchase_date_raw"]))

    total_amount = ""
    if data.get("total_amount"):
        total_amount = _format_order_total(
            str(data["total_amount"]),
            str(data.get("currency_code", "USD")),
        )

    return TrackingSnapshot(
        source=source,
        available=True,
        status=status if not processing else "PROCESSING",
        carrier=str(data.get("company", "")),
        tracking_number=tn,
        tracking_url=str(data.get("tracking_url", "")),
        eta=str(data.get("eta", "Pending carrier update")),
        last_event=str(data.get("last_event", "")),
        current_location=str(data.get("current_location", "")),
        current_hub=str(data.get("current_hub", "")),
        events=list(data.get("events") or []),
        order_number=str(data.get("order_number", "")),
        purchase_date=purchase_date,
        product_names=list(data.get("product_names") or []),
        total_amount=total_amount,
        looked_up_at=_now_iso(),
    )


def parse_order_or_email(raw: str) -> tuple[str, str]:
    """Split customer input into (order_number, email). Either may be empty."""
    from delivery_intake import is_plausible_email, is_plausible_order_id  # noqa: WPS433

    text = (raw or "").strip()
    if not text:
        return "", ""
    if is_plausible_email(text):
        return "", text
    clean = text.replace("#", "").strip()
    if is_plausible_order_id(clean):
        return clean, ""
    return "", ""


def lookup_by_tracking_number(tracking_number: str, domain: str) -> TrackingSnapshot:
    """Track123 (+ AfterShip fallback) using the customer-supplied tracking number."""
    tn = (tracking_number or "").strip()
    if not tn:
        return TrackingSnapshot(
            source="unavailable",
            available=False,
            error="Please enter a tracking number.",
            looked_up_at=_now_iso(),
        )

    logistics = _lazy_logistics()
    store = logistics.get_store_config(domain)
    enriched = logistics.enrich_tracking_from_track123(tn, store)

    if not enriched:
        carrier = logistics.resolve_carrier_name("", tn, "")
        enriched = logistics.enrich_tracking_from_aftership(carrier, tn) or {}

    if not enriched:
        return TrackingSnapshot(
            source="unavailable",
            available=False,
            tracking_number=tn,
            error="We could not verify this tracking number with the carrier right now.",
            looked_up_at=_now_iso(),
        )

    data: Dict[str, Any] = {
        "status": enriched.get("status", "IN_TRANSIT"),
        "company": enriched.get("company") or logistics.resolve_carrier_name("", tn, ""),
        "tracking_number": tn,
        "tracking_url": enriched.get("tracking_url", ""),
        "current_location": enriched.get("current_location", "Carrier network"),
        "current_hub": enriched.get("current_hub", ""),
        "eta": enriched.get("eta", "Pending carrier update"),
        "last_event": enriched.get("last_event", "Latest carrier update"),
        "events": enriched.get("events", []),
    }
    snap = _snapshot_from_tracking_data(data, source="track123", tracking_number=tn)
    snap.available = True
    return snap


def lookup_by_order_or_email(raw: str, domain: str) -> TrackingSnapshot:
    """Shopify order lookup — order number OR email is enough."""
    order, email = parse_order_or_email(raw)
    if not order and not email:
        return TrackingSnapshot(
            source="unavailable",
            available=False,
            error="Please provide your order number or checkout email.",
            looked_up_at=_now_iso(),
        )

    logistics = _lazy_logistics()
    data = logistics.fetch_shopify_order_status(order, email, domain)
    return _snapshot_from_tracking_data(data, source="shopify")


_CARRIER_PATTERNS = [
    (r"^1Z[A-Z0-9]{16}$", "ups"),
    (r"^9[2-5]\d{20,}$", "usps"),
    (r"^(94|93|92|95)\d{18,}$", "usps"),
    (r"^\d{20,22}$", "usps"),
    (r"^\d{12,15}$", "fedex"),
    (r"^C\d{8,}$", "ontrac"),
    (r"^1LS\d+$", "lasership"),
    (r"^TBA\d+$", "amazon"),
]


def _infer_carrier_slug(tracking_number: str, carrier: str = "") -> str:
    carrier_lower = (carrier or "").lower()
    for slug in _CARRIER_TRACKING_URLS:
        if slug in carrier_lower:
            return slug

    tn = re.sub(r"\s+", "", (tracking_number or "").strip())
    for pattern, slug in _CARRIER_PATTERNS:
        if re.match(pattern, tn, re.IGNORECASE):
            return slug
    return ""


def build_carrier_tracking_url(tracking_number: str, carrier: str = "") -> str:
    """Best-effort direct carrier tracking URL when live API lookup fails."""
    tn = re.sub(r"\s+", "", (tracking_number or "").strip())
    if not tn:
        return ""

    slug = _infer_carrier_slug(tn, carrier)
    template = _CARRIER_TRACKING_URLS.get(slug)
    if template:
        return template.format(tn=tn)
    return ""


def build_self_service_lookup_links(
    *,
    domain: str,
    lookup_kind: str,
    raw_input: str = "",
) -> List[Tuple[str, str]]:
    """
    Return (label, url) pairs the customer can use when automatic lookup fails.
    """
    links: List[Tuple[str, str]] = []
    store_url = get_order_tracking_page_url(domain)
    if store_url:
        links.append(("Track on our website", store_url))

    if lookup_kind == "tracking":
        carrier_url = build_carrier_tracking_url(raw_input)
        if carrier_url:
            links.append(("Track with the carrier", carrier_url))

    return links


def format_warranty_tracking_message(
    snapshot: TrackingSnapshot,
    *,
    domain: str = "",
    lookup_kind: str = "",
    raw_input: str = "",
    continue_with_questions: bool = True,
) -> str:
    """Customer-facing English message for the warranty chat (no sales footer).

    When ``continue_with_questions`` is False (status-only lookup path), close with
    a follow-up note instead of promising more warranty questions.
    """
    closing = (
        "We'll continue with your delivery warranty questions below."
        if continue_with_questions
        else (
            "That's the latest status we have. "
            "Our team will follow up shortly if you need more help."
        )
    )

    if not snapshot.available:
        lines = [
            "We couldn't verify your delivery details automatically right now.",
            "Our support team will look this up and follow up with you.",
        ]
        if snapshot.error:
            lines.append(f"\n({snapshot.error})")

        help_links = build_self_service_lookup_links(
            domain=domain,
            lookup_kind=lookup_kind,
            raw_input=raw_input,
        )
        if help_links:
            lines.append("")
            lines.append("You can also check directly here:")
            for label, url in help_links:
                lines.append(f"- [{label}]({url})")

        return "\n".join(lines)

    order_lines = _order_details_lines(snapshot)

    if snapshot.status in ("PROCESSING", "UNFULFILLED") or not snapshot.tracking_number:
        lines = order_lines or ["We found your order!"]
        lines.extend(
            [
                "",
                f"- Order Status: **{snapshot.status or 'PROCESSING'}** (in preparation)",
                "- A tracking number will be emailed once the carrier picks up the shipment.",
                "- Typical processing time before pickup: **1–3 business days**.",
                "",
                closing,
            ]
        )
        return "\n".join(lines)

    lines = list(order_lines)
    if lines:
        lines.append("")
    lines.extend(
        [
            "Here is your latest delivery update:",
            f"- Current Status: {snapshot.status}",
            f"- Carrier: {snapshot.carrier or 'Carrier'}",
            f"- Tracking Number: {snapshot.tracking_number}",
        ]
    )
    if snapshot.current_location:
        lines.append(f"- Current Location: {snapshot.current_location}")
    if snapshot.eta:
        lines.append(f"- Estimated Delivery: {snapshot.eta}")
    if snapshot.last_event:
        lines.append(f"- Last Carrier Event: {snapshot.last_event}")
    if snapshot.tracking_url:
        lines.append(f"- Live Tracking: {snapshot.tracking_url}")

    if snapshot.events:
        lines.append("")
        lines.append("Recent updates:")
        for event in snapshot.events[-3:]:
            t = event.get("time", "")
            loc = event.get("location", "")
            msg = event.get("event", "")
            lines.append(f"- {t} | {loc} | {msg}")

    lines.append("")
    lines.append(closing)
    return "\n".join(lines)


def persist_snapshot(ticket_id: str, snapshot: TrackingSnapshot) -> None:
    from warranty_eligibility import evaluate_purchase_eligibility  # noqa: WPS433
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    eligibility = evaluate_purchase_eligibility(snapshot.purchase_date)

    with warranty_db_session() as db:
        ticket = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket:
            ticket.set_collected("tracking_snapshot", json.dumps(snapshot.to_dict()))
            ticket.set_collected("warranty_eligibility", json.dumps(eligibility.to_dict()))
            if eligibility.purchase_date_iso:
                ticket.set_collected("purchase_date", eligibility.purchase_date_iso)
            ticket.set_collected("warranty_eligibility_status", eligibility.status)


def append_eligibility_to_tracking_message(message: str, ticket_id: str) -> str:
    """Append a soft eligibility note when purchase date was evaluated."""
    from warranty_eligibility import customer_eligibility_note  # noqa: WPS433
    from warranty_models import WarrantyTicket, warranty_db_session  # noqa: WPS433

    with warranty_db_session() as db:
        ticket = (
            db.query(WarrantyTicket)
            .filter(WarrantyTicket.ticket_id == ticket_id)
            .first()
        )
        if ticket is None:
            return message
        raw = ticket.get_collected().get("warranty_eligibility")
    if not raw:
        return message
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return message
    if not isinstance(data, dict):
        return message
    from warranty_eligibility import EligibilityResult  # noqa: WPS433

    result = EligibilityResult(
        status=str(data.get("status") or "unknown"),
        purchase_date=str(data.get("purchase_date") or ""),
        purchase_date_iso=str(data.get("purchase_date_iso") or ""),
        eligibility_years=int(data.get("eligibility_years") or 3),
        expires_on=str(data.get("expires_on") or ""),
        days_remaining=data.get("days_remaining"),
        summary=str(data.get("summary") or ""),
    )
    note = customer_eligibility_note(result)
    if not note:
        return message
    return f"{message.rstrip()}{note}"
