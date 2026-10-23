# Sales action tracking

The Sales metrics distinguish a recommendation/link **offered by the AI** from
an observed product click, successful cart operation, and checkout start.
Browser events are observations and may be missing or blocked. They are never
proof of payment. Paid orders, cancellations, and refunds come from authenticated
Shopify webhooks only.

## API and token

Register `sales_events.router` with the FastAPI app. Sales chat replies can
include `analytics_token`, issued by `issue_event_token(session_id, domain)` for
the existing store/session. The token expires after 24 hours, contains no customer
email/phone, and cannot be used for another store/session. It is an analytics
capability, not authentication for customer account or payment operations.

Production uses a private signing key of at least 32 characters from
`SALES_ANALYTICS_SIGNING_SECRET`, falling back to the existing
`ADMIN_SESSION_SECRET`. Use the same signing key on every worker. When neither
is usable in production, analytics is unavailable and normal Sales chat continues.
Local development can use an ephemeral key; its tokens stop working after restart.
Never put signing keys in browser code. The browser receives only its short-lived
session token.

`POST /api/v1/sales/events` accepts exactly:

```json
{
  "event_id": "c695e77b-929a-49b8-bc38-1717c6d32904",
  "session_id": "opaque-sales-session-id",
  "domain": "osakiusa.com",
  "event_token": "token-from-this-session-chat-response",
  "event_type": "product_click",
  "product_url": "https://osakiusa.com/products/product-handle"
}
```

Allowed event types are `tracking_started`, `product_click`, `add_to_cart`, and
`checkout_started`. `tracking_started` means the adapter was enabled and received
a valid chat-session token; it contains no product URL and establishes measurement
coverage, not a shopping action.
A product click requires a same-store HTTPS product URL. Cart/checkout events
may omit the URL. Query strings/fragments are discarded; free-text metadata,
customer identifiers, cart contents, arbitrary event names, and payment claims
are rejected. The server records receipt time, checks the existing session's
store and the browser Origin, limits request/session volume, and deduplicates
the same UUID. Retry an event with its original UUID rather than generating a
new observation for the same action.

## Storefront / Tidio adapter

Serve `frontend/public/sales-analytics.js` with the existing frontend deployment.
Loading it alone sends nothing. Initialize it through the storefront's existing
analytics/privacy decision; the adapter does not infer permission, enable itself,
or add cookies automatically.

```html
<script src="https://help.osakichair.com/sales-analytics.js" defer></script>
```

After the script is loaded, initialize with the current storefront and API origin:

```javascript
window.SalesAnalytics.init({
  enabled: siteAnalyticsIsEnabled,
  domain: window.location.hostname,
  apiBase: "https://help.osakichair.com",
  persistSession: false
});
// Call with the real reply from /api/v1/sales/chat:
window.SalesAnalytics.adoptResponse(salesReply);
```

Adopting a valid response sends one `tracking_started` observation per session in
the current page context. With tab persistence, its UUID is retained and reused
after navigation so the server deduplicates it. A session with tracking enabled
and no click is therefore distinguishable from a session whose storefront never
installed or enabled the adapter.

`persistSession: true` optionally retains only the opaque session/token/store in
tab-local `sessionStorage`, so the same tab can connect a chat recommendation to
later cart actions. Use it only in accordance with the site's existing storage
policy, initialize with the same settings after navigation, and call `clear()`
when ending/resetting the conversation. Tokens expire after 24 hours; get a new
token through the current session's chat reply. No customer identity values should
be used as session IDs.

Mark only actual Sales recommendation links with `data-sales-product`. Clicking
those links sends a product-click event without blocking navigation:

```html
<a data-sales-product href="https://osakiusa.com/products/product-handle">
  View recommended chair
</a>
```

For Tidio iframe buttons, the host page cannot inspect cross-origin DOM. Use the
supported Tidio/button click callback to call
`SalesAnalytics.trackProductClick(actualProductUrl)` on an actual click. Do not
call it when generating a button or delivering an AI answer.

Call `SalesAnalytics.recordCart(productUrl)` only after the real Shopify cart API
confirms success, and `SalesAnalytics.recordCheckout()` on actual checkout start.
These are explicit integration hooks; the adapter does not intercept every fetch
request or assume that clicking an Add button succeeded. Shopify Customer Events
pixels may run in a sandbox and require a separate approved adapter using these
same session/token/event fields. Wiring the external storefront/Tidio/checkout is
a deployment step; adding this repository asset does not activate those sites.

Browser events can be blocked, delayed, or omitted. A checkout on another device
will not acquire an earlier tab's analytics capability automatically.

## Reading the funnel

`aggregate_sales_events(session_ids, start, end, domain=None)` returns:

- `event_counts`: accepted event totals per type.
- `observed_sessions`: distinct cohort sessions with tracking readiness/each action, including
  incomplete or out-of-order sequences.
- `ordered_sessions`: clicks, carts following observed clicks, and checkouts
  following observed carts. Missing preceding actions remain unknown.
- `measured_sessions`: sessions with accepted tracking readiness or any action
  (existing action-only observations remain measurable).
- `measurement_coverage_pct`: measured sessions divided by all cohort sessions.
- `rates_pct`: measured-consultation-to-click, click-to-cart, cart-to-checkout
  percentages. A zero denominator returns null, meaning unmeasured/not applicable;
  a measured session with no click contributes a genuine zero-click outcome.
- `denominators`: cohort, measured, clicked, and cart session counts.

The consultation-to-click denominator includes measured sessions only, not every
session in an uninstrumented storefront. Always display measurement coverage next
to funnel rates; do not interpret an unmeasured rate as 0% performance.

These counts describe the supplied consultation cohort and selected event window.
They are associations, not evidence that the AI caused a purchase. Order/revenue
measurement remains separate. Events are ordered by server receipt time; the
browser adapter serializes sends to reduce overtaking. Offline or separately
implemented clients can still produce incomplete sequences.

`GET /admin/sales/events?session_id=...&limit=100&offset=0` requires the existing
admin key and provides a paginated diagnostic event list. Public clients cannot
read other customers' events. Normal event ingestion logs no event token, IP,
customer email/phone, or cart contents.

## Validation before activation

Run `pytest -q tests/test_sales_events.py` and the frontend adapter tests. In
staging, complete one real recommendation click, one successful cart addition,
and one checkout start; confirm the corresponding ordered funnel and verify a
failed cart operation does not create an event. Separately verify paid/cancelled/
refunded orders through the authenticated webhook. Live storefront and Tidio
integration require deployment credentials and are not established by unit tests.
