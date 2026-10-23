"use client";

import type { SalesMetricsResponse } from "@/lib/adminTypes";

function Kpi({
  label,
  value,
  hint,
  alert = false,
}: {
  label: string;
  value: string;
  hint: string;
  alert?: boolean;
}) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">
        {label}
      </p>
      <p className={`mt-1 text-2xl font-semibold ${alert ? "text-red-700" : "text-gray-900"}`}>
        {value}
      </p>
      <p className="mt-1 text-xs text-gray-500">{hint}</p>
    </div>
  );
}

function Breakdown({
  title,
  rows,
  keyName,
}: {
  title: string;
  rows: Array<Record<string, string | number>>;
  keyName: string;
}) {
  const max = Math.max(1, ...rows.map((row) => Number(row.count)));
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-gray-900">{title}</h2>
      <div className="space-y-2">
        {rows.length === 0 ? (
          <p className="text-sm text-gray-500">No data in this reporting period.</p>
        ) : (
          rows.map((row) => (
            <div key={String(row[keyName])} className="flex items-center gap-3">
              <span className="w-36 truncate text-xs text-gray-700">
                {String(row[keyName])}
              </span>
              <div className="h-4 flex-1 overflow-hidden rounded bg-gray-100">
                <div
                  className="h-full bg-brand-500"
                  style={{ width: `${Math.max(3, (Number(row.count) / max) * 100)}%` }}
                />
              </div>
              <span className="w-10 text-right text-xs tabular-nums text-gray-600">
                {Number(row.count)}
              </span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export default function SalesMetricsDashboard({
  data,
}: {
  data: SalesMetricsResponse;
}) {
  const t = data.totals;
  const allTime = data.range.period === "all";
  const startDate = data.range.start_date ?? data.range.start?.slice(0, 10) ?? null;
  const endDate = data.range.end_date ?? data.range.end.slice(0, 10);
  return (
    <div className="space-y-6">
      <section aria-label="Reporting period" className="rounded-xl border border-gray-200 bg-white p-4 text-sm text-gray-600">
        <p className="font-semibold text-gray-900">{allTime ? "All time" : `Last ${data.range.days} days`}</p>
        <p className="mt-1">
          {startDate ? `Saved Sales activity from ${startDate} through ${endDate}.` : `Recording start date is unavailable. Reporting through ${endDate}.`}
        </p>
        <p className="mt-1 text-xs">
          {allTime ? "The metrics below include the full stored history." : "The metrics below cover the selected time window."} Sessions are conversations, not unique customers. Visits without a saved conversation and deleted records are not included.
        </p>
      </section>
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label={allTime ? "Sessions (all time)" : `Sessions (last ${data.range.days}d)`} value={String(t.started)} hint="Sales conversations started in the reporting period" />
        <Kpi
          label="Engagement"
          value={`${t.engagement_rate_pct}%`}
          hint={`${t.engaged} sessions reached a second customer turn`}
        />
        <Kpi
          label="Recommendations"
          value={`${t.recommend_rate_pct}%`}
          hint={`${t.recommended} sessions reached product picks`}
        />
        <Kpi
          label="No fit"
          value={`${t.nofit_rate_pct}%`}
          hint={`${t.nofit} honest no-fit outcomes`}
        />
        <Kpi
          label="Human handoff"
          value={`${t.handoff_rate_pct}%`}
          hint={`${t.handoffs} sessions transferred`}
        />
        <Kpi
          label="Lead capture"
          value={`${t.lead_rate_pct}%`}
          hint={`${t.leads} leads captured`}
        />
        <Kpi
          label="Lead delivery failures"
          value={String(t.lead_forward_failed)}
          hint={`${t.lead_forward_failure_rate_pct}% of captured leads`}
          alert={t.lead_forward_failed > 0}
        />
        <Kpi
          label="Turns"
          value={`${t.user_turns} / ${t.assistant_turns}`}
          hint="Customer / assistant messages"
        />
        <Kpi
          label="Sales data"
          value={data.artifacts.ok ? "Healthy" : "Needs attention"}
          hint={`Current catalog · ${data.artifacts.models} models · ${data.artifacts.doorway_models} doorway specs`}
          alert={!data.artifacts.ok}
        />
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="font-semibold">Observed shopper actions &amp; paid orders</h2>
        <p className="mt-1 text-xs text-gray-500">Actions are recorded only after storefront analytics is enabled. Offered links are not clicks. Missing observations do not establish that no action occurred.</p>
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Kpi label="Analytics coverage" value={data.behavior_funnel?.measurement_coverage_pct == null ? "—" : `${data.behavior_funnel.measurement_coverage_pct}%`} hint={`${data.behavior_funnel?.measured_sessions ?? 0} conversations with an accepted tracking signal; other conversations are unmeasured`} />
          <Kpi label="Product clicks" value={String(data.behavior_funnel?.observed_sessions.product_click ?? "—")} hint="Sessions with an observed product link click" />
          <Kpi label="Added to cart" value={String(data.behavior_funnel?.ordered_sessions.add_to_cart ?? "—")} hint="Observed cart success after a product click" />
          <Kpi label="Checkout started" value={String(data.behavior_funnel?.ordered_sessions.checkout_started ?? "—")} hint="Observed checkout start after cart success" />
          <Kpi label="Paid orders" value={String(t.paid_orders_seen ?? "—")} hint="Authenticated payment evidence; cancelled and fully refunded orders excluded" />
          <Kpi label="Paid conversations" value={String(t.paid_sessions ?? "—")} hint={`${t.session_to_order_rate_pct ?? "—"}% of the selected conversation cohort has a paid order`} />
          <Kpi label="Payment unknown" value={String(t.payment_unknown_orders ?? "—")} hint="Legacy or missing payment evidence; excluded from paid revenue" />
          <Kpi label="Helpful answers" value={t.csat_score_pct === undefined || !t.csat_rated ? "—" : `${t.csat_score_pct}%`} hint={`${t.csat_rated ?? 0} rated answers; unrated answers remain unknown`} />
          <Kpi label="Net attributed USD" value={t.attributed_revenue_usd === undefined ? "—" : `$${t.attributed_revenue_usd.toFixed(2)}`} hint="Verified USD net payment only; attribution does not prove AI caused the sale" />
        </div>
        {data.attributed_revenue_by_currency && Object.keys(data.attributed_revenue_by_currency).length > 0 && <p className="mt-3 text-xs text-gray-600">Net attributed amounts by original currency: {Object.entries(data.attributed_revenue_by_currency).map(([currency, amount]) => `${currency} ${amount.toFixed(2)}`).join(" · ")}</p>}
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Breakdown title="Intent mix" rows={data.by_intent} keyName="intent" />
        <Breakdown title="Storefronts" rows={data.by_domain} keyName="domain" />
        <Breakdown title="Session status" rows={data.by_status} keyName="status" />
        <Breakdown title="Lead delivery" rows={data.lead_delivery} keyName="status" />
      </section>
    </div>
  );
}
