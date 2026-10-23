"use client";

import type { SalesMetricsResponse } from "@/lib/adminTypes";
import Kpi from "./MetricCard";
import AdminJourneyPanel from "./AdminJourneyPanel";
import { metricCount, metricRate, recordedRate } from "@/lib/adminMetricDisplay";

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
                  style={{ width: `${Math.max(0, (Number(row.count) / max) * 100)}%` }}
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
  const outcomes = data.outcomes;
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
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-3" aria-label="Saved conversation coverage">
        <Kpi label={allTime ? "Sessions (all time)" : `Sessions (last ${data.range.days}d)`} value={String(t.started)} hint="Sales conversations started in the reporting period" />
        <Kpi label="Latest state: answer provided" value={metricCount(outcomes?.conversation.answer_provided_sessions)} hint="A saved response is not proof it was helpful, read, or completed the customer's task" />
        <Kpi label="Latest state unknown" value={metricCount(outcomes?.conversation.unknown_sessions)} hint="No conclusive saved state. A one-turn answer can be a normal end; its helpfulness also remains unverified without feedback." />
      </section>

      {outcomes ? <section className="space-y-3" aria-labelledby="sales-outcomes-title">
        <h2 id="sales-outcomes-title" className="font-semibold text-gray-900">Purpose-specific Sales outcomes</h2>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <Kpi label="Recommendation requests completed" value={metricRate(outcomes.recommendation.completed_sessions, outcomes.recommendation.denominator_sessions)} hint={`${outcomes.recommendation.completed_sessions} / ${outcomes.recommendation.denominator_sessions} eligible recommendation conversations; ${outcomes.recommendation.recommended_sessions} with picks, ${outcomes.recommendation.no_verified_match_sessions} honest no-match outcomes`} />
          <Kpi label="Helpful ratings" value={metricRate(outcomes.quality.helpful, outcomes.quality.rated)} hint={`${outcomes.quality.helpful} helpful / ${outcomes.quality.rated} ratings · ${outcomes.quality.unrated_sessions} conversations without a rating; helpfulness is unknown for them`} />
          <Kpi label="Repeated unclear conversations" value={String(outcomes.quality.repeated_unclear_sessions)} hint={`${outcomes.quality.unclear_turns} unclear replies / ${outcomes.quality.assistant_turns} assistant replies; inspect the conversation before classifying an error`} alert={outcomes.quality.repeated_unclear_sessions > 0} />
          <Kpi label="Support requests / redirects saved" value={String(outcomes.human.requested_sessions)} hint={`${outcomes.human.explicit_requested_sessions} explicit requests · ${outcomes.human.staff_confirmed_sessions === null ? "staff acceptance unmeasured" : `${outcomes.human.staff_confirmed_sessions} conversations with authenticated staff acceptance observed; other staff activity unknown`}`} />
          <Kpi label="SMTP submissions accepted" value={metricRate(outcomes.follow_up.sent, outcomes.follow_up.sent + outcomes.follow_up.failed)} hint={`${outcomes.follow_up.sent} accepted / ${outcomes.follow_up.sent + outcomes.follow_up.failed} completed send attempts · ${outcomes.follow_up.pending} pending · ${outcomes.follow_up.failed} failed. Acceptance does not confirm inbox delivery or staff response.`} alert={outcomes.follow_up.failed > 0} />
          <Kpi label="Idle awaiting an answer" value={String(outcomes.conversation.idle_missing_followup_sessions)} hint={`No later reply after ${outcomes.conversation.idle_after_hours}h · ${outcomes.conversation.active_waiting_sessions} still active at a question; not confirmed abandonment`} />
        </div>
        <p className="text-xs text-gray-500">Contact follow-up: {outcomes.follow_up.lead_sessions} conversations have a saved lead. Recommendation results are evaluated only among eligible recommendation requests. No verified match can be an appropriate result.</p>
        <p className="text-xs text-gray-500">Rating coverage: {recordedRate(outcomes.quality.rating_coverage_pct, outcomes.quality.response_sessions)} of {outcomes.quality.response_sessions} conversations with an assistant reply had a rating. {outcomes.quality.rated_sessions} distinct conversations were rated; rating records are not unique people.</p>
      </section> : <p className="rounded-xl border border-gray-200 bg-white p-4 text-sm text-gray-600">Purpose-specific outcome measurements are unavailable from this response. Historical conversation counts remain available.</p>}

      <AdminJourneyPanel journey={data.journey} />

      <section className="rounded-xl border border-gray-200 bg-white p-4">
        <h2 className="font-semibold text-gray-900">Observed question stages</h2>
        <p className="mt-1 text-xs text-gray-600">Latest recorded occurrence of each question per conversation. Questions can be optional or occur in different orders. An answer or human request can be a valid next action; missing follow-up is not a confirmed exit.</p>
        {data.question_funnel?.some((row) => row.reached > 0) ? <div className="mt-3 overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr className="border-b text-gray-500">{["Question", "Reached", "Continued", "Answer provided", "Human requested", "Active waiting", "Idle, no reply", "Unknown"].map((label) => <th className="whitespace-nowrap px-2 py-2 font-medium" key={label}>{label}</th>)}</tr></thead><tbody>{data.question_funnel.filter((row) => row.reached > 0).map((row) => <tr key={row.stage} className="border-b border-gray-100"><th className="px-2 py-2 font-medium">{row.label}</th>{[row.reached, row.continued, row.answer_provided, row.human_requested, row.active_waiting, row.idle_missing_followup, row.unknown].map((count, index) => <td key={index} className="px-2 py-2 tabular-nums">{count}</td>)}</tr>)}</tbody></table></div> : <p className="mt-3 text-sm text-gray-500">No saved question-stage observations for this cohort.</p>}
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="font-semibold">Observed shopper actions &amp; paid orders</h2>
        <p className="mt-1 text-xs text-gray-500">Actions are recorded only after storefront analytics is enabled. Offered links are not clicks. Missing observations do not establish that no action occurred.</p>
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Kpi label="Analytics coverage" value={recordedRate(data.behavior_funnel?.measurement_coverage_pct, t.started)} hint={`${metricCount(data.behavior_funnel?.measured_sessions)} / ${t.started} conversations with an accepted tracking signal; other conversations are unmeasured`} />
          <Kpi label="Product clicks" value={metricCount(data.behavior_funnel?.ordered_sessions.product_click)} hint={`${recordedRate(data.behavior_funnel?.rates_pct.consultation_to_click, data.behavior_funnel?.denominators.measured_sessions)} of ${metricCount(data.behavior_funnel?.denominators.measured_sessions)} measured conversations; unmeasured clicks remain unknown`} />
          <Kpi label="Added to cart" value={metricCount(data.behavior_funnel?.ordered_sessions.add_to_cart)} hint={`${recordedRate(data.behavior_funnel?.rates_pct.click_to_cart, data.behavior_funnel?.denominators.clicked_sessions)} of ${metricCount(data.behavior_funnel?.denominators.clicked_sessions)} observed clicking conversations, in recorded order`} />
          <Kpi label="Checkout started" value={metricCount(data.behavior_funnel?.ordered_sessions.checkout_started)} hint={`${recordedRate(data.behavior_funnel?.rates_pct.cart_to_checkout, data.behavior_funnel?.denominators.cart_sessions)} of ${metricCount(data.behavior_funnel?.denominators.cart_sessions)} observed cart conversations, in recorded order`} />
          <Kpi label="Paid orders" value={String(t.paid_orders_seen ?? "—")} hint="Authenticated payment evidence; cancelled and fully refunded orders excluded" />
          <Kpi label="Paid conversations" value={String(t.paid_sessions ?? "—")} hint={`${metricRate(t.paid_sessions, t.started)} of the selected conversation cohort has a paid order`} />
          <Kpi label="Payment unknown" value={String(t.payment_unknown_orders ?? "—")} hint="Legacy or missing payment evidence; excluded from paid revenue" />
          <Kpi label="Net attributed USD" value={t.attributed_revenue_usd === undefined ? "—" : `$${t.attributed_revenue_usd.toFixed(2)}`} hint="Verified USD net payment only; attribution does not prove AI caused the sale" />
        </div>
        {data.attributed_revenue_by_currency && Object.keys(data.attributed_revenue_by_currency).length > 0 && <p className="mt-3 text-xs text-gray-600">Net attributed amounts by original currency: {Object.entries(data.attributed_revenue_by_currency).map(([currency, amount]) => `${currency} ${amount.toFixed(2)}`).join(" · ")}</p>}
      </section>

      {outcomes && <section className="rounded-xl border border-gray-200 bg-white p-4">
        <h2 className="font-semibold text-gray-900">Follow-up submission reliability</h2>
        <p className="mt-2 text-sm">{outcomes.follow_up.lead_records} saved lead records · {outcomes.follow_up.sent} SMTP accepted · {outcomes.follow_up.failed} failed · {outcomes.follow_up.pending} pending · {outcomes.follow_up.unknown} status unknown</p>
        <p className="mt-1 text-xs text-gray-600">SMTP acceptance confirms a submission to the mail service. Inbox arrival, shopper receipt and a staff response are unconfirmed.</p>
        <p className="mt-2 text-xs text-gray-600">Transport: {outcomes.follow_up.transport ? outcomes.follow_up.transport.configured ? "Configured" : "Not configured" : "Unmeasured"}{outcomes.follow_up.transport?.failure_code ? ` · ${outcomes.follow_up.transport.failure_code.replace(/_/g, " ")}` : ""}</p>
        {Boolean(outcomes.follow_up.failure_codes?.length) && <ul className="mt-2 space-y-1 text-xs text-amber-800">{outcomes.follow_up.failure_codes?.map((row) => <li key={row.code}>{row.code.replace(/_/g, " ")}: {row.count}</li>)}</ul>}
      </section>}

      <section className="space-y-3" aria-labelledby="sales-reference-title">
        <h2 id="sales-reference-title" className="font-semibold text-gray-900">Conversation &amp; service reference</h2>
        <p className="text-xs text-gray-600">Longer conversations and more contact records do not establish answer quality. Catalog health below describes the current files, not the historical release cohort.</p>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <Kpi label="Two or more customer turns" value={metricRate(t.engaged, t.started)} hint={`${t.engaged} / ${t.started} saved conversations; repeat questions can increase this count`} />
          <Kpi label="Sessions with product picks" value={String(t.recommended)} hint="All-purpose reference count; recommendation eligibility is measured separately above" />
          <Kpi label="Sessions with a handoff flag" value={String(t.handoffs)} hint="Legacy state flag; does not prove staff receipt or a completed transfer" />
          <Kpi label="Saved lead records" value={String(t.leads)} hint="Records can repeat within a conversation; not unique customers or confirmed sales" />
          <Kpi label="Customer / assistant messages" value={`${t.user_turns} / ${t.assistant_turns}`} hint="Saved message counts" />
          <Kpi label="Current catalog health" value={data.artifacts.ok ? "Healthy" : "Needs attention"} hint={`${data.artifacts.models} models · ${data.artifacts.doorway_models} doorway specifications; current artifacts only`} alert={!data.artifacts.ok} />
        </div>
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
