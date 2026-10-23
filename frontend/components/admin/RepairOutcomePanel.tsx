import type { RepairOutcomeMetrics } from "@/lib/adminTypes";
import { metricRate } from "@/lib/adminMetricDisplay";
import MetricCard from "./MetricCard";

export default function RepairOutcomePanel({ repair }: { repair?: RepairOutcomeMetrics }) {
  if (!repair) return <section className="rounded-xl border border-gray-200 bg-white p-4 text-sm text-gray-600">Latest repair outcomes are unavailable from this response. Historical workflow completion is not a confirmed fix.</section>;
  const outcomes = repair.latest_outcomes;
  return <section className="space-y-3" aria-labelledby="repair-outcomes-title">
    <div><h2 id="repair-outcomes-title" className="font-semibold text-gray-900">Latest repair outcomes</h2><p className="mt-1 text-xs text-gray-600">{repair.consultations} setup or malfunction consultations. Delivery checks and deferred cases are separate. Each consultation uses the latest customer result; unreported results remain unknown.</p></div>
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
      <MetricCard label="Confirmed fixed / repair consultations" value={metricRate(outcomes.fixed, repair.consultations)} hint={`${outcomes.fixed} confirmed fixes / ${repair.consultations} repair consultations`} />
      <MetricCard label="Confirmed fixed / reported results" value={metricRate(outcomes.fixed, repair.result_reported)} hint={`${outcomes.fixed} fixed / ${repair.result_reported} fixed-or-not-fixed results; no samples = —`} />
      <MetricCard label="Not fixed" value={String(outcomes.not_fixed)} hint="Latest result says the issue remains" />
      <MetricCard label="Unable to attempt" value={String(outcomes.unable_to_attempt)} hint="Could not try the steps; not counted as a reported fix result" />
      <MetricCard label="Outcome unknown" value={String(outcomes.unknown)} hint={`${outcomes.unknown} / ${repair.consultations} repair consultations; includes unfinished or unreported results`} />
      <MetricCard label="Deferred" value={String(outcomes.deferred)} hint="Customer postponed the case; not a confirmed fix" />
    </div>
    <p className="text-xs text-gray-500">{repair.followed_up} consultations with a later customer report · {repair.reopened_after_fix} reported a problem after an earlier fix. Follow-up reports do not reopen a claim or approve service.</p>
  </section>;
}
