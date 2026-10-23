"use client";

/**
 * Warranty completion-rate dashboard — Client Component.
 *
 * Kept purely presentational: the server component upstream fetches the
 * metrics with the admin API key (never exposed to the browser) and passes
 * the resulting payload down as a plain prop.
 */

import Link from "next/link";
import { useMemo } from "react";
import MetricCard from "./MetricCard";
import RepairOutcomePanel from "./RepairOutcomePanel";
import AdminJourneyPanel from "./AdminJourneyPanel";
import { metricCount, metricRate } from "@/lib/adminMetricDisplay";
import type {
  WarrantyMetricsResponse,
  MetricsDailyRow,
  MetricsStatusRow,
  MetricsIssueRow,
  MetricsDomainRow,
  MetricsTerminalRow,
} from "@/lib/adminTypes";

interface Props {
  data: WarrantyMetricsResponse;
  days: number | null;
}

const STATUS_LABELS: Record<string, string> = {
  in_progress: "In progress",
  awaiting_admin_review: "Awaiting admin",
  awaiting_evidence: "Awaiting evidence",
  send_info: "Self-service guidance issued",
  sales_handoff: "Sales request saved",
  admin_reviewing: "Admin reviewing",
  need_more_information: "Needs more info",
  resolved: "Marked closed",
  closed: "Closed",
};

const ISSUE_LABELS: Record<string, string> = {
  installation: "Installation",
  delivery: "Delivery",
  defect: "Defect",
  unknown: "Not classified",
};

function statusLabel(key: string): string {
  return STATUS_LABELS[key] ?? key;
}

function issueLabel(key: string): string {
  return ISSUE_LABELS[key] ?? key;
}

function formatDay(day: string): string {
  return day.slice(0, 10);
}

function BarRow({
  label,
  count,
  extra,
  pct,
  color = "bg-brand-500",
}: {
  label: string;
  count: number;
  extra?: string;
  pct: number;
  color?: string;
}) {
  return (
    <li className="flex items-center gap-3 py-1.5">
      <div className="w-40 shrink-0 truncate text-sm text-gray-700" title={label}>
        {label}
      </div>
      <div className="relative h-6 flex-1 overflow-hidden rounded-md bg-gray-100">
        <div
          className={`h-full ${color}`}
            style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
        />
      </div>
      <div className="w-24 shrink-0 text-right text-xs text-gray-600 tabular-nums">
        {count}
        {extra ? <span className="text-gray-400"> · {extra}</span> : null}
      </div>
    </li>
  );
}

function DailyTrend({ rows }: { rows: MetricsDailyRow[] }) {
  const max = useMemo(
    () => Math.max(1, ...rows.map((r) => r.started)),
    [rows]
  );
  if (rows.length === 0) return null;
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-900">Started per day</h3>
        <div className="text-xs text-gray-500">
          Peak: {max} · Last {rows.length} days
        </div>
      </div>
      <div className="flex h-32 items-end gap-1">
        {rows.map((r) => {
          const startedH = (r.started / max) * 100;
          const completedH = (r.completed / max) * 100;
          return (
            <div
              key={r.day}
              className="group relative flex flex-1 flex-col items-center justify-end"
              title={`${r.day}\nStarted: ${r.started}\nWorkflow endpoint: ${r.completed}`}
            >
              <div
                className="w-full rounded-t-sm bg-gray-200"
                style={{ height: `${Math.max(0, startedH)}%` }}
              />
              <div
                className="absolute bottom-0 w-full rounded-t-sm bg-brand-500"
                style={{ height: `${Math.max(0, completedH)}%` }}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex justify-between text-[10px] text-gray-500">
        <span>{formatDay(rows[0].day)}</span>
        <span>{formatDay(rows[rows.length - 1].day)}</span>
      </div>
      <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm bg-gray-200" /> Started
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm bg-brand-500" /> Reached a workflow endpoint
        </span>
      </div>
    </div>
  );
}

export default function AdminMetricsDashboard({ data, days }: Props) {
  const { totals, by_status, by_issue_type, by_domain, top_terminals, daily_started } =
    data;

  const started = totals.started;
  const allTime = data.range.all_time ?? days === null;
  const repair = data.repair;
  const delivery = data.delivery;

  return (
    <div className="space-y-6">
      <p className="text-sm text-gray-600">
        {allTime ? "All stored warranty consultations" : "Stored warranty consultations"}
        {started > 0 && ` · ${formatDay(data.range.start)} to ${formatDay(data.range.end)}`}
        . Counts cover saved tickets; visits without a saved consultation and deleted records are not included.
      </p>
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4" aria-label="Saved consultation coverage">
        <MetricCard label={allTime ? "Consultations (all time)" : `Consultations (last ${days}d)`} value={String(started)} hint="Saved consultations, not site visits or verified unique people" />
        <MetricCard label="Repair / delivery consultations" value={`${metricCount(repair?.consultations)} / ${metricCount(delivery?.consultations)}`} hint={`${metricCount(data.purpose_counts?.other)} other · ${metricCount(data.purpose_counts?.unknown)} purpose unknown; separate outcome denominators`} />
        <MetricCard label="Identified customers (email addresses)" value={metricCount(totals.unique_identified_customers)} hint="Distinct email addresses; repeat consultations with the same email count once" />
        <MetricCard label="Anonymous sessions" value={metricCount(totals.anonymous_sessions)} hint="Sessions without an email; these are not verified unique people" />
      </section>

      <RepairOutcomePanel repair={repair} />

      <section className="space-y-3" aria-labelledby="delivery-outcomes-title">
        <h2 id="delivery-outcomes-title" className="font-semibold text-gray-900">Delivery outcomes</h2>
        <p className="text-xs text-gray-600">Delivery updates and postponed cases are not repair fixes. A provider update returned by the API does not prove the customer read it or that the shipment was delivered.</p>
        {delivery ? <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <MetricCard label="Latest lookup available" value={String(delivery.latest_lookup_available)} hint={`${delivery.latest_lookup_available} / ${delivery.consultations} delivery consultations returned a provider update`} />
          <MetricCard label="Latest lookup unavailable" value={String(delivery.latest_lookup_unavailable)} hint="The latest attempted lookup could not verify an update" />
          <MetricCard label="No lookup attempted" value={String(delivery.lookup_not_attempted)} hint="No saved tracking lookup; not evidence of an unsuccessful shipment" />
          <MetricCard label="Delivery question answered" value={String(delivery.answered)} hint="Customer confirmed the delivery question was answered; not a repair fix or proof of shipment delivery" />
          <MetricCard label="Delivery deferred" value={String(delivery.deferred)} hint="Customer will return later; not a self-resolution" />
          <MetricCard label="Delivery outcome unknown" value={String(delivery.unknown)} hint="No confirmed delivery outcome was recorded" />
        </div> : <p className="rounded-lg bg-white p-4 text-sm text-gray-600">Purpose-specific delivery measurements are unavailable.</p>}
        {delivery && <p className="text-xs text-gray-500">{delivery.needs_review} delivery cases need team review · {delivery.unable_to_attempt} could not proceed. Review requests do not confirm staff receipt.</p>}
      </section>

      <AdminJourneyPanel journey={data.journey} />

      <section className="rounded-xl border border-gray-200 bg-white p-4">
        <h2 className="font-semibold text-gray-900">Support requests saved</h2>
        <p className="mt-2 text-2xl font-semibold">{metricCount(data.handoffs?.requested)}</p>
        <p className="mt-1 text-xs text-gray-600">Saved requests do not confirm a notification, staff receipt, or callback. Department records show the intended contact route.</p>
        {data.handoffs && <div className="mt-3 grid gap-3 sm:grid-cols-2"><div><h3 className="text-xs font-medium text-gray-500">Intended department</h3><ul className="mt-1 text-sm">{data.handoffs.by_department.map((row) => <li key={row.department}>{row.department || "Not recorded"}: {row.count}</li>)}</ul></div><div><h3 className="text-xs font-medium text-gray-500">Reason</h3><ul className="mt-1 text-sm">{data.handoffs.by_reason.map((row) => <li key={row.reason}>{row.reason.replace(/_/g, " ")}: {row.count}</li>)}</ul></div></div>}
      </section>

      <section className="space-y-3" aria-labelledby="workflow-reference-title">
        <h2 id="workflow-reference-title" className="font-semibold text-gray-900">Workflow &amp; queue reference</h2>
        <p className="text-xs text-gray-600">Workflow endpoints, saved emails, and closed statuses do not establish customer resolution. Idle activity is an observation, not confirmed abandonment.</p>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <MetricCard label="Idle in-progress consultations" value={String(totals.idle_in_progress ?? totals.abandoned)} hint={`No activity >${data.range.abandon_threshold_hours}h · ${metricRate(totals.idle_in_progress ?? totals.abandoned, started)} of saved consultations; departure and outcome unknown`} />
          <MetricCard label="Repair diagnosis completed" value={metricRate(repair?.diagnosis_completed, repair?.consultations)} hint={`${metricCount(repair?.diagnosis_completed)} / ${metricCount(repair?.consultations)} repair consultations reached a diagnosis endpoint`} />
          <MetricCard label="Workflow endpoint reached" value={metricRate(totals.reached_terminal, started)} hint={`${totals.reached_terminal} / ${started} saved consultations; not confirmed fixes`} />
          <MetricCard label="Tickets marked closed" value={String(totals.resolved)} hint="Administrative and customer closures; not confirmed fixes" />
          <MetricCard label="Saved email coverage" value={metricRate(totals.contact_captured, started)} hint={`${totals.contact_captured} / ${started} saved consultations; visits before contact collection are not measured`} />
          <MetricCard label="Median recorded turns to endpoint" value={totals.reached_terminal > 0 ? String(totals.median_turns_to_terminal) : "—"} hint="Recorded workflow turns; fewer turns do not establish a better outcome" />
        </div>
      </section>

      {/* Trend chart */}
      <section>
        {allTime && (
          <p className="mb-2 text-xs text-gray-500">
            Daily trend: most recent {data.range.trend_days ?? daily_started.length} days. Totals above include the full stored history.
          </p>
        )}
        <DailyTrend rows={daily_started} />
      </section>

      {/* Breakdowns */}
      <section className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-gray-900">
            Current queue statuses
          </h3>
          {by_status.length === 0 ? (
            <p className="text-sm text-gray-500">No tickets in this window.</p>
          ) : (
            <ul>
              {by_status.map((row: MetricsStatusRow) => (
                <BarRow
                  key={row.status}
                  label={statusLabel(row.status)}
                  count={row.count}
                  pct={started > 0 ? (row.count / started) * 100 : 0}
                  color={
                    row.status === "resolved"
                      ? "bg-slate-400"
                      : row.status === "in_progress"
                      ? "bg-gray-400"
                      : row.status === "sales_handoff"
                      ? "bg-indigo-500"
                      : "bg-brand-500"
                  }
                />
              ))}
            </ul>
          )}
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-gray-900">
            Workflow endpoint by issue type
          </h3>
          {by_issue_type.length === 0 ? (
            <p className="text-sm text-gray-500">No tickets in this window.</p>
          ) : (
            <ul>
              {by_issue_type.map((row: MetricsIssueRow) => (
                <BarRow
                  key={row.issue_type}
                  label={issueLabel(row.issue_type)}
                  count={row.count}
                  extra={metricRate(row.completed, row.count)}
                  pct={row.completion_rate_pct}
                  color="bg-brand-500"
                />
              ))}
            </ul>
          )}
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-gray-900">
            Workflow endpoint by domain
          </h3>
          {by_domain.length === 0 ? (
            <p className="text-sm text-gray-500">No tickets in this window.</p>
          ) : (
            <ul>
              {by_domain.slice(0, 8).map((row: MetricsDomainRow) => (
                <BarRow
                  key={row.domain}
                  label={row.domain}
                  count={row.count}
                  extra={metricRate(row.completed, row.count)}
                  pct={row.completion_rate_pct}
                  color="bg-teal-500"
                />
              ))}
            </ul>
          )}
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-900">
              Most-reached terminals
            </h3>
            <span className="text-xs text-gray-500">Top 10</span>
          </div>
          {top_terminals.length === 0 ? (
            <p className="text-sm text-gray-500">
              No customer has reached a terminal node yet.
            </p>
          ) : (
            <ul>
              {top_terminals.map((row: MetricsTerminalRow) => (
                <BarRow
                  key={row.node_id}
                  label={row.node_id}
                  count={row.count}
                  pct={
                    top_terminals[0].count > 0
                      ? (row.count / top_terminals[0].count) * 100
                      : 0
                  }
                  color="bg-amber-500"
                />
              ))}
            </ul>
          )}
        </div>
      </section>

      <p className="text-xs text-gray-500">
        Auto-refreshed on every page load ·{" "}
        <Link href="/admin/warranty" className="underline">
          Back to ticket queue
        </Link>
      </p>
    </div>
  );
}
