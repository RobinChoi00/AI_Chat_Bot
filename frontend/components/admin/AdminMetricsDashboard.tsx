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
  days: number;
}

const STATUS_LABELS: Record<string, string> = {
  in_progress: "In progress",
  awaiting_admin_review: "Awaiting admin",
  awaiting_evidence: "Awaiting evidence",
  send_info: "Self-service",
  sales_handoff: "Sent to sales",
  admin_reviewing: "Admin reviewing",
  need_more_information: "Needs more info",
  resolved: "Resolved",
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
  const [_, month, dom] = day.split("-");
  return `${month}/${dom}`;
}

function KpiCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const toneClass =
    tone === "good"
      ? "text-emerald-700"
      : tone === "warn"
      ? "text-amber-700"
      : tone === "bad"
      ? "text-red-700"
      : "text-gray-900";
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">
        {label}
      </div>
      <div className={`mt-1 text-2xl font-semibold ${toneClass}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-gray-500">{hint}</div>}
    </div>
  );
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
          style={{ width: `${Math.max(2, Math.min(100, pct))}%` }}
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
              title={`${r.day}\nStarted: ${r.started}\nCompleted: ${r.completed}`}
            >
              <div
                className="w-full rounded-t-sm bg-gray-200"
                style={{ height: `${Math.max(2, startedH)}%` }}
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
          <span className="inline-block h-3 w-3 rounded-sm bg-brand-500" /> Reached a terminal
        </span>
      </div>
    </div>
  );
}

export default function AdminMetricsDashboard({ data, days }: Props) {
  const { totals, by_status, by_issue_type, by_domain, top_terminals, daily_started } =
    data;

  const started = totals.started;
  const completionTone: "default" | "good" | "warn" | "bad" =
    started === 0
      ? "default"
      : totals.completion_rate_pct >= 70
      ? "good"
      : totals.completion_rate_pct >= 40
      ? "warn"
      : "bad";

  return (
    <div className="space-y-6">
      {/* KPI row */}
      <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <KpiCard
          label={`Started (last ${days}d)`}
          value={started.toString()}
        />
        <KpiCard
          label="Completion rate"
          value={`${totals.completion_rate_pct}%`}
          hint={`${totals.reached_terminal} / ${started} reached a terminal`}
          tone={completionTone}
        />
        <KpiCard
          label="Contact captured"
          value={`${totals.contact_rate_pct}%`}
          hint={`${totals.contact_captured} customers left an email`}
        />
        <KpiCard
          label="Admin resolved"
          value={`${totals.resolved} / ${totals.reached_terminal}`}
          hint={
            totals.reached_terminal > 0
              ? `${totals.resolved_rate_pct}% of terminal-reached tickets closed`
              : "No terminal-reached tickets yet"
          }
        />
        <KpiCard
          label="Abandoned"
          value={`${totals.abandoned}`}
          hint={`In-progress with no activity >${data.range.abandon_threshold_hours}h (${totals.abandoned_rate_pct}%)`}
          tone={totals.abandoned > 0 ? "warn" : "default"}
        />
        <KpiCard
          label="Median steps to terminal"
          value={`${totals.median_turns_to_terminal}`}
          hint="Fewer = faster diagnosis"
        />
        <KpiCard
          label="Awaiting admin"
          value={`${by_status.find((r: MetricsStatusRow) => r.status === "awaiting_admin_review")?.count ?? 0}`}
          hint="Needs a decision"
        />
        <KpiCard
          label="Sales handoffs"
          value={`${by_status.find((r: MetricsStatusRow) => r.status === "sales_handoff")?.count ?? 0}`}
          hint="Passed to sales team"
        />
      </section>

      {/* Trend chart */}
      <section>
        <DailyTrend rows={daily_started} />
      </section>

      {/* Breakdowns */}
      <section className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-gray-900">
            Where tickets end up
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
                      ? "bg-emerald-500"
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
            Completion by issue type
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
                  extra={`${row.completion_rate_pct}%`}
                  pct={row.completion_rate_pct}
                  color="bg-brand-500"
                />
              ))}
            </ul>
          )}
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-gray-900">
            Completion by domain
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
                  extra={`${row.completion_rate_pct}%`}
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
