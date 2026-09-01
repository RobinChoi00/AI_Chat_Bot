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
          <p className="text-sm text-gray-500">No data in this window.</p>
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
  return (
    <div className="space-y-6">
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label="Sessions" value={String(t.started)} hint="Sales conversations started" />
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
          hint={`${data.artifacts.models} models · ${data.artifacts.doorway_models} doorway specs`}
          alert={!data.artifacts.ok}
        />
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
