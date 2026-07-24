"use client";

interface CostSummary {
  days: number;
  requests?: number;
  openai_calls?: number;
  prompt_tokens?: number;
  cached_tokens?: number;
  completion_tokens?: number;
  cache_hit_ratio?: number;
  total_cost_usd?: number;
  avg_cost_per_request_usd?: number;
  by_model?: Record<
    string,
    {
      requests: number;
      calls: number;
      cost_usd: number;
    }
  >;
  summary?: string;
}

interface FeedbackSummary {
  summary: Record<
    string,
    { up: number; down: number; total: number; up_ratio: number }
  >;
}

interface FeedbackRow {
  id: number;
  session_id: string;
  rating: string;
  comment?: string | null;
  message_content: string;
  context: string;
  ticket_id?: string | null;
  created_at?: string | null;
}

export default function AdminOpsDashboard({
  cost,
  feedbackSummary,
  downVotes,
  days,
}: {
  cost: CostSummary | null;
  feedbackSummary: FeedbackSummary | null;
  downVotes: FeedbackRow[];
  days: number;
}) {
  const contexts = Object.entries(feedbackSummary?.summary ?? {});

  return (
    <div className="space-y-6">
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi
          label={`OpenAI spend (${days}d)`}
          value={
            cost?.total_cost_usd != null
              ? `$${cost.total_cost_usd.toFixed(2)}`
              : "—"
          }
          hint={
            cost?.requests
              ? `${cost.requests} logged requests`
              : cost?.summary || "No usage in window"
          }
        />
        <Kpi
          label="Avg $/request"
          value={
            cost?.avg_cost_per_request_usd != null
              ? `$${cost.avg_cost_per_request_usd.toFixed(4)}`
              : "—"
          }
          hint={
            cost?.openai_calls != null
              ? `${cost.openai_calls} OpenAI calls`
              : undefined
          }
        />
        <Kpi
          label="Cache hit ratio"
          value={
            cost?.cache_hit_ratio != null
              ? `${Math.round(cost.cache_hit_ratio * 100)}%`
              : "—"
          }
          hint={
            cost?.cached_tokens != null
              ? `${cost.cached_tokens.toLocaleString()} cached tokens`
              : undefined
          }
        />
        <Kpi
          label="Thumbs-down (listed)"
          value={`${downVotes.length}`}
          hint="Most recent down votes"
          tone={downVotes.length > 0 ? "warn" : "good"}
        />
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold text-gray-900">Spend by model</h2>
        {!cost?.by_model || Object.keys(cost.by_model).length === 0 ? (
          <p className="text-sm text-gray-500">No model breakdown in this window.</p>
        ) : (
          <ul className="divide-y divide-gray-100">
            {Object.entries(cost.by_model)
              .sort((a, b) => b[1].cost_usd - a[1].cost_usd)
              .map(([model, row]) => (
                <li
                  key={model}
                  className="flex items-center justify-between py-2 text-sm"
                >
                  <span className="font-mono text-xs text-gray-800">{model}</span>
                  <span className="text-gray-600">
                    ${row.cost_usd.toFixed(4)} · {row.requests} req
                  </span>
                </li>
              ))}
          </ul>
        )}
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold text-gray-900">
          Feedback summary
        </h2>
        {contexts.length === 0 ? (
          <p className="text-sm text-gray-500">No feedback yet.</p>
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2">
            {contexts.map(([ctx, row]) => (
              <li
                key={ctx}
                className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-sm"
              >
                <div className="font-medium capitalize text-gray-900">{ctx}</div>
                <div className="mt-1 text-gray-600">
                  👍 {row.up} · 👎 {row.down} ·{" "}
                  {Math.round((row.up_ratio || 0) * 100)}% up
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold text-gray-900">
          Recent thumbs-down
        </h2>
        {downVotes.length === 0 ? (
          <p className="text-sm text-gray-500">No down votes in this list.</p>
        ) : (
          <ul className="space-y-3">
            {downVotes.map((row) => (
              <li
                key={row.id}
                className="rounded-lg border border-red-100 bg-red-50/40 px-3 py-2 text-sm"
              >
                <div className="mb-1 flex flex-wrap gap-2 text-[11px] text-red-700">
                  <span className="font-semibold uppercase">{row.context}</span>
                  {row.created_at ? <span>{row.created_at}</span> : null}
                  {row.ticket_id ? (
                    <a
                      href={`/admin/warranty/${row.ticket_id}`}
                      className="ml-auto underline"
                    >
                      Ticket
                    </a>
                  ) : null}
                </div>
                <p className="whitespace-pre-wrap text-gray-800">
                  {(row.message_content || "").slice(0, 400)}
                  {(row.message_content || "").length > 400 ? "…" : ""}
                </p>
                {row.comment ? (
                  <p className="mt-1 text-xs text-gray-600">
                    Comment: {row.comment}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function Kpi({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "warn" | "good";
}) {
  const toneClass =
    tone === "warn"
      ? "border-amber-200 bg-amber-50"
      : tone === "good"
        ? "border-emerald-200 bg-emerald-50"
        : "border-gray-200 bg-white";
  return (
    <div className={`rounded-xl border p-4 shadow-sm ${toneClass}`}>
      <div className="text-xs font-medium uppercase tracking-wide text-gray-500">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-gray-900">{value}</div>
      {hint ? <div className="mt-1 text-xs text-gray-500">{hint}</div> : null}
    </div>
  );
}
