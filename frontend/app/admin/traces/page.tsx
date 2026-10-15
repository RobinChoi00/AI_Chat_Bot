import Link from "next/link";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();

type TraceRow = {
  request_id: string;
  session_id: string;
  domain: string;
  user_query: string;
  short_circuit: string | null;
  scope_blocked: boolean;
  cache_hit: boolean;
  guard_sanitized: boolean;
  turn_count: number;
  tools_called: string[];
  total_latency_ms: number;
  created_at: string | null;
};

type TraceTurn = {
  turn: number;
  tool_name: string | null;
  tool_success: boolean;
  tool_result_len: number;
  llm_latency_ms: number;
};

type TraceDetail = TraceRow & {
  model: string;
  final_response: string;
  forced_first_tool: string | null;
  turns: TraceTurn[];
};

async function fetchJson<T>(path: string): Promise<{ data: T | null; error: string | null }> {
  const adminKey = process.env.ADMIN_API_KEY;
  if (!adminKey) {
    return { data: null, error: "ADMIN_API_KEY is not configured on the Next.js server." };
  }
  try {
    const res = await fetch(`${BACKEND}${path}`, {
      headers: { "X-Admin-Key": adminKey },
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    if (!res.ok) {
      return { data: null, error: `Backend returned ${res.status}` };
    }
    return { data: (await res.json()) as T, error: null };
  } catch (err) {
    return { data: null, error: err instanceof Error ? err.message : "Fetch failed" };
  }
}

function flags(row: TraceRow): string {
  const bits = [
    row.guard_sanitized ? "guard" : "",
    row.scope_blocked ? "blocked" : "",
    row.cache_hit ? "cache" : "",
    row.short_circuit ? row.short_circuit : "",
  ].filter(Boolean);
  return bits.join(", ") || "—";
}

export default async function AdminTracesPage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string; issues?: string }>;
}) {
  const { id, issues } = await searchParams;
  const issuesOnly = issues === "1";
  const listPath = `/admin/traces?limit=50${issuesOnly ? "&issues=1" : ""}`;
  const listRes = await fetchJson<{ rows: TraceRow[] }>(listPath);
  const detailRes = id
    ? await fetchJson<TraceDetail>(`/admin/traces/${encodeURIComponent(id)}`)
    : { data: null, error: null };
  const error = listRes.error || detailRes.error;
  const rows = listRes.data?.rows ?? [];

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="mx-auto max-w-7xl">
          <h1 className="text-lg font-semibold text-gray-900">Chat traces</h1>
          <p className="text-xs text-gray-500">
            Each customer message, the tool that ran, and whether the guard changed the answer
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-6">
        <div className="mb-4 flex gap-2">
          <Link
            href="/admin/traces"
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              issuesOnly
                ? "border border-gray-300 bg-white text-gray-700"
                : "bg-brand-600 text-white"
            }`}
          >
            Latest 50
          </Link>
          <Link
            href="/admin/traces?issues=1"
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              issuesOnly
                ? "bg-brand-600 text-white"
                : "border border-gray-300 bg-white text-gray-700"
            }`}
          >
            Guard or blocked
          </Link>
        </div>

        {error ? (
          <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        ) : null}

        {rows.length === 0 && !error ? (
          <p className="text-sm text-gray-500">No chat traces yet. New customer messages will show up here.</p>
        ) : null}

        {rows.length > 0 ? (
          <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-gray-200 text-xs uppercase text-gray-500">
                <tr>
                  <th className="px-3 py-2">When</th>
                  <th className="px-3 py-2">Customer</th>
                  <th className="px-3 py-2">Tools</th>
                  <th className="px-3 py-2">Flags</th>
                  <th className="px-3 py-2">ms</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.request_id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="whitespace-nowrap px-3 py-2 text-xs text-gray-500">
                      {row.created_at || "—"}
                    </td>
                    <td className="max-w-md px-3 py-2">
                      <Link href={`/admin/traces?id=${encodeURIComponent(row.request_id)}${issuesOnly ? "&issues=1" : ""}`} className="text-indigo-700 hover:underline">
                        {row.user_query || "(empty)"}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-xs text-gray-700">
                      {(row.tools_called || []).join(", ") || "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-gray-700">{flags(row)}</td>
                    <td className="px-3 py-2 text-xs text-gray-500">{row.total_latency_ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        {detailRes.data ? (
          <section className="mt-6 rounded-xl border border-gray-200 bg-white p-4">
            <h2 className="text-sm font-semibold text-gray-900">This turn</h2>
            <p className="mt-2 text-sm text-gray-800">
              <span className="font-medium">Customer: </span>
              {detailRes.data.user_query}
            </p>
            <p className="mt-2 whitespace-pre-wrap text-sm text-gray-800">
              <span className="font-medium">Answer: </span>
              {detailRes.data.final_response || "(empty)"}
            </p>
            <ul className="mt-3 space-y-1 text-xs text-gray-600">
              {(detailRes.data.turns || []).map((turn) => (
                <li key={turn.turn}>
                  Turn {turn.turn}: {turn.tool_name || "final answer"}
                  {turn.tool_name ? ` · ${turn.tool_success ? "ok" : "failed"} · ${turn.tool_result_len} chars` : ""}
                  {turn.llm_latency_ms ? ` · ${turn.llm_latency_ms} ms` : ""}
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </main>
    </div>
  );
}
