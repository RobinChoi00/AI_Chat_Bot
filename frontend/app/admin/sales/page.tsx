import Link from "next/link";
import SalesMetricsDashboard from "@/components/admin/SalesMetricsDashboard";
import type { SalesMetricsResponse } from "@/lib/adminTypes";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();
const ALLOWED_DAYS = [7, 14, 30, 60, 90] as const;
export const dynamic = "force-dynamic";

type SalesPeriod = { period: "all"; days: null } | { period: "range"; days: number };

function coercePeriod(rawPeriod: unknown, rawDays: unknown): SalesPeriod {
  const days = ALLOWED_DAYS.find((value) => String(value) === rawDays);
  if (rawPeriod === "all") return { period: "all", days: null };
  if (rawPeriod === "range") return { period: "range", days: days ?? 30 };
  if (rawPeriod === undefined && days) return { period: "range", days };
  return { period: "all", days: null };
}

async function fetchMetrics(selection: SalesPeriod) {
  const key = process.env.ADMIN_API_KEY;
  if (!key) {
    return { data: null, error: "ADMIN_API_KEY is not configured." };
  }
  try {
    const query = selection.period === "all" ? "period=all" : `period=range&days=${selection.days}`;
    const response = await fetch(`${BACKEND}/admin/sales/metrics?${query}`, {
      headers: { "X-Admin-Key": key },
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    if (response.status === 401 || response.status === 403) {
      return { data: null, error: "Admin access was rejected by the backend. Check the server admin key." };
    }
    if (!response.ok) {
      return { data: null, error: `Backend returned ${response.status}` };
    }
    const data = (await response.json()) as SalesMetricsResponse;
    if (selection.period === "all" && data.range.period !== "all") {
      return { data: null, error: "The backend must be updated before all-time Sales metrics are available." };
    }
    return {
      data,
      error: null,
    };
  } catch {
    return {
      data: null,
      error: "Sales metrics could not be loaded. Check the backend connection and try again.",
    };
  }
}

export default async function AdminSalesPage({
  searchParams,
}: {
  searchParams: Promise<{ period?: string; days?: string }>;
}) {
  const { period: rawPeriod, days: rawDays } = await searchParams;
  const selection = coercePeriod(rawPeriod, rawDays);
  const { data, error } = await fetchMetrics(selection);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="mx-auto max-w-7xl">
          <h1 className="text-lg font-semibold text-gray-900">Sales AI dashboard</h1>
          <p className="text-xs text-gray-500">
            Recommendation funnel, handoffs, and lead-delivery reliability
          </p>
          <Link href="/admin/history?source=sales" className="mt-2 inline-block text-sm text-indigo-700 underline">
            View all Sales conversations →
          </Link>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-6">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="text-sm text-gray-600">Time window:</span>
          <Link
            href="/admin/sales?period=all"
            aria-current={selection.period === "all" ? "page" : undefined}
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              selection.period === "all"
                ? "bg-brand-600 text-white"
                : "border border-gray-300 bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            All time
          </Link>
          {ALLOWED_DAYS.map((n) => (
            <Link
              key={n}
              href={`/admin/sales?period=range&days=${n}`}
              aria-current={selection.period === "range" && n === selection.days ? "page" : undefined}
              className={`rounded-full px-3 py-1 text-xs font-medium ${
                selection.period === "range" && n === selection.days
                  ? "bg-brand-600 text-white"
                  : "border border-gray-300 bg-white text-gray-700 hover:bg-gray-100"
              }`}
            >
              Last {n} days
            </Link>
          ))}
        </div>
        {error ? (
          <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        ) : null}
        {data ? <SalesMetricsDashboard data={data} /> : null}
      </main>
    </div>
  );
}
