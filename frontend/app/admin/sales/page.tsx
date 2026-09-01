import Link from "next/link";
import SalesMetricsDashboard from "@/components/admin/SalesMetricsDashboard";
import type { SalesMetricsResponse } from "@/lib/adminTypes";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();
const ALLOWED_DAYS = [7, 14, 30, 60, 90] as const;

function coerceDays(raw: string | undefined): number {
  const parsed = Number.parseInt(raw ?? "", 10);
  return ALLOWED_DAYS.includes(parsed as (typeof ALLOWED_DAYS)[number])
    ? parsed
    : 30;
}

async function fetchMetrics(days: number) {
  const key = process.env.ADMIN_API_KEY;
  if (!key) {
    return { data: null, error: "ADMIN_API_KEY is not configured." };
  }
  try {
    const response = await fetch(`${BACKEND}/admin/sales/metrics?days=${days}`, {
      headers: { "X-Admin-Key": key },
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) {
      return { data: null, error: `Backend returned ${response.status}` };
    }
    return {
      data: (await response.json()) as SalesMetricsResponse,
      error: null,
    };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Fetch failed",
    };
  }
}

export default async function AdminSalesPage({
  searchParams,
}: {
  searchParams: Promise<{ days?: string }>;
}) {
  const { days: rawDays } = await searchParams;
  const days = coerceDays(rawDays);
  const { data, error } = await fetchMetrics(days);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="mx-auto max-w-7xl">
          <h1 className="text-lg font-semibold text-gray-900">Sales AI dashboard</h1>
          <p className="text-xs text-gray-500">
            Recommendation funnel, handoffs, and lead-delivery reliability
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-6">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="text-sm text-gray-600">Time window:</span>
          {ALLOWED_DAYS.map((n) => (
            <Link
              key={n}
              href={`/admin/sales?days=${n}`}
              className={`rounded-full px-3 py-1 text-xs font-medium ${
                n === days
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
