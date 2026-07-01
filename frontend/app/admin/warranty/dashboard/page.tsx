/**
 * Admin warranty completion-rate dashboard — Server Component.
 *
 * Auth model mirrors /admin/warranty: the ADMIN_API_KEY lives in the Next.js
 * server environment only. The browser never sees the key; instead this
 * server component fetches the backend metrics endpoint and passes plain
 * JSON down to a client component for rendering.
 */

import Link from "next/link";
import AdminMetricsDashboard from "@/components/admin/AdminMetricsDashboard";
import type { WarrantyMetricsResponse } from "@/lib/adminTypes";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();

const ALLOWED_DAYS = [7, 14, 30, 60, 90] as const;
type AllowedDays = (typeof ALLOWED_DAYS)[number];

function coerceDays(raw: string | undefined): AllowedDays {
  const parsed = Number.parseInt(raw ?? "", 10);
  const found = ALLOWED_DAYS.find((n) => n === parsed);
  return (found ?? 30) as AllowedDays;
}

async function fetchMetrics(
  days: number
): Promise<{ data: WarrantyMetricsResponse | null; error: string | null }> {
  const adminKey = process.env.ADMIN_API_KEY;
  if (!adminKey) {
    return {
      data: null,
      error: "ADMIN_API_KEY is not configured on the Next.js server.",
    };
  }

  try {
    const res = await fetch(
      `${BACKEND}/admin/warranty/metrics?days=${days}`,
      {
        headers: { "X-Admin-Key": adminKey },
        cache: "no-store",
        signal: AbortSignal.timeout(15_000),
      }
    );
    if (res.status === 401 || res.status === 503) {
      return { data: null, error: "Admin key rejected by backend." };
    }
    if (!res.ok) {
      return {
        data: null,
        error: `Backend returned ${res.status} ${res.statusText}`,
      };
    }
    const body = (await res.json()) as WarrantyMetricsResponse;
    return { data: body, error: null };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Fetch failed";
    return { data: null, error: message };
  }
}

export default async function AdminMetricsPage({
  searchParams,
}: {
  searchParams: Promise<{ days?: string }>;
}) {
  const { days: rawDays } = await searchParams;
  const days = coerceDays(rawDays);
  const { data, error } = await fetchMetrics(days);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white px-6 py-3 shadow-sm">
        <div className="mx-auto flex max-w-7xl items-center gap-3">
          <Link
            href="/admin/warranty"
            className="text-sm text-gray-400 hover:text-gray-700"
          >
            ← Ticket queue
          </Link>
          <span className="text-gray-300">|</span>
          <div>
            <h1 className="text-base font-semibold text-gray-900">
              Warranty completion dashboard
            </h1>
            <p className="text-xs font-medium text-red-600">
              INTERNAL USE ONLY — Do not share this page
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="text-sm text-gray-600">Time window:</span>
          {ALLOWED_DAYS.map((n) => (
            <Link
              key={n}
              href={`/admin/warranty/dashboard?days=${n}`}
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

        {error && (
          <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        )}

        {data && <AdminMetricsDashboard data={data} days={days} />}
      </main>
    </div>
  );
}
