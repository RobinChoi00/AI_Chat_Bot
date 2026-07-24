/**
 * Admin ops page — OpenAI cost + thumbs-down feedback.
 * Server Component; ADMIN_API_KEY never reaches the browser.
 */

import Link from "next/link";
import AdminOpsDashboard from "@/components/admin/AdminOpsDashboard";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();
const ALLOWED_DAYS = [7, 14, 30, 60, 90] as const;
type AllowedDays = (typeof ALLOWED_DAYS)[number];

function coerceDays(raw: string | undefined): AllowedDays {
  const parsed = Number.parseInt(raw ?? "", 10);
  const found = ALLOWED_DAYS.find((n) => n === parsed);
  return (found ?? 14) as AllowedDays;
}

async function fetchJson<T>(
  path: string
): Promise<{ data: T | null; error: string | null }> {
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
    return {
      data: null,
      error: err instanceof Error ? err.message : "Fetch failed",
    };
  }
}

export default async function AdminOpsPage({
  searchParams,
}: {
  searchParams: Promise<{ days?: string }>;
}) {
  const { days: rawDays } = await searchParams;
  const days = coerceDays(rawDays);

  const [costRes, summaryRes, downRes] = await Promise.all([
    fetchJson<Record<string, unknown>>(`/admin/cost_summary?days=${days}`),
    fetchJson<{ summary: Record<string, unknown> }>("/admin/feedback/summary"),
    fetchJson<{ rows: unknown[] }>("/admin/feedback?rating=down&limit=40"),
  ]);

  const error = costRes.error || summaryRes.error || downRes.error;

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
              Ops — cost & feedback
            </h1>
            <p className="text-xs font-medium text-red-600">
              INTERNAL USE ONLY — Do not share this page
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="text-sm text-gray-600">Cost window:</span>
          {ALLOWED_DAYS.map((n) => (
            <Link
              key={n}
              href={`/admin/ops?days=${n}`}
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

        <AdminOpsDashboard
          cost={costRes.data as never}
          feedbackSummary={summaryRes.data as never}
          downVotes={((downRes.data?.rows as never[]) || []) as never}
          days={days}
        />
      </main>
    </div>
  );
}
