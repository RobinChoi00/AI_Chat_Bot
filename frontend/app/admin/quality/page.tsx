import AIQualityDashboard from "@/components/admin/AIQualityDashboard";
import type { AIHealth, KnowledgeReview, OutcomeQuality } from "@/lib/qualityTypes";
import { getBackendUrl } from "@/lib/backendUrl";
import MetricCohortFilter from "@/components/admin/MetricCohortFilter";
import { metricCohort, withMetricCohort } from "@/lib/adminMetricCohort";

// Admin data and runtime credentials must be read per request, never baked into a build.
export const dynamic = "force-dynamic";

async function readAdmin<T>(path: string): Promise<{ data: T | null; error: string | null }> {
  const key = process.env.ADMIN_API_KEY;
  if (!key) return { data: null, error: "ADMIN_API_KEY is not configured." };
  try {
    const response = await fetch(`${getBackendUrl()}${path}`, {
      headers: { "X-Admin-Key": key }, cache: "no-store", signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) throw new Error(`Quality service returned HTTP ${response.status}.`);
    return { data: await response.json() as T, error: null };
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : "Quality service unavailable." };
  }
}

export default async function AIQualityPage({ searchParams }: { searchParams?: Promise<{ cohort?: string }> }) {
  const cohort = metricCohort((await searchParams)?.cohort);
  const [outcomes, reviews, health] = await Promise.all([
    readAdmin<OutcomeQuality>(withMetricCohort("/admin/warranty/quality?all_time=true", cohort)),
    readAdmin<{ reviews: KnowledgeReview[] }>("/admin/warranty/knowledge-reviews?limit=100"),
    readAdmin<AIHealth>("/admin/ai/health?days=7"),
  ]);
  const cohortMismatch = outcomes.data && cohort !== "all" && outcomes.data.cohort_metadata?.selected !== cohort;
  return (
    <div className="mx-auto max-w-7xl space-y-5 px-6 py-6">
      <h1 className="text-xl font-semibold">AI quality &amp; knowledge review</h1>
      <p className="text-sm text-gray-600">Customer-confirmed outcomes, measured response times, and human-reviewed troubleshooting guidance.</p>
      <MetricCohortFilter action="/admin/quality" selected={cohort} metadata={outcomes.data?.cohort_metadata} />
      {cohortMismatch && <p role="alert" className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">The backend must be updated before release-cohort outcomes are available.</p>}
      {[outcomes.error, reviews.error, health.error].filter(Boolean).map((message, index) => (
        <p key={index} role="alert" className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">{message}</p>
      ))}
      <AIQualityDashboard outcomes={cohortMismatch ? null : outcomes.data} reviews={reviews.data?.reviews ?? null} health={health.data} />
    </div>
  );
}
