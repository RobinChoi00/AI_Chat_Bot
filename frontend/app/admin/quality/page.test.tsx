import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OutcomeQuality } from "@/lib/qualityTypes";

vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: vi.fn() }) }));
vi.mock("@/lib/backendUrl", () => ({ getBackendUrl: () => "http://backend.test" }));
import AIQualityPage, { dynamic } from "./page";

function outcomes(selected = "legacy"): OutcomeQuality {
  return {
    total_consultations: 2, latest_confirmed_fixed: 0, latest_still_broken: 0, outcome_unknown: 2, followed_up: 0, reopened_after_fix: 0, confirmed_fix_rate_all_pct: 0, confirmed_fix_rate_reported_pct: 0,
    repair: { consultations: 2, diagnosis_completed: 0, diagnosis_completion_rate_pct: 0, latest_outcomes: { fixed: 0, not_fixed: 0, unable_to_attempt: 0, deferred: 0, unknown: 2 }, result_reported: 0, confirmed_fix_rate_all_pct: 0, confirmed_fix_rate_reported_pct: null, followed_up: 0, reopened_after_fix: 0 },
    cohort_metadata: { selected, current_release: "release-current", available_releases: ["release-current"], measurement_started_at: null, legacy_note: "Legacy cases have no verified release." },
  };
}

const fetchMock = vi.fn();
beforeEach(() => {
  vi.stubEnv("ADMIN_API_KEY", "quality-server-secret");
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.resetAllMocks(); });

function backend(body: OutcomeQuality) {
  fetchMock.mockImplementation(async (raw: string) => {
    if (raw.includes("/warranty/quality?")) return Response.json(body);
    if (raw.includes("knowledge-reviews")) return Response.json({ reviews: [] });
    return Response.json({ models: { agent: "current-agent", router: "current-router", embedding: "current-embedding" }, latency: { warranty: { samples: 0, p50_ms: 0, p95_ms: 0 } }, stock_cache: {}, warranty_embeddings: {} });
  });
}

describe("quality release cohorts", () => {
  it("filters outcome evidence while clearly keeping current runtime health separate", async () => {
    backend(outcomes());
    const html = renderToStaticMarkup(await AIQualityPage({ searchParams: Promise.resolve({ cohort: "legacy" }) }));
    expect(dynamic).toBe("force-dynamic");
    expect(fetchMock).toHaveBeenCalledWith("http://backend.test/admin/warranty/quality?all_time=true&cohort=legacy", expect.objectContaining({ cache: "no-store", headers: { "X-Admin-Key": "quality-server-secret" } }));
    expect(html).toContain("Outcome unknown");
    expect(html).toContain("0 fixed / 0 fixed-or-not-fixed results");
    expect(html).toContain("runtime view is not filtered by the selected release cohort");
    expect(html).toContain("Model names describe the current configuration");
    expect(html).not.toContain("median 0 ms");
    expect(html).not.toContain("quality-server-secret");
  });

  it("does not label all-time outcome evidence as a cohort an older backend ignored", async () => {
    backend(outcomes("all"));
    const html = renderToStaticMarkup(await AIQualityPage({ searchParams: Promise.resolve({ cohort: "current" }) }));
    expect(html).toContain("backend must be updated before release-cohort outcomes");
    expect(html).not.toContain("Latest repair outcomes");
    expect(html).toContain("Current runtime");
  });
});
