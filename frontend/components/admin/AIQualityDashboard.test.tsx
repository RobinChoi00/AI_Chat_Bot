import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import AIQualityDashboard from "./AIQualityDashboard";
import type { AIHealth } from "@/lib/qualityTypes";

vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: vi.fn() }) }));

describe("AI quality dashboard", () => {
  it("keeps unknown outcomes distinct from confirmed fixes and claim decisions", () => {
    const html = renderToStaticMarkup(<AIQualityDashboard outcomes={{
      total_consultations: 10, latest_confirmed_fixed: 2, latest_still_broken: 3,
      outcome_unknown: 5, followed_up: 2, reopened_after_fix: 1,
      confirmed_fix_rate_all_pct: 20, confirmed_fix_rate_reported_pct: 40,
      repair: {
        consultations: 10, diagnosis_completed: 6, diagnosis_completion_rate_pct: 60,
        latest_outcomes: { fixed: 2, not_fixed: 3, unable_to_attempt: 0, deferred: 0, unknown: 5 },
        result_reported: 5, confirmed_fix_rate_all_pct: 20, confirmed_fix_rate_reported_pct: 40,
        followed_up: 2, reopened_after_fix: 1,
      },
    }} health={null} reviews={[]} />);
    expect(html).toContain("Outcome unknown");
    expect(html).toContain("20%");
    expect(html).toContain("40%");
    expect(html).toContain("do not reopen a claim or approve service");
    expect(html).toContain("Submit for review");
  });

  it("requires review of drafts and does not expose approval controls for accepted entries", () => {
    const review = { id: 1, status: "pending" as const, title: "Remote power check", category: "remote",
      model_names: ["Example Model"], diagnostic: "The remote has no power.",
      customer_steps: ["Check the external power connection."], source_url: "https://support.example/manual",
      review_until: "2027-01-01", reviewed_by: null };
    const pending = renderToStaticMarkup(<AIQualityDashboard outcomes={null} health={null} reviews={[review]} />);
    expect(pending).toContain("Approve verified guidance");
    const approved = renderToStaticMarkup(<AIQualityDashboard outcomes={null} health={null} reviews={[{ ...review, status: "approved", reviewed_by: "reviewer" }]} />);
    expect(approved).not.toContain("Approve verified guidance");
    expect(approved).toContain("Reviewed by");
  });

  it("keeps configured models separate from unmeasured provider access", () => {
    const health: AIHealth = {
      models: { agent: "candidate-alias", router: "router-alias", embedding: "embedding-model" },
      app_release_id: "release-current",
      model_configuration: {
        agent: { effective_model: "candidate-alias", default_model: "code-default", source: "env" },
        router: { effective_model: "router-alias", default_model: "router-alias", source: "default" },
      },
      model_calls: { days: 7, retention_days: 30, max_retained_calls: 10000, scope: "Recorded attempts only.", roles: {
        agent: { calls: 0, successful_calls: 0, failed_calls: 0, latest_call: null, latest_successful_call: null },
        router: { calls: 0, successful_calls: 0, failed_calls: 0, latest_call: null, latest_successful_call: null },
      } },
      latency: {}, stock_cache: {}, warranty_embeddings: {},
    };
    const html = renderToStaticMarkup(<AIQualityDashboard outcomes={null} reviews={null} health={health} />);
    expect(html).toContain("environment override");
    expect(html).toContain("code default");
    expect(html).toContain("release-current");
    expect(html).toContain("Provider model is unknown");
    expect(html).toContain("Model access is unverified");
  });

  it("shows a failed latest request alongside the older successful provider identity", () => {
    const health: AIHealth = {
      models: { agent: "new-candidate", router: "router-alias", embedding: "embedding-model" },
      model_calls: { days: 7, retention_days: 30, max_retained_calls: 10000, scope: "Past success does not prove current access.", roles: {
        agent: {
          calls: 2, successful_calls: 1, failed_calls: 1,
          latest_call: { requested_model: "new-candidate", provider_model: null, succeeded: false, error_code: "permission", release_id: "release-new", observed_at: null },
          latest_successful_call: { requested_model: "old-alias", provider_model: "old-snapshot", succeeded: true, error_code: null, release_id: "release-old", observed_at: null },
        },
        router: { calls: 0, successful_calls: 0, failed_calls: 0, latest_call: null, latest_successful_call: null },
      } },
      latency: {}, stock_cache: {}, warranty_embeddings: {},
    };
    const html = renderToStaticMarkup(<AIQualityDashboard outcomes={null} reviews={null} health={health} />);
    expect(html).toContain("Latest request: new-candidate");
    expect(html).toContain("failed (permission)");
    expect(html).toContain("Latest successful request: old-alias");
    expect(html).toContain("provider returned old-snapshot");
    expect(html).toContain("release-old");
    expect(html).toContain("Past success does not prove current access");
  });
});
