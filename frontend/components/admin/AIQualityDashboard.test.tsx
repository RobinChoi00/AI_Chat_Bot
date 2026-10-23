import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import AIQualityDashboard from "./AIQualityDashboard";

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
});
