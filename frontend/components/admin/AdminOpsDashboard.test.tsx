import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import AdminOpsDashboard from "./AdminOpsDashboard";

describe("operations cost measurement", () => {
  it("renders unknown costs safely and labels priced totals as incomplete", () => {
    const html = renderToStaticMarkup(<AdminOpsDashboard days={14} feedbackSummary={null} downVotes={[]} cost={{
      days: 14, requests: 3, openai_calls: 3, total_cost_usd: null,
      avg_cost_per_request_usd: null, unknown_cost_requests: 1, known_cost_usd: .005,
      by_model: {
        "unpriced-model": { requests: 1, calls: 1, cost_usd: null, unknown_cost_requests: 1 },
        "zero-cost-no-call": { requests: 1, calls: 0, cost_usd: 0, unknown_cost_requests: 0 },
        "priced-model": { requests: 1, calls: 2, cost_usd: .005, unknown_cost_requests: 0 },
      },
    }} />);
    expect(html).toContain("Cost is unmeasured for 1 logged request.");
    expect(html).toContain("Priced subtotal: $0.0050.");
    expect(html).toContain("it is not complete spend");
    expect(html.match(/Unmeasured/g)).toHaveLength(3);
    expect(html).not.toContain("NaN");
    expect(html.indexOf("priced-model")).toBeLessThan(html.indexOf("zero-cost-no-call"));
    expect(html.indexOf("zero-cost-no-call")).toBeLessThan(html.indexOf("unpriced-model"));
  });

  it("preserves a measured zero when every logged request has a known cost", () => {
    const html = renderToStaticMarkup(<AdminOpsDashboard days={14} feedbackSummary={null} downVotes={[]} cost={{
      days: 14, requests: 1, total_cost_usd: 0, avg_cost_per_request_usd: 0,
      unknown_cost_requests: 0, known_cost_usd: 0,
      by_model: { "no-model-call": { requests: 1, calls: 0, cost_usd: 0 } },
    }} />);
    expect(html).toContain(">$0.00<");
    expect(html).toContain("Logged general-chat cost (14d)");
    expect(html).toContain("router, embeddings, OCR and speech are not included");
    expect(html).not.toContain("OpenAI spend");
    expect(html).toContain(">$0.0000<");
    expect(html).not.toContain("Unmeasured");
    expect(html).not.toContain("not complete spend");
  });

  it("does not report a complete total when unknown model costs accompany a numeric subtotal", () => {
    const html = renderToStaticMarkup(<AdminOpsDashboard days={14} feedbackSummary={null} downVotes={[]} cost={{
      days: 14, requests: 1, total_cost_usd: 0, avg_cost_per_request_usd: 0,
      unknown_cost_requests: 1, known_cost_usd: 0,
      by_model: { "unpriced-model": { requests: 1, calls: 1, cost_usd: null } },
    }} />);
    expect(html.match(/Unmeasured/g)).toHaveLength(3);
    expect(html).toContain("Priced subtotal: $0.0000.");
    expect(html).not.toContain(">$0.00<");
  });
});
