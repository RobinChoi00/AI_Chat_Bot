import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { WarrantyMetricsResponse } from "@/lib/adminTypes";
import AdminMetricsPage from "./page";

vi.mock("@/lib/backendUrl", () => ({
  getBackendUrl: () => "http://backend.test",
}));

function metrics(): WarrantyMetricsResponse {
  return {
    range: {
      days: null,
      all_time: true,
      start: "2024-01-01T00:00:00-06:00",
      end: "2026-10-23T10:00:00-05:00",
      trend_days: 180,
      abandon_threshold_hours: 6,
    },
    totals: {
      started: 200,
      unique_identified_customers: 80,
      unique_sessions: 160,
      anonymous_sessions: 60,
      reached_terminal: 100,
      completion_rate_pct: 50,
      contact_captured: 110,
      contact_rate_pct: 55,
      admin_decided: 30,
      resolved: 40,
      resolved_rate_pct: 40,
      self_service_started: 50,
      self_service_resolved: 10,
      self_service_resolution_rate_pct: 20,
      self_service_overall_resolution_rate_pct: 5,
      escalated_after_self_service: 40,
      abandoned: 20,
      abandoned_rate_pct: 10,
      median_turns_to_terminal: 4,
    },
    by_status: [],
    by_issue_type: [],
    by_domain: [],
    top_terminals: [],
    daily_started: [],
    purpose_counts: { repair: 100, delivery: 80, other: 20, unknown: 0 },
    repair: {
      consultations: 100, diagnosis_completed: 60, diagnosis_completion_rate_pct: 60,
      latest_outcomes: { fixed: 10, not_fixed: 40, unable_to_attempt: 10, deferred: 5, unknown: 35 },
      result_reported: 50, confirmed_fix_rate_all_pct: 10, confirmed_fix_rate_reported_pct: 20,
      followed_up: 3, reopened_after_fix: 1,
    },
    delivery: { consultations: 80, latest_lookup_available: 30, latest_lookup_unavailable: 10, lookup_not_attempted: 40, answered: 20, needs_review: 10, unable_to_attempt: 5, deferred: 15, unknown: 30 },
    handoffs: { requested: 10, by_department: [{ department: "sales", count: 4 }, { department: "warranty", count: 6 }], by_reason: [{ reason: "customer_request", count: 10 }] },
  };
}

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubEnv("ADMIN_API_KEY", "test-admin-key");
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockResolvedValue(Response.json(metrics()));
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.resetAllMocks();
});

async function render(days?: string, cohort?: string) {
  return renderToStaticMarkup(
    await AdminMetricsPage({ searchParams: Promise.resolve({ days, cohort }) })
  );
}

describe("warranty history dashboard", () => {
  it("requests the full stored history by default and shows coverage dates", async () => {
    const html = await render();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://backend.test/admin/warranty/metrics?all_time=true",
      expect.objectContaining({ cache: "no-store" })
    );
    expect(html).toContain("Consultations (all time)");
    expect(html).toContain("2024-01-01 to 2026-10-23");
    expect(html).toContain("Identified customers");
    expect(html).toContain("Anonymous sessions");
    expect(html).toContain("not verified unique people");
    expect(html).toContain("most recent 180 days");
    expect(html).not.toContain("test-admin-key");
  });

  it("preserves recent-period filters and provides a link back to all time", async () => {
    const body = metrics();
    body.range.all_time = false;
    body.range.days = 90;
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render("90");
    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://backend.test/admin/warranty/metrics?days=90"
    );
    expect(html).toContain("Consultations (last 90d)");
    expect(html).toContain('href="/admin/warranty/dashboard?days=all"');
  });

  it.each(["all", "30junk", "999"])("uses all history for %s", async (days) => {
    await render(days);
    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://backend.test/admin/warranty/metrics?all_time=true"
    );
  });

  it("distinguishes confirmed fixes from closures and reports both denominators", async () => {
    const html = await render();
    expect(html).toContain("10 confirmed fixes / 100 repair consultations");
    expect(html).toContain("10 fixed / 50 fixed-or-not-fixed results");
    expect(html).toContain("Tickets marked closed");
    expect(html).toContain("not confirmed fixes");
    expect(html).toContain("Delivery deferred");
    expect(html).toContain("Customer confirmed the delivery question was answered");
    expect(html).toContain("Idle in-progress consultations");
    expect(html).not.toContain(">Abandoned<");
    expect(html).not.toContain(">Completion rate<");
    expect(html).toContain("staff receipt");
  });

  it("preserves release cohorts in backend requests, date links, and the filter form", async () => {
    const body = metrics();
    body.range = { ...body.range, all_time: false, days: 7 };
    body.cohort_metadata = { selected: "current", current_release: "release-2026-10-23", available_releases: ["release-2026-10-23", "release-old"], measurement_started_at: "2026-10-23T12:00:00Z", legacy_note: "Old releases were not recorded." };
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render("7", "current");
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/warranty/metrics?days=7&cohort=current");
    expect(html).toContain('href="/admin/warranty/dashboard?days=all&amp;cohort=current"');
    expect(html).toContain('href="/admin/warranty/dashboard?days=30&amp;cohort=current"');
    expect(html).toContain('value="current" selected=""');
    expect(html).toContain('name="days" value="7"');
    expect(html).toContain("release-2026-10-23");
    expect(html).toContain("Legacy records have no verified release");
    expect(html).not.toContain("test-admin-key");
  });

  it("does not present unfiltered counts as a requested release cohort", async () => {
    const html = await render("all", "legacy");
    expect(html).toContain("backend must be updated before release-cohort metrics");
    expect(html).not.toContain("10 confirmed fixes / 100 repair consultations");
  });

  it("does not invent a cohort from malformed query values", async () => {
    await render("all", "<script>bad</script>");
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/warranty/metrics?all_time=true");
  });

  it("shows configuration failures instead of zero historical usage", async () => {
    vi.stubEnv("ADMIN_API_KEY", "");
    const html = await render();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(html).toContain("ADMIN_API_KEY is not configured");
    expect(html).not.toContain("Consultations (all time)");
  });

  it("does not label limited results from an older backend as all-time usage", async () => {
    const body = metrics();
    delete body.range.all_time;
    body.range.days = 30;
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render();
    expect(html).toContain("backend must be updated");
    expect(html).not.toContain("Consultations (all time)");
  });
});
