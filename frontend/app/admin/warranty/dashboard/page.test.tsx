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

async function render(days?: string) {
  return renderToStaticMarkup(
    await AdminMetricsPage({ searchParams: Promise.resolve({ days }) })
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
    expect(html).toContain("10 customer-confirmed fixes / 200 consultations");
    expect(html).toContain("10 / 50 consultations with a reported troubleshooting outcome");
    expect(html).toContain("Tickets marked closed");
    expect(html).toContain("not confirmed fixes");
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
