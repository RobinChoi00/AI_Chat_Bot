import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SalesMetricsResponse } from "@/lib/adminTypes";

vi.mock("next/link", () => ({
  default: ({ href, children, "aria-current": current }: { href: string; children: ReactNode; "aria-current"?: "page" }) => <a href={href} aria-current={current}>{children}</a>,
}));
vi.mock("@/lib/backendUrl", () => ({ getBackendUrl: () => "http://backend.test" }));

import AdminSalesPage, { dynamic } from "./page";

function metrics(): SalesMetricsResponse {
  return {
    range: { period: "all", days: null, start: "2020-01-02T10:30:00-06:00", end: "2026-10-23T10:00:00-05:00", start_date: "2020-01-02", end_date: "2026-10-23" },
    totals: {
      started: 205, engaged: 100, engagement_rate_pct: 48.8,
      recommended: 90, recommend_rate_pct: 43.9, nofit: 2, nofit_rate_pct: 2.2,
      handoffs: 10, handoff_rate_pct: 4.9, leads: 8, lead_rate_pct: 3.9,
      lead_forward_failed: 0, lead_forward_failure_rate_pct: 0, user_turns: 1000, assistant_turns: 1000,
      paid_orders_seen: 2, paid_sessions: 1, session_to_order_rate_pct: 0.5, attributed_revenue_usd: 2600,
    },
    by_status: [], by_intent: [], by_domain: [], by_channel: [], lead_delivery: [], daily: [],
    artifacts: { ok: true, models: 177, doorway_models: 175, files: {} },
  };
}

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubEnv("ADMIN_API_KEY", "sales-page-test-admin-secret");
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockResolvedValue(Response.json(metrics()));
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.resetAllMocks();
});

async function render(params: { period?: string; days?: string; cohort?: string } = {}) {
  return renderToStaticMarkup(await AdminSalesPage({ searchParams: Promise.resolve(params) }));
}

describe("Sales all-time dashboard", () => {
  it("is dynamic and loads the full stored history by default", async () => {
    expect(dynamic).toBe("force-dynamic");
    const html = await render();
    expect(fetchMock).toHaveBeenCalledWith("http://backend.test/admin/sales/metrics?period=all", expect.objectContaining({
      cache: "no-store", headers: { "X-Admin-Key": "sales-page-test-admin-secret" },
    }));
    expect(html).toContain('href="/admin/sales?period=all" aria-current="page"');
    expect(html).toContain("Sessions (all time)");
    expect(html).toContain("2020-01-02 through 2026-10-23");
    expect(html).toContain("full stored history");
    expect(html).toContain('href="/admin/history?source=sales"');
    expect(html).toContain("View all Sales conversations");
    expect(html).not.toContain("sales-page-test-admin-secret");
  });

  it("honors explicit All time even when a legacy days value is present", async () => {
    const html = await render({ period: "all", days: "7" });
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/sales/metrics?period=all");
    expect(html).toContain("Sessions (all time)");
    expect(html).not.toContain("Sessions (last 7d)");
  });

  it.each(["7", "14", "30", "60", "90"])("retains the %s-day filter and explicit range links", async (days) => {
    const body = metrics();
    body.range = { ...body.range, period: "range", days: Number(days), start_date: "2026-10-17", start: "2026-10-17T00:00:00-05:00" };
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render({ period: "range", days });
    expect(fetchMock.mock.calls[0][0]).toBe(`http://backend.test/admin/sales/metrics?period=range&days=${days}`);
    expect(html).toContain(`href="/admin/sales?period=range&amp;days=${days}" aria-current="page"`);
    expect(html).toContain(`Sessions (last ${days}d)`);
    expect(html).not.toContain("Sessions (all time)");
    expect(html).toContain('href="/admin/sales?period=all"');
  });

  it("keeps legacy days-only links on their selected range", async () => {
    const body = metrics();
    body.range = { ...body.range, period: "range", days: 7 };
    fetchMock.mockResolvedValue(Response.json(body));
    expect(await render({ days: "7" })).toContain("Sessions (last 7d)");
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/sales/metrics?period=range&days=7");
  });

  it.each(["7junk", "07", "14.0", "999"])("does not partially parse invalid days %s", async (days) => {
    await render({ days });
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/sales/metrics?period=all");
  });

  it("defaults an explicit range with invalid days to 30 days", async () => {
    const body = metrics();
    body.range = { ...body.range, period: "range", days: 30 };
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render({ period: "range", days: "7junk" });
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/sales/metrics?period=range&days=30");
    expect(html).toContain("Sessions (last 30d)");
  });

  it("renders empty lifetime dates without null, fake epoch dates or NaN", async () => {
    const body = metrics();
    body.range.start = null;
    body.range.start_date = null;
    body.totals.started = 0;
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render();
    expect(html).toContain("Recording start date is unavailable. Reporting through 2026-10-23");
    expect(html).toContain("Sessions (all time)");
    expect(html).not.toContain("1970-");
    expect(html).not.toContain("NaN");
    expect(html).not.toContain("null through");
    expect(html).not.toContain("undefined");
  });

  it("does not mistake undated legacy activity for an empty history", async () => {
    const body = metrics();
    body.range.start = null;
    body.range.start_date = null;
    body.totals.started = 5;
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render();
    expect(html).toContain("Recording start date is unavailable");
    expect(html).toContain("Sessions (all time)");
    expect(html).not.toContain("No stored Sales activity");
  });

  it("does not mislabel finite metrics from an older backend as all time", async () => {
    const body = metrics();
    body.range.period = "range";
    body.range.days = 30;
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render();
    expect(html).toContain("backend must be updated before all-time Sales metrics");
    expect(html).not.toContain("Sessions (all time)");
  });

  it("keeps configuration, access and backend errors separate from empty history", async () => {
    vi.stubEnv("ADMIN_API_KEY", "");
    const missing = await render();
    expect(missing).toContain("ADMIN_API_KEY is not configured");
    expect(fetchMock).not.toHaveBeenCalled();
    vi.stubEnv("ADMIN_API_KEY", "sales-page-test-admin-secret");
    fetchMock.mockResolvedValue(new Response(null, { status: 403 }));
    expect(await render()).toContain("Admin access was rejected");
    fetchMock.mockResolvedValue(new Response("private backend details", { status: 500 }));
    const failed = await render();
    expect(failed).toContain("Backend returned 500");
    expect(failed).not.toContain("private backend details");
    expect(failed).not.toContain("No stored Sales activity");
    fetchMock.mockRejectedValue(new Error("private transport details"));
    expect(await render()).not.toContain("private transport details");
  });

  it("keeps the selected cohort in all-time and recent-period links", async () => {
    const body = metrics();
    body.cohort_metadata = { selected: "mixed", current_release: "release-new", available_releases: ["release-new", "release-old"], measurement_started_at: "2026-10-23T12:00:00Z", legacy_note: "Historical releases are unknown." };
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await render({ cohort: "mixed" });
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/sales/metrics?period=all&cohort=mixed");
    expect(html).toContain('href="/admin/sales?period=range&amp;days=7&amp;cohort=mixed"');
    expect(html).toContain('name="period" value="all"');
    expect(html).toContain('value="mixed" selected=""');
    expect(html).toContain("Mixed releases");
    expect(html).not.toContain("sales-page-test-admin-secret");
  });

  it("rejects an older backend silently ignoring the release cohort", async () => {
    const html = await render({ cohort: "current" });
    expect(html).toContain("backend must be updated before release-cohort metrics");
    expect(html).not.toContain("Sessions (all time)");
  });
});
