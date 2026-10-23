import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a>,
}));
vi.mock("next/navigation", () => ({
  redirect: (href: string) => { throw new Error(`REDIRECT:${href}`); },
}));
vi.mock("@/components/admin/AdminTicketQueue", () => ({ default: () => null }));
vi.mock("@/components/admin/AdminQueueFilters", () => ({ default: () => null }));
vi.mock("@/components/admin/AdminFreshdeskSync", () => ({ default: () => null }));

import AdminWarrantyQueuePage from "./page";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("admin queue historical pages", () => {
  it("requests the second page and renders working filtered navigation", async () => {
    vi.stubEnv("ADMIN_API_KEY", "page-test-key");
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ total: 205, offset: 100, tickets: [] })));
    vi.stubGlobal("fetch", fetchMock);
    const markup = renderToStaticMarkup(await AdminWarrantyQueuePage({
      searchParams: Promise.resolve({ page: "2", status: "resolved", channel: "web" }),
    }));
    const url = new URL(fetchMock.mock.calls[0][0]);
    expect(url.searchParams.get("offset")).toBe("100");
    expect(url.searchParams.get("limit")).toBe("100");
    expect(url.searchParams.get("status")).toBe("resolved");
    expect(url.searchParams.get("channel")).toBe("web");
    expect(markup).toContain("Showing 101–200 of 205 stored tickets");
    expect(markup).toContain('href="/admin/warranty?status=resolved&amp;channel=web"');
    expect(markup).toContain('href="/admin/warranty?status=resolved&amp;channel=web&amp;page=3"');
  });

  it("redirects a stale out-of-range page to the last matching page", async () => {
    vi.stubEnv("ADMIN_API_KEY", "page-test-key");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ total: 105, offset: 500, tickets: [] }))));
    await expect(AdminWarrantyQueuePage({
      searchParams: Promise.resolve({ page: "6", channel: "phone" }),
    })).rejects.toThrow("REDIRECT:/admin/warranty?channel=phone&page=2");
  });

  it("shows an access error instead of reporting zero stored tickets", async () => {
    vi.stubEnv("ADMIN_API_KEY", "page-test-key");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 401 })));
    const markup = renderToStaticMarkup(await AdminWarrantyQueuePage({ searchParams: Promise.resolve({}) }));
    expect(markup).toContain("Backend rejected admin access");
    expect(markup).not.toContain("Showing 0–0");
    expect(markup).not.toContain('aria-label="Ticket pages"');
  });
});
