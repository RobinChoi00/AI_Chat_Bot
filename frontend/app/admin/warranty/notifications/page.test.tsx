import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a>,
}));

import WarrantyNotificationsPage from "./page";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("read-only handoff notification queue", () => {
  it("uses protected GET, filters failures and renders old causes without exposing arbitrary provider text", async () => {
    vi.stubEnv("ADMIN_API_KEY", "test-secret-key");
    const fetchMock = vi.fn().mockResolvedValue(Response.json({
      counts: { dead_letter: 91 }, worker: { enabled: true, alive: true }, has_more: true,
      smtp: { configured: true },
      handoff: { new_job_channel: "freshdesk", settings_present: true },
      items: [
        { id: 41, ticket_id: "synthetic", status: "dead_letter", attempts: 8, failure_code: "SalesEmailError" },
        { id: 40, ticket_id: "synthetic2", status: "dead_letter", attempts: 8, failure_code: "bad-value@example.invalid" },
      ],
    }));
    vi.stubGlobal("fetch", fetchMock);
    const html = renderToStaticMarkup(await WarrantyNotificationsPage({ searchParams: Promise.resolve({ offset: "50" }) }));
    const url = new URL(fetchMock.mock.calls[0][0]);
    expect(url.searchParams.get("kind")).toBe("handoff");
    expect(url.searchParams.get("status")).toBe("dead_letter");
    expect(url.searchParams.get("offset")).toBe("50");
    expect(fetchMock.mock.calls[0][1].headers["X-Admin-Key"]).toBe("test-secret-key");
    expect(fetchMock.mock.calls[0][1].method).toBeUndefined();
    expect(html).toContain("91 jobs exhausted");
    expect(html).toContain("Older record: SMTP failure details not captured");
    expect(html).toContain("Other delivery failure");
    expect(html).toContain("New handoffs use the internal Freshdesk support queue");
    expect(html).toContain("status=dead_letter&amp;offset=100");
    expect(html).not.toContain("test-secret-key");
    expect(html).not.toContain("bad-value@example.invalid");
  });

  it("distinguishes helpdesk acceptance from SMTP and failed SMTP from unavailable provider details", async () => {
    vi.stubEnv("ADMIN_API_KEY", "test-secret-key");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({
      counts: { completed: 2 }, worker: { enabled: true, alive: true }, has_more: false,
      items: [
        { id: 2, ticket_id: "synthetic", status: "completed", attempts: 1, delivery_channel: "freshdesk", submitted_via: "freshdesk", helpdesk_ticket_id: 808 },
        { id: 1, ticket_id: "synthetic2", status: "completed", attempts: 1, delivery_channel: "smtp", submitted_via: "smtp" },
      ],
    })));
    const html = renderToStaticMarkup(await WarrantyNotificationsPage({ searchParams: Promise.resolve({ status: "completed" }) }));
    expect(html).toContain("Accepted by helpdesk");
    expect(html).toContain("Freshdesk #808");
    expect(html).toContain("Accepted by mail server");
    expect(html).toContain("does not confirm a staff response or callback booking");
  });
});
