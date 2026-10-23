import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { lookupDeliveryTracking } from "@/lib/api";
import DeliveryTrackingPanel, { DeliveryTrackingResult } from "./DeliveryTrackingPanel";

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("delivery status lookup", () => {
  it("asks for both checkout identity fields and identifies the selected store", () => {
    const html = renderToStaticMarkup(<DeliveryTrackingPanel storeDomain="titanchair.com" summary={null} onLookup={vi.fn()} />);
    expect(html).toContain("titanchair.com");
    expect(html).toContain("Order number");
    expect(html).toContain("Email used at checkout");
    expect(html).toContain('type="email"');
    expect(html.match(/required=""/g)).toHaveLength(2);
    expect(html).toContain("Both must match");
    expect(html).toContain("damage or missing parts");
    expect(html).not.toContain("Latest delivery update");
  });

  it("shows the returned status, estimate and check time without extra order details", () => {
    const html = renderToStaticMarkup(<DeliveryTrackingResult summary={{
      available: true, message: "Update found", status: "In transit", carrier: "UPS", eta: "Tomorrow",
      fetched_at: "2026-10-23T13:00:00Z", tracking_url: "https://www.ups.com/track?tracknum=123",
      source: "private-provider-name", self_service_links: [{ label: "Store order status", url: "https://titanchair.com/pages/track" }],
    }} />);
    expect(html).toContain("In transit");
    expect(html).toContain("UPS");
    expect(html).toContain("Tomorrow");
    expect(html).toContain("Checked:");
    expect(html).toContain("Delivery estimates can change");
    expect(html).toContain('href="https://www.ups.com/track?tracknum=123"');
    expect(html).not.toContain("private-provider-name");
    expect(html).not.toContain("Order total");
  });

  it("never presents stale carrier details as current after an unavailable lookup", () => {
    const html = renderToStaticMarkup(<DeliveryTrackingResult summary={{
      available: false, message: "We could not verify a current update.",
      status: "STALE_STATUS", carrier: "STALE_CARRIER", eta: "STALE_DATE", tracking_url: "https://carrier.test/stale",
      fetched_at: "2026-10-23T13:00:00Z", self_service_links: [{ label: "Unsafe link", url: "javascript:alert(1)" }],
      support_contact: { department: "sales", phone: "+1-888-848-2630 ext. 2", phone_href: "tel:+18888482630;ext=2", hours: "Sales hours" },
    }} />);
    expect(html).toContain("Current delivery update unavailable");
    expect(html).toContain("Last attempt:");
    expect(html).not.toContain("STALE_");
    expect(html).not.toContain("carrier.test/stale");
    expect(html).not.toContain("javascript:");
    expect(html).toContain("contact our Sales team");
    expect(html).toContain('href="tel:+18888482630;ext=2"');
    expect(html).not.toContain("tel:+18888482630,3");
  });

  it("omits tracking URLs containing credentials or unsupported schemes", () => {
    for (const tracking_url of ["javascript:alert(1)", "https://secret:password@carrier.test/track", "http://carrier.test/track"]) {
      const html = renderToStaticMarkup(<DeliveryTrackingResult summary={{ available: true, message: "", tracking_url }} />);
      expect(html).not.toContain("Open carrier tracking");
      expect(html).not.toContain("secret:password");
    }
  });

  it("performs a new private POST for every check and refresh", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://backend.test");
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(Response.json({ session_id: "session/1", ticket: null, tracking_summary: { available: true, status: "In transit", message: "" } })));
    vi.stubGlobal("fetch", fetch);
    const identity = { order_number: "12345", email: "checkout@example.test", domain: "titanchair.com" };
    await lookupDeliveryTracking("session/1", identity);
    await lookupDeliveryTracking("session/1", identity);
    expect(fetch).toHaveBeenCalledTimes(2);
    for (const [url, options] of fetch.mock.calls) {
      expect(url).toBe("http://backend.test/api/v1/warranty/session/session%2F1/delivery-tracking");
      expect(url).not.toContain("checkout@");
      expect(options).toMatchObject({ method: "POST", cache: "no-store" });
      expect(JSON.parse(options.body)).toEqual(identity);
      expect(options.headers).not.toHaveProperty("X-Admin-Key");
    }
  });

  it("rejects a failed check rather than reporting it as a verified update", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://backend.test");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ detail: "Checkout identity could not be verified." }, { status: 400 })));
    await expect(lookupDeliveryTracking("session", { order_number: "wrong", email: "checkout@example.test", domain: "osakiusa.com" })).rejects.toThrow("could not be verified");
  });
});
