import { readFileSync } from "node:fs";
import path from "node:path";
import { runInNewContext } from "node:vm";
import { describe, expect, it, vi } from "vitest";

type Analytics = {
  init: (config: Record<string, unknown>) => boolean;
  adoptResponse: (response: Record<string, unknown>) => boolean;
  trackProductClick: (url: string) => Promise<boolean>;
  recordCart: (url?: string) => Promise<boolean>;
  recordCheckout: () => Promise<boolean>;
  clear: () => void;
};

const source = readFileSync(path.resolve("public/sales-analytics.js"), "utf8");

function browser(storage = new Map<string, string>()) {
  const fetchMock = vi.fn().mockResolvedValue({ ok: true });
  let uuid = 0;
  const listeners = new Map<string, (event: { target: { closest: () => { href: string } | null } }) => void>();
  const window = {
    SalesAnalytics: null as Analytics | null,
    location: { hostname: "osakiusa.com", origin: "https://osakiusa.com" },
    crypto: { randomUUID: () => `00000000-0000-4000-8000-${String(++uuid).padStart(12, "0")}` },
    sessionStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    },
    fetch: fetchMock,
  };
  runInNewContext(source, {
    window,
    document: { addEventListener: (kind: string, callback: (event: { target: { closest: () => { href: string } | null } }) => void) => listeners.set(kind, callback) },
    URL,
  });
  return { analytics: window.SalesAnalytics!, fetchMock, listeners, storage };
}

function enabled(analytics: Analytics, extra: Record<string, unknown> = {}) {
  analytics.init({ enabled: true, domain: "osakiusa.com", apiBase: "https://help.osakichair.com", ...extra });
  expect(analytics.adoptResponse({ session_id: "opaque-session", analytics_token: "signed-event-token" })).toBe(true);
}

describe("storefront Sales analytics adapter", () => {
  it("sends nothing on load, without explicit enablement, or without a session token", async () => {
    const { analytics, fetchMock } = browser();
    expect(await analytics.recordCheckout()).toBe(false);
    analytics.init({ enabled: false });
    expect(analytics.adoptResponse({ session_id: "opaque-session", analytics_token: "token" })).toBe(false);
    expect(await analytics.recordCheckout()).toBe(false);
    analytics.init({ enabled: true });
    expect(await analytics.recordCheckout()).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("tracks actual product clicks with opaque context and strips customer-bearing URL data", async () => {
    const { analytics, fetchMock } = browser();
    enabled(analytics);
    expect(await analytics.trackProductClick("https://osakiusa.com/products/chair?email=private@example.test#customer")).toBe(true);
    expect(fetchMock.mock.calls[1][0]).toBe("https://help.osakichair.com/api/v1/sales/events");
    const options = fetchMock.mock.calls[1][1];
    expect(options.keepalive).toBe(true);
    expect(options.credentials).toBe("omit");
    const payload = JSON.parse(options.body);
    expect(payload).toMatchObject({ event_type: "product_click", session_id: "opaque-session", product_url: "https://osakiusa.com/products/chair" });
    expect(options.body).not.toContain("private");
    expect(options.body).not.toContain("customer");
    expect(await analytics.trackProductClick("https://osakiusa.com/account")).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("serializes observed click, successful cart and checkout sends", async () => {
    const { analytics, fetchMock } = browser();
    enabled(analytics);
    let release: () => void = () => undefined;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => {
      release = () => resolve({ ok: true });
    }));
    const clicked = analytics.trackProductClick("https://osakiusa.com/products/chair");
    const cart = analytics.recordCart("https://osakiusa.com/products/chair");
    const checkout = analytics.recordCheckout();
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    release();
    await Promise.all([clicked, cart, checkout]);
    expect(fetchMock.mock.calls.map((call) => JSON.parse(call[1].body).event_type)).toEqual([
      "tracking_started", "product_click", "add_to_cart", "checkout_started",
    ]);
  });

  it("binds only explicitly marked recommendation clicks and never blocks navigation", async () => {
    const { analytics, fetchMock, listeners } = browser();
    enabled(analytics);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const listener = listeners.get("click")!;
    listener({ target: { closest: () => null } });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    listener({ target: { closest: () => ({ href: "https://osakiusa.com/products/chair" }) } });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("persists context only when requested and clears the current tab session", async () => {
    const first = browser();
    enabled(first.analytics);
    expect(first.storage.size).toBe(0);
    enabled(first.analytics, { persistSession: true });
    expect(first.storage.size).toBe(1);
    const nextPage = browser(first.storage);
    expect(nextPage.analytics.init({ enabled: true, persistSession: true, domain: "osakiusa.com" })).toBe(true);
    expect(await nextPage.analytics.recordCheckout()).toBe(true);
    nextPage.analytics.clear();
    expect(first.storage.size).toBe(0);
    expect(await nextPage.analytics.recordCheckout()).toBe(false);
  });

  it("absorbs tracking failures without breaking later storefront actions", async () => {
    const { analytics, fetchMock } = browser();
    enabled(analytics);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    fetchMock.mockRejectedValueOnce(new Error("offline"));
    expect(await analytics.recordCart()).toBe(false);
    expect(await analytics.recordCheckout()).toBe(true);
  });

  it("reports tracking coverage once per adopted session and reuses its UUID after navigation", async () => {
    const first = browser();
    enabled(first.analytics, { persistSession: true });
    first.analytics.adoptResponse({ session_id: "opaque-session", analytics_token: "updated-signed-token" });
    await vi.waitFor(() => expect(first.fetchMock).toHaveBeenCalledTimes(1));
    const readiness = JSON.parse(first.fetchMock.mock.calls[0][1].body);
    expect(readiness.event_type).toBe("tracking_started");
    expect(readiness.product_url).toBeNull();
    const nextPage = browser(first.storage);
    nextPage.analytics.init({ enabled: true, persistSession: true, domain: "osakiusa.com" });
    await vi.waitFor(() => expect(nextPage.fetchMock).toHaveBeenCalledTimes(1));
    expect(JSON.parse(nextPage.fetchMock.mock.calls[0][1].body).event_id).toBe(readiness.event_id);
  });
});
