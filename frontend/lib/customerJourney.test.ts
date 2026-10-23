import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { reportCustomerJourney } from "./customerJourney";

beforeEach(() => vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://backend.test"));
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("consented browser journey reporting", () => {
  it("does not measure visitors before recording consent", () => {
    const fetch = vi.fn(); vi.stubGlobal("fetch", fetch);
    reportCustomerJourney({ consentAccepted: false, sessionId: "session", domain: "osakiusa.com", stage: "consent", event: "shown" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("sends only the event allowlist and a unique event ID", async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json({ recorded: true })); vi.stubGlobal("fetch", fetch);
    const context = { consentAccepted: true, sessionId: "session", domain: "osakiusa.com", stage: "resume" as const, event: "requested" as const, email: "must-not-be-sent@example.test", message: "must not be sent", resume_url: "must not be sent" };
    reportCustomerJourney(context);
    reportCustomerJourney(context);
    const payloads = fetch.mock.calls.map(([url, options]) => {
      expect(url).toBe("https://backend.test/api/v1/journey/events");
      expect(options.cache).toBe("no-store");
      const payload = JSON.parse(options.body);
      expect(Object.keys(payload).sort()).toEqual(["domain", "event", "event_id", "session_id", "stage"]);
      expect(payload.event_id).toMatch(/^[0-9a-f-]{36}$/i);
      expect(payload).toMatchObject({ session_id: "session", domain: "osakiusa.com", stage: "resume", event: "requested" });
      return payload;
    });
    expect(payloads[0].event_id).not.toBe(payloads[1].event_id);
  });

  it("keeps a failed measurement request out of the customer flow", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    expect(() => reportCustomerJourney({ consentAccepted: true, sessionId: "session", domain: "osakiusa.com", stage: "evidence", event: "failed" })).not.toThrow();
    await new Promise<void>((resolve) => setImmediate(resolve));
  });
});
