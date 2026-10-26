import { afterEach, describe, expect, it, vi } from "vitest";
import { getWarrantyConversationMessages } from "./api";

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("customer conversation history requests", () => {
  it("requests an earlier private page scoped to the current session without caching it", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://backend.example.test");
    const page = { session_id: "session/one", ticket_id: "case", messages: [{ id: 7, role: "user", content: "Earlier answer" }], has_more: false, before_id: 7 };
    const fetch = vi.fn().mockResolvedValue(Response.json(page));
    vi.stubGlobal("fetch", fetch);
    await expect(getWarrantyConversationMessages("session/one", { beforeId: 10 })).resolves.toEqual(page);
    expect(fetch).toHaveBeenCalledWith("https://backend.example.test/api/v1/warranty/session/session%2Fone/messages?before_id=10&limit=100", { cache: "no-store" });
  });

  it("keeps a failed history request retryable instead of treating it as an empty conversation", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ detail: "Unavailable" }, { status: 503 })));
    await expect(getWarrantyConversationMessages("session", { beforeId: 10 })).rejects.toThrow("Could not load earlier messages. Please try again.");
  });
});
