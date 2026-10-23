import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import OutcomeFollowup from "./OutcomeFollowup";
import { submitWarrantyFollowup } from "@/lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("customer outcome follow-up", () => {
  it("offers an optional later update without exposing its access token", () => {
    const html = renderToStaticMarkup(<OutcomeFollowup ticketId="case-1" token="private-followup-capability" />);
    expect(html).toContain("Update your result");
    expect(html).toContain("It is working now");
    expect(html).toContain("I still have a problem");
    expect(html).not.toContain("private-followup-capability");
    expect(html).toContain("<details");
    expect(html).not.toContain(" open=");
  });

  it("sends a signed, deduplicated follow-up without customer contact details", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://backend.test");
    const fetch = vi.fn().mockResolvedValue(Response.json({ ok: true, outcome: "still_broken", observation_id: 7 }));
    vi.stubGlobal("fetch", fetch);
    await expect(submitWarrantyFollowup("case/1", "signed-token", "still_broken", "event-1234")).resolves.toEqual({
      ok: true, outcome: "still_broken", observation_id: 7,
    });
    expect(fetch.mock.calls[0][0]).toBe("http://backend.test/api/v1/warranty/case%2F1/follow-up");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ token: "signed-token", outcome: "still_broken", event_id: "event-1234" });
  });

  it("does not treat an expired or rejected update as saved", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ detail: "Invalid or expired follow-up access." }, { status: 401 })));
    await expect(submitWarrantyFollowup("case-1", "expired", "fixed", "event-1234")).rejects.toThrow("expired");
  });
});
