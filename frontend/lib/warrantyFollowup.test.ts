import { afterEach, describe, expect, it, vi } from "vitest";
import { readWarrantyFollowup, saveWarrantyFollowup } from "./warrantyFollowup";

afterEach(() => vi.unstubAllGlobals());

describe("follow-up browser restoration", () => {
  it("restores only the signed capability belonging to the current session", () => {
    const values = new Map<string, string>();
    vi.stubGlobal("window", {});
    vi.stubGlobal("sessionStorage", {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    });
    const access = { sessionId: "session-a", ticketId: "case-a", token: "signed", issueType: "defect" };
    saveWarrantyFollowup(access);
    expect(readWarrantyFollowup("session-a")).toEqual(access);
    expect(readWarrantyFollowup("session-b")).toBeNull();
    values.set("warranty_followup:session-b", JSON.stringify(access));
    expect(readWarrantyFollowup("session-b")).toBeNull();
  });

  it("ignores corrupted data and unavailable browser storage", () => {
    vi.stubGlobal("window", {});
    vi.stubGlobal("sessionStorage", { getItem: () => "{bad json", setItem: () => { throw new Error("disabled"); } });
    expect(readWarrantyFollowup("session-a")).toBeNull();
    expect(() => saveWarrantyFollowup({ sessionId: "session-a", ticketId: "case-a", token: "signed", issueType: "defect" })).not.toThrow();
  });
});
