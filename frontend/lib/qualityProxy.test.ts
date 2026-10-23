import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ADMIN_COOKIE_NAME, createAdminSessionToken } from "./adminSession";
import { proxyKnowledgeWrite } from "./qualityProxy";

beforeEach(() => {
  vi.stubEnv("ADMIN_USERNAME", "reviewer");
  vi.stubEnv("ADMIN_SESSION_SECRET", "test-secret-only-0123456789abcdef0123456789");
  vi.stubEnv("ADMIN_API_KEY", "test-backend-key");
  vi.stubEnv("BACKEND_INTERNAL_URL", "http://backend.test");
});
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

function request(body: unknown, origin = "https://admin.test") {
  return new Request("https://admin.test/api/admin/quality/knowledge/1/decision", {
    method: "POST", body: JSON.stringify(body),
    headers: { cookie: `${ADMIN_COOKIE_NAME}=${createAdminSessionToken()}`, origin },
  });
}

describe("knowledge review proxy", () => {
  it("requires a signed admin session before touching the backend", async () => {
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    const response = await proxyKnowledgeWrite(new Request("https://admin.test/api/admin/quality/knowledge", { method: "POST", body: "{}" }));
    expect(response.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("rejects cross-origin writes", async () => {
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    expect((await proxyKnowledgeWrite(request({}, "https://other.test"), "1")).status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("uses a fixed backend path and trusted reviewer while keeping the key private", async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json({ id: 1, status: "approved" }));
    vi.stubGlobal("fetch", fetchMock);
    const response = await proxyKnowledgeWrite(request({ decision: "approved", reviewed_by: "spoofed" }), "1");
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/warranty/knowledge-reviews/1/decision");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).reviewed_by).toBe("reviewer");
    expect(await response.text()).not.toContain("test-backend-key");
    expect((await proxyKnowledgeWrite(request({}), "../secrets")).status).toBe(400);
  });
});
