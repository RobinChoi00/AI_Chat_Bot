import { afterAll, beforeEach, describe, expect, it } from "vitest";
import {
  ADMIN_COOKIE_NAME,
  createAdminSessionToken,
  isAdminApiRequestAuthenticated,
  verifyAdminSessionToken,
} from "./adminSession";

const originalUsername = process.env.ADMIN_USERNAME;
const originalSecret = process.env.ADMIN_SESSION_SECRET;

beforeEach(() => {
  process.env.ADMIN_USERNAME = "operator";
  process.env.ADMIN_SESSION_SECRET = "a".repeat(64);
});

afterAll(() => {
  if (originalUsername === undefined) delete process.env.ADMIN_USERNAME;
  else process.env.ADMIN_USERNAME = originalUsername;
  if (originalSecret === undefined) delete process.env.ADMIN_SESSION_SECRET;
  else process.env.ADMIN_SESSION_SECRET = originalSecret;
});

describe("admin session token", () => {
  it("accepts a correctly signed, unexpired token", () => {
    const now = Math.floor(Date.now() / 1000);
    const token = createAdminSessionToken(now);
    expect(verifyAdminSessionToken(token, now + 1)).toBe(true);

    const request = new Request("https://example.test/api/admin/warranty/tickets", {
      headers: { cookie: `${ADMIN_COOKIE_NAME}=${encodeURIComponent(token)}` },
    });
    expect(isAdminApiRequestAuthenticated(request)).toBe(true);
  });

  it("rejects expired and tampered tokens", () => {
    const token = createAdminSessionToken(1_000);
    expect(verifyAdminSessionToken(token, 1_000 + 8 * 60 * 60)).toBe(false);
    expect(verifyAdminSessionToken(`${token}tampered`, 1_001)).toBe(false);
  });

  it("fails closed when the signing secret is too short", () => {
    process.env.ADMIN_SESSION_SECRET = "short";
    expect(verifyAdminSessionToken("anything", 1_001)).toBe(false);
    expect(() => createAdminSessionToken(1_000)).toThrow();
  });
});
