/**
 * Next.js Proxy — protects all /admin/* pages.  (replaces middleware.ts in Next.js 16)
 *
 * Uses Web Crypto API (available in the Edge Runtime) instead of Node.js
 * `crypto` module so no runtime warning is emitted.
 *
 * How it works:
 * 1. On every request to /admin/*, read the admin_session cookie.
 * 2. Re-compute the expected HMAC-SHA256 token from env vars.
 * 3. Compare cookie vs expected with a constant-time string comparison.
 * 4. If valid → allow through. Otherwise → redirect to /admin/login.
 *
 * Excluded from protection:
 * - /admin/login            (the login page itself)
 * - /api/admin/auth/*       (login / logout API routes)
 */
import { NextRequest, NextResponse } from "next/server";

export const config = {
  matcher: ["/admin/:path*"],
};

const COOKIE_NAME = "admin_session";

async function computeToken(
  username: string,
  password: string,
  secret: string
): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(`${username}:${password}`));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Constant-time string comparison (prevents timing attacks). */
function safeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

export async function proxy(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Pass through login page and auth API routes unconditionally
  if (
    pathname === "/admin/login" ||
    pathname.startsWith("/api/admin/auth/")
  ) {
    return NextResponse.next();
  }

  const username = process.env.ADMIN_USERNAME;
  const password = process.env.ADMIN_PASSWORD;
  const secret   = process.env.ADMIN_SESSION_SECRET;

  if (!username || !password || !secret) {
    // Env vars not configured — redirect to login (will show "not configured" message)
    const loginUrl = req.nextUrl.clone();
    loginUrl.pathname = "/admin/login";
    loginUrl.search = "";
    return NextResponse.redirect(loginUrl);
  }

  const sessionCookie = req.cookies.get(COOKIE_NAME)?.value ?? "";
  const expected = await computeToken(username, password, secret);

  if (safeCompare(sessionCookie, expected)) {
    return NextResponse.next();
  }

  const loginUrl = req.nextUrl.clone();
  loginUrl.pathname = "/admin/login";
  loginUrl.search = "";
  return NextResponse.redirect(loginUrl);
}
