/**
 * Next.js Proxy — protects all /admin/* pages.  (replaces middleware.ts in Next.js 16)
 *
 * Uses Web Crypto API (available in the Edge Runtime) instead of Node.js
 * `crypto` module so no runtime warning is emitted.
 *
 * How it works:
 * 1. On every request to /admin/* and /api/admin/*, read the admin_session cookie.
 * 2. Verify the HMAC-SHA256 signature and embedded expiry.
 * 3. Compare signatures with a constant-time string comparison.
 * 4. If valid → allow through. Otherwise → redirect to /admin/login.
 *
 * Excluded from protection:
 * - /admin/login            (the login page itself)
 * - /api/admin/auth/*       (login / logout API routes)
 */
import { NextRequest, NextResponse } from "next/server";

export const config = {
  matcher: ["/admin/:path*", "/api/admin/:path*"],
};

const COOKIE_NAME = "admin_session";

async function computeSignature(payload: string, secret: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(payload));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function decodePayload(encoded: string): { sub?: unknown; exp?: unknown } | null {
  try {
    const base64 = encoded.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=");
    return JSON.parse(atob(padded)) as { sub?: unknown; exp?: unknown };
  } catch {
    return null;
  }
}

async function validSession(token: string, username: string, secret: string): Promise<boolean> {
  const [encoded, suppliedSignature, extra] = token.split(".");
  if (!encoded || !suppliedSignature || extra) return false;
  const expectedSignature = await computeSignature(encoded, secret);
  if (!safeCompare(suppliedSignature, expectedSignature)) return false;
  const payload = decodePayload(encoded);
  return Boolean(
    payload &&
      payload.sub === username &&
      typeof payload.exp === "number" &&
      Number.isFinite(payload.exp) &&
      payload.exp > Math.floor(Date.now() / 1000)
  );
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
    pathname === "/api/admin/auth/login" ||
    pathname === "/api/admin/auth/logout"
  ) {
    return NextResponse.next();
  }

  const username = process.env.ADMIN_USERNAME?.trim();
  const secret = process.env.ADMIN_SESSION_SECRET?.trim();

  if (!username || !secret || secret.length < 32) {
    if (pathname.startsWith("/api/admin/")) {
      return NextResponse.json(
        { detail: "Admin authentication is not configured." },
        { status: 503 }
      );
    }
    // Env vars not configured — redirect to login (will show "not configured" message)
    const loginUrl = req.nextUrl.clone();
    loginUrl.pathname = "/admin/login";
    loginUrl.search = "";
    return NextResponse.redirect(loginUrl);
  }

  const sessionCookie = req.cookies.get(COOKIE_NAME)?.value ?? "";

  if (await validSession(sessionCookie, username, secret)) {
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/admin/")) {
    return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  }

  const loginUrl = req.nextUrl.clone();
  loginUrl.pathname = "/admin/login";
  loginUrl.search = "";
  return NextResponse.redirect(loginUrl);
}
