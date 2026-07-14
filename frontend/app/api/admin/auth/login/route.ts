/**
 * POST /api/admin/auth/login
 *
 * Validates admin username + password from process.env and sets an
 * HTTP-only session cookie.
 *
 * Required env vars:
 *   ADMIN_USERNAME        — admin user ID
 *   ADMIN_PASSWORD        — admin password
 *   ADMIN_SESSION_SECRET  — secret used to sign the session token (min 32 chars)
 *
 * The cookie contains a signed username + expiry payload. The proxy verifies
 * both the HMAC and expiry, so copied cookies cannot be replayed indefinitely.
 */
import { timingSafeEqual } from "crypto";
import { NextRequest, NextResponse } from "next/server";
import {
  ADMIN_COOKIE_NAME,
  ADMIN_SESSION_MAX_AGE_SECONDS,
  createAdminSessionToken,
} from "@/lib/adminSession";

export const runtime = "nodejs";

const LOGIN_WINDOW_MS = 15 * 60 * 1000;
const LOGIN_MAX_FAILURES = 5;
const attempts = new Map<string, { count: number; resetAt: number }>();

function safeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(Buffer.from(a), Buffer.from(b));
}

function clientAddress(req: NextRequest): string {
  return (
    req.headers.get("cf-connecting-ip") ||
    req.headers.get("x-real-ip") ||
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    "unknown"
  );
}

function isRateLimited(key: string, now: number): boolean {
  const current = attempts.get(key);
  if (!current || current.resetAt <= now) {
    attempts.delete(key);
    return false;
  }
  return current.count >= LOGIN_MAX_FAILURES;
}

function recordFailure(key: string, now: number): void {
  const current = attempts.get(key);
  if (!current || current.resetAt <= now) {
    attempts.set(key, { count: 1, resetAt: now + LOGIN_WINDOW_MS });
  } else {
    current.count += 1;
  }
  if (attempts.size > 10_000) {
    for (const [candidate, value] of attempts) {
      if (value.resetAt <= now) attempts.delete(candidate);
    }
  }
}

export async function POST(req: NextRequest) {
  const adminUsername = process.env.ADMIN_USERNAME?.trim();
  const adminPassword = process.env.ADMIN_PASSWORD;
  const sessionSecret = process.env.ADMIN_SESSION_SECRET?.trim();

  if (!adminUsername || !adminPassword || !sessionSecret || sessionSecret.length < 32) {
    return NextResponse.json(
      { detail: "Admin credentials are not configured on the server." },
      { status: 503 }
    );
  }

  const address = clientAddress(req);
  const now = Date.now();
  if (isRateLimited(address, now)) {
    return NextResponse.json(
      { detail: "Too many login attempts. Try again later." },
      { status: 429, headers: { "Retry-After": "900" } }
    );
  }

  let body: { username?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: "Invalid request body." }, { status: 400 });
  }

  const { username = "", password = "" } = body;

  const usernameMatch = safeCompare(username, adminUsername);
  const passwordMatch = safeCompare(password, adminPassword);

  if (!usernameMatch || !passwordMatch) {
    recordFailure(address, now);
    return NextResponse.json(
      { detail: "Invalid username or password." },
      { status: 401 }
    );
  }

  attempts.delete(address);
  const token = createAdminSessionToken();

  const response = NextResponse.json({ ok: true });
  response.cookies.set(ADMIN_COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/",
    maxAge: ADMIN_SESSION_MAX_AGE_SECONDS,
  });
  return response;
}
