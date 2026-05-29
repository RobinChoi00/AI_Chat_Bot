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
 * The cookie value is an HMAC-SHA256 of "username:password" using
 * ADMIN_SESSION_SECRET.  The middleware re-computes the same HMAC and
 * compares — no server-side session store is required.
 */
import { createHmac, timingSafeEqual } from "crypto";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const COOKIE_NAME = "admin_session";
const COOKIE_MAX_AGE = 60 * 60 * 8; // 8 hours

function sessionToken(username: string, password: string, secret: string): string {
  return createHmac("sha256", secret)
    .update(`${username}:${password}`)
    .digest("hex");
}

function safeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(Buffer.from(a), Buffer.from(b));
}

export async function POST(req: NextRequest) {
  const adminUsername = process.env.ADMIN_USERNAME;
  const adminPassword = process.env.ADMIN_PASSWORD;
  const sessionSecret = process.env.ADMIN_SESSION_SECRET;

  if (!adminUsername || !adminPassword || !sessionSecret) {
    return NextResponse.json(
      { detail: "Admin credentials are not configured on the server." },
      { status: 503 }
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
    return NextResponse.json(
      { detail: "Invalid username or password." },
      { status: 401 }
    );
  }

  const token = sessionToken(adminUsername, adminPassword, sessionSecret);

  const response = NextResponse.json({ ok: true });
  response.cookies.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: COOKIE_MAX_AGE,
  });
  return response;
}
