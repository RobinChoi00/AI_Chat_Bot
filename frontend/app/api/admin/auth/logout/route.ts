/**
 * POST /api/admin/auth/logout
 *
 * Clears the admin session cookie.
 */
import { NextResponse } from "next/server";
import { ADMIN_COOKIE_NAME } from "@/lib/adminSession";

export async function POST() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(ADMIN_COOKIE_NAME, "", {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/",
    maxAge: 0,
  });
  return response;
}
