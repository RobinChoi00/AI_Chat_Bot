/**
 * GET/POST /api/admin/warranty/chat-email-prompt
 *
 * Server-side proxy for the customer email prompt switch.
 */
import { NextResponse } from "next/server";
import { getBackendUrl } from "@/lib/backendUrl";
import { isAdminApiRequestAuthenticated } from "@/lib/adminSession";

const BACKEND = getBackendUrl();

function requireAdminKey(): string {
  const key = process.env.ADMIN_API_KEY;
  if (!key) throw new Error("ADMIN_API_KEY is not configured on the server.");
  return key;
}

export async function GET(request: Request) {
  if (!isAdminApiRequestAuthenticated(request)) {
    return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  }
  try {
    const upstream = await fetch(`${BACKEND}/admin/warranty/chat-email-prompt`, {
      headers: { "X-Admin-Key": requireAdminKey() },
      cache: "no-store",
    });
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}

export async function POST(request: Request) {
  if (!isAdminApiRequestAuthenticated(request)) {
    return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  }
  try {
    const body = await request.json();
    const upstream = await fetch(`${BACKEND}/admin/warranty/chat-email-prompt`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": requireAdminKey(),
      },
      body: JSON.stringify({ enabled: Boolean(body?.enabled) }),
      cache: "no-store",
    });
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
