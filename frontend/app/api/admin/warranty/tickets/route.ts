/**
 * GET /api/admin/warranty/tickets
 *
 * Server-side proxy to FastAPI GET /admin/warranty/tickets.
 * ADMIN_API_KEY is read from process.env here — never exposed to the browser.
 */
import { NextRequest, NextResponse } from "next/server";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();

function requireAdminKey(): string {
  const key = process.env.ADMIN_API_KEY;
  if (!key) throw new Error("ADMIN_API_KEY is not configured on the server.");
  return key;
}

export async function GET(req: NextRequest) {
  try {
    const adminKey = requireAdminKey();

    // Forward any query params (status, limit, offset, …)
    const { searchParams } = new URL(req.url);
    const qs = new URLSearchParams();
    for (const [k, v] of searchParams.entries()) qs.set(k, v);

    const url = `${BACKEND}/admin/warranty/tickets${qs.size ? `?${qs}` : ""}`;
    const upstream = await fetch(url, {
      headers: { "X-Admin-Key": adminKey },
      cache: "no-store",
    });

    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
