/**
 * GET /api/admin/warranty/freshdesk-status
 *
 * Server-side proxy to FastAPI GET /admin/warranty/freshdesk-status.
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
    const adminKey = requireAdminKey();
    const url = new URL(request.url);
    const probe = url.searchParams.get("probe") ?? "true";
    const upstream = await fetch(
      `${BACKEND}/admin/warranty/freshdesk-status?probe=${encodeURIComponent(probe)}`,
      {
        headers: { "X-Admin-Key": adminKey },
        cache: "no-store",
        signal: AbortSignal.timeout(30_000),
      }
    );

    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
