/**
 * GET /api/admin/warranty/freshdesk-solutions-probe
 *
 * Proxy: cheap read-only probe of Freshdesk Solutions (KB) — reports
 * category/folder/article counts without downloading article bodies.
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
    const upstream = await fetch(
      `${BACKEND}/admin/warranty/freshdesk-solutions/probe`,
      {
        headers: { "X-Admin-Key": adminKey },
        cache: "no-store",
        signal: AbortSignal.timeout(60_000),
      },
    );
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
