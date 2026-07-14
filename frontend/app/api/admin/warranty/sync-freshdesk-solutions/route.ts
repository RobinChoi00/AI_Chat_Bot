/**
 * POST /api/admin/warranty/sync-freshdesk-solutions
 *
 * Proxy: pulls Freshdesk KB (Solutions) articles into the warranty
 * knowledge base. Optionally schedules a FAISS rebuild.
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

export async function POST(request: Request) {
  if (!isAdminApiRequestAuthenticated(request)) {
    return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  }
  try {
    const adminKey = requireAdminKey();
    const url = new URL(request.url);
    const qs = new URLSearchParams();
    for (const key of ["max_articles", "rebuild_faiss"]) {
      const v = url.searchParams.get(key);
      if (v !== null) qs.set(key, v);
    }
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    const upstream = await fetch(
      `${BACKEND}/admin/warranty/sync-freshdesk-solutions${suffix}`,
      {
        method: "POST",
        headers: { "X-Admin-Key": adminKey },
        cache: "no-store",
        signal: AbortSignal.timeout(180_000),
      },
    );
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
