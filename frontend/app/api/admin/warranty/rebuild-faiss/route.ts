/**
 * /api/admin/warranty/rebuild-faiss
 *
 * POST — schedule (or optionally block on) a freshdesk_qa FAISS rebuild.
 * GET  — report the current rebuild status.
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
    const wait = url.searchParams.get("wait");
    const suffix = wait ? `?wait=${wait}` : "";
    const upstream = await fetch(
      `${BACKEND}/admin/warranty/rebuild-faiss${suffix}`,
      {
        method: "POST",
        headers: { "X-Admin-Key": adminKey },
        cache: "no-store",
        signal: AbortSignal.timeout(300_000),
      },
    );
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}

export async function GET(request: Request) {
  if (!isAdminApiRequestAuthenticated(request)) {
    return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  }
  try {
    const adminKey = requireAdminKey();
    const upstream = await fetch(
      `${BACKEND}/admin/warranty/faiss/status`,
      {
        headers: { "X-Admin-Key": adminKey },
        cache: "no-store",
        signal: AbortSignal.timeout(15_000),
      },
    );
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
