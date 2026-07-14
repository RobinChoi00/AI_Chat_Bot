/**
 * GET /api/admin/warranty/tickets/[ticketId]
 *
 * Server-side proxy to FastAPI GET /admin/warranty/tickets/{ticket_id}.
 * Strips `file_path` from evidence records before sending to the browser.
 * ADMIN_API_KEY is read from process.env here — never exposed to the browser.
 */
import { NextRequest, NextResponse } from "next/server";
import { getBackendUrl } from "@/lib/backendUrl";
import { isAdminApiRequestAuthenticated } from "@/lib/adminSession";

const BACKEND = getBackendUrl();

function requireAdminKey(): string {
  const key = process.env.ADMIN_API_KEY;
  if (!key) throw new Error("ADMIN_API_KEY is not configured on the server.");
  return key;
}

/** Recursively remove file_path from any object/array so internal paths
 *  never reach the browser. */
function stripFilePath(data: unknown): unknown {
  if (Array.isArray(data)) return data.map(stripFilePath);
  if (data !== null && typeof data === "object") {
    const copy = { ...(data as Record<string, unknown>) };
    delete copy.file_path;
    for (const key of Object.keys(copy)) {
      copy[key] = stripFilePath(copy[key]);
    }
    return copy;
  }
  return data;
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ ticketId: string }> }
) {
  if (!isAdminApiRequestAuthenticated(req)) {
    return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  }
  try {
    const { ticketId } = await params;
    const adminKey = requireAdminKey();

    const url = `${BACKEND}/admin/warranty/tickets/${encodeURIComponent(ticketId)}`;
    const upstream = await fetch(url, {
      headers: { "X-Admin-Key": adminKey },
      cache: "no-store",
    });

    const data: unknown = await upstream.json();
    return NextResponse.json(stripFilePath(data), { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
