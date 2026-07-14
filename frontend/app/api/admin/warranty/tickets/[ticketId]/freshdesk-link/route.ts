/**
 * POST /api/admin/warranty/tickets/[ticketId]/freshdesk-link
 *
 * Server-side proxy to FastAPI POST /admin/warranty/{ticket_id}/freshdesk-link.
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

export async function POST(
  request: Request,
  context: { params: Promise<{ ticketId: string }> }
) {
  if (!isAdminApiRequestAuthenticated(request)) {
    return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  }
  try {
    const adminKey = requireAdminKey();
    const { ticketId } = await context.params;
    const upstream = await fetch(
      `${BACKEND}/admin/warranty/${encodeURIComponent(ticketId)}/freshdesk-link`,
      {
        method: "POST",
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
