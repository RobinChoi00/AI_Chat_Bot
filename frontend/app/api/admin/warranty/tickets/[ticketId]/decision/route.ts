/**
 * POST /api/admin/warranty/tickets/[ticketId]/decision
 *
 * Server-side proxy to FastAPI POST /admin/warranty/{ticket_id}/decision.
 * ADMIN_API_KEY is read from process.env here — never exposed to the browser.
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

function requireAdminKey(): string {
  const key = process.env.ADMIN_API_KEY;
  if (!key) throw new Error("ADMIN_API_KEY is not configured on the server.");
  return key;
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ ticketId: string }> }
) {
  try {
    const { ticketId } = await params;
    const adminKey = requireAdminKey();
    const body: unknown = await req.json();

    // NOTE: backend path is /admin/warranty/{id}/decision (not under /tickets/)
    const url = `${BACKEND}/admin/warranty/${encodeURIComponent(ticketId)}/decision`;
    const upstream = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": adminKey,
      },
      body: JSON.stringify(body),
    });

    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
