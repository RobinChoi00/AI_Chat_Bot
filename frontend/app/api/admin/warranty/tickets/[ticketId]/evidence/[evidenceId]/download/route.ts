/**
 * GET /api/admin/warranty/tickets/[ticketId]/evidence/[evidenceId]/download
 *
 * Server-side proxy to FastAPI GET /admin/warranty/{ticket_id}/evidence/{evidence_id}/download.
 *
 * Security:
 * - ADMIN_API_KEY is read from process.env — never sent to the browser.
 * - file_path is handled entirely by the FastAPI backend — never exposed here.
 * - Path traversal protection is enforced by the FastAPI layer.
 * - Content-Disposition is forwarded so the browser can offer a filename.
 */
import { NextRequest, NextResponse } from "next/server";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();

function requireAdminKey(): string {
  const key = process.env.ADMIN_API_KEY;
  if (!key) throw new Error("ADMIN_API_KEY is not configured on the server.");
  return key;
}

export async function GET(
  _req: NextRequest,
  {
    params,
  }: { params: Promise<{ ticketId: string; evidenceId: string }> }
) {
  try {
    const { ticketId, evidenceId } = await params;
    const adminKey = requireAdminKey();

    const url = `${BACKEND}/admin/warranty/${encodeURIComponent(ticketId)}/evidence/${encodeURIComponent(evidenceId)}/download`;

    const upstream = await fetch(url, {
      headers: { "X-Admin-Key": adminKey },
      cache: "no-store",
    });

    // Non-2xx: surface the FastAPI error body as JSON
    if (!upstream.ok) {
      const body: unknown = await upstream.json().catch(() => ({
        detail: `HTTP ${upstream.status}`,
      }));
      return NextResponse.json(body, { status: upstream.status });
    }

    // Stream the file bytes back to the browser
    const contentType =
      upstream.headers.get("content-type") ?? "application/octet-stream";
    const contentDisposition =
      upstream.headers.get("content-disposition") ?? "";

    const fileBytes = await upstream.arrayBuffer();

    const headers: Record<string, string> = {
      "Content-Type": contentType,
      // Prevent browsers from sniffing MIME type
      "X-Content-Type-Options": "nosniff",
    };
    if (contentDisposition) {
      headers["Content-Disposition"] = contentDisposition;
    }

    return new NextResponse(fileBytes, { status: 200, headers });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ detail }, { status: 500 });
  }
}
