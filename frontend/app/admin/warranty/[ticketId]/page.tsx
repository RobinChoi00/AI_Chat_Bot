/**
 * Admin warranty ticket detail page — Server Component.
 *
 * Data is fetched server-side with ADMIN_API_KEY from process.env.
 * ADMIN_API_KEY is NEVER sent to the browser.
 * After client-side mutations (decision / note), components call
 * router.refresh() to re-run this Server Component with fresh data.
 */
import { notFound } from "next/navigation";
import Link from "next/link";
import AdminTicketDetail from "@/components/admin/AdminTicketDetail";
import AdminDecisionPanel from "@/components/admin/AdminDecisionPanel";
import AdminNoteForm from "@/components/admin/AdminNoteForm";
import type { TicketDetailResponse } from "@/lib/adminTypes";

const BACKEND = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

/** Remove file_path from evidence items — must never reach the browser. */
function stripFilePaths(data: unknown): unknown {
  if (Array.isArray(data)) return data.map(stripFilePaths);
  if (data !== null && typeof data === "object") {
    const copy = { ...(data as Record<string, unknown>) };
    delete copy.file_path;
    for (const key of Object.keys(copy)) copy[key] = stripFilePaths(copy[key]);
    return copy;
  }
  return data;
}

async function fetchTicket(ticketId: string): Promise<TicketDetailResponse> {
  const adminKey = process.env.ADMIN_API_KEY;
  if (!adminKey) {
    throw new Error("ADMIN_API_KEY is not configured on the server.");
  }

  const res = await fetch(
    `${BACKEND}/admin/warranty/tickets/${encodeURIComponent(ticketId)}`,
    { headers: { "X-Admin-Key": adminKey }, cache: "no-store" }
  );

  if (res.status === 404) notFound();
  if (!res.ok) throw new Error(`Backend returned HTTP ${res.status}`);

  const raw: unknown = await res.json();
  return stripFilePaths(raw) as TicketDetailResponse;
}

export default async function AdminTicketDetailPage({
  params,
}: {
  params: Promise<{ ticketId: string }>;
}) {
  const { ticketId } = await params;
  const adminConfigured = !!process.env.ADMIN_API_KEY;

  let data: TicketDetailResponse | null = null;
  let fetchError: string | null = null;

  if (adminConfigured) {
    try {
      data = await fetchTicket(ticketId);
    } catch (err) {
      fetchError = err instanceof Error ? err.message : "Failed to load ticket.";
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white px-6 py-3 shadow-sm">
        <div className="mx-auto flex max-w-5xl items-center gap-3">
          <Link
            href="/admin/warranty"
            className="text-sm text-gray-400 hover:text-gray-700"
          >
            ← Ticket Queue
          </Link>
          <span className="text-gray-300">|</span>
          <div>
            <h1 className="text-base font-semibold text-gray-900">
              🔐 Ticket Detail
            </h1>
            <p className="text-xs font-medium text-red-600">
              INTERNAL USE ONLY
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-6">
        {!adminConfigured && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            ⚠️ <strong>ADMIN_API_KEY is not configured.</strong> Set it in{" "}
            <code>.env.local</code> on the Next.js server.
          </div>
        )}

        {fetchError && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            ⚠️ {fetchError}
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {/* Ticket info + turns + evidence */}
            <AdminTicketDetail
              ticket={data.ticket}
              turns={data.turns}
              evidence={data.evidence}
            />

            {/* ── Admin decision panel ─────────────────────────── */}
            <section className="rounded-xl border border-indigo-200 bg-white p-5 shadow-sm">
              <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-indigo-600">
                Admin Decision
              </h2>
              <p className="mb-4 text-xs text-gray-500">
                Only approved / rejected may be set here. These decisions are
                permanent and affect the customer&apos;s warranty ticket status.
              </p>
              <AdminDecisionPanel
                ticketId={data.ticket.ticket_id}
                currentStatus={data.ticket.status}
              />
            </section>

            {/* ── Admin note panel ─────────────────────────────── */}
            <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-gray-500">
                Internal Notes
              </h2>
              <p className="mb-4 text-xs text-gray-500">
                Notes are internal only and not visible to the customer.
              </p>
              <AdminNoteForm
                ticketId={data.ticket.ticket_id}
                currentNote={data.ticket.admin_note}
              />
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
