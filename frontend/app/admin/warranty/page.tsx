/**
 * Admin warranty queue page — Server Component.
 *
 * Data is fetched server-side with ADMIN_API_KEY from process.env.
 * ADMIN_API_KEY is NEVER sent to the browser.
 * Filters and pagination live in the URL.
 */
import { redirect } from "next/navigation";
import Link from "next/link";
import AdminTicketQueue from "@/components/admin/AdminTicketQueue";
import AdminQueueFilters from "@/components/admin/AdminQueueFilters";
import AdminFreshdeskSync from "@/components/admin/AdminFreshdeskSync";
import AdminEmailPromptToggle from "@/components/admin/AdminEmailPromptToggle";
import type { TicketListResponse } from "@/lib/adminTypes";
import { getBackendUrl } from "@/lib/backendUrl";
import {
  parseTicketPage,
  ticketPagination,
  ticketQueueHref,
  TICKET_PAGE_SIZE,
} from "@/lib/adminTicketPagination";

const BACKEND = getBackendUrl();

type HandoffAttention = {
  total: number;
  safety_reviews: number;
  items: Array<{ticket_id: string; case_reference: string; safety_review: boolean; department: "warranty" | "sales";
    notification_state: "failed" | "dead_letter" | "not_recorded"}>;
};

async function fetchHandoffAttention(): Promise<HandoffAttention | null> {
  const adminKey = process.env.ADMIN_API_KEY;
  if (!adminKey) return null;
  try {
    const res = await fetch(`${BACKEND}/admin/warranty/handoff-attention?limit=8`, {
      headers: { "X-Admin-Key": adminKey }, cache: "no-store", signal: AbortSignal.timeout(6000),
    });
    if (!res.ok) return null;
    const data = await res.json() as HandoffAttention;
    return typeof data.total === "number" && Array.isArray(data.items) ? data : null;
  } catch {
    return null;
  }
}

async function fetchTickets(page: number, status?: string, channel?: string): Promise<TicketListResponse> {
  const adminKey = process.env.ADMIN_API_KEY;
  if (!adminKey) {
    throw new Error("ADMIN_API_KEY is not configured on the Next.js server.");
  }

  const qs = new URLSearchParams({
    limit: String(TICKET_PAGE_SIZE),
    offset: String((page - 1) * TICKET_PAGE_SIZE),
  });
  if (status) qs.set("status", status);
  if (channel) qs.set("channel", channel);

  const res = await fetch(`${BACKEND}/admin/warranty/tickets?${qs}`, {
    headers: { "X-Admin-Key": adminKey },
    cache: "no-store",
    signal: AbortSignal.timeout(15_000),
  });

  if (res.status === 401 || res.status === 403) {
    throw new Error("Backend rejected admin access. Check the server admin key.");
  }
  if (!res.ok) throw new Error(`Failed to load tickets (backend HTTP ${res.status}).`);

  return res.json() as Promise<TicketListResponse>;
}

export default async function AdminWarrantyQueuePage({
  searchParams,
}: {
  searchParams: Promise<{ status?: string; channel?: string; page?: string }>;
}) {
  const { status, channel, page: rawPage } = await searchParams;
  const page = parseTicketPage(rawPage);
  const adminConfigured = !!process.env.ADMIN_API_KEY;

  let data: TicketListResponse = { total: 0, offset: 0, tickets: [] };
  let fetchError: string | null = null;
  const tickets = fetchTickets(page, status, channel);
  const attentionRequest = fetchHandoffAttention();

  try {
    data = await tickets;
  } catch (err) {
    fetchError = err instanceof Error ? err.message : "Failed to load tickets from the backend.";
  }

  const pagination = ticketPagination(page, data.total);
  // Deleted tickets or changed filters can leave an old URL past the last page.
  if (!fetchError && page > pagination.pageCount) {
    redirect(ticketQueueHref(pagination.pageCount, status, channel));
  }
  const attention = await attentionRequest;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white px-6 py-3 shadow-sm">
        <div className="mx-auto flex max-w-7xl items-center gap-3">
          <Link href="/" className="text-sm text-gray-400 hover:text-gray-700">
            ← Home
          </Link>
          <span className="text-gray-300">|</span>
          <div className="flex-1">
            <h1 className="text-base font-semibold text-gray-900">
              🔐 Admin — Warranty Tickets
            </h1>
            <p className="text-xs font-medium text-red-600">
              INTERNAL USE ONLY — Do not share this page
            </p>
          </div>
          <Link
            href="/admin/warranty/dashboard"
            className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-100"
          >
            Usage &amp; resolution →
          </Link>
          <Link
            href="/admin/ops"
            className="rounded-md border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs font-medium text-indigo-800 hover:bg-indigo-100"
          >
            Cost & feedback →
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        {/* Config warning */}
        {!adminConfigured && (
          <div className="mb-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            ⚠️ <strong>ADMIN_API_KEY is not configured.</strong> Set it in{" "}
            <code>.env.local</code> on the Next.js server.
          </div>
        )}

        <AdminEmailPromptToggle />

        {attention && attention.total > 0 ? (
          <section aria-label="Support requests needing review" className="mb-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">
            <h2 className="font-semibold">{attention.total} open support requests without confirmed submission</h2>
            <p className="my-1">{attention.safety_reviews} marked for safety review. Reconcile with Freshdesk and staff before any individual retry. Nothing on this page sends notifications.</p>
            <ul className="space-y-1 pt-1">
              {attention.items.map((item) => (
                <li key={item.ticket_id}>
                  <Link className="font-medium underline" href={`/admin/warranty/${encodeURIComponent(item.ticket_id)}`}>{item.case_reference}</Link>
                  {item.safety_review ? " · Safety review" : ""} · {item.department === "sales" ? "Sales" : "Warranty"} · {item.notification_state === "not_recorded" ? "No matching job" : item.notification_state === "dead_letter" ? "Failed or unconfirmed" : "Retry pending"}
                </li>
              ))}
            </ul>
          </section>
        ) : adminConfigured && !attention && !fetchError ? (
          <p className="mb-5 text-sm text-gray-600">Support notification status could not be loaded.</p>
        ) : null}

        {/* Admin decision warning */}
        <div className="mb-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          ⚠️{" "}
          <strong>Admin decisions are permanent</strong> — approvals and
          rejections directly affect the customer&apos;s warranty ticket status.
          Review all evidence before making a final decision.
        </div>

        {/* Filters + refresh */}
        {!fetchError && (
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <AdminQueueFilters currentStatus={status} currentChannel={channel} total={data.total} />
            <AdminFreshdeskSync />
          </div>
        )}

        {/* Fetch error */}
        {fetchError && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            ⚠️ {fetchError}
          </div>
        )}

        {/* Ticket table */}
        {!fetchError && (
          <div className="mt-4">
            <AdminTicketQueue tickets={data.tickets} />
            <nav
              aria-label="Ticket pages"
              className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm"
            >
              <p className="text-gray-600">
                Showing {pagination.first}–{pagination.last} of {data.total} stored tickets
                {data.total > 0 && ` · Page ${page} of ${pagination.pageCount}`}
              </p>
              <div className="flex items-center gap-2">
                {pagination.hasPrevious ? (
                  <Link
                    href={ticketQueueHref(page - 1, status, channel)}
                    className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-gray-700 hover:bg-gray-50"
                  >
                    ← Previous
                  </Link>
                ) : (
                  <span aria-disabled="true" className="rounded-lg border border-gray-100 px-3 py-1.5 text-gray-400">
                    ← Previous
                  </span>
                )}
                {pagination.hasNext ? (
                  <Link
                    href={ticketQueueHref(page + 1, status, channel)}
                    className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-gray-700 hover:bg-gray-50"
                  >
                    Next →
                  </Link>
                ) : (
                  <span aria-disabled="true" className="rounded-lg border border-gray-100 px-3 py-1.5 text-gray-400">
                    Next →
                  </span>
                )}
              </div>
            </nav>
          </div>
        )}
      </main>
    </div>
  );
}
