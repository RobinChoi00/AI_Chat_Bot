/**
 * Admin warranty queue page — Server Component.
 *
 * Data is fetched server-side with ADMIN_API_KEY from process.env.
 * ADMIN_API_KEY is NEVER sent to the browser.
 * Filter state lives in the URL (searchParams.status).
 */
import { notFound } from "next/navigation";
import Link from "next/link";
import AdminTicketQueue from "@/components/admin/AdminTicketQueue";
import AdminQueueFilters from "@/components/admin/AdminQueueFilters";
import AdminFreshdeskSync from "@/components/admin/AdminFreshdeskSync";
import type { TicketListResponse } from "@/lib/adminTypes";
import { getBackendUrl } from "@/lib/backendUrl";

const BACKEND = getBackendUrl();

async function fetchTickets(status?: string, channel?: string): Promise<TicketListResponse> {
  const adminKey = process.env.ADMIN_API_KEY;
  if (!adminKey) {
    return { total: 0, offset: 0, tickets: [] };
  }

  const qs = new URLSearchParams({ limit: "100" });
  if (status) qs.set("status", status);
  if (channel) qs.set("channel", channel);

  const res = await fetch(`${BACKEND}/admin/warranty/tickets?${qs}`, {
    headers: { "X-Admin-Key": adminKey },
    cache: "no-store",
    signal: AbortSignal.timeout(15_000),
  });

  if (res.status === 401 || res.status === 503) {
    return { total: 0, offset: 0, tickets: [] };
  }
  if (!res.ok) notFound();

  return res.json() as Promise<TicketListResponse>;
}

export default async function AdminWarrantyQueuePage({
  searchParams,
}: {
  searchParams: Promise<{ status?: string; channel?: string }>;
}) {
  const { status, channel } = await searchParams;
  const adminConfigured = !!process.env.ADMIN_API_KEY;

  let data: TicketListResponse = { total: 0, offset: 0, tickets: [] };
  let fetchError: string | null = null;

  try {
    data = await fetchTickets(status, channel);
  } catch {
    fetchError = "Failed to load tickets from the backend.";
  }

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
            Completion dashboard →
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

        {/* Admin decision warning */}
        <div className="mb-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          ⚠️{" "}
          <strong>Admin decisions are permanent</strong> — approvals and
          rejections directly affect the customer&apos;s warranty ticket status.
          Review all evidence before making a final decision.
        </div>

        {/* Filters + refresh */}
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <AdminQueueFilters currentStatus={status} currentChannel={channel} total={data.total} />
          <AdminFreshdeskSync />
        </div>

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
          </div>
        )}
      </main>
    </div>
  );
}
