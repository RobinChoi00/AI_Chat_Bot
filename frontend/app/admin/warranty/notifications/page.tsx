import Link from "next/link";
import { getBackendUrl } from "@/lib/backendUrl";

export const dynamic = "force-dynamic";

const path = "/admin/warranty/notifications";
const pageSize = 50;
const filters = ["dead_letter", "failed", "pending", "processing", "completed", "all"] as const;
type Filter = (typeof filters)[number];
type Status = Exclude<Filter, "all">;

const labels: Record<Filter, string> = {
  dead_letter: "Needs manual review",
  failed: "Failed; retry scheduled",
  pending: "Queued",
  processing: "Sending",
  completed: "Accepted by a provider",
  all: "All jobs",
};

const failureLabels: Record<string, string> = {
  SalesEmailError: "Older record: SMTP failure details not captured",
  smtp_authentication_failed: "SMTP authentication rejected",
  smtp_connection_failed: "SMTP connection unavailable",
  smtp_not_configured: "SMTP settings missing",
  smtp_invalid_configuration: "Invalid SMTP configuration",
  smtp_invalid_recipient: "Invalid destination address",
  smtp_recipient_rejected: "Destination rejected by mail server",
  sales_inbox_unconfigured: "Sales destination not configured",
  smtp_tls_unavailable: "SMTP TLS unavailable",
  smtp_submission_failed: "Mail submission failed",
  smtp_transport_error: "Unclassified mail transport error",
  send_not_confirmed: "Mail acceptance not confirmed",
  attempts_exhausted: "Retry limit reached",
  freshdesk_handoff_not_configured: "Freshdesk handoff settings are missing or invalid",
  freshdesk_authentication_failed: "Freshdesk access was rejected",
  freshdesk_request_rejected: "Freshdesk did not accept the handoff",
  freshdesk_rate_limited: "Freshdesk rate limit reached; check the case before acting",
  freshdesk_submission_uncertain: "Freshdesk acceptance is unknown; reconcile the case before any repeat action",
  handoff_source_no_longer_pending: "The linked request is no longer pending or was superseded",
};

interface NotificationReport {
  counts: Partial<Record<Status, number>>;
  worker: { enabled: boolean; alive: boolean };
  smtp?: { configured: boolean };
  handoff?: { new_job_channel: "smtp" | "freshdesk"; settings_present: boolean };
  items: {
    id: number; ticket_id: string; status: Status; attempts: number; failure_code: string | null;
    next_attempt_at: string | null; delivery_channel?: "smtp" | "freshdesk";
    submitted_via?: "smtp" | "freshdesk" | null; dispatch_state?: string | null;
    helpdesk_ticket_id?: number | null;
  }[];
  has_more: boolean;
}

function pageLink(status: Filter, offset = 0) {
  const query = new URLSearchParams({ status });
  if (offset > 0) query.set("offset", String(offset));
  return `${path}?${query}`;
}

export default async function WarrantyNotificationsPage({ searchParams }: {
  searchParams: Promise<{ status?: string; offset?: string }>;
}) {
  const query = await searchParams;
  const status = filters.find((value) => value === query.status) ?? "dead_letter";
  const parsedOffset = /^\d{1,7}$/.test(query.offset ?? "") ? Number(query.offset) : 0;
  const offset = Math.floor(parsedOffset / pageSize) * pageSize;
  const key = process.env.ADMIN_API_KEY;
  let report: NotificationReport | undefined;
  if (key) {
    const params = new URLSearchParams({ kind: "handoff", limit: String(pageSize), offset: String(offset) });
    if (status !== "all") params.set("status", status);
    try {
      const response = await fetch(`${getBackendUrl()}/admin/warranty/notification-jobs?${params}`, {
        headers: { "X-Admin-Key": key }, cache: "no-store", signal: AbortSignal.timeout(10_000),
      });
      if (response.ok) report = await response.json() as NotificationReport;
    } catch {
      // The error banner is intentionally independent of provider diagnostics.
    }
  }

  return (
    <main className="mx-auto max-w-6xl space-y-4 px-6 py-6">
      <Link href="/admin/warranty" className="text-sm text-indigo-700">← Warranty tickets</Link>
      <h1 className="text-xl font-semibold text-gray-900">Handoff notification queue</h1>
      <p className="text-sm text-gray-600">
        Provider acceptance does not confirm a staff response or callback booking.
        This view is read-only; opening it never retries an existing notification.
      </p>
      {!report ? <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-red-900">Notification queue could not be loaded.</p> : <>
        {(!report.worker.enabled || !report.worker.alive) && <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-red-900">
          Notification worker is unavailable. Check queued handoffs manually.
        </p>}
        {!!report.counts.dead_letter && <p role="alert" className="rounded-lg bg-amber-50 px-4 py-3 text-amber-900">
          {report.counts.dead_letter} jobs exhausted delivery attempts. Check the linked case before taking action; do not resend in bulk.
        </p>}
        {report.handoff?.new_job_channel === "freshdesk" && <p className="text-sm text-gray-600">New handoffs use the internal Freshdesk support queue. Older jobs are unchanged.</p>}
        {report.handoff?.new_job_channel === "freshdesk" && !report.handoff.settings_present && <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-red-900">Freshdesk handoff settings are not configured.</p>}
        {report.smtp?.configured === false && report.handoff?.new_job_channel !== "freshdesk" && <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-red-900">SMTP is not configured.</p>}
        <nav aria-label="Handoff job status" className="flex flex-wrap gap-2 text-sm">
          {filters.map((filter) => <Link key={filter} href={pageLink(filter)} aria-current={filter === status ? "page" : undefined}
            className={`rounded-md border px-3 py-1.5 ${filter === status ? "border-indigo-600 bg-indigo-50" : "border-gray-300 bg-white"}`}>
            {labels[filter]}{filter === "all" ? "" : ` (${report.counts[filter] ?? 0})`}
          </Link>)}
        </nav>
        {report.items.length ? <div className="overflow-x-auto rounded-lg border bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-gray-50"><tr>{["Case", "State", "Channel", "Attempts", "Last failure", "Next scheduled retry (UTC)"].map((column) =>
              <th key={column} scope="col" className="px-3 py-2 font-medium">{column}</th>)}</tr></thead>
            <tbody>{report.items.map((job) => <tr key={job.id} className="border-t border-gray-100">
              <td className="px-3 py-2"><Link href={`/admin/warranty/${encodeURIComponent(job.ticket_id)}`} className="text-indigo-700 underline">Open case · job {job.id}</Link></td>
              <td className="px-3 py-2">{job.submitted_via === "freshdesk" ? "Accepted by helpdesk" : job.submitted_via === "smtp" ? "Accepted by mail server" : labels[job.status] ?? "Unknown"}</td>
              <td className="px-3 py-2">{job.delivery_channel === "freshdesk" ? (job.helpdesk_ticket_id ? `Freshdesk #${job.helpdesk_ticket_id}` : "Freshdesk") : "SMTP"}</td>
              <td className="px-3 py-2">{job.attempts}</td>
              <td className="px-3 py-2">{job.failure_code ? failureLabels[job.failure_code] ?? "Other delivery failure" : "—"}</td>
              <td className="px-3 py-2">{job.next_attempt_at?.replace("T", " ") ?? "—"}</td>
            </tr>)}</tbody>
          </table>
        </div> : <p className="text-sm text-gray-600">No jobs match this view.</p>}
        <nav className="flex gap-4 text-sm text-indigo-700" aria-label="Handoff queue pages">
          {offset > 0 && <Link href={pageLink(status, offset - pageSize)}>← Previous</Link>}
          {report.has_more && <Link href={pageLink(status, offset + pageSize)}>Next →</Link>}
        </nav>
      </>}
    </main>
  );
}
