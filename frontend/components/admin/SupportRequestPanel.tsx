import Link from "next/link";
import type { SupportRequestMetrics } from "@/lib/adminTypes";
import { metricCount } from "@/lib/adminMetricDisplay";
import MetricCard from "./MetricCard";

export default function SupportRequestPanel({ handoffs }: { handoffs?: SupportRequestMetrics }) {
  const state = handoffs?.notification;
  return (
    <section className="space-y-3 rounded-xl border border-gray-200 bg-white p-4" aria-labelledby="support-notification-title">
      <h2 id="support-notification-title" className="font-semibold text-gray-900">Support requests and notification status</h2>
      <p className="text-xs text-gray-600">Saved requests do not confirm staff receipt or a callback. A Freshdesk link and email acceptance are separate from a recorded staff action.</p>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricCard label="Support requests saved" value={metricCount(handoffs?.requested)} hint="One per consultation with a recorded support request" />
        {state && <>
          <MetricCard label="Notification in queue" value={metricCount(state.queued)} hint={`${metricCount(state.retry_scheduled)} additional requests scheduled to retry after a failed attempt`} />
          <MetricCard label="Accepted by email service" value={metricCount(state.smtp_accepted)} hint="SMTP accepted the notification; inbox delivery is not confirmed" />
          <MetricCard label="Submitted to Freshdesk" value={metricCount(state.freshdesk_submitted)} hint="Freshdesk returned a ticket ID; no staff action is inferred" />
          <MetricCard label="Notification failed permanently" value={metricCount(state.dead_letter)} hint="Dead-lettered; no automatic retry remains" />
          <MetricCard label="Notification delivery uncertain" value={metricCount(state.delivery_uncertain)} hint="Confirm whether the provider received it before any retry to avoid duplicate tickets" />
          <MetricCard label="Notification status not recorded" value={metricCount(state.not_recorded)} hint="No matching job or verifiable status; notification outcome is unknown" />
          <MetricCard label="Freshdesk linked" value={metricCount(state.freshdesk_linked)} hint="Linked cases only, not proof of staff acknowledgement" />
          <MetricCard label="Staff acknowledgement recorded" value={metricCount(state.staff_acknowledged)} hint={`${metricCount(state.staff_acknowledgement_unknown)} requests without a recorded authenticated staff action; other staff activity unknown`} />
        </>}
      </div>
      {!state && <p className="text-xs text-gray-600">Notification and staff acknowledgement measurements are unavailable from this response.</p>}
      {state && (state.dead_letter > 0 || state.delivery_uncertain > 0) && <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
        <Link className="font-medium underline" href="/admin/warranty/notifications">Review warranty notifications</Link>. Check failed or uncertain notifications against Freshdesk or other confirmed contact before retrying individual requests.
      </p>}
      {handoffs && <div className="grid gap-3 sm:grid-cols-2"><div><h3 className="text-xs font-medium text-gray-500">Intended department</h3><ul className="mt-1 text-sm">{handoffs.by_department.map((row) => <li key={row.department}>{row.department || "Not recorded"}: {row.count}</li>)}</ul></div><div><h3 className="text-xs font-medium text-gray-500">Reason</h3><ul className="mt-1 text-sm">{handoffs.by_reason.map((row) => <li key={row.reason}>{row.reason.replace(/_/g, " ")}: {row.count}</li>)}</ul></div></div>}
    </section>
  );
}
