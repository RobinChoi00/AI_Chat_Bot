import type { JourneyMetrics } from "@/lib/adminTypes";

function stageLabel(value: string): string {
  if (value === "lead_delivery") return "Lead email submission";
  if (value === "delivery") return "Delivery lookup";
  if (value === "response") return "API response";
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function AdminJourneyPanel({ journey }: { journey?: JourneyMetrics }) {
  const hasSamples = Boolean(journey?.measured && journey.sessions_observed > 0 && journey.stages.length > 0);
  return <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
    <h2 className="font-semibold text-gray-900">Recorded journey stages</h2>
    <p className="mt-1 text-xs text-gray-600">Counts are distinct observed sessions at each stage. A session can contain more than one saved consultation. Stages can be optional or occur in different orders. Without completion means no completion event was recorded; it does not confirm abandonment.</p>
    <p className="mt-2 text-xs text-gray-500">{journey?.measurement_start === "after_recording_consent" ? "Measurement begins after recording consent. Visits before consent are not measured here." : journey?.measurement_start === "server_conversation_requests" ? "Server conversation requests are measured; storefront visits and unreported browser actions are not implied." : "Stage measurement coverage is not available for this response."}</p>
    <p className="mt-1 text-xs text-gray-500">A response event records an API reply, not a verified helpful answer.</p>
    {journey && <p className="mt-1 text-xs text-gray-500">{journey.unlinked_sessions} unlinked observed sessions cannot be assigned to a saved consultation or exact release cohort. {journey.note}</p>}
    {journey?.measurement_started_at && <p className="mt-1 text-xs text-gray-500">First stage event in this view: {journey.measurement_started_at.replace("T", " ")}</p>}
    {hasSamples && journey ? <>
      <p className="mt-3 text-xs text-gray-600">{journey.sessions_observed} observed sessions · {journey.client_events} browser events · {journey.server_events} server events</p>
      <div className="mt-3 overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr className="border-b border-gray-200 text-gray-500">
        {["Stage", "Shown", "Requested", "Completed", "Skipped", "Failed", "Resumed", "Without completion"].map((title) => <th key={title} className="whitespace-nowrap px-2 py-2 font-medium">{title}</th>)}
      </tr></thead><tbody>{journey.stages.map((stage) => <tr key={stage.stage} className="border-b border-gray-100"><th className="px-2 py-2 font-medium text-gray-800">{stageLabel(stage.stage)}</th>{[stage.shown, stage.requested, stage.completed, stage.skipped, stage.failed, stage.resumed, stage.without_completion].map((count, index) => <td className="px-2 py-2 tabular-nums" key={index}>{count}</td>)}</tr>)}</tbody></table></div>
    </> : <p className="mt-3 rounded-lg bg-gray-50 p-3 text-sm text-gray-600">No stage measurements for this cohort yet. Saved historical consultations remain available; missing telemetry is unknown.</p>}
  </section>;
}
