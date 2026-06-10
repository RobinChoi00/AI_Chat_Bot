import type { ReactNode } from "react";
import type {
  AdminWarrantyEvidence,
  AdminWarrantyTicket,
  AdminWarrantyTurn,
} from "@/lib/adminTypes";
import AdminStatusBadge from "./AdminStatusBadge";
import AdminEvidenceList from "./AdminEvidenceList";

interface Props {
  ticket: AdminWarrantyTicket;
  turns: AdminWarrantyTurn[];
  evidence: AdminWarrantyEvidence[];
}

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase tracking-wide text-gray-400">
        {label}
      </dt>
      <dd className="mt-0.5 text-sm text-gray-900">{value ?? "—"}</dd>
    </div>
  );
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function AdminTicketDetail({ ticket, turns, evidence }: Props) {
  const collectedEntries = Object.entries(ticket.collected_data ?? {}).filter(
    ([key]) => key !== "customer_contact_email" && key !== "tracking_snapshot"
  );
  const customerEmail = ticket.customer_email;

  return (
    <div className="space-y-6">
      {/* ── Customer contact ─────────────────────────────────────── */}
      <section className="rounded-xl border border-emerald-200 bg-emerald-50 p-5 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-emerald-700">
          Customer Contact
        </h2>
        {customerEmail ? (
          <div className="flex flex-wrap items-center gap-3">
            <a
              href={`mailto:${customerEmail}`}
              className="text-lg font-semibold text-emerald-900 underline-offset-2 hover:underline"
            >
              {customerEmail}
            </a>
            <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-medium text-emerald-800">
              Follow up within 24 hours
            </span>
          </div>
        ) : (
          <p className="text-sm text-emerald-800">
            No customer email captured yet. Check conversation answers or ask the
            customer to upload evidence with their email address.
          </p>
        )}
      </section>

      {/* ── Ticket info ─────────────────────────────────────────── */}
      <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Ticket Information
        </h2>
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <Field
            label="Ticket ID"
            value={
              <span className="break-all font-mono text-xs">{ticket.ticket_id}</span>
            }
          />
          <Field
            label="Status"
            value={<AdminStatusBadge status={ticket.status} size="md" />}
          />
          <Field label="Session ID" value={
            <span className="break-all font-mono text-xs">{ticket.session_id}</span>
          } />
          <Field label="Domain" value={ticket.domain} />
          <Field label="Issue Type" value={ticket.issue_type} />
          <Field label="Defect Type" value={ticket.defect_type} />
          <Field label="Model Name" value={ticket.model_name} />
          <Field label="Current Node" value={
            <span className="font-mono text-xs">{ticket.current_node_id}</span>
          } />
          <Field label="Created" value={formatDate(ticket.created_at)} />
          <Field label="Updated" value={formatDate(ticket.updated_at)} />
        </dl>
      </section>

      {/* ── Admin review ─────────────────────────────────────────── */}
      {(ticket.admin_decision || ticket.admin_note || ticket.decided_by) && (
        <section className="rounded-xl border border-indigo-200 bg-indigo-50 p-5">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-indigo-600">
            Admin Review
          </h2>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Field label="Decision" value={ticket.admin_decision} />
            <Field label="Decided By" value={ticket.decided_by} />
            {ticket.customer_message && (
              <div className="col-span-full">
                <Field
                  label="Customer Message"
                  value={
                    <p className="whitespace-pre-wrap text-sm">
                      {ticket.customer_message}
                    </p>
                  }
                />
              </div>
            )}
            {ticket.admin_note && (
              <div className="col-span-full">
                <Field
                  label="Admin Note"
                  value={
                    <p className="whitespace-pre-wrap text-sm">
                      {ticket.admin_note}
                    </p>
                  }
                />
              </div>
            )}
          </dl>
        </section>
      )}

      {/* ── Collected data ───────────────────────────────────────── */}
      {collectedEntries.length > 0 && (
        <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500">
            Collected Data
          </h2>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {collectedEntries.map(([k, v]) => (
              <Field key={k} label={k.replace(/_/g, " ")} value={v} />
            ))}
          </dl>
        </section>
      )}

      {/* ── Conversation turns ───────────────────────────────────── */}
      <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Conversation ({turns.length} step{turns.length !== 1 ? "s" : ""})
        </h2>
        {turns.length === 0 ? (
          <p className="text-sm text-gray-400 italic">No turns recorded yet.</p>
        ) : (
          <ol className="space-y-3">
            {turns.map((turn, i) => (
              <li key={turn.id} className="flex gap-3">
                <span className="mt-1 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-700">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1 rounded-lg border border-gray-100 bg-gray-50 p-3 text-xs">
                  <p className="mb-1 font-medium text-gray-700">
                    <span className="rounded bg-gray-200 px-1 py-0.5 font-mono text-[10px]">
                      {turn.node_id}
                    </span>
                    {turn.node_type && (
                      <span className="ml-2 text-gray-400">
                        ({turn.node_type})
                      </span>
                    )}
                  </p>
                  {turn.node_prompt && (
                    <p className="mb-1 text-gray-600">
                      <span className="font-medium">Prompt:</span>{" "}
                      {turn.node_prompt}
                    </p>
                  )}
                  {turn.customer_answer && (
                    <p className="text-gray-800">
                      <span className="font-medium">Answer:</span>{" "}
                      {turn.customer_answer}
                      {turn.answer_key && turn.answer_key !== turn.customer_answer && (
                        <span className="ml-2 text-gray-400">
                          (key: {turn.answer_key})
                        </span>
                      )}
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
      </section>

      {/* ── Evidence ─────────────────────────────────────────────── */}
      <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Evidence ({evidence.length} file{evidence.length !== 1 ? "s" : ""})
        </h2>
        <AdminEvidenceList evidence={evidence} />
      </section>
    </div>
  );
}
