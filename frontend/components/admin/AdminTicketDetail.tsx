import type { ReactNode } from "react";
import type {
  AdminWarrantyEvidence,
  AdminWarrantyTicket,
  AdminWarrantyTurn,
} from "@/lib/adminTypes";
import AdminStatusBadge from "./AdminStatusBadge";
import AdminChannelBadge from "./AdminChannelBadge";
import AdminEvidenceList from "./AdminEvidenceList";
import AdminFreshdeskLinkButton from "./AdminFreshdeskLinkButton";

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

function formatCollectedValue(value: unknown): ReactNode {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return "—";
    return (
      <ul className="list-disc space-y-1 pl-4 text-xs">
        {value.map((item, index) => (
          <li key={index}>{formatCollectedValue(item)}</li>
        ))}
      </ul>
    );
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const entries = Object.entries(record);
    if (entries.length === 0) return "—";
    return (
      <dl className="space-y-1 text-xs">
        {entries.map(([key, nested]) => (
          <div key={key}>
            <dt className="font-medium text-gray-500">{key.replace(/_/g, " ")}</dt>
            <dd className="text-gray-800">{formatCollectedValue(nested)}</dd>
          </div>
        ))}
      </dl>
    );
  }
  return String(value);
}

const ANSWER_KEY_LABELS: Record<string, string> = {
  warranty: "Warranty issue",
  sales: "Sales inquiry",
  installation: "Setup / installation",
  delivery: "Delivery",
  defect: "Chair malfunction",
  power: "Power issue",
  remote: "Remote / controller",
  air: "Air / inflation",
  rolling: "Massage mechanism",
  recline: "Recline / position",
  footrest: "Footrest",
  cosmetic: "Cosmetic damage",
  heat: "Heat",
  voice: "Voice control",
  general_setup: "General setup help",
  footrest_or_no_air: "Footrest / no air",
  yes: "Yes",
  no: "No",
};

function humanAnswer(turn: AdminWarrantyTurn): string {
  const raw = (turn.customer_answer || "").trim();
  const key = (turn.answer_key || "").trim();
  if (raw && raw !== key) return raw;
  if (key && ANSWER_KEY_LABELS[key]) return ANSWER_KEY_LABELS[key];
  if (raw) return ANSWER_KEY_LABELS[raw] || raw;
  if (key) return key.replace(/_/g, " ");
  return "—";
}

function formatTurnTime(iso: string | null): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function AdminTicketDetail({ ticket, turns, evidence }: Props) {
  const collectedEntries = Object.entries(ticket.collected_data ?? {}).filter(
    ([key]) =>
      ![
        "customer_contact_email",
        "tracking_snapshot",
        "channel",
        "caller_phone",
        "followup_sent_at",
        "error_code",
        "fonz_meaning",
        "fonz_parts_internal",
        "fonz_severity",
        "fonz_lookup_failed",
        "fonz_category_aligned",
        "error_code_gate_completed",
        "pending_terminal",
        "troubleshooting_history",
        "chat_timeline",
        "warranty_eligibility",
        "warranty_eligibility_status",
        "purchase_date",
        "unmapped_phrases",
        "delivery_lookup_input",
        "delivery_lookup_kind",
        "delivery_lookup_failed",
        "delivery_lookup_error",
        "tracking_number",
        "order_number",
        "checkout_email",
      ].includes(key)
  );
  const fonz = ticket.fonz_diagnostics;
  const customerEmail = ticket.customer_email;
  const isPhone = (ticket.channel || "").toLowerCase() === "phone";
  const troubleshootingHistory = ticket.collected_data?.troubleshooting_history;
  const hasTroubleshootingHistory =
    troubleshootingHistory !== undefined &&
    troubleshootingHistory !== null &&
    troubleshootingHistory !== "" &&
    !(Array.isArray(troubleshootingHistory) && troubleshootingHistory.length === 0);

  type EligibilityInfo = {
    status?: string;
    purchase_date?: string;
    summary?: string;
    expires_on?: string;
    days_remaining?: number | null;
  };
  let eligibility: EligibilityInfo | null = null;
  const eligibilityRaw = ticket.collected_data?.warranty_eligibility;
  if (typeof eligibilityRaw === "string" && eligibilityRaw.trim()) {
    try {
      eligibility = JSON.parse(eligibilityRaw) as EligibilityInfo;
    } catch {
      eligibility = null;
    }
  } else if (eligibilityRaw && typeof eligibilityRaw === "object") {
    eligibility = eligibilityRaw as EligibilityInfo;
  }

  return (
    <div className="space-y-6">
      {/* ── Customer contact ─────────────────────────────────────── */}
      <section className="rounded-xl border border-emerald-200 bg-emerald-50 p-5 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-emerald-700">
          Customer Contact
        </h2>
        {isPhone && ticket.caller_phone ? (
          <div className="flex flex-wrap items-center gap-3">
            <AdminChannelBadge channel="phone" />
            <a
              href={`tel:${ticket.caller_phone}`}
              className="text-lg font-semibold font-mono text-emerald-900 underline-offset-2 hover:underline"
            >
              {ticket.caller_phone}
            </a>
            <span className="rounded-full bg-sky-100 px-2.5 py-0.5 text-xs font-medium text-sky-800">
              After-hours phone IVR
            </span>
          </div>
        ) : customerEmail ? (
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
        ) : ticket.intake_email_gate_status === "skipped" ||
          ticket.collected_data?.intake_email_gate_status === "skipped" ? (
          <p className="text-sm text-amber-900">
            Customer skipped the intake email step. Ask them to upload evidence with
            an email, or use the final handoff contact form if they reach it.
          </p>
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
            label="Case Reference"
            value={
              ticket.case_reference ? (
                <span className="font-mono text-sm font-semibold">{ticket.case_reference}</span>
              ) : (
                "—"
              )
            }
          />
          <Field
            label="Freshdesk"
            value={
              <AdminFreshdeskLinkButton
                ticketId={ticket.ticket_id}
                freshdeskUrl={ticket.freshdesk_url}
                freshdeskTicketId={ticket.freshdesk_ticket_id}
                createError={ticket.freshdesk_create_error}
                createErrorDetail={ticket.freshdesk_create_error_detail}
                createFailedAt={ticket.freshdesk_create_failed_at}
                createAttemptCount={ticket.freshdesk_create_attempt_count}
              />
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
          <Field
            label="Channel"
            value={<AdminChannelBadge channel={ticket.channel} />}
          />
          <Field
            label="Caller Phone"
            value={
              ticket.caller_phone ? (
                <a href={`tel:${ticket.caller_phone}`} className="font-mono text-sm text-sky-700 underline-offset-2 hover:underline">
                  {ticket.caller_phone}
                </a>
              ) : (
                "—"
              )
            }
          />
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

      {eligibility?.status ? (
        <section
          className={`rounded-xl border p-5 shadow-sm ${
            eligibility.status === "possibly_expired"
              ? "border-amber-200 bg-amber-50"
              : eligibility.status === "unknown"
                ? "border-gray-200 bg-gray-50"
                : "border-sky-200 bg-sky-50"
          }`}
        >
          <h2
            className={`mb-3 text-sm font-semibold uppercase tracking-wide ${
              eligibility.status === "possibly_expired"
                ? "text-amber-800"
                : eligibility.status === "unknown"
                  ? "text-gray-600"
                  : "text-sky-800"
            }`}
          >
            Warranty eligibility (soft — confirm plan; does not block)
          </h2>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Field label="Status" value={eligibility.status} />
            <Field label="Purchase date" value={eligibility.purchase_date || "Unknown"} />
            <Field
              label="Rough review horizon"
              value={eligibility.expires_on || "—"}
            />
          </dl>
          <p className="mt-3 text-sm text-gray-700">
            {eligibility.status === "unknown"
              ? "Purchase date unknown — confirm plan and purchase channel in NetSuite. Soft signal only; does not block the case."
              : eligibility.summary || ""}
          </p>
          <p className="mt-2 text-xs text-gray-500">
            Plans: Standard (1 yr L+P then parts), Extended (3 yr L+P then parts),
            Adjusted (rare), Brand extended / Mattress Firm / Johnson Fitness (5 yr L+P).
            Unauthorized or third-party purchase: no service / no parts.
          </p>
        </section>
      ) : null}

      {(() => {
        const raw = ticket.collected_data?.unmapped_phrases;
        let phrases: { node_id?: string; text?: string }[] = [];
        if (Array.isArray(raw)) {
          phrases = raw;
        } else if (typeof raw === "string" && raw.trim()) {
          try {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) phrases = parsed;
          } catch {
            phrases = [];
          }
        }
        if (!phrases.length) return null;
        return (
          <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
              Unmapped customer phrases
            </h2>
            <p className="mb-3 text-sm text-gray-600">
              Typed answers that did not match a menu option. The customer stayed
              on that step.
            </p>
            <ul className="space-y-2 text-sm text-gray-800">
              {phrases.map((row, idx) => (
                <li key={`${row.node_id || "node"}-${idx}`}>
                  <span className="font-mono text-xs text-gray-500">
                    {row.node_id || "—"}
                  </span>
                  {": "}
                  {row.text || "—"}
                </li>
              ))}
            </ul>
          </section>
        );
      })()}

      {ticket.collected_data?.delivery_lookup_failed === "1" && (
        <section className="rounded-xl border border-amber-200 bg-amber-50 p-5 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-amber-800">
            Delivery lookup failed
          </h2>
          <p className="mb-3 text-sm text-amber-900">
            Automatic Shopify / carrier lookup did not return a status. The
            customer&apos;s case continued. Look this up manually.
          </p>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Field
              label="Customer entered"
              value={String(ticket.collected_data?.delivery_lookup_input || "—")}
            />
            <Field
              label="Kind"
              value={String(ticket.collected_data?.delivery_lookup_kind || "—")}
            />
            <Field
              label="Tracking / order"
              value={String(
                ticket.collected_data?.tracking_number ||
                  ticket.collected_data?.order_number ||
                  "—"
              )}
            />
            <Field
              label="Failure reason"
              value={String(ticket.collected_data?.delivery_lookup_error || "—")}
            />
          </dl>
        </section>
      )}

      {/* ── Fonz error-code diagnostics (internal) ───────────────── */}
      {fonz?.error_code && (
        <section className="rounded-xl border border-amber-200 bg-amber-50 p-5 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-amber-800">
            Fonz Error Code (Internal)
          </h2>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Field label="Error Code" value={<span className="font-mono font-semibold">{fonz.error_code}</span>} />
            <Field label="Severity" value={fonz.severity} />
            <Field label="Gate" value={fonz.gate_completed} />
            <Field
              label="Category Aligned"
              value={fonz.category_aligned ? "Yes — matches defect path" : "No / unknown"}
            />
            <Field
              label="Lookup"
              value={
                fonz.lookup_failed
                  ? "Not found in Fonz list for this model"
                  : fonz.match_model
                    ? `Matched ${fonz.match_model}`
                    : "—"
              }
            />
            {fonz.meaning && (
              <div className="col-span-full">
                <Field
                  label="Meaning"
                  value={<p className="whitespace-pre-wrap text-sm">{fonz.meaning}</p>}
                />
              </div>
            )}
            {fonz.parts_internal && (
              <div className="col-span-full">
                <Field
                  label="Parts (internal)"
                  value={<p className="whitespace-pre-wrap text-sm">{fonz.parts_internal}</p>}
                />
              </div>
            )}
          </dl>
        </section>
      )}

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
      {(collectedEntries.length > 0 || hasTroubleshootingHistory) && (
        <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500">
            Collected Data
          </h2>
          {hasTroubleshootingHistory && (
            <div className="mb-4">
              <Field
                label="Troubleshooting history"
                value={formatCollectedValue(troubleshootingHistory)}
              />
            </div>
          )}
          {collectedEntries.length > 0 && (
            <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {collectedEntries.map(([k, v]) => (
                <Field key={k} label={k.replace(/_/g, " ")} value={formatCollectedValue(v)} />
              ))}
            </dl>
          )}
        </section>
      )}

      {/* ── Conversation turns ───────────────────────────────────── */}
      <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Conversation ({turns.length} step{turns.length !== 1 ? "s" : ""}
          {ticket.current_node_id ? " + current" : ""})
        </h2>
        {turns.length === 0 && !ticket.current_node_prompt ? (
          <p className="text-sm text-gray-400 italic">No turns recorded yet.</p>
        ) : (
          <ol className="space-y-3">
            {turns.map((turn, i) => (
              <li key={turn.id} className="flex gap-3">
                <span className="mt-1 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-700">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1 rounded-lg border border-gray-100 bg-gray-50 p-3 text-xs">
                  <div className="mb-1 flex flex-wrap items-center gap-2">
                    <span className="rounded bg-gray-200 px-1 py-0.5 font-mono text-[10px] font-medium text-gray-700">
                      {turn.node_id}
                    </span>
                    {turn.node_type && (
                      <span className="text-gray-400">({turn.node_type})</span>
                    )}
                    {turn.created_at && (
                      <span className="ml-auto text-[10px] text-gray-400">
                        {formatTurnTime(turn.created_at)}
                      </span>
                    )}
                  </div>
                  {turn.node_prompt && (
                    <p className="mb-2 text-gray-600">
                      <span className="font-medium text-gray-500">Bot asked:</span>{" "}
                      {turn.node_prompt}
                    </p>
                  )}
                  <p className="text-sm text-gray-900">
                    <span className="font-medium text-indigo-700">Customer:</span>{" "}
                    {humanAnswer(turn)}
                  </p>
                  {turn.answer_key &&
                    turn.answer_key !== turn.customer_answer &&
                    humanAnswer(turn) !== turn.answer_key && (
                      <p className="mt-1 text-[10px] text-gray-400">
                        key: {turn.answer_key}
                      </p>
                    )}
                </div>
              </li>
            ))}

            {ticket.current_node_id && (
              <li className="flex gap-3">
                <span className="mt-1 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-amber-100 text-xs font-bold text-amber-800">
                  →
                </span>
                <div className="min-w-0 flex-1 rounded-lg border border-amber-200 bg-amber-50/80 p-3 text-xs">
                  <div className="mb-1 flex flex-wrap items-center gap-2">
                    <span className="rounded bg-amber-200/80 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-900">
                      Current step
                    </span>
                    <span className="rounded bg-white/80 px-1 py-0.5 font-mono text-[10px] text-amber-900">
                      {ticket.current_node_id}
                    </span>
                  </div>
                  {ticket.current_node_prompt ? (
                    <p className="whitespace-pre-wrap text-sm text-amber-950">
                      <span className="font-medium text-amber-800">Bot is showing:</span>{" "}
                      {ticket.current_node_prompt}
                    </p>
                  ) : (
                    <p className="text-amber-800/80 italic">
                      Waiting on this node (prompt not available).
                    </p>
                  )}
                </div>
              </li>
            )}
          </ol>
        )}

        {hasTroubleshootingHistory && (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Extra troubleshooting notes
            </p>
            <div className="text-xs text-slate-800">
              {formatCollectedValue(troubleshootingHistory)}
            </div>
          </div>
        )}

        {Array.isArray(ticket.collected_data?.chat_timeline) &&
          (ticket.collected_data.chat_timeline as unknown[]).length > 0 && (
            <div className="mt-4 rounded-lg border border-indigo-100 bg-indigo-50/50 p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-indigo-600">
                Extra chat (tips & side questions)
              </p>
              <ul className="space-y-2">
                {(ticket.collected_data.chat_timeline as Array<Record<string, unknown>>).map(
                  (event, idx) => {
                    const role = String(event.role || "");
                    const kind = String(event.kind || "");
                    const text = String(event.text || "");
                    const when = formatTurnTime(
                      typeof event.created_at === "string" ? event.created_at : null
                    );
                    return (
                      <li
                        key={`${idx}-${when}`}
                        className="rounded-md border border-indigo-100 bg-white px-3 py-2 text-xs"
                      >
                        <div className="mb-1 flex flex-wrap gap-2 text-[10px] text-indigo-500">
                          <span className="font-semibold uppercase">
                            {role === "user" ? "Customer" : "Bot"} · {kind}
                          </span>
                          {when ? <span className="ml-auto">{when}</span> : null}
                        </div>
                        <p className="whitespace-pre-wrap text-gray-800">{text}</p>
                      </li>
                    );
                  }
                )}
              </ul>
            </div>
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
