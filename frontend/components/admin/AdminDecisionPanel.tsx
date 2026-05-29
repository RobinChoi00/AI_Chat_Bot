"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitAdminDecision } from "@/lib/adminApi";
import type { AdminDecision } from "@/lib/adminTypes";
import AdminStatusBadge from "./AdminStatusBadge";

interface Props {
  ticketId: string;
  currentStatus: string;
}

interface DecisionConfig {
  label: string;
  description: string;
  btnCls: string;
  destructive?: boolean;
}

const DECISIONS: Record<AdminDecision, DecisionConfig> = {
  admin_reviewing: {
    label: "Mark as Reviewing",
    description: "You have picked up this ticket and are actively reviewing it.",
    btnCls: "bg-indigo-600 hover:bg-indigo-700 text-white",
  },
  need_more_information: {
    label: "Request More Info",
    description: "Ticket needs additional information from the customer.",
    btnCls: "bg-yellow-500 hover:bg-yellow-600 text-white",
  },
  approved: {
    label: "Approve",
    description:
      "Approve the warranty claim. This action will be recorded. Warranty decisions are final.",
    btnCls: "bg-green-600 hover:bg-green-700 text-white",
    destructive: true,
  },
  rejected: {
    label: "Reject",
    description:
      "Reject the warranty claim. This action will be recorded and cannot be reversed automatically.",
    btnCls: "bg-red-600 hover:bg-red-700 text-white",
    destructive: true,
  },
  closed: {
    label: "Close Case",
    description: "Close this case without further action.",
    btnCls: "bg-gray-600 hover:bg-gray-700 text-white",
  },
};

export default function AdminDecisionPanel({ ticketId, currentStatus }: Props) {
  const router = useRouter();
  const [selected, setSelected] = useState<AdminDecision | null>(null);
  const [note, setNote] = useState("");
  const [customerMessage, setCustomerMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const cfg = selected ? DECISIONS[selected] : null;

  function handleSelect(decision: AdminDecision) {
    setSelected(decision);
    setNote("");
    setCustomerMessage("");
    setError(null);
    setDone(false);
  }

  function handleCancel() {
    setSelected(null);
    setError(null);
  }

  async function handleConfirm() {
    if (!selected) return;
    setLoading(true);
    setError(null);

    try {
      await submitAdminDecision(ticketId, {
        decision: selected,
        note: note.trim() || undefined,
        customer_message: customerMessage.trim() || undefined,
        decided_by: "admin",
      });
      setDone(true);
      setSelected(null);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* Current status */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-500">Current status:</span>
        <AdminStatusBadge status={currentStatus} size="md" />
      </div>

      {done && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-800">
          ✅ Decision recorded successfully.
        </div>
      )}

      {/* Decision buttons */}
      {!selected && (
        <div className="flex flex-wrap gap-2">
          {(Object.entries(DECISIONS) as [AdminDecision, DecisionConfig][]).map(
            ([key, d]) => (
              <button
                key={key}
                onClick={() => handleSelect(key)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition active:scale-95 ${d.btnCls}`}
              >
                {d.label}
              </button>
            )
          )}
        </div>
      )}

      {/* Confirmation panel */}
      {selected && cfg && (
        <div
          className={`rounded-xl border p-4 ${
            cfg.destructive
              ? "border-red-300 bg-red-50"
              : "border-indigo-200 bg-indigo-50"
          }`}
        >
          {cfg.destructive && (
            <div className="mb-3 rounded-lg border border-red-300 bg-red-100 px-3 py-2 text-xs font-medium text-red-800">
              ⚠️ This is a permanent decision. Please review before confirming.
            </div>
          )}

          <p className="mb-3 text-sm font-medium text-gray-800">
            Decision: <span className="font-bold">{cfg.label}</span>
          </p>
          <p className="mb-4 text-xs text-gray-600">{cfg.description}</p>

          {/* Internal note */}
          <div className="mb-3">
            <label className="mb-1 block text-xs font-medium text-gray-600">
              Internal note (optional)
            </label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Add an internal note about this decision…"
              rows={2}
              disabled={loading}
              className="w-full resize-none rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-400 disabled:opacity-60"
            />
          </div>

          {/* Customer message */}
          <div className="mb-4">
            <label className="mb-1 block text-xs font-medium text-gray-600">
              Message for customer (optional)
            </label>
            <textarea
              value={customerMessage}
              onChange={(e) => setCustomerMessage(e.target.value)}
              placeholder="Message to surface to the customer in the chat (if supported)…"
              rows={2}
              disabled={loading}
              className="w-full resize-none rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-400 disabled:opacity-60"
            />
          </div>

          {error && (
            <p className="mb-3 text-xs text-red-600">⚠️ {error}</p>
          )}

          <div className="flex gap-2">
            <button
              onClick={handleConfirm}
              disabled={loading}
              className={`flex-1 rounded-lg py-2 text-sm font-medium text-white transition active:scale-95 disabled:opacity-60 ${
                cfg.destructive
                  ? "bg-red-600 hover:bg-red-700"
                  : "bg-indigo-600 hover:bg-indigo-700"
              }`}
            >
              {loading ? "Saving…" : `Confirm: ${cfg.label}`}
            </button>
            <button
              onClick={handleCancel}
              disabled={loading}
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 disabled:opacity-60"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
