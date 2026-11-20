"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { acknowledgeHandoff } from "@/lib/adminApi";

interface Props {
  ticketId: string;
  alreadyAcknowledged: boolean;
  acceptedAt?: string | null;
}

export default function AdminHandoffAckButton({
  ticketId,
  alreadyAcknowledged,
  acceptedAt,
}: Props) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(alreadyAcknowledged);
  const [note, setNote] = useState("");

  async function handleAck() {
    setLoading(true);
    setError(null);
    try {
      await acknowledgeHandoff(ticketId, { note: note.trim() || undefined });
      setDone(true);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not record staff acknowledgement.");
    } finally {
      setLoading(false);
    }
  }

  if (done || alreadyAcknowledged) {
    return (
      <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
        Staff acknowledgement recorded
        {acceptedAt ? ` · ${acceptedAt.replace("T", " ").slice(0, 19)}` : ""}.
        This case no longer appears in the attention queue.
      </div>
    );
  }

  return (
    <div className="space-y-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
      <div>
        <h3 className="text-sm font-semibold text-amber-950">
          Clear from handoff attention
        </h3>
        <p className="mt-1 text-xs text-amber-900/90">
          Use after you reconciled Freshdesk or confirmed staff received the
          request. This does not approve or reject the warranty claim and does
          not resend notifications.
        </p>
      </div>
      <label className="block text-xs font-medium text-amber-950">
        Optional note
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={2}
          disabled={loading}
          placeholder="e.g. Freshdesk #… already open; staff notified"
          className="mt-1 w-full resize-none rounded-lg border border-amber-300 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-1 focus:ring-amber-400 disabled:opacity-60"
        />
      </label>
      {error && <p className="text-xs text-red-700">{error}</p>}
      <button
        type="button"
        onClick={handleAck}
        disabled={loading}
        className="rounded-lg border border-amber-600 bg-amber-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-800 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Saving…" : "Mark staff received"}
      </button>
    </div>
  );
}
