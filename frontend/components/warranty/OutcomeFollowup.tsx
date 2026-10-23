"use client";

import { useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { submitWarrantyFollowup } from "@/lib/api";
import { WARRANTY_CONTACT_EMAIL } from "@/lib/evidenceMessage";

interface Props {
  ticketId: string;
  token: string;
}

/** Optional later update: keeps an unresolved recurrence visible after closure. */
export default function OutcomeFollowup({ ticketId, token }: Props) {
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<"fixed" | "still_broken" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<{ outcome: "fixed" | "still_broken"; eventId: string } | null>(null);

  async function report(outcome: "fixed" | "still_broken") {
    if (saving) return;
    if (pending.current?.outcome !== outcome) {
      pending.current = { outcome, eventId: uuidv4() };
    }
    setSaving(true);
    setError(null);
    try {
      await submitWarrantyFollowup(ticketId, token, outcome, pending.current.eventId);
      setResult(outcome);
      pending.current = null;
    } catch (err) {
      setError(err instanceof Error ? err.message : "We could not save your update.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <details className="rounded-xl border border-gray-200 bg-white p-3 text-sm">
      <summary className="cursor-pointer font-medium text-gray-800">Update your result</summary>
      <p className="mt-3 text-gray-700">How is your chair doing now?</p>
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" disabled={saving} onClick={() => report("fixed")}
          className="rounded-lg border border-emerald-300 px-3 py-2 text-emerald-800 disabled:opacity-50">
          It is working now
        </button>
        <button type="button" disabled={saving} onClick={() => report("still_broken")}
          className="rounded-lg border border-gray-300 px-3 py-2 text-gray-800 disabled:opacity-50">
          I still have a problem
        </button>
      </div>
      {result && <p role="status" className="mt-2 text-gray-700">
        {result === "fixed" ? "Thank you — we recorded that your issue is resolved." :
          `Thank you — we recorded that you still need help. To request support, start a new case or contact ${WARRANTY_CONTACT_EMAIL}.`}
      </p>}
      {error && <p role="alert" className="mt-2 text-red-700">{error}</p>}
    </details>
  );
}
