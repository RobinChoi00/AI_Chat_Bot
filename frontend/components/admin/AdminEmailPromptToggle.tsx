"use client";

import { useEffect, useState } from "react";

interface PromptState {
  enabled: boolean;
  source: "admin" | "default" | string;
}

/** Turns the customer email step on or off. Recording consent stays in place. */
export default function AdminEmailPromptToggle() {
  const [state, setState] = useState<PromptState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch("/api/admin/warranty/chat-email-prompt", { cache: "no-store" });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error((body as { detail?: string }).detail || "Could not load the email prompt setting.");
        }
        if (!cancelled) setState(body as PromptState);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load the email prompt setting.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function setEnabled(enabled: boolean) {
    setSaving(true);
    setError(null);
    try {
      const response = await fetch("/api/admin/warranty/chat-email-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error((body as { detail?: string }).detail || "Could not update the email prompt.");
      }
      setState(body as PromptState);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update the email prompt.");
    } finally {
      setSaving(false);
    }
  }

  const enabled = state?.enabled === true;

  return (
    <div className={`mb-5 rounded-xl border px-4 py-3 text-sm ${enabled ? "border-sky-200 bg-sky-50 text-sky-950" : "border-orange-200 bg-orange-50 text-orange-950"}`}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-semibold">
            Customer email prompt is {state ? (enabled ? "on" : "off") : "loading"}
          </p>
          <p className="mt-1 text-xs leading-relaxed">
            {enabled
              ? "Customers enter an email after I Agree. Turn this off only for a training session."
              : "Training mode: the chat opens after I Agree and does not ask for an email. Turn it on before live customers use this chat."}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setEnabled(!enabled)}
          disabled={saving || state === null}
          className="shrink-0 rounded-lg border border-current bg-white px-3 py-1.5 text-xs font-semibold hover:bg-white/70 disabled:opacity-50"
        >
          {saving ? "Saving…" : enabled ? "Turn off email prompt" : "Ask customers for email"}
        </button>
      </div>
      {error && <p role="alert" className="mt-2 text-xs text-red-700">{error}</p>}
    </div>
  );
}
