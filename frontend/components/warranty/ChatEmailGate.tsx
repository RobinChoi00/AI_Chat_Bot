"use client";

import { FormEvent, useState } from "react";

interface Props {
  disabled?: boolean;
  onContinue: (email: string) => Promise<void> | void;
}

const EMAIL_RE = /^[\w.+-]+@[\w.-]+\.\w+$/;

/** Required email step shown after I Agree, before chat is unlocked. */
export default function ChatEmailGate({ disabled, onContinue }: Props) {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const trimmed = email.trim();
  const valid = EMAIL_RE.test(trimmed);

  async function handleContinue(e: FormEvent) {
    e.preventDefault();
    if (!valid || busy || disabled) {
      setError(valid ? "" : "Please enter a valid email address.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await onContinue(trimmed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save email.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-3 mb-4 rounded-xl border border-sky-200 bg-sky-50/90 px-4 py-4 sm:mx-4">
      <p className="text-sm font-semibold text-sky-950">
        Where should we email follow-up?
      </p>
      <p className="mt-1 text-sm text-sky-900">
        Please enter your email so our warranty team can reply about your case —
        parts, technician updates, and support.
      </p>
      <p className="mt-2 rounded-lg border border-sky-100 bg-white/70 px-3 py-2 text-xs leading-snug text-sky-800">
        We do <strong>not</strong> sell or share your email for advertising or marketing
        lists. It is used only to follow up on this warranty case (and related support
        tools like our helpdesk and email). Please avoid shared or public computers when
        possible, and use <strong>Start over</strong> when you are done on a shared device.
      </p>
      <form onSubmit={handleContinue} className="mt-3 space-y-2">
        <label className="block text-xs font-medium text-sky-900" htmlFor="chat-email-gate">
          Email address <span className="text-red-500">*</span>
        </label>
        <input
          id="chat-email-gate"
          type="email"
          required
          autoComplete="email"
          inputMode="email"
          value={email}
          disabled={busy || disabled}
          onChange={(e) => {
            setEmail(e.target.value);
            if (error) setError("");
          }}
          placeholder="you@example.com"
          className="w-full rounded-lg border border-sky-200 bg-white px-3 py-2.5 text-sm text-gray-900 outline-none ring-sky-300 placeholder:text-gray-400 focus:ring-2 disabled:opacity-60"
        />
        {error ? <p className="text-xs text-red-600">{error}</p> : null}
        <button
          type="submit"
          disabled={busy || disabled || !valid}
          className="w-full rounded-full bg-sky-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-sky-800 disabled:opacity-50"
        >
          {busy ? "Saving…" : "Continue"}
        </button>
      </form>
    </div>
  );
}
