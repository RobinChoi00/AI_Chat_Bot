"use client";

import { FormEvent, useState } from "react";

interface Props {
  disabled?: boolean;
  onContinue: (email: string) => Promise<void> | void;
  onSkip: () => Promise<void> | void;
}

const EMAIL_RE = /^[\w.+-]+@[\w.-]+\.\w+$/;

/** Soft-required email step shown after I Agree, before chat is unlocked. */
export default function ChatEmailGate({ disabled, onContinue, onSkip }: Props) {
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

  async function handleSkip() {
    if (busy || disabled) return;
    setBusy(true);
    setError("");
    try {
      await onSkip();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not continue.");
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
        We use this only so our warranty team can reply about your case — parts,
        technician updates, and support. You can skip for now and add it later if
        needed.
      </p>
      <p className="mt-2 rounded-lg border border-sky-100 bg-white/70 px-3 py-2 text-xs leading-snug text-sky-800">
        We do <strong>not</strong> sell or share your email with third parties, and we
        do <strong>not</strong> use it for advertising or marketing lists. It is only
        for warranty support on this case.
      </p>
      <form onSubmit={handleContinue} className="mt-3 space-y-2">
        <label className="block text-xs font-medium text-sky-900" htmlFor="chat-email-gate">
          Email address
        </label>
        <input
          id="chat-email-gate"
          type="email"
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
      <button
        type="button"
        disabled={busy || disabled}
        onClick={handleSkip}
        className="mt-2 w-full text-center text-xs font-medium text-sky-800 underline-offset-2 hover:underline disabled:opacity-50"
      >
        Skip for now
      </button>
    </div>
  );
}
