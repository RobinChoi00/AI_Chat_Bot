"use client";

import { useState } from "react";
import { sendWarrantyResumeLink } from "@/lib/api";

interface Props {
  sessionId: string;
  disabled?: boolean;
}

type Status = "idle" | "open" | "sending" | "sent" | "error";

/**
 * Header button that emails the customer a signed resume URL.
 *
 * Collapsed by default; expanding shows a slim email row inline so the user
 * never leaves the chat context. On success the row shrinks to a confirmation.
 */
export default function SaveProgressButton({ sessionId, disabled }: Props) {
  const [status, setStatus] = useState<Status>("idle");
  const [email, setEmail] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [sentTo, setSentTo] = useState("");

  const emailValid = /^[\w.+-]+@[\w.-]+\.\w+$/.test(email.trim());

  async function submit() {
    if (!emailValid) return;
    setStatus("sending");
    setErrorMsg("");
    try {
      const resp = await sendWarrantyResumeLink(sessionId, email.trim());
      setSentTo(resp.customer_email);
      setStatus("sent");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Could not send link.");
      setStatus("error");
    }
  }

  if (status === "sent") {
    return (
      <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs text-emerald-800">
        Link sent to {sentTo}
      </span>
    );
  }

  if (status === "open" || status === "sending" || status === "error") {
    return (
      <div className="flex flex-wrap items-center gap-1.5">
        <input
          type="email"
          autoFocus
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          className="min-h-[32px] w-44 rounded-full border border-gray-200 bg-white px-3 py-1 text-xs text-gray-900 placeholder-gray-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
        />
        <button
          type="button"
          onClick={submit}
          disabled={!emailValid || status === "sending"}
          className="rounded-full bg-brand-600 px-3 py-1 text-xs font-medium text-white hover:bg-brand-700 disabled:opacity-60"
        >
          {status === "sending" ? "Sending…" : "Email link"}
        </button>
        <button
          type="button"
          onClick={() => {
            setStatus("idle");
            setErrorMsg("");
          }}
          className="text-xs text-gray-500 hover:text-gray-700"
        >
          Cancel
        </button>
        {errorMsg && (
          <span className="w-full text-[11px] text-red-600">{errorMsg}</span>
        )}
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setStatus("open")}
      disabled={disabled}
      title="Email me a link to continue this warranty case later"
      className="rounded-full border border-gray-200 bg-white px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
    >
      📧 Save my progress
    </button>
  );
}
