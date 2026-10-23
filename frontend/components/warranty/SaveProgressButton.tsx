"use client";

import { useRef, useState } from "react";
import { sendWarrantyResumeLink } from "@/lib/api";

interface Props {
  sessionId: string;
  disabled?: boolean;
  initialCustomerEmail?: string;
  onSaved?: (status: "sent" | "queued" | "saved") => void;
  onError?: () => void;
  onCustomerEmailConfirmed?: (email: string) => void;
}

type Status = "idle" | "open" | "sending" | "sent" | "queued" | "saved" | "error";

/**
 * Header button that emails the customer a signed resume URL.
 *
 * Collapsed by default; expanding shows a slim email row inline so the user
 * never leaves the chat context. On success the row shrinks to a confirmation.
 */
export default function SaveProgressButton({ sessionId, disabled, initialCustomerEmail = "", onSaved, onError, onCustomerEmailConfirmed }: Props) {
  const [status, setStatus] = useState<Status>("idle");
  const [email, setEmail] = useState(initialCustomerEmail);
  const [errorMsg, setErrorMsg] = useState("");
  const [sentTo, setSentTo] = useState("");
  const [resumeLink, setResumeLink] = useState("");
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const requestInFlight = useRef(false);

  const emailValid = /^[\w.+-]+@[\w.-]+\.\w+$/.test(email.trim());
  const knownEmail = /^[\w.+-]+@[\w.-]+\.\w+$/.test(initialCustomerEmail.trim()) ? initialCustomerEmail.trim() : "";

  async function submit(destination = email.trim()) {
    if (disabled || requestInFlight.current || status === "sending" || !/^[\w.+-]+@[\w.-]+\.\w+$/.test(destination)) return;
    requestInFlight.current = true;
    setEmail(destination);
    setStatus("sending");
    setErrorMsg("");
    try {
      const resp = await sendWarrantyResumeLink(sessionId, destination);
      let link = "";
      if (resp.resume_url) {
        try {
          const url = new URL(resp.resume_url);
          const localHttp = url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
          if ((url.protocol === "https:" || localHttp) && !url.username && !url.password) link = url.href;
        } catch { /* An invalid link must not become clickable. */ }
      }
      if (!resp.sent && !resp.queued && !(resp.email_saved && link)) throw new Error("The resume link was not saved. Please try again.");
      setSentTo(resp.customer_email);
      setResumeLink(link);
      const next = resp.sent ? "sent" : resp.queued ? "queued" : "saved";
      setStatus(next);
      onCustomerEmailConfirmed?.(destination);
      onSaved?.(next);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Could not send link.");
      setStatus("error");
      onError?.();
    } finally {
      requestInFlight.current = false;
    }
  }

  if (status === "sent" || status === "queued" || status === "saved") {
    return (
      <div className="max-w-full rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-800">
        <p role="status">{status === "sent" ? `Link sent to ${sentTo}` : status === "queued" ? resumeLink ? "Email requested. You can also copy your resume link now." : "Email requested. Delivery has not been confirmed." : "Progress saved. Copy your resume link to continue later."}</p>
        {resumeLink && <div className="mt-2 flex flex-wrap items-center gap-3">
          <button type="button" className="min-h-[40px] font-semibold underline" onClick={async () => {
            try { await navigator.clipboard.writeText(resumeLink); setCopied(true); setCopyFailed(false); }
            catch { setCopyFailed(true); }
          }}>{copied ? "Link copied" : "Copy resume link"}</button>
          <a href={resumeLink} target="_blank" rel="noopener noreferrer" referrerPolicy="no-referrer" className="min-h-[40px] py-3 underline">Open resume link</a>
          {copyFailed && <label className="w-full">Select and copy this link:
            <input aria-label="Resume link" value={resumeLink} readOnly onFocus={(event) => event.target.select()} className="mt-1 min-h-[44px] w-full rounded border border-emerald-200 px-2 text-xs" />
          </label>}
        </div>}
      </div>
    );
  }

  if (status === "open" || status === "sending" || status === "error") {
    return (
      <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
        <input
          type="email"
          autoFocus
          aria-label="Email for your resume link"
          autoComplete="email"
          inputMode="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          disabled={disabled || status === "sending"}
          className="min-h-[44px] min-w-0 flex-1 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 sm:w-44 sm:flex-none"
        />
        <button
          type="button"
          onClick={() => submit()}
          disabled={disabled || !emailValid || status === "sending"}
          className="min-h-[44px] rounded-xl bg-brand-600 px-3 py-2 text-xs font-medium text-white hover:bg-brand-700 disabled:opacity-60"
        >
          {status === "sending" ? "Sending…" : "Email link"}
        </button>
        <button
          type="button"
          onClick={() => {
            setStatus("idle");
            setErrorMsg("");
          }}
          disabled={status === "sending"}
          className="text-xs text-gray-500 hover:text-gray-700"
        >
          Cancel
        </button>
        {errorMsg && (
          <span role="alert" className="w-full text-xs text-red-600">{errorMsg}</span>
        )}
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => knownEmail ? submit(knownEmail) : setStatus("open")}
      disabled={disabled}
      title="Email me a link to continue this warranty case later"
      className="min-h-[44px] rounded-xl border border-gray-200 bg-white px-3 py-2 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
    >
      {knownEmail ? "Email a save link" : "Save my progress"}
    </button>
  );
}
