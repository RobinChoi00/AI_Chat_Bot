"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { linkFreshdeskTicket } from "@/lib/adminApi";

interface Props {
  ticketId: string;
  freshdeskUrl?: string | null;
  freshdeskTicketId?: string | null;
  createError?: string | null;
  createErrorDetail?: string | null;
  createFailedAt?: string | null;
  createAttemptCount?: number | null;
}

export default function AdminFreshdeskLinkButton({
  ticketId,
  freshdeskUrl,
  freshdeskTicketId,
  createError,
  createErrorDetail,
  createFailedAt,
  createAttemptCount,
}: Props) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const hasLink = Boolean(freshdeskUrl);
  const hasCreateFailure = Boolean(createError) && !hasLink;

  async function handleLink() {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const result = await linkFreshdeskTicket(ticketId);
      if (result.freshdesk?.created) {
        setMessage("Freshdesk ticket created and linked.");
      } else if (result.freshdesk?.reason === "already_linked") {
        setMessage("Freshdesk ticket was already linked.");
      } else if (result.ok) {
        setMessage("Freshdesk link confirmed.");
      } else {
        setError(result.freshdesk?.reason ?? "Could not link Freshdesk ticket.");
      }
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Freshdesk link failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {freshdeskUrl ? (
          <a
            href={freshdeskUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-indigo-300 bg-white px-3 py-1.5 text-xs font-medium text-indigo-800 hover:bg-indigo-50"
          >
            Open Freshdesk #{freshdeskTicketId}
          </a>
        ) : hasCreateFailure ? (
          <span className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-900">
            Freshdesk create failed
          </span>
        ) : (
          <span className="text-xs text-gray-500">No Freshdesk ticket linked yet.</span>
        )}
        <button
          type="button"
          onClick={handleLink}
          disabled={loading}
          className={`rounded-lg border px-3 py-1.5 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-60 ${
            hasCreateFailure
              ? "border-amber-500 bg-amber-600 hover:bg-amber-700"
              : "border-indigo-300 bg-indigo-600 hover:bg-indigo-700"
          }`}
        >
          {loading
            ? "Linking…"
            : hasLink
              ? "Retry / refresh link"
              : hasCreateFailure
                ? "Retry Freshdesk create"
                : "Create Freshdesk ticket"}
        </button>
      </div>
      {hasCreateFailure && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-950">
          <p className="font-medium">Last create error: {createError}</p>
          {createErrorDetail ? (
            <p className="mt-1 break-words text-amber-900/90">{createErrorDetail}</p>
          ) : null}
          <p className="mt-1 text-amber-800/80">
            {createFailedAt ? `Failed at ${createFailedAt}` : null}
            {typeof createAttemptCount === "number"
              ? `${createFailedAt ? " · " : ""}Attempts: ${createAttemptCount}`
              : null}
          </p>
        </div>
      )}
      {message && <p className="text-xs text-green-700">{message}</p>}
      {error && <p className="text-xs text-red-700">{error}</p>}
    </div>
  );
}
