"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { linkFreshdeskTicket } from "@/lib/adminApi";

interface Props {
  ticketId: string;
  freshdeskUrl?: string | null;
  freshdeskTicketId?: string | null;
}

export default function AdminFreshdeskLinkButton({
  ticketId,
  freshdeskUrl,
  freshdeskTicketId,
}: Props) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

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
        ) : (
          <span className="text-xs text-gray-500">No Freshdesk ticket linked yet.</span>
        )}
        <button
          type="button"
          onClick={handleLink}
          disabled={loading}
          className="rounded-lg border border-indigo-300 bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading
            ? "Linking…"
            : freshdeskUrl
              ? "Retry / refresh link"
              : "Create Freshdesk ticket"}
        </button>
      </div>
      {message && (
        <p className="text-xs text-green-700">{message}</p>
      )}
      {error && (
        <p className="text-xs text-red-700">{error}</p>
      )}
    </div>
  );
}
