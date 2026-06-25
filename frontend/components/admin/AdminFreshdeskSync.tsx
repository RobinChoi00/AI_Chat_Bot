"use client";

import { useState } from "react";

interface SyncResult {
  ok?: boolean;
  ticket_count?: number;
  knowledge_freshdesk_entries?: number;
  knowledge_total_entries?: number;
  message?: string;
  detail?: string;
}

export default function AdminFreshdeskSync() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SyncResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSync() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/admin/warranty/sync-freshdesk", {
        method: "POST",
      });
      const data = (await res.json()) as SyncResult;
      if (!res.ok) {
        throw new Error(data.detail ?? data.message ?? `HTTP ${res.status}`);
      }
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Sync failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
      <button
        type="button"
        onClick={handleSync}
        disabled={loading}
        className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs font-medium text-indigo-800 transition hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Syncing Freshdesk…" : "↻ Sync Freshdesk knowledge"}
      </button>

      {result?.ok && (
        <span className="text-xs text-green-700">
          {result.ticket_count ?? 0} tickets → {result.knowledge_freshdesk_entries ?? 0}{" "}
          knowledge entries
        </span>
      )}

      {result && result.ok === false && (
        <span className="text-xs text-amber-700">{result.message ?? "No rows saved."}</span>
      )}

      {error && <span className="text-xs text-red-600">{error}</span>}
    </div>
  );
}
