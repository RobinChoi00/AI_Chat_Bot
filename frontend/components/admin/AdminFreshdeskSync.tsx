"use client";

import { useEffect, useState } from "react";

interface TicketSyncResult {
  ok?: boolean;
  ticket_count?: number;
  knowledge_freshdesk_entries?: number;
  knowledge_freshdesk_kb_entries?: number;
  knowledge_total_entries?: number;
  resolved_scanned?: number;
  usable_qa_pairs?: number;
  search_pages_fetched?: number;
  month_windows_scanned?: number;
  fetch_mode?: string;
  llm_rescue_enabled?: boolean;
  llm_rescue_stats?: {
    processed?: number;
    rescued?: number;
    cached?: number;
    skipped?: number;
    errors?: number;
  };
  faiss_rebuild_scheduled?: boolean;
  message?: string;
  detail?: string;
}

interface SolutionsSyncResult {
  ok?: boolean;
  article_count?: number;
  knowledge_freshdesk_kb_entries?: number;
  knowledge_total_entries?: number;
  faiss_rebuild_scheduled?: boolean;
  message?: string;
  detail?: string;
}

interface KbProbeResult {
  reachable?: boolean;
  categories?: number;
  folders?: number;
  articles?: number;
  error?: string;
  detail?: string;
}

interface FaissStatus {
  running?: boolean;
  ok?: boolean;
  ticket_docs?: number;
  kb_docs?: number;
  csv_docs?: number;
  total_docs?: number;
  finished_at?: number;
  error?: string;
  scheduled?: boolean;
  message?: string;
  detail?: string;
}

function formatTs(ts?: number): string {
  if (!ts) return "";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "";
  }
}

export default function AdminFreshdeskSync() {
  const [ticketLoading, setTicketLoading] = useState(false);
  const [kbLoading, setKbLoading] = useState(false);
  const [probeLoading, setProbeLoading] = useState(false);
  const [rebuildLoading, setRebuildLoading] = useState(false);

  const [llmRescue, setLlmRescue] = useState(true);
  const [rebuildAfter, setRebuildAfter] = useState(false);

  const [ticketResult, setTicketResult] = useState<TicketSyncResult | null>(null);
  const [kbResult, setKbResult] = useState<SolutionsSyncResult | null>(null);
  const [probe, setProbe] = useState<KbProbeResult | null>(null);
  const [faiss, setFaiss] = useState<FaissStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void refreshFaiss();
  }, []);

  async function refreshFaiss() {
    try {
      const res = await fetch("/api/admin/warranty/rebuild-faiss", {
        method: "GET",
      });
      const data = (await res.json()) as FaissStatus;
      if (res.ok) setFaiss(data);
    } catch {
      // Non-fatal.
    }
  }

  async function handleTicketSync() {
    setTicketLoading(true);
    setError(null);
    setTicketResult(null);
    try {
      const qs = new URLSearchParams({
        llm_rescue: llmRescue ? "true" : "false",
        rebuild_faiss: rebuildAfter ? "true" : "false",
      });
      const res = await fetch(`/api/admin/warranty/sync-freshdesk?${qs}`, {
        method: "POST",
      });
      const data = (await res.json()) as TicketSyncResult;
      if (!res.ok) {
        throw new Error(data.detail ?? data.message ?? `HTTP ${res.status}`);
      }
      setTicketResult(data);
      if (rebuildAfter) setTimeout(() => void refreshFaiss(), 2000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Ticket sync failed.");
    } finally {
      setTicketLoading(false);
    }
  }

  async function handleKbSync() {
    setKbLoading(true);
    setError(null);
    setKbResult(null);
    try {
      const qs = new URLSearchParams({
        rebuild_faiss: rebuildAfter ? "true" : "false",
      });
      const res = await fetch(`/api/admin/warranty/sync-freshdesk-solutions?${qs}`, {
        method: "POST",
      });
      const data = (await res.json()) as SolutionsSyncResult;
      if (!res.ok) {
        throw new Error(data.detail ?? data.message ?? `HTTP ${res.status}`);
      }
      setKbResult(data);
      if (rebuildAfter) setTimeout(() => void refreshFaiss(), 2000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "KB sync failed.");
    } finally {
      setKbLoading(false);
    }
  }

  async function handleProbe() {
    setProbeLoading(true);
    setError(null);
    setProbe(null);
    try {
      const res = await fetch("/api/admin/warranty/freshdesk-solutions-probe");
      const data = (await res.json()) as KbProbeResult;
      if (!res.ok) {
        throw new Error(data.detail ?? data.error ?? `HTTP ${res.status}`);
      }
      setProbe(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "KB probe failed.");
    } finally {
      setProbeLoading(false);
    }
  }

  async function handleRebuild() {
    setRebuildLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/warranty/rebuild-faiss", {
        method: "POST",
      });
      const data = (await res.json()) as FaissStatus;
      if (!res.ok) {
        throw new Error(data.detail ?? data.message ?? `HTTP ${res.status}`);
      }
      setTimeout(() => void refreshFaiss(), 2000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Rebuild failed.");
    } finally {
      setRebuildLoading(false);
    }
  }

  return (
    <div className="space-y-3 rounded-xl border border-indigo-100 bg-indigo-50/50 p-4">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 text-indigo-900">
          <input
            type="checkbox"
            checked={llmRescue}
            onChange={(event) => setLlmRescue(event.target.checked)}
            className="h-3.5 w-3.5"
          />
          <span>LLM rescue (skip regex-drops)</span>
        </label>
        <label className="flex items-center gap-1.5 text-indigo-900">
          <input
            type="checkbox"
            checked={rebuildAfter}
            onChange={(event) => setRebuildAfter(event.target.checked)}
            className="h-3.5 w-3.5"
          />
          <span>Rebuild FAISS after sync</span>
        </label>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={handleTicketSync}
          disabled={ticketLoading}
          className="rounded-lg border border-indigo-300 bg-white px-3 py-1.5 text-xs font-medium text-indigo-800 transition hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {ticketLoading ? "Syncing tickets…" : "↻ Sync Freshdesk tickets"}
        </button>
        <button
          type="button"
          onClick={handleKbSync}
          disabled={kbLoading}
          className="rounded-lg border border-emerald-300 bg-white px-3 py-1.5 text-xs font-medium text-emerald-800 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {kbLoading ? "Syncing KB…" : "📚 Sync Freshdesk KB (Solutions)"}
        </button>
        <button
          type="button"
          onClick={handleProbe}
          disabled={probeLoading}
          className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {probeLoading ? "Probing…" : "🔎 Probe KB counts"}
        </button>
        <button
          type="button"
          onClick={handleRebuild}
          disabled={rebuildLoading || faiss?.running}
          className="rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-medium text-amber-800 transition hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {rebuildLoading
            ? "Scheduling…"
            : faiss?.running
              ? "FAISS rebuilding…"
              : "🧱 Rebuild FAISS (freshdesk_qa)"}
        </button>
      </div>

      {ticketResult?.ok && (
        <div className="rounded-md border border-indigo-200 bg-white px-3 py-2 text-xs text-indigo-900">
          <strong>{ticketResult.ticket_count ?? 0}</strong> Q&A saved
          {ticketResult.resolved_scanned !== undefined && (
            <>
              {" "}
              (scanned <strong>{ticketResult.resolved_scanned}</strong> resolved
              {ticketResult.search_pages_fetched !== undefined && (
                <> · {ticketResult.search_pages_fetched} search pages</>
              )}
              )
            </>
          )}
          {" → "}
          <strong>{ticketResult.knowledge_freshdesk_entries ?? 0}</strong> knowledge entries
          {ticketResult.knowledge_freshdesk_kb_entries !== undefined && (
            <>
              {" "}
              (+ <strong>{ticketResult.knowledge_freshdesk_kb_entries}</strong> KB)
            </>
          )}
          {ticketResult.llm_rescue_stats && (
            <>
              {" "}
              · LLM rescued <strong>{ticketResult.llm_rescue_stats.rescued ?? 0}</strong>
              , cached {ticketResult.llm_rescue_stats.cached ?? 0}, skipped{" "}
              {ticketResult.llm_rescue_stats.skipped ?? 0}
            </>
          )}
          {ticketResult.faiss_rebuild_scheduled && (
            <> · FAISS rebuild scheduled</>
          )}
        </div>
      )}
      {ticketResult && ticketResult.ok === false && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {ticketResult.message ?? "No tickets saved."}
        </div>
      )}

      {kbResult?.ok && (
        <div className="rounded-md border border-emerald-200 bg-white px-3 py-2 text-xs text-emerald-900">
          <strong>{kbResult.article_count ?? 0}</strong> KB articles →{" "}
          <strong>{kbResult.knowledge_freshdesk_kb_entries ?? 0}</strong> KB entries
          {kbResult.faiss_rebuild_scheduled && <> · FAISS rebuild scheduled</>}
        </div>
      )}
      {kbResult && kbResult.ok === false && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {kbResult.message ?? "No KB articles saved."}
        </div>
      )}

      {probe && (
        <div className="rounded-md border border-gray-200 bg-white px-3 py-2 text-xs text-gray-800">
          KB probe: {probe.reachable ? "reachable" : "unreachable"} —{" "}
          <strong>{probe.categories ?? 0}</strong> categories,{" "}
          <strong>{probe.folders ?? 0}</strong> folders,{" "}
          <strong>{probe.articles ?? 0}</strong> published articles
          {probe.error && <span className="text-red-600"> · {probe.error}</span>}
        </div>
      )}

      {faiss && (
        <div className="rounded-md border border-amber-200 bg-white px-3 py-2 text-xs text-amber-900">
          FAISS:{" "}
          {faiss.running ? (
            <strong>running…</strong>
          ) : faiss.ok ? (
            <>
              last built with <strong>{faiss.total_docs ?? 0}</strong> docs (
              tickets {faiss.ticket_docs ?? 0}, KB {faiss.kb_docs ?? 0}, CSV{" "}
              {faiss.csv_docs ?? 0}) at {formatTs(faiss.finished_at)}
            </>
          ) : faiss.error ? (
            <span className="text-red-600">last error: {faiss.error}</span>
          ) : (
            "no rebuild yet"
          )}
        </div>
      )}

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {error}
        </div>
      )}
    </div>
  );
}
