"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import type { AIHealth, KnowledgeReview, OutcomeQuality } from "@/lib/qualityTypes";
import RepairOutcomePanel from "./RepairOutcomePanel";

export default function AIQualityDashboard({ outcomes, reviews, health }: {
  outcomes: OutcomeQuality | null; reviews: KnowledgeReview[] | null; health: AIHealth | null;
}) {
  const router = useRouter();
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function write(path: string, body: unknown) {
    setBusy(true); setMessage("");
    try {
      const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const result = await response.json() as { detail?: unknown };
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `Request failed (${response.status}); check the draft fields.`);
      setMessage("Saved. Knowledge is used only after approval and before its review expiry.");
      router.refresh();
      return true;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Save failed.");
      return false;
    } finally { setBusy(false); }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const fields = new FormData(form);
    const value = (name: string) => String(fields.get(name) ?? "").trim();
    const saved = await write("/api/admin/quality/knowledge", {
      title: value("title"), category: value("category"), diagnostic: value("diagnostic"),
      model_names: value("models").split(",").map((item) => item.trim()).filter(Boolean),
      customer_steps: value("steps").split("\n").map((item) => item.trim()).filter(Boolean),
      source_url: value("source"), review_until: value("review_until"),
    });
    if (saved) form.reset();
  }

  const input = "w-full rounded border border-gray-300 bg-white p-2 text-sm";
  const modelCalls = health?.model_calls;
  return <div className="space-y-6">
    {outcomes && <RepairOutcomePanel repair={outcomes.repair} />}
    {health && <section className="rounded-xl border bg-white p-4">
      <h2 className="font-semibold">Current runtime · response times in the last 7 days</h2>
      <p className="mt-2 text-xs text-gray-600">Agent: {health.models.agent} · Router: {health.models.router} · Embedding: {health.models.embedding}</p>
      <p className="mt-1 break-all text-xs text-gray-500">Code release: {health.app_release_id ?? "Unmeasured"}</p>
      {health.model_configuration && <div className="mt-3 rounded-lg bg-gray-50 p-3 text-xs text-gray-700">
        <h3 className="font-semibold">Configured model selection</h3>
        {(["agent", "router"] as const).map((role) => {
          const selection = health.model_configuration?.[role];
          return <p key={role} className="mt-1 break-all">{role}: {selection?.effective_model ?? "Unknown"} · {selection?.source === "env" ? "environment override" : selection?.source === "default" ? "code default" : "selection source unknown"} · code default {selection?.default_model ?? "Unknown"}</p>;
        })}
        <p className="mt-2">Configuration alone does not confirm model access or a completed request.</p>
      </div>}
      <div className="mt-4 rounded-lg border border-gray-200 p-3 text-xs text-gray-700">
        <h3 className="font-semibold">Observed model calls{modelCalls ? ` · last ${modelCalls.days} days` : ""}</h3>
        {modelCalls ? <>
          {(["agent", "router"] as const).map((role) => {
            const observed = modelCalls.roles[role];
            const latest = observed.latest_call;
            const successful = observed.latest_successful_call;
            return <div key={role} className="mt-3 space-y-1 break-words">
              <p className="font-medium">{role}: {observed.calls} recorded calls · {observed.successful_calls} successful · {observed.failed_calls} failed</p>
              {latest ? <p>Latest request: {latest.requested_model} · {latest.succeeded ? "successful" : `failed (${latest.error_code ?? "unknown category"})`} · {latest.observed_at ?? "time unknown"} · release {latest.release_id}</p> : <p>No recorded request. Provider model is unknown.</p>}
              {successful ? <p>Latest successful request: {successful.requested_model} · {successful.provider_model ? `provider returned ${successful.provider_model}` : "provider model ID unknown"} · {successful.observed_at ?? "time unknown"} · release {successful.release_id}</p> : <p>No successful call recorded. Model access is unverified.</p>}
            </div>;
          })}
          <p className="mt-3 text-gray-500">{modelCalls.scope}</p>
          <p className="mt-1 text-gray-500">Counts cover retained observations: at most {modelCalls.max_retained_calls.toLocaleString("en-US")} calls for {modelCalls.retention_days} days.</p>
        </> : <p className="mt-2">Model-call evidence is unavailable. Configured names do not establish successful access.</p>}
      </div>
      {Object.entries(health.latency).map(([surface, timing]) => <p key={surface} className="mt-2 text-sm">
        {surface}: {timing.samples} successful requests · median {timing.samples <= 0 || timing.p50_ms === null ? "—" : `${timing.p50_ms} ms`} · P95 {timing.samples <= 0 || timing.p95_ms === null ? "—" : `${timing.p95_ms} ms`}
      </p>)}
      <p className="mt-3 text-sm">Stock cache: {String(health.stock_cache.fresh_entries ?? 0)} fresh entries · TTL {String(health.stock_cache.ttl_seconds ?? "—")} seconds · Last provider fetch {String(health.stock_cache.latest_fetched_at ?? "unmeasured")}</p>
      <p className="mt-2 text-sm">Warranty semantic index: {health.warranty_embeddings.ready ? "Ready" : "Not loaded"} · {String(health.warranty_embeddings.entries ?? 0)} entries · Source {String(health.warranty_embeddings.source ?? "none")}</p>
      <p className="mt-3 text-xs text-gray-500">This runtime view is not filtered by the selected release cohort. Model names describe the current configuration, not versions used in past conversations. No samples means unmeasured. Streamed chat completion is measured separately in chat traces and live evaluations. Cache status does not verify external service access.</p>
    </section>}
    <section className="rounded-xl border bg-white p-4">
      <h2 className="font-semibold">Submit a knowledge draft</h2>
      <p className="mt-1 text-xs text-gray-500">Use verified customer-safe instructions with a source and applicable models. Do not paste customer transcripts or personal information.</p>
      <form onSubmit={submit} className="mt-4 grid gap-3 md:grid-cols-2">
        <label className="text-sm">Title<input name="title" required maxLength={160} className={input} /></label>
        <label className="text-sm">Applicable models (comma separated)<input name="models" required className={input} /></label>
        <label className="text-sm">Category<select name="category" className={input}>{["power", "remote", "air", "mech", "footrest", "heat", "voice", "misc", "general"].map((category) => <option key={category}>{category}</option>)}</select></label>
        <label className="text-sm">Review again by<input name="review_until" type="date" required className={input} /></label>
        <label className="text-sm md:col-span-2">Source URL<input name="source" type="url" required placeholder="https://..." className={input} /></label>
        <label className="text-sm md:col-span-2">Symptom / diagnostic context<textarea name="diagnostic" required minLength={10} maxLength={1000} className={input} /></label>
        <label className="text-sm md:col-span-2">Customer-safe steps (one per line)<textarea name="steps" required rows={4} className={input} /></label>
        <button disabled={busy} className="rounded bg-indigo-600 px-4 py-2 text-sm text-white disabled:opacity-50">Submit for review</button>
      </form>
      {message && <p role="status" className="mt-3 text-sm">{message}</p>}
    </section>
    {reviews !== null && <section className="rounded-xl border bg-white p-4">
      <h2 className="font-semibold">Knowledge review · latest {reviews.length} entries</h2>
      {reviews.length === 0 && <p className="mt-3 text-sm text-gray-500">No drafts submitted.</p>}
      <div className="mt-3 divide-y">{reviews.map((review) => <article key={review.id} className="space-y-2 py-4">
        <h3 className="font-medium">{review.title} · {review.status}</h3>
        <p className="text-xs text-gray-500">{review.model_names.join(", ")} · {review.category} · Review by {review.review_until}</p>
        <p className="text-sm">{review.diagnostic}</p>
        <ol className="list-inside list-decimal text-sm">{review.customer_steps.map((step, index) => <li key={index}>{step}</li>)}</ol>
        <p className="break-all text-xs text-gray-500">Source: {review.source_url}</p>
        {review.status === "pending" ? <div className="flex gap-2">{["approved", "rejected"].map((decision) => <button
          key={decision} disabled={busy} onClick={() => write(`/api/admin/quality/knowledge/${review.id}/decision`, { decision })}
          className="rounded border px-3 py-1 text-sm disabled:opacity-50"
        >{decision === "approved" ? "Approve verified guidance" : "Reject"}</button>)}</div> : <p className="text-xs">Reviewed by {review.reviewed_by ?? "—"}</p>}
      </article>)}</div>
    </section>}
  </div>;
}
