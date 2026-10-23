"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import type { AIHealth, KnowledgeReview, OutcomeQuality } from "@/lib/qualityTypes";

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
  return <div className="space-y-6">
    {outcomes && <section className="rounded-xl border bg-white p-4">
      <h2 className="font-semibold">All stored warranty outcomes</h2>
      <dl className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
        {[
          ["Consultations", outcomes.total_consultations], ["Latest report: fixed", outcomes.latest_confirmed_fixed],
          ["Latest report: still broken", outcomes.latest_still_broken], ["Outcome unknown", outcomes.outcome_unknown],
          ["Customers followed up", outcomes.followed_up], ["Reported broken after a fix", outcomes.reopened_after_fix],
          ["Fixed / all consultations", `${outcomes.confirmed_fix_rate_all_pct}%`],
          ["Fixed / reported outcomes", `${outcomes.confirmed_fix_rate_reported_pct}%`],
        ].map(([label, value]) => <div key={String(label)}><dt className="text-xs text-gray-500">{label}</dt><dd className="text-xl font-semibold">{value}</dd></div>)}
      </dl>
      <p className="mt-3 text-xs text-gray-500">Latest customer reports can change earlier results. A follow-up does not reopen a claim or approve service. Unreported results remain unknown.</p>
    </section>}
    {health && <section className="rounded-xl border bg-white p-4">
      <h2 className="font-semibold">Response times · last 7 days</h2>
      <p className="mt-2 text-xs text-gray-600">Agent: {health.models.agent} · Router: {health.models.router} · Embedding: {health.models.embedding}</p>
      {Object.entries(health.latency).map(([surface, timing]) => <p key={surface} className="mt-2 text-sm">
        {surface}: {timing.samples} successful requests · median {timing.p50_ms === null ? "—" : `${timing.p50_ms} ms`} · P95 {timing.p95_ms === null ? "—" : `${timing.p95_ms} ms`}
      </p>)}
      <p className="mt-3 text-sm">Stock cache: {String(health.stock_cache.fresh_entries ?? 0)} fresh entries · TTL {String(health.stock_cache.ttl_seconds ?? "—")} seconds · Last provider fetch {String(health.stock_cache.latest_fetched_at ?? "unmeasured")}</p>
      <p className="mt-2 text-sm">Warranty semantic index: {health.warranty_embeddings.ready ? "Ready" : "Not loaded"} · {String(health.warranty_embeddings.entries ?? 0)} entries · Source {String(health.warranty_embeddings.source ?? "none")}</p>
      <p className="mt-3 text-xs text-gray-500">No samples means unmeasured. Streamed chat completion is measured separately in chat traces and live evaluations. Cache status does not verify external service access.</p>
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
