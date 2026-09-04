"use client";

import { useState, type FormEvent } from "react";
import { lookupWarrantyCaseStatus } from "@/lib/api";
import type { WarrantyStatusLookupResponse } from "@/lib/types";
import WarrantyTeamContactFooter from "./WarrantyTeamContactFooter";

interface Props {
  initialCaseReference?: string;
  compact?: boolean;
}

export default function WarrantyCaseLookup({
  initialCaseReference = "",
  compact = false,
}: Props) {
  const [caseReference, setCaseReference] = useState(initialCaseReference);
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<WarrantyStatusLookupResponse | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const payload = await lookupWarrantyCaseStatus(caseReference, email);
      setResult(payload);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "We couldn't find a case with those details."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={compact ? "space-y-3" : "space-y-4"}>
      <form onSubmit={handleSubmit} className="space-y-3">
        <label className="block text-left">
          <span className="text-xs font-medium text-gray-700">Case reference</span>
          <input
            type="text"
            autoComplete="off"
            value={caseReference}
            onChange={(e) => setCaseReference(e.target.value)}
            placeholder="WR-20260904-ABCDEF"
            className="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
        </label>
        <label className="block text-left">
          <span className="text-xs font-medium text-gray-700">Email on the case</span>
          <input
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
        </label>
        <button
          type="submit"
          disabled={loading || !caseReference.trim() || !email.trim()}
          className="min-h-[44px] w-full rounded-xl bg-brand-600 px-4 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-400"
        >
          {loading ? "Checking…" : "Check case"}
        </button>
      </form>

      {error && (
        <p className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          {error}
        </p>
      )}

      {result && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-left">
          <p className="text-xs font-medium uppercase tracking-wide text-emerald-800">
            {result.status_label}
          </p>
          <p className="mt-1 font-mono text-sm font-semibold text-emerald-950">
            {result.case_reference}
          </p>
          {result.model_name && (
            <p className="mt-1 text-sm text-emerald-900">{result.model_name}</p>
          )}
          <p className="mt-2 text-sm text-emerald-900">{result.next_step}</p>
          {result.customer_message && (
            <p className="mt-2 whitespace-pre-wrap text-sm text-emerald-950">
              {result.customer_message}
            </p>
          )}
        </div>
      )}

      {!compact && <WarrantyTeamContactFooter />}
    </div>
  );
}
