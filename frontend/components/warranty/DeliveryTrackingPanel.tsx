"use client";

import { useState, type FormEvent } from "react";
import type { TrackingSummary } from "@/lib/types";
import SupportContactActions from "./SupportContactActions";

export interface DeliveryLookupIdentity {
  order_number: string;
  email: string;
}

function safeTrackingUrl(value?: string): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password ? url.href : null;
  } catch {
    return null;
  }
}

function checkedAt(value?: string): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleString("en-US", { timeZoneName: "short" });
}

export function DeliveryTrackingResult({ summary }: { summary: TrackingSummary }) {
  const trackingUrl = summary.available ? safeTrackingUrl(summary.tracking_url) : null;
  const when = checkedAt(summary.fetched_at);
  const links = (summary.self_service_links ?? []).flatMap((link) => {
    const url = safeTrackingUrl(link.url);
    return url ? [{ label: link.label, url }] : [];
  });
  return (
    <div className="mt-3 rounded-lg border border-sky-100 bg-white p-3" aria-live="polite">
      {summary.available ? <>
        <p className="text-sm font-semibold text-sky-950">Latest delivery update</p>
        <dl className="mt-2 space-y-1 text-sm text-gray-800">
          <div><dt className="inline font-medium">Status: </dt><dd className="inline">{summary.status || "Check tracking for details"}</dd></div>
          {summary.carrier && <div><dt className="inline font-medium">Carrier: </dt><dd className="inline">{summary.carrier}</dd></div>}
          {summary.eta && <div><dt className="inline font-medium">Delivery estimate: </dt><dd className="inline">{summary.eta}</dd></div>}
        </dl>
        <p className="mt-2 text-xs text-gray-600">Delivery estimates can change. Carrier scans may take time to appear.</p>
        {trackingUrl && <a className="mt-2 inline-block text-sm font-medium text-sky-800 underline" href={trackingUrl} target="_blank" rel="noopener noreferrer">Open carrier tracking ↗</a>}
      </> : <>
        <p className="text-sm font-semibold text-amber-900">Current delivery update unavailable</p>
        <p className="mt-1 text-sm text-gray-700">{summary.message || "We could not verify the latest status. Check your order number and checkout email, or try again later."}</p>
      </>}
      <p className="mt-2 text-xs text-gray-500">{when ? `${summary.available ? "Checked" : "Last attempt"}: ${when}` : "Check time was not recorded. Refresh to request another update."}</p>
      {links.length > 0 && <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-sky-800">
        {links.map((link, index) => <li key={`${link.url}-${index}`}><a href={link.url} target="_blank" rel="noopener noreferrer" className="underline">{link.label} ↗</a></li>)}
      </ul>}
      {!summary.available && summary.support_contact && <div className="mt-3 border-t border-gray-200 pt-2 text-gray-800"><p className="text-xs">For delivery help, contact our Sales team with your order number.</p><SupportContactActions contact={summary.support_contact} /></div>}
    </div>
  );
}

export default function DeliveryTrackingPanel({ storeDomain, initialEmail = "", summary, disabled = false, onLookup }: {
  storeDomain: string;
  initialEmail?: string;
  summary: TrackingSummary | null;
  disabled?: boolean;
  onLookup: (identity: DeliveryLookupIdentity) => Promise<void>;
}) {
  const [orderNumber, setOrderNumber] = useState("");
  const [email, setEmail] = useState(initialEmail);
  const [lastIdentity, setLastIdentity] = useState<DeliveryLookupIdentity | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const identityChanged = Boolean(lastIdentity && (
    orderNumber.trim() !== lastIdentity.order_number || email.trim() !== lastIdentity.email
  ));

  async function lookup(identity: DeliveryLookupIdentity) {
    if (disabled || busy) return;
    setBusy(true);
    setError(null);
    setLastIdentity(identity);
    try {
      await onLookup(identity);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not check delivery status. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const identity = { order_number: orderNumber.trim(), email: email.trim() };
    if (identity.order_number && identity.email) void lookup(identity);
  }

  return (
    <section className="mt-4 rounded-xl border border-sky-200 bg-sky-50/70 p-4" aria-labelledby="delivery-tracking-heading">
      <h2 id="delivery-tracking-heading" className="text-sm font-semibold text-sky-950">Check your delivery</h2>
      <p className="mt-1 text-xs text-sky-900">Enter your order number and the email used at checkout for <strong>{storeDomain}</strong>. Both must match to show delivery details.</p>
      <form onSubmit={submit} className="mt-3 space-y-3">
        <label className="block text-xs font-medium text-gray-700">Order number
          <input required maxLength={40} autoComplete="off" name="delivery-order-number" value={orderNumber} onChange={(event) => { setOrderNumber(event.target.value); setError(null); }} disabled={disabled || busy} placeholder="e.g. 12345" className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm disabled:opacity-60" />
        </label>
        <label className="block text-xs font-medium text-gray-700">Email used at checkout
          <input required type="email" maxLength={254} autoComplete="email" name="delivery-checkout-email" value={email} onChange={(event) => { setEmail(event.target.value); setError(null); }} disabled={disabled || busy} placeholder="you@example.com" className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm disabled:opacity-60" />
        </label>
        <div className="flex flex-wrap gap-2">
          <button type="submit" disabled={disabled || busy} className="rounded-full bg-sky-800 px-4 py-2 text-sm font-medium text-white hover:bg-sky-900 disabled:opacity-50">{busy ? "Checking delivery…" : "Check delivery status"}</button>
          {lastIdentity && !identityChanged && <button type="button" onClick={() => void lookup(lastIdentity)} disabled={disabled || busy} className="rounded-full border border-sky-300 bg-white px-4 py-2 text-sm font-medium text-sky-900 disabled:opacity-50">Refresh</button>}
        </div>
      </form>
      <p className="mt-2 text-xs text-gray-600">Each check requests a new update. For damage or missing parts, you can continue with the delivery help questions below.</p>
      {error && <p role="alert" className="mt-3 text-sm text-red-800">{error} We have not confirmed a current delivery update.</p>}
      {identityChanged && <p className="mt-3 text-sm text-sky-900">Order details changed. Check delivery status to load an update for these details.</p>}
      {summary && !error && !identityChanged && <DeliveryTrackingResult summary={summary} />}
      {summary && !lastIdentity && <p className="mt-2 text-xs text-gray-600">To refresh this update, enter your order number and checkout email above.</p>}
    </section>
  );
}
