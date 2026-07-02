/**
 * adminApi.ts
 * ===========
 * Client-side API helpers for the admin warranty dashboard.
 *
 * All calls go to Next.js Route Handlers (/api/admin/warranty/…),
 * which proxy to the FastAPI backend with ADMIN_API_KEY server-side.
 * ADMIN_API_KEY is NEVER read here or in any client component.
 */

import type {
  AdminWarrantyTicket,
  DecisionRequest,
  DecisionResponse,
  FreshdeskLinkResponse,
  NoteRequest,
  TicketDetailResponse,
  TicketListResponse,
} from "./adminTypes";

const PROXY = "/api/admin/warranty";

// ------------------------------------------------------------
// Ticket list
// ------------------------------------------------------------

export async function fetchAdminTickets(params?: {
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<TicketListResponse> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.limit != null) qs.set("limit", String(params.limit));
  if (params?.offset != null) qs.set("offset", String(params.offset));

  const url = `${PROXY}/tickets${qs.size ? `?${qs}` : ""}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { detail?: string };
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<TicketListResponse>;
}

// ------------------------------------------------------------
// Ticket detail
// ------------------------------------------------------------

export async function fetchAdminTicket(
  ticketId: string
): Promise<TicketDetailResponse> {
  const res = await fetch(
    `${PROXY}/tickets/${encodeURIComponent(ticketId)}`,
    { cache: "no-store" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { detail?: string };
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<TicketDetailResponse>;
}

// ------------------------------------------------------------
// Admin decision
// ------------------------------------------------------------

export async function submitAdminDecision(
  ticketId: string,
  body: DecisionRequest
): Promise<DecisionResponse> {
  const res = await fetch(
    `${PROXY}/tickets/${encodeURIComponent(ticketId)}/decision`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { detail?: string };
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<DecisionResponse>;
}

// ------------------------------------------------------------
// Freshdesk link
// ------------------------------------------------------------

export async function linkFreshdeskTicket(
  ticketId: string
): Promise<FreshdeskLinkResponse> {
  const res = await fetch(
    `${PROXY}/tickets/${encodeURIComponent(ticketId)}/freshdesk-link`,
    { method: "POST" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { detail?: string };
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<FreshdeskLinkResponse>;
}

// ------------------------------------------------------------
// Admin note
// ------------------------------------------------------------

export async function submitAdminNote(
  ticketId: string,
  body: NoteRequest
): Promise<{ ticket: AdminWarrantyTicket }> {
  const res = await fetch(
    `${PROXY}/tickets/${encodeURIComponent(ticketId)}/note`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { detail?: string };
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<{ ticket: AdminWarrantyTicket }>;
}
