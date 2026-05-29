/**
 * api.ts
 * ======
 * Centralized API client for the Titan / Osaki Warranty Chat frontend.
 *
 * All network calls go through this module.
 * No production URLs are hardcoded here.
 */

import type {
  ChatRequest,
  EvidenceType,
  EvidenceUploadResponse,
  WarrantySessionResponse,
} from "./types";

// ---------------------------------------------------------------------------
// Base URL
// ---------------------------------------------------------------------------

/**
 * Resolve the backend base URL from the environment variable.
 *
 * In development, defaults to localhost:8000.
 * In Docker/production, set NEXT_PUBLIC_API_BASE_URL in .env.local or
 * docker-compose.yml.
 *
 * When using Next.js rewrites (next.config.mjs), calls to /api/* are
 * transparently proxied to the backend, so API_BASE can be an empty string
 * for browser-side code. Set it explicitly for server-side calls.
 */
function getApiBase(): string {
  const base = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!base) {
    if (typeof window !== "undefined") {
      // Browser: rely on Next.js rewrite proxy — use relative paths
      return "";
    }
    // Server-side fallback
    console.warn(
      "[api.ts] NEXT_PUBLIC_API_BASE_URL is not set. Falling back to http://localhost:8000"
    );
    return "http://localhost:8000";
  }
  return base.replace(/\/$/, "");
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/**
 * Stream a chat response from the FastAPI backend.
 *
 * Returns an AsyncGenerator that yields text chunks as they arrive.
 * The caller is responsible for assembling the full response.
 *
 * @throws Error on non-2xx HTTP status
 */
export async function* streamChat(
  req: ChatRequest
): AsyncGenerator<string, void, unknown> {
  const url = `${getApiBase()}/api/v1/chat`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      detail = err.detail ?? detail;
    } catch {
      // ignore JSON parse failure
    }
    throw new Error(detail);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder("utf-8");
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    if (chunk) yield chunk;
  }
}

// ---------------------------------------------------------------------------
// Warranty session state
// ---------------------------------------------------------------------------

/**
 * Fetch the current warranty session state for a session_id.
 *
 * Returns ticket=null if no active warranty ticket exists for this session.
 * Call this after each chat turn to get structured node/option data.
 *
 * CONTRACT: GET /api/v1/warranty/session/{session_id}
 */
export async function getWarrantySession(
  sessionId: string
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch warranty session: HTTP ${res.status}`);
  }
  return res.json() as Promise<WarrantySessionResponse>;
}

// ---------------------------------------------------------------------------
// Evidence upload
// ---------------------------------------------------------------------------

/**
 * Upload an evidence file for a warranty ticket.
 *
 * CONTRACT: POST /api/v1/warranty/{ticket_id}/evidence (multipart/form-data)
 *
 * Allowed file types: jpg, jpeg, png, pdf, mp4, mov (max 20 MB)
 * Does NOT send email.
 * Does NOT expose internal server paths to the caller.
 *
 * @throws Error on validation failure (422), ticket not found (404), etc.
 */
export async function uploadEvidence(
  ticketId: string,
  evidenceType: EvidenceType,
  file: File
): Promise<EvidenceUploadResponse> {
  const url = `${getApiBase()}/api/v1/warranty/${encodeURIComponent(ticketId)}/evidence`;

  const formData = new FormData();
  formData.append("evidence_type", evidenceType);
  formData.append("file", file, file.name);

  const res = await fetch(url, {
    method: "POST",
    body: formData,
    // Do NOT set Content-Type manually — let the browser set the multipart boundary
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  const data = (await res.json()) as EvidenceUploadResponse & {
    saved_path?: string;
  };

  // Strip internal server path before returning to UI
  // TODO(contract-gap): Ideally the backend should not return saved_path at all
  //   in the customer-facing response. Tracked in docs/warranty_api_contract.md.
  const { saved_path: _stripped, ...safe } = data;
  return safe;
}
