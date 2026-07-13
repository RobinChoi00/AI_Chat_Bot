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
  WarrantyContactResponse,
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

/**
 * Register chair model — required before issue-type selection.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/register-model
 */
export async function registerWarrantyModel(
  sessionId: string,
  model: string,
  domain = "osaki.com"
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}/register-model`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, domain }),
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
  return res.json() as Promise<WarrantySessionResponse>;
}

/**
 * Confirm or correct inferred model after smart-start.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/confirm-model
 */
export async function confirmWarrantyModel(
  sessionId: string,
  payload: { confirmed?: boolean; model?: string; domain?: string }
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}/confirm-model`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      confirmed: payload.confirmed ?? true,
      model: payload.model,
      domain: payload.domain ?? "osaki.com",
    }),
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
  return res.json() as Promise<WarrantySessionResponse>;
}

/**
 * Start warranty intake at Installation / Delivery / Defect — no LLM call.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/quick-start
 */
export async function quickStartWarranty(
  sessionId: string,
  issueType: "installation" | "delivery" | "defect",
  domain = "osaki.com"
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}/quick-start`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ issue_type: issueType, domain }),
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
  return res.json() as Promise<WarrantySessionResponse>;
}

/**
 * Start warranty intake from natural language — server maps to issue type.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/natural-start
 */
export async function naturalStartWarranty(
  sessionId: string,
  message: string,
  domain = "osaki.com"
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}/natural-start`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, domain }),
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
  return res.json() as Promise<WarrantySessionResponse>;
}

/**
 * Smart-start: free-text intake that fast-forwards multiple flowchart steps
 * when the LLM is confident. Returns the same session payload as the other
 * warranty endpoints plus a `smart_start` metadata block.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/smart-start
 */
export async function smartStartWarranty(
  sessionId: string,
  message: string,
  domain = "osaki.com"
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}/smart-start`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, domain }),
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
  return res.json() as Promise<WarrantySessionResponse>;
}

/**
 * Abandon any in-progress warranty ticket so the customer can start over.
 *
 * Returns a session response with `ticket: null` (and `restarted: true`).
 * The caller should also rotate the local `warranty_session_id` so any
 * resolved/abandoned tickets stay out of the way of the new conversation.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/restart
 */
export async function restartWarrantySession(
  sessionId: string,
  domain = "osaki.com"
): Promise<WarrantySessionResponse & { restarted?: boolean; closed_ticket_count?: number }> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}/restart`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ domain }),
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
  return res.json();
}

/**
 * Advance the warranty workflow by one step — NLP maps free text when needed.
 *
 * CONTRACT: POST /api/v1/warranty/{ticket_id}/answer
 */
export async function submitWarrantyAnswer(
  ticketId: string,
  answer: string
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/${encodeURIComponent(ticketId)}/answer`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer }),
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
  return res.json() as Promise<WarrantySessionResponse>;
}

/**
 * Undo the last warranty workflow answer (one step back).
 *
 * CONTRACT: POST /api/v1/warranty/{ticket_id}/back
 */
export async function goBackWarranty(
  ticketId: string
): Promise<WarrantySessionResponse> {
  const url = `${getApiBase()}/api/v1/warranty/${encodeURIComponent(ticketId)}/back`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
  return res.json() as Promise<WarrantySessionResponse>;
}

/**
 * Notify the warranty team when the customer leaves their email in chat.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/notify-email
 */
export async function notifyWarrantyEmail(
  sessionId: string,
  message: string,
  chatMessages: { role: string; content: string }[]
): Promise<{ sent: boolean; customer_email?: string; reason?: string }> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(sessionId)}/notify-email`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, chat_messages: chatMessages }),
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
  return res.json();
}

// ---------------------------------------------------------------------------
// Evidence upload
// ---------------------------------------------------------------------------

/**
 * Final-step contact — email only (N/A for photo/video).
 *
 * CONTRACT: POST /api/v1/warranty/{ticket_id}/contact
 */
export async function submitWarrantyContact(
  ticketId: string,
  customerEmail: string
): Promise<WarrantyContactResponse> {
  const url = `${getApiBase()}/api/v1/warranty/${encodeURIComponent(ticketId)}/contact`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_email: customerEmail, evidence_na: true }),
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
  return res.json() as Promise<WarrantyContactResponse>;
}

/**
 * Send a "save & continue later" email link for the current warranty session.
 *
 * CONTRACT: POST /api/v1/warranty/session/{session_id}/resume-link
 */
export async function sendWarrantyResumeLink(
  sessionId: string,
  customerEmail: string
): Promise<{ sent: boolean; customer_email: string; expires_in_days: number }> {
  const url = `${getApiBase()}/api/v1/warranty/session/${encodeURIComponent(
    sessionId
  )}/resume-link`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_email: customerEmail }),
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
  return res.json() as Promise<{
    sent: boolean;
    customer_email: string;
    expires_in_days: number;
  }>;
}

/**
 * Exchange a signed resume token for the underlying session/ticket mapping.
 *
 * CONTRACT: GET /api/v1/warranty/resume/{token}
 */
export async function resumeWarrantyFromToken(token: string): Promise<{
  ticket_id: string;
  session_id: string;
  status: string;
  domain?: string;
  expires_at: number;
}> {
  const url = `${getApiBase()}/api/v1/warranty/resume/${encodeURIComponent(token)}`;
  const res = await fetch(url);
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
  return res.json() as Promise<{
    ticket_id: string;
    session_id: string;
    status: string;
    domain?: string;
    expires_at: number;
  }>;
}

/**
 * Submit thumbs-up / thumbs-down feedback for an assistant message.
 *
 * CONTRACT: POST /api/v1/feedback
 * Body: { session_id, rating: "up"|"down", message_content, comment?, context?, ticket_id? }
 */
export interface ChatFeedbackPayload {
  sessionId: string;
  rating: "up" | "down";
  messageContent: string;
  comment?: string;
  context?: "warranty" | "chat";
  domain?: string;
  ticketId?: string;
}

export async function submitChatFeedback(
  payload: ChatFeedbackPayload
): Promise<{ ok: boolean; feedback_id: number; rating: "up" | "down" }> {
  const url = `${getApiBase()}/api/v1/feedback`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: payload.sessionId,
      rating: payload.rating,
      message_content: payload.messageContent,
      comment: payload.comment,
      context: payload.context ?? "warranty",
      domain: payload.domain,
      ticket_id: payload.ticketId,
    }),
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
  return res.json() as Promise<{
    ok: boolean;
    feedback_id: number;
    rating: "up" | "down";
  }>;
}

/**
 * Append a customer follow-up note after the warranty workflow ends.
 *
 * CONTRACT: POST /api/v1/warranty/{ticket_id}/customer-note
 * Body: { note: string }  → { customer_notes: {text, created_at}[] }
 *
 * Used when the customer hides the contact form and types extra context
 * into the chat input during the terminal step.
 */
export interface CustomerNoteEntry {
  text: string;
  created_at: string;
}

export async function submitCustomerNote(
  ticketId: string,
  note: string
): Promise<{ ticket_id: string; customer_notes: CustomerNoteEntry[] }> {
  const url = `${getApiBase()}/api/v1/warranty/${encodeURIComponent(ticketId)}/customer-note`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note }),
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
  return res.json() as Promise<{
    ticket_id: string;
    customer_notes: CustomerNoteEntry[];
  }>;
}

// ---------------------------------------------------------------------------
// Serial-label OCR
// ---------------------------------------------------------------------------

export interface SerialOcrResponse {
  model_name: string | null;
  serial_number: string | null;
  raw_text: string;
  confidence: "high" | "medium" | "low";
}

/**
 * Send a photo of the warranty sticker on the chair to the backend so it can
 * auto-detect the chair model + serial number.
 *
 * CONTRACT: POST /api/v1/warranty/ocr/serial (multipart/form-data)
 *   file: image/*  (≤ 8 MB)
 *
 * The backend never persists the uploaded photo — it just returns the
 * extracted fields. The caller is responsible for showing them back to the
 * customer for confirmation before starting a warranty intake.
 */
export async function extractSerialFromPhoto(file: File): Promise<SerialOcrResponse> {
  const url = `${getApiBase()}/api/v1/warranty/ocr/serial`;
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(url, { method: "POST", body: formData });
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
  return res.json() as Promise<SerialOcrResponse>;
}

/**
 * Upload an evidence file for a warranty ticket.
 *
 * CONTRACT: POST /api/v1/warranty/{ticket_id}/evidence (multipart/form-data)
 *
 * Allowed file types: jpg, jpeg, png, pdf, mp4, mov (max 20 MB)
 * Requires customer_email in the form.
 * Notifies the warranty evidence team distribution list (file attached).
 * Does NOT expose internal server paths to the caller.
 *
 * @throws Error on validation failure (422), ticket not found (404), etc.
 */
export async function uploadEvidence(
  ticketId: string,
  evidenceType: string,
  file: File,
  customerEmail: string
): Promise<EvidenceUploadResponse> {
  const url = `${getApiBase()}/api/v1/warranty/${encodeURIComponent(ticketId)}/evidence`;

  const formData = new FormData();
  formData.append("evidence_type", evidenceType);
  formData.append("customer_email", customerEmail.trim());
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

  return (await res.json()) as EvidenceUploadResponse;
}
