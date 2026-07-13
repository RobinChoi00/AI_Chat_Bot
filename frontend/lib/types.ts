// ============================================================
// Shared TypeScript types for the Titan / Osaki Warranty UI
// ============================================================

// ------------------------------------------------------------
// Chat
// ------------------------------------------------------------

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  /**
   * When true, the ChatMessageBubble reveals text one chunk at a time so the
   * response *feels* streamed (real backend SSE can slot in later without any
   * UI change — the same animation ends as soon as the full content lands).
   */
  animate?: boolean;
}

export interface ChatRequest {
  session_id: string;
  user_query: string;
  chat_history: ChatMessage[];
  current_domain: string;
}

// ------------------------------------------------------------
// Warranty ticket
// ------------------------------------------------------------

/**
 * Ticket statuses — mirrors warranty_workflow.py TicketStatus values.
 *
 * "approved" and "rejected" are stored in admin_decision, NOT in status.
 * The ticket status becomes "resolved" when admin sets approved/rejected/closed.
 */
export type TicketStatus =
  | "in_progress"
  | "awaiting_admin_review"
  | "awaiting_evidence"
  | "send_info"
  | "sales_handoff"
  | "admin_reviewing"
  | "need_more_information"
  | "resolved";

export interface AnswerOption {
  answer_key: string;
  label: string;
}

export interface WarrantyNode {
  node_id: string | null;
  node_type: string | null;
  prompt: string | null;
  options: AnswerOption[];
  is_terminal: boolean;
  evidence_required?: string[];
  evidence_email?: string | null;
}

export interface WarrantyTicketState {
  ticket_id: string;
  case_reference?: string | null;
  status: TicketStatus;
  issue_type: string;
  model_name: string;
  model_confirmed?: boolean;
  needs_model_confirmation?: boolean;
  ready_for_issue_type?: boolean;
  needs_customer_reply?: boolean;
  customer_message?: string | null;
  admin_decision?: string | null;
  current_node: WarrantyNode | null;
}

export interface TrackingSummary {
  available: boolean;
  message: string;
  snapshot?: Record<string, unknown>;
}

export interface TerminalEnrichment {
  message?: string;
  install_video?: { url: string; label: string };
  self_help?: string | null;
  diagnosis?: {
    summary?: string;
    steps?: string[];
    sources?: string[];
    top_match?: string | null;
  } | null;
  phase?: "awaiting_help_consent" | "contact";
  help_offer_options?: AnswerOption[];
  show_contact_form?: boolean;
  defer_email?: boolean;
}

/** Freshdesk-backed context on non-terminal workflow steps (button path). */
export interface StepEnrichment {
  message?: string;
  phase?: "workflow_step";
  sources?: string[];
  top_match?: string | null;
  tips?: string[];
  /** True when a small LLM rewrote the draft (question kept verbatim). */
  paraphrased?: boolean;
}

export interface SmartStartRoutingConfirmation {
  inferred_issue_type: string;
  applied_count: number;
  summary: string;
  message: string;
}

export interface SmartStartMetadata {
  source: "llm" | "empty" | string;
  summary: string;
  applied_keys: string[];
  skipped_keys: string[];
  stopped_reason: string;
  model_name_hint?: string;
  routing_confirmation?: SmartStartRoutingConfirmation | null;
}

export interface ModelConfirmationMetadata {
  model_name: string;
  message: string;
}

export interface WarrantySessionResponse {
  session_id: string;
  ticket: WarrantyTicketState | null;
  tracking_summary?: TrackingSummary | null;
  email_notified?: boolean;
  nlp_interpreted?: boolean;
  interpreted_issue_type?: string;
  assistant_message?: string | null;
  terminal_enrichment?: TerminalEnrichment | null;
  step_enrichment?: StepEnrichment | null;
  model_registered?: boolean;
  resolved_model?: string;
  smart_start?: SmartStartMetadata | null;
  model_confirmation?: ModelConfirmationMetadata | null;
  side_question?: boolean;
}

export interface WarrantyContactResponse {
  ticket_id: string;
  customer_email: string;
  evidence_type: string;
  evidence_na: boolean;
  email_notified?: boolean;
}

// ------------------------------------------------------------
// Evidence
// ------------------------------------------------------------

export type EvidenceType =
  | "damage_photos"
  | "video_of_issue"
  | "proof_of_purchase"
  | "photo_of_chair"
  | "photo_of_defect"
  | "proof_of_delivery"
  | "assembly_photo"
  | "remote_photo"
  | "other";

export interface EvidenceUploadResponse {
  evidence_id: number;
  ticket_id: string;
  ticket_status: string;
  evidence_type: string;
  original_filename: string;
  customer_email?: string;
  mime_type?: string;
  file_size_bytes: number;
}

// ------------------------------------------------------------
// API error shape
// ------------------------------------------------------------

export interface ApiError {
  detail: string;
}
