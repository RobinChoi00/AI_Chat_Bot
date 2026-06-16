// ============================================================
// Shared TypeScript types for the Titan / Osaki Warranty UI
// ============================================================

// ------------------------------------------------------------
// Chat
// ------------------------------------------------------------

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
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
  status: TicketStatus;
  issue_type: string;
  model_name: string;
  current_node: WarrantyNode | null;
}

export interface TrackingSummary {
  available: boolean;
  message: string;
  snapshot?: Record<string, unknown>;
}

export interface WarrantySessionResponse {
  session_id: string;
  ticket: WarrantyTicketState | null;
  tracking_summary?: TrackingSummary | null;
  email_notified?: boolean;
  nlp_interpreted?: boolean;
  interpreted_issue_type?: string;
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
  file_size_bytes: number;
  // NOTE: saved_path is returned by the backend but MUST NOT be displayed
  // to the customer — it is an internal server path.
}

// ------------------------------------------------------------
// API error shape
// ------------------------------------------------------------

export interface ApiError {
  detail: string;
}
