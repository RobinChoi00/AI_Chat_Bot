// ============================================================
// Admin-side TypeScript types — mirrors warranty_models.py to_dict()
// ============================================================

// All status values the backend may return for a warranty ticket.
export type AdminTicketStatus =
  | "in_progress"
  | "awaiting_admin_review"
  | "awaiting_evidence"
  | "send_info"
  | "sales_handoff"
  | "admin_reviewing"
  | "need_more_information"
  | "approved"
  | "rejected"
  | "resolved"
  | "closed";

// The five decisions an admin may submit.
export type AdminDecision =
  | "admin_reviewing"
  | "need_more_information"
  | "approved"
  | "rejected"
  | "closed";

// ------------------------------------------------------------
// Core models
// ------------------------------------------------------------

export interface AdminWarrantyTicket {
  ticket_id: string;
  case_reference?: string | null;
  session_id: string;
  domain: string;
  current_node_id: string;
  status: AdminTicketStatus;
  issue_type: string | null;
  defect_type: string | null;
  model_name: string | null;
  collected_data: Record<string, string>;
  customer_email: string | null;
  admin_decision: string | null;
  admin_note: string | null;
  decided_by: string | null;
  customer_message: string | null;
  freshdesk_ticket_id?: string | null;
  freshdesk_url?: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface AdminWarrantyTurn {
  id: number;
  ticket_id: string;
  node_id: string;
  node_type: string | null;
  node_prompt: string | null;
  customer_answer: string | null;
  answer_key: string | null;
  created_at: string | null;
}

/**
 * Evidence record as seen by the admin UI.
 * NOTE: `file_path` is intentionally absent — the Route Handler proxy
 * strips it before sending data to the browser. Never expose raw server paths.
 */
export interface AdminWarrantyEvidence {
  id: number;
  ticket_id: string;
  evidence_type: string;
  original_filename: string | null;
  mime_type: string | null;
  file_size_bytes: number;
  customer_email: string | null;
  emailed: boolean;
  created_at: string | null;
}

// ------------------------------------------------------------
// API response shapes
// ------------------------------------------------------------

export interface TicketListResponse {
  total: number;
  offset: number;
  tickets: AdminWarrantyTicket[];
}

export interface TicketDetailResponse {
  ticket: AdminWarrantyTicket;
  turns: AdminWarrantyTurn[];
  evidence: AdminWarrantyEvidence[];
}

// ------------------------------------------------------------
// Warranty completion-rate dashboard
// ------------------------------------------------------------

export interface MetricsRange {
  days: number;
  start: string;
  end: string;
  abandon_threshold_hours: number;
}

export interface MetricsTotals {
  started: number;
  reached_terminal: number;
  completion_rate_pct: number;
  contact_captured: number;
  contact_rate_pct: number;
  admin_decided: number;
  resolved: number;
  resolved_rate_pct: number;
  abandoned: number;
  abandoned_rate_pct: number;
  median_turns_to_terminal: number;
}

export interface MetricsStatusRow {
  status: string;
  count: number;
}

export interface MetricsIssueRow {
  issue_type: string;
  count: number;
  completed: number;
  completion_rate_pct: number;
}

export interface MetricsDomainRow {
  domain: string;
  count: number;
  completed: number;
  completion_rate_pct: number;
}

export interface MetricsTerminalRow {
  node_id: string;
  count: number;
}

export interface MetricsDailyRow {
  day: string;
  started: number;
  completed: number;
}

export interface WarrantyMetricsResponse {
  range: MetricsRange;
  totals: MetricsTotals;
  by_status: MetricsStatusRow[];
  by_issue_type: MetricsIssueRow[];
  by_domain: MetricsDomainRow[];
  top_terminals: MetricsTerminalRow[];
  daily_started: MetricsDailyRow[];
}

// ------------------------------------------------------------
// Request bodies
// ------------------------------------------------------------

export interface DecisionRequest {
  decision: AdminDecision;
  note?: string;
  customer_message?: string;
  decided_by?: string;
}

export interface NoteRequest {
  note: string;
  added_by?: string;
}

export type CustomerEmailSkipReason =
  | "decision_not_notifiable"
  | "no_customer_message"
  | "no_customer_email"
  | "smtp_not_configured"
  | "send_failed";

export interface DecisionResponse {
  ticket: AdminWarrantyTicket;
  customer_email_sent: boolean;
  customer_email_skip_reason: CustomerEmailSkipReason | null;
}
