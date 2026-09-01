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

export interface AdminFonzDiagnostics {
  error_code: string | null;
  meaning: string | null;
  parts_internal: string | null;
  severity: string | null;
  lookup_failed: boolean;
  category_aligned: boolean;
  gate_completed: string | null;
  match_model: string | null;
  match_code: string | null;
}

export interface AdminWarrantyTicket {
  ticket_id: string;
  case_reference?: string | null;
  session_id: string;
  domain: string;
  current_node_id: string;
  current_node_prompt?: string | null;
  status: AdminTicketStatus;
  issue_type: string | null;
  defect_type: string | null;
  model_name: string | null;
  collected_data: Record<string, unknown>;
  customer_email: string | null;
  intake_email_gate_status?: string | null;
  channel?: string | null;
  caller_phone?: string | null;
  admin_decision: string | null;
  admin_note: string | null;
  decided_by: string | null;
  customer_message: string | null;
  freshdesk_ticket_id?: string | null;
  freshdesk_url?: string | null;
  freshdesk_create_error?: string | null;
  freshdesk_create_error_detail?: string | null;
  freshdesk_create_failed_at?: string | null;
  freshdesk_create_attempt_count?: number | null;
  fonz_diagnostics?: AdminFonzDiagnostics | null;
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
  email_gate_provided?: number;
  email_gate_skipped?: number;
  email_gate_provide_rate_pct?: number;
  admin_decided: number;
  resolved: number;
  resolved_rate_pct: number;
  self_service_started: number;
  self_service_resolved: number;
  self_service_resolution_rate_pct: number;
  escalated_after_self_service: number;
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
// Sales funnel dashboard
// ------------------------------------------------------------

export interface SalesMetricsResponse {
  range: {
    days: number;
    start: string;
    end: string;
  };
  totals: {
    started: number;
    engaged: number;
    engagement_rate_pct: number;
    recommended: number;
    recommend_rate_pct: number;
    nofit: number;
    nofit_rate_pct: number;
    handoffs: number;
    handoff_rate_pct: number;
    leads: number;
    lead_rate_pct: number;
    lead_forward_failed: number;
    lead_forward_failure_rate_pct: number;
    user_turns: number;
    assistant_turns: number;
  };
  by_status: Array<{ status: string; count: number }>;
  by_intent: Array<{ intent: string; count: number }>;
  by_domain: Array<{ domain: string; count: number }>;
  by_channel: Array<{ channel: string; count: number }>;
  lead_delivery: Array<{ status: string; count: number }>;
  daily: Array<{
    day: string;
    started: number;
    recommended: number;
    leads: number;
  }>;
  artifacts: {
    ok: boolean;
    models: number;
    doorway_models: number;
    files: Record<
      string,
      { exists: boolean; size_bytes: number; modified_at: string | null }
    >;
  };
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
  freshdesk_sync?: {
    synced?: boolean;
    skipped?: boolean;
    reason?: string;
    freshdesk_ticket_id?: string;
    freshdesk_url?: string;
    error?: string;
  } | null;
}

export interface FreshdeskLinkResponse {
  ok: boolean;
  freshdesk: {
    created?: boolean;
    skipped?: boolean;
    reason?: string;
    freshdesk_ticket_id?: string;
    freshdesk_url?: string;
    error?: string;
    detail?: string;
  };
  ticket: AdminWarrantyTicket;
}
