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
  history_record_id?: string | null;
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
  days: number | null;
  all_time?: boolean;
  start: string;
  end: string;
  trend_start?: string;
  trend_days?: number;
  abandon_threshold_hours: number;
}

export interface MetricsTotals {
  started: number;
  unique_identified_customers?: number;
  unique_sessions?: number;
  anonymous_sessions?: number;
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
  self_service_overall_resolution_rate_pct?: number;
  escalated_after_self_service: number;
  abandoned: number;
  idle_in_progress?: number;
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
  cohort_metadata?: MetricCohortMetadata;
  journey?: JourneyMetrics;
  range: MetricsRange;
  totals: MetricsTotals;
  by_status: MetricsStatusRow[];
  by_issue_type: MetricsIssueRow[];
  by_domain: MetricsDomainRow[];
  top_terminals: MetricsTerminalRow[];
  daily_started: MetricsDailyRow[];
  purpose_counts?: { repair: number; delivery: number; other: number; unknown: number };
  repair?: RepairOutcomeMetrics;
  delivery?: DeliveryOutcomeMetrics;
  handoffs?: SupportRequestMetrics;
}

export interface MetricCohortMetadata {
  selected: string;
  current_release: string;
  available_releases: string[];
  measurement_started_at: string | null;
  legacy_note: string;
}

export interface JourneyMetrics {
  measured: boolean;
  measurement_start: "after_recording_consent" | "server_conversation_requests";
  measurement_started_at: string | null;
  unlinked_sessions: number;
  note: string;
  sessions_observed: number;
  client_events: number;
  server_events: number;
  stages: { stage: string; shown: number; completed: number; skipped: number; failed: number; resumed: number; requested: number; without_completion: number }[];
}

export interface RepairOutcomeMetrics {
  consultations: number;
  diagnosis_completed: number;
  diagnosis_completion_rate_pct: number | null;
  latest_outcomes: { fixed: number; not_fixed: number; unable_to_attempt: number; deferred: number; unknown: number };
  result_reported: number;
  confirmed_fix_rate_all_pct: number | null;
  confirmed_fix_rate_reported_pct: number | null;
  followed_up: number;
  reopened_after_fix: number;
}

export interface DeliveryOutcomeMetrics {
  consultations: number;
  latest_lookup_available: number;
  latest_lookup_unavailable: number;
  lookup_not_attempted: number;
  answered: number;
  needs_review: number;
  unable_to_attempt: number;
  deferred: number;
  unknown: number;
}

export interface SupportRequestMetrics {
  requested: number;
  by_reason: { reason: string; count: number }[];
  by_department: { department: string; count: number }[];
}

export interface SalesOutcomeMetrics {
  recommendation: {
    requested_sessions: number; recommended_sessions: number; no_verified_match_sessions: number;
    completed_sessions: number; completion_rate_pct: number | null; denominator_sessions: number; note?: string;
  };
  human: { requested_sessions: number; explicit_requested_sessions: number; request_rate_pct: number | null; staff_confirmed_sessions: number | null; staff_confirmation_status: string; note?: string };
  follow_up: {
    lead_sessions: number; lead_records: number; pending: number; sent: number; failed: number; unknown: number;
    submission_failure_rate_pct: number | null; staff_confirmed: number | null; note?: string;
    transport?: { configured: boolean; failure_code: string | null; acceptance_is_staff_confirmation: false; tls_mode?: string; timeout_seconds?: number };
    failure_codes?: { code: string; count: number }[];
  };
  conversation: { answer_provided_sessions: number; active_waiting_sessions: number; idle_missing_followup_sessions: number; human_requested_sessions: number; no_verified_match_sessions: number; unknown_sessions: number; idle_after_hours: number; note?: string };
  quality: {
    repeated_unclear_sessions: number; unclear_turns: number; assistant_turns: number;
    helpful: number; not_helpful: number; rated: number;
    rated_sessions: number; response_sessions: number; unrated_sessions: number;
    helpful_rate_pct: number | null; rating_coverage_pct: number | null;
    repeated_unclear_rate_pct?: number | null;
  };
}

// ------------------------------------------------------------
// Sales funnel dashboard
// ------------------------------------------------------------

export interface SalesMetricsResponse {
  cohort_metadata?: MetricCohortMetadata;
  journey?: JourneyMetrics;
  outcomes?: SalesOutcomeMetrics;
  question_funnel?: {
    stage: string; label: string; reached: number; advanced: number; continued: number;
    active_waiting: number; idle_missing_followup: number; answer_provided: number; human_requested: number;
    unknown: number; dropped: null; drop_off_pct: null; measurement: string;
  }[];
  range: {
    period: "all" | "range";
    days: number | null;
    start: string | null;
    end: string;
    start_date: string | null;
    end_date: string;
    daily_mode?: "observed" | "calendar";
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
    csat_rated?: number;
    csat_score_pct?: number;
    paid_orders_seen?: number;
    orders_attributed?: number;
    payment_unknown_orders?: number;
    attributed_revenue_usd?: number;
    paid_sessions?: number;
    session_to_order_rate_pct?: number;
  };
  behavior_funnel?: {
    event_counts: Record<string, number>;
    observed_sessions: Record<string, number>;
    ordered_sessions: Record<string, number>;
    rates_pct: Record<string, number | null>;
    denominators: Record<string, number>;
    measured_sessions?: number;
    measurement_coverage_pct?: number | null;
  };
  attributed_revenue_by_currency?: Record<string, number>;
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
