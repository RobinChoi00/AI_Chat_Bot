export interface OutcomeQuality {
  total_consultations: number;
  latest_confirmed_fixed: number;
  latest_still_broken: number;
  outcome_unknown: number;
  followed_up: number;
  reopened_after_fix: number;
  confirmed_fix_rate_all_pct: number;
  confirmed_fix_rate_reported_pct: number;
}

export interface KnowledgeReview {
  id: number;
  status: "pending" | "approved" | "rejected";
  title: string;
  category: string;
  model_names: string[];
  diagnostic: string;
  customer_steps: string[];
  source_url: string;
  review_until: string;
  reviewed_by: string | null;
}

export interface AIHealth {
  models: { agent: string; router: string; embedding: string };
  latency: Record<string, { samples: number; p50_ms: number | null; p95_ms: number | null }>;
  stock_cache: Record<string, unknown>;
  warranty_embeddings: Record<string, unknown>;
}
