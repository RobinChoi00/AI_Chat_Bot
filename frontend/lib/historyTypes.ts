export const HISTORY_SOURCES = ["warranty", "sales", "general"] as const;
export type HistorySource = (typeof HISTORY_SOURCES)[number];
export type HistoryView = "conversation" | "workflow";
export type HistoryTranscriptMode = "api_recorded" | "legacy_reconstruction" | "workflow_reference" | "stored_messages";
export type HistoryMessageOrigin = "customer_input" | "user_action" | "api_response" | "workflow_prompt" | "workflow_answer" | "legacy_enrichment" | "sales_message" | "general_log";

export interface HistoryConversation {
  source: HistorySource;
  record_id: string;
  session_id: string | null;
  domain: string | null;
  status: string | null;
  created_at: string | null;
  updated_at: string | null;
  message_count: number;
  title: string;
  ticket_id?: string | null;
}

export interface HistoryListResponse {
  total: number;
  offset: number;
  limit: number;
  counts: Record<HistorySource, number>;
  items: HistoryConversation[];
}

export interface HistoryDetailResponse {
  conversation: HistoryConversation;
  messages: {
    id: string;
    role: "user" | "assistant";
    content: string;
    created_at: string | null;
    origin?: HistoryMessageOrigin;
  }[];
  total: number;
  offset: number;
  limit: number;
  transcript_mode?: HistoryTranscriptMode;
  recorded_since?: string | null;
  alternate_view_available?: boolean;
}
