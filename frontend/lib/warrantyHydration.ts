import { formatTerminalPrompt } from "@/lib/evidenceMessage";
import type {
  TerminalEnrichment,
  WarrantySessionResponse,
  WarrantyTicketState,
} from "@/lib/types";

type AssistantResponse = Pick<
  WarrantySessionResponse,
  "assistant_message" | "terminal_enrichment"
>;

export function assistantContentFromResponse(
  ticket: WarrantyTicketState | null,
  resp: AssistantResponse
): string | null {
  const node = ticket?.current_node;
  if (!node?.prompt && !resp.assistant_message) return null;
  return formatTerminalPrompt(
    node?.prompt ?? "",
    node?.evidence_required,
    node?.evidence_email,
    resp.assistant_message ?? resp.terminal_enrichment?.message
  );
}

/** Restore the last assistant bubble after refresh / resume (uses enrichment when present). */
export function hydrationAssistantContent(
  ticket: WarrantyTicketState | null,
  resp: WarrantySessionResponse
): string | null {
  const node = ticket?.current_node;
  if (!node) return null;

  if (node.is_terminal) {
    return assistantContentFromResponse(ticket, resp) ?? node.prompt ?? null;
  }

  if (resp.assistant_message?.trim()) {
    return resp.assistant_message.trim();
  }

  return node.prompt ?? null;
}

export function hasStepEnrichmentPanel(
  enrichment: WarrantySessionResponse["step_enrichment"]
): boolean {
  if (!enrichment || enrichment.phase !== "workflow_step") return false;
  return Boolean(
    enrichment.top_match ||
      (enrichment.tips && enrichment.tips.length > 0) ||
      (enrichment.sources && enrichment.sources.length > 0)
  );
}

export function formatEnrichmentSource(source: string): string {
  switch (source) {
    case "freshdesk":
      return "Past support ticket";
    case "freshdesk_kb":
      return "Knowledge base";
    case "freshdesk_qa":
      return "Support Q&A";
    default:
      return source.replace(/_/g, " ");
  }
}
