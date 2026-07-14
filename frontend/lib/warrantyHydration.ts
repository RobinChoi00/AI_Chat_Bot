import { formatTerminalPrompt } from "@/lib/evidenceMessage";
import type {
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

  // A sales handoff is terminal in the warranty state machine, but it must not
  // receive warranty evidence, email, phone, or follow-up wording.
  if (ticket?.status === "sales_handoff") {
    return resp.assistant_message?.trim() || node?.prompt?.trim() || null;
  }

  // Contact/evidence formatting is terminal-only. Applying it to an ordinary
  // question tells customers to submit email before troubleshooting begins.
  if (!node?.is_terminal) {
    return resp.assistant_message?.trim() || node?.prompt?.trim() || null;
  }

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
