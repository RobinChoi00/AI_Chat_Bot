import { formatTerminalPrompt } from "@/lib/evidenceMessage";
import { WARRANTY_WELCOME_MESSAGE } from "@/lib/welcomeMessage";
import type {
  ChatMessage,
  WarrantySessionResponse,
  WarrantyTicketState,
} from "@/lib/types";

type AssistantResponse = Pick<
  WarrantySessionResponse,
  "assistant_message" | "terminal_enrichment" | "handoff"
>;

export function assistantContentFromResponse(
  ticket: WarrantyTicketState | null,
  resp: AssistantResponse
): string | null {
  if (resp.handoff?.requested) {
    return resp.handoff.message?.trim() || resp.assistant_message?.trim() || null;
  }
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
  if (resp.handoff?.requested) return assistantContentFromResponse(ticket, resp);
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

/** Restore actual customer-visible replies without reanimating or repeating them. */
export function hydrationMessages(resp: WarrantySessionResponse): ChatMessage[] {
  const saved = (resp.conversation_messages ?? [])
    .filter((message) => (message.role === "user" || message.role === "assistant") && message.content.trim())
    .map(({ id, role, content }) => ({ id, role, content }));
  if (saved.length && saved[saved.length - 1].role === "assistant") return saved;

  const ticket = resp.ticket;
  let current: string | null = null;
  if (resp.handoff?.requested || ticket?.current_node?.is_terminal) {
    current = hydrationAssistantContent(ticket, resp);
  } else if (ticket?.ready_for_issue_type && ticket.model_name) {
    current = `Great — I have **${ticket.model_name}** on file.\n\nWhat type of issue can we help you with? Please choose an option below.`;
  } else if (ticket?.current_node?.prompt) {
    current = hydrationAssistantContent(ticket, resp);
  } else if (!ticket?.ticket_id) {
    current = WARRANTY_WELCOME_MESSAGE;
  }

  if (saved.length) return current ? [...saved, { role: "assistant", content: current }] : saved;
  if (!current) return [];
  if (ticket?.ready_for_issue_type && ticket.model_name && !ticket.current_node?.is_terminal && !resp.handoff?.requested) {
    return [{ role: "assistant", content: WARRANTY_WELCOME_MESSAGE }, { role: "assistant", content: current }];
  }
  return [{ role: "assistant", content: current }];
}
