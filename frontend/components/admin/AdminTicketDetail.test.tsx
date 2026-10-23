import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { AdminWarrantyTicket, AdminWarrantyTurn } from "@/lib/adminTypes";

vi.mock("next/link", () => ({ default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a> }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: vi.fn() }) }));
import AdminTicketDetail from "./AdminTicketDetail";

const ticket: AdminWarrantyTicket = {
  ticket_id: "ticket-uuid", history_record_id: "42", session_id: "session", domain: "osakiusa.com",
  current_node_id: "internal-current", current_node_prompt: "Current internal workflow instruction", status: "in_progress",
  issue_type: "defect", defect_type: "power", model_name: "Chair", collected_data: {
    chat_timeline: [{ role: "assistant", kind: "terminal_enrichment", text: "Legacy enrichment text", created_at: null }],
  }, customer_email: null, admin_decision: null, admin_note: null, decided_by: null, customer_message: null, created_at: null, updated_at: null,
};
const turns: AdminWarrantyTurn[] = [{
  id: 1, ticket_id: ticket.ticket_id, node_id: "internal-previous", node_type: "question", node_prompt: "Previous internal workflow instruction",
  customer_answer: "Actual saved customer answer", answer_key: "power", created_at: null,
}];

describe("admin warranty workflow reference", () => {
  it("labels raw node text as internal and links to the separately recorded conversation", () => {
    const html = renderToStaticMarkup(<AdminTicketDetail ticket={ticket} turns={turns} evidence={[]} />);
    expect(html).toContain("Internal workflow prompt:");
    expect(html).toContain("Current workflow prompt (internal):");
    expect(html).toContain("may differ from the reply returned to the customer");
    expect(html).toContain('href="/admin/history/warranty/42"');
    expect(html).toContain("Actual saved customer answer");
    expect(html).toContain("Legacy reply / enrichment");
    expect(html).not.toContain("Bot asked:");
    expect(html).not.toContain("Bot is showing:");
  });

  it("does not build an unsafe direct history path from an invalid record ID", () => {
    const html = renderToStaticMarkup(<AdminTicketDetail ticket={{ ...ticket, history_record_id: "../unsafe" }} turns={[]} evidence={[]} />);
    expect(html).toContain('href="/admin/history?source=warranty"');
    expect(html).not.toContain("../unsafe");
  });
});
