import { describe, expect, it } from "vitest";
import {
  assistantContentFromResponse,
  hydrationAssistantContent,
} from "./warrantyHydration";
import type { WarrantySessionResponse, WarrantyTicketState } from "./types";

const midFlowTicket: WarrantyTicketState = {
  ticket_id: "t-1",
  status: "in_progress",
  issue_type: "defect",
  model_name: "Maestro",
  current_node: {
    node_id: "defect_problem_type",
    node_type: "question",
    prompt: "Which part is affected?",
    is_terminal: false,
    options: [{ answer_key: "air", label: "Air" }],
  },
};

describe("hydrationAssistantContent", () => {
  it("does not append terminal contact or email text to a live question", () => {
    const content = assistantContentFromResponse(midFlowTicket, {
      assistant_message: undefined,
      terminal_enrichment: null,
    });
    expect(content).toBe("Which part is affected?");
    expect(content).not.toContain("final step");
    expect(content).not.toContain("888-848-2630");
  });

  it("prefers assistant_message on non-terminal refresh", () => {
    const resp: WarrantySessionResponse = {
      session_id: "s-1",
      ticket: midFlowTicket,
      assistant_message: "Freshdesk tip\n\nWhich part is affected?",
    };
    expect(hydrationAssistantContent(midFlowTicket, resp)).toBe(
      "Freshdesk tip\n\nWhich part is affected?"
    );
  });

  it("falls back to node prompt when assistant_message is empty", () => {
    const resp: WarrantySessionResponse = {
      session_id: "s-1",
      ticket: midFlowTicket,
    };
    expect(hydrationAssistantContent(midFlowTicket, resp)).toBe(
      "Which part is affected?"
    );
  });

  it("uses terminal enrichment on terminal nodes", () => {
    const terminalTicket: WarrantyTicketState = {
      ...midFlowTicket,
      current_node: {
        node_id: "terminal_x",
        node_type: "terminal",
        prompt: "Email us.",
        is_terminal: true,
        options: [],
      },
    };
    const resp: WarrantySessionResponse = {
      session_id: "s-1",
      ticket: terminalTicket,
      assistant_message: "Try these steps first.",
    };
    expect(hydrationAssistantContent(terminalTicket, resp)).toBe(
      "Try these steps first."
    );
  });

  it("does not append warranty contact text to a sales handoff", () => {
    const salesTicket: WarrantyTicketState = {
      ...midFlowTicket,
      status: "sales_handoff",
      current_node: {
        node_id: "sales_routing",
        node_type: "terminal",
        prompt: "How can I help you today?",
        is_terminal: true,
        options: [],
      },
    };
    const content = assistantContentFromResponse(salesTicket, {
      assistant_message: undefined,
      terminal_enrichment: null,
    });
    expect(content).toBe("How can I help you today?");
    expect(content).not.toContain("888-848-2630");
    expect(content).not.toContain("final step");
  });
});
