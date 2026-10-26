import { describe, expect, it } from "vitest";
import {
  assistantContentFromResponse,
  hydrationAssistantContent,
  hydrationMessages,
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

describe("saved conversation hydration", () => {
  it("restores user answers and exact saved replies instead of replacing them with the current question", () => {
    const conversation = [
      { id: 11, role: "user" as const, content: "The left airbag stopped inflating." },
      { id: 12, role: "assistant" as const, content: "Check the left hose first, then tell me what happens." },
      { id: 13, role: "user" as const, content: "The hose is connected." },
      { id: 14, role: "assistant" as const, content: "Thanks for checking. Which part is affected?" },
    ];
    const restored = hydrationMessages({ session_id: "saved", ticket: midFlowTicket, conversation_messages: conversation });
    expect(restored).toEqual(conversation);
    expect(restored.filter((message) => message.content.includes("Which part is affected?"))).toHaveLength(1);
    expect(restored.every((message) => !message.animate)).toBe(true);
  });

  it("keeps an unanswered saved turn and adds the current question when no assistant reply was recorded", () => {
    const saved = { id: 21, role: "user" as const, content: "It is the left side." };
    expect(hydrationMessages({ session_id: "partial", ticket: midFlowTicket, conversation_messages: [saved] })).toEqual([
      saved,
      { role: "assistant", content: "Which part is affected?" },
    ]);
  });

  it("retains a usable current prompt when an older backend has no saved-conversation field", () => {
    expect(hydrationMessages({ session_id: "legacy", ticket: midFlowTicket })).toEqual([
      { role: "assistant", content: "Which part is affected?" },
    ]);
  });

  it("does not add warranty instructions after a saved handoff reply", () => {
    const messages = [{ id: 31, role: "assistant" as const, content: "Your request for a person is saved. Call our team for assistance." }];
    expect(hydrationMessages({ session_id: "handoff", ticket: { ...midFlowTicket, status: "sales_handoff" }, conversation_messages: messages })).toEqual(messages);
  });
});
