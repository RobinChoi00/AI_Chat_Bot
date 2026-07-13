import { describe, expect, it } from "vitest";
import {
  formatEnrichmentSource,
  hasStepEnrichmentPanel,
  hydrationAssistantContent,
} from "./warrantyHydration";
import type { WarrantySessionResponse, WarrantyTicketState } from "./types";

const midFlowTicket: WarrantyTicketState = {
  ticket_id: "t-1",
  session_id: "s-1",
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
  it("prefers assistant_message on non-terminal refresh", () => {
    const resp: WarrantySessionResponse = {
      session_id: "s-1",
      ticket: midFlowTicket,
      assistant_message: "Freshdesk tip\n\nWhich part is affected?",
      step_enrichment: { phase: "workflow_step", tips: ["Check power"] },
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
});

describe("hasStepEnrichmentPanel", () => {
  it("returns true when tips or sources exist", () => {
    expect(
      hasStepEnrichmentPanel({
        phase: "workflow_step",
        tips: ["Unplug for 30 seconds"],
        sources: ["freshdesk"],
      })
    ).toBe(true);
  });

  it("returns false for empty enrichment", () => {
    expect(hasStepEnrichmentPanel(null)).toBe(false);
    expect(hasStepEnrichmentPanel({ phase: "workflow_step" })).toBe(false);
  });
});

describe("formatEnrichmentSource", () => {
  it("maps known sources to friendly labels", () => {
    expect(formatEnrichmentSource("freshdesk")).toBe("Past support ticket");
    expect(formatEnrichmentSource("freshdesk_kb")).toBe("Knowledge base");
  });
});
