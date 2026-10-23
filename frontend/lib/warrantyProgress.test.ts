import { describe, expect, it } from "vitest";
import { applyWarrantyOutcome, canAccessWarrantyHelp, canSaveWarrantyProgress } from "./warrantyProgress";
import type { WarrantyPrivacyState, WarrantyTicketState } from "./types";

const privacy: WarrantyPrivacyState = { consent_required: true, consent_recorded: true, email_required_before_help: false, email_gate_status: "pending", contact_email_available: false };
const ticket: WarrantyTicketState = { ticket_id: "case", status: "awaiting_evidence", issue_type: "delivery", model_name: "", current_node: { node_id: "delivery_terminal", node_type: "terminal", prompt: "Review", is_terminal: true, options: [] } };

describe("warranty progress boundaries", () => {
  it("permits basic help without email only after consent and an explicit server mode", () => {
    expect(canAccessWarrantyHelp(true, false, privacy)).toBe(true);
    expect(canAccessWarrantyHelp(false, true, privacy)).toBe(false);
    expect(canAccessWarrantyHelp(true, true, { ...privacy, consent_recorded: false })).toBe(false);
    expect(canAccessWarrantyHelp(true, false)).toBe(false);
    expect(canAccessWarrantyHelp(true, false, { ...privacy, email_required_before_help: true })).toBe(false);
    expect(canAccessWarrantyHelp(true, true, { ...privacy, email_required_before_help: true })).toBe(true);
  });

  it("keeps deferred cases open and distinct from customer-confirmed fixes", () => {
    const updated = applyWarrantyOutcome(ticket, { ticket_id: ticket.ticket_id, outcome: "deferred", status: "awaiting_evidence", self_service_resolved: false, deferred: true });
    expect(updated.status).toBe("awaiting_evidence");
    expect(updated.admin_decision).not.toBe("self_resolved");
    expect(updated.troubleshooting_outcome).toBe("deferred");
    expect(canSaveWarrantyProgress(updated)).toBe(true);
  });

  it("uses service-result confirmation without inventing a repaired-chair outcome", () => {
    const updated = applyWarrantyOutcome(ticket, { ticket_id: ticket.ticket_id, outcome: "delivery_answered", status: "resolved", self_service_resolved: false, service_result: true });
    expect(updated.admin_decision).toBe("delivery_answered");
    expect(updated.admin_decision).not.toBe("self_resolved");
    expect(canSaveWarrantyProgress(updated)).toBe(false);
  });

  it("offers resume for evidence and review states, and not for closed or sales handoffs", () => {
    expect(canSaveWarrantyProgress(ticket)).toBe(true);
    expect(canSaveWarrantyProgress({ ...ticket, status: "awaiting_admin_review" })).toBe(true);
    expect(canSaveWarrantyProgress({ ...ticket, status: "sales_handoff" })).toBe(false);
    expect(canSaveWarrantyProgress(null)).toBe(false);
  });
});
