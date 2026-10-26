import type { WarrantyPrivacyState, WarrantyTicketState, WarrantyTroubleshootingResponse } from "./types";

/** A current server session must confirm consent and the saved contact email. */
export function canAccessWarrantyHelp(consentAccepted: boolean, emailProvided: boolean, privacy?: WarrantyPrivacyState | null): boolean {
  if (!consentAccepted || !privacy?.consent_recorded) return false;
  return privacy.email_required_before_help === false || (
    emailProvided && privacy.email_gate_status === "provided" && privacy.contact_email_available
  );
}

export function canSaveWarrantyProgress(ticket: WarrantyTicketState | null): boolean {
  return Boolean(ticket?.ticket_id && ticket.status !== "resolved" && ticket.status !== "sales_handoff");
}

/** Administrative status and confirmed fixes come from the response, not a button label. */
export function applyWarrantyOutcome(ticket: WarrantyTicketState, result: WarrantyTroubleshootingResponse): WarrantyTicketState {
  return {
    ...ticket,
    status: result.status,
    admin_decision: result.self_service_resolved ? "self_resolved" : result.service_result ? "delivery_answered" : ticket.admin_decision,
    troubleshooting_outcome: result.outcome,
    followup_token: result.followup_token ?? ticket.followup_token,
  };
}
