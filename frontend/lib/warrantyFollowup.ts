export interface WarrantyFollowupAccess {
  sessionId: string;
  ticketId: string;
  token: string;
  issueType: string;
}

export function readWarrantyFollowup(sessionId: string): WarrantyFollowupAccess | null {
  if (typeof window === "undefined") return null;
  try {
    const value = JSON.parse(sessionStorage.getItem(`warranty_followup:${sessionId}`) || "null");
    if (value?.sessionId !== sessionId || typeof value.ticketId !== "string" ||
      typeof value.token !== "string" || typeof value.issueType !== "string") return null;
    return value;
  } catch {
    return null;
  }
}

export function saveWarrantyFollowup(value: WarrantyFollowupAccess): void {
  if (typeof window === "undefined") return;
  try {
    // Store only this session's temporary capability, never contact details.
    sessionStorage.setItem(`warranty_followup:${value.sessionId}`, JSON.stringify(value));
  } catch {
    // Feedback still works in this page when browser storage is unavailable.
  }
}
