import { v4 as uuidv4 } from "uuid";
import { sendCustomerJourneyEvent } from "./api";

export type JourneyStage = "consent" | "email" | "model" | "issue" | "help" | "outcome" | "evidence" | "handoff" | "delivery" | "resume" | "response" | "recommendation" | "lead";
export type JourneyEvent = "shown" | "completed" | "failed" | "resumed" | "requested" | "skipped";

/** Browser-reported UI events; never include customer content or identity fields. */
export function reportCustomerJourney({ consentAccepted, sessionId, domain, stage, event }: {
  consentAccepted: boolean;
  sessionId: string;
  domain: string;
  stage: JourneyStage;
  event: JourneyEvent;
}): void {
  if (!consentAccepted) return;
  void sendCustomerJourneyEvent({ event_id: uuidv4(), session_id: sessionId, domain, stage, event }).catch(() => {
    // Measurement cannot block help, and must not log customer/session content.
  });
}
