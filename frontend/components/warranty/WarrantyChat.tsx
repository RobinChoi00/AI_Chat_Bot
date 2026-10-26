"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  confirmWarrantyModel,
  getWarrantySession,
  goBackFromWarrantyContact,
  goBackWarranty,
  lookupDeliveryTracking,
  quickStartWarranty,
  naturalStartWarranty,
  recordWarrantyChatConsent,
  recordWarrantySessionContactEmail,
  requestWarrantyHandoff,
  registerWarrantyModel,
  restartWarrantySession,
  resumeWarrantyFromToken,
  smartStartWarranty,
  submitCustomerNote,
  submitTroubleshootingOutcome,
  submitWarrantyAnswer,
} from "@/lib/api";
import type {
  AnswerOption,
  ChatMessage,
  TerminalEnrichment,
  TroubleshootingOutcome,
  WarrantySessionResponse,
  WarrantyTicketState,
  TrackingSummary,
  WarrantyHandoff,
  WarrantyPrivacyState,
} from "@/lib/types";
import {
  assistantContentFromResponse,
  hydrationAssistantContent,
} from "@/lib/warrantyHydration";
import ChatMessageBubble from "./ChatMessageBubble";
import AnswerOptions from "./AnswerOptions";
import CollapsibleOptionPanel from "./CollapsibleOptionPanel";
import EvidenceUploader from "./EvidenceUploader";
import SaveProgressButton from "./SaveProgressButton";
import SerialPhotoButton from "./SerialPhotoButton";
import TicketStatusBadge from "./TicketStatusBadge";
import TroubleshootingGate, { resolutionGateLabels } from "./TroubleshootingGate";
import {
  CHAT_CONSENT_STORAGE_KEY,
  CHAT_CONTACT_EMAIL_STORAGE_KEY,
  CHAT_EMAIL_GATE_STORAGE_KEY,
  resolveStorePolicyUrls,
  WARRANTY_WELCOME_MESSAGE,
} from "@/lib/welcomeMessage";
import ChatRecordingNoticeBanner from "./ChatRecordingNoticeBanner";
import ChatEmailGate from "./ChatEmailGate";
import WarrantyCaseLookup from "./WarrantyCaseLookup";
import WarrantyTeamContactFooter from "./WarrantyTeamContactFooter";
import OutcomeFollowup from "./OutcomeFollowup";
import DeliveryTrackingPanel, { type DeliveryLookupIdentity } from "./DeliveryTrackingPanel";
import HumanHandoffNotice from "./HumanHandoffNotice";
import { readWarrantyFollowup, saveWarrantyFollowup, type WarrantyFollowupAccess } from "@/lib/warrantyFollowup";

import { resolveWarrantyStoreDomain } from "@/lib/warrantyStoreDomain";
import { buildWarrantySubmissionMessage } from "@/lib/warrantyContact";
import { applyWarrantyOutcome, canAccessWarrantyHelp, canSaveWarrantyProgress } from "@/lib/warrantyProgress";
import { reportCustomerJourney, type JourneyEvent, type JourneyStage } from "@/lib/customerJourney";

// Short pause before the assistant bubble. Kept small because ChatMessageBubble
// already types the response out.
const THINKING_DELAY_MS = 700;

function assistantMessage(content: string): ChatMessage {
  return { role: "assistant", content, animate: true };
}

const SELF_HELP_CLOSING =
  "Great — this request is now closed as self-resolved. If you still need us, you can start a new case anytime. " +
  "Our warranty team is also available by phone during business hours.";

function selfHelpClosing(issueType?: string | null): string {
  if (issueType === "installation") {
    return (
      "Great — glad the chair is set up. If you still need install help, you can start a new case anytime. " +
      "Our warranty team is also available by phone during business hours."
    );
  }
  if (issueType === "delivery") {
    return (
      "No problem — come back anytime when you have the photos or delivery paperwork. " +
      "Our warranty team is also available by phone during business hours."
    );
  }
  return SELF_HELP_CLOSING;
}

const DEFECT_MODEL_PROMPT =
  "To troubleshoot warranty defects accurately, please **enter your chair model** in the box below (for example OS-4000T or 3D LTX). I'll continue with the defect questions right after.";

const INITIAL_ISSUE_OPTIONS: AnswerOption[] = [
  { answer_key: "installation", label: "Setup & installation" },
  { answer_key: "delivery", label: "Delivery & tracking" },
  { answer_key: "defect", label: "Chair malfunction" },
];

const ISSUE_TYPE_KEYS = new Set(["installation", "delivery", "defect"]);

function resolveSuggestedIssueType(
  resp: Pick<WarrantySessionResponse, "smart_start" | "suggested_issue_type">
): string | null {
  const raw =
    resp.smart_start?.suggested_issue_type ||
    resp.smart_start?.routing_confirmation?.inferred_issue_type ||
    resp.suggested_issue_type ||
    "";
  const key = String(raw).trim().toLowerCase();
  return ISSUE_TYPE_KEYS.has(key) ? key : null;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Drop the last user message and any assistant replies after it. */
function trimMessagesBeforeLastUser(messages: ChatMessage[]): ChatMessage[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      return messages.slice(0, i);
    }
  }
  return messages;
}

export default function WarrantyChat({
  embed = false,
  storeDomain: storeDomainProp,
}: {
  embed?: boolean;
  storeDomain?: string;
}) {
  const storeDomain = storeDomainProp?.trim() || resolveWarrantyStoreDomain();
  const [sessionId, setSessionId] = useState<string>(() => {
    if (typeof window === "undefined") return uuidv4();
    const stored = sessionStorage.getItem("warranty_session_id");
    if (stored) return stored;
    const newId = uuidv4();
    sessionStorage.setItem("warranty_session_id", newId);
    return newId;
  });
  const [resumeStatus, setResumeStatus] = useState<
    "idle" | "resuming" | "resumed" | "failed"
  >(() => {
    if (typeof window === "undefined") return "idle";
    return new URLSearchParams(window.location.search).has("resume")
      ? "resuming"
      : "idle";
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warrantyState, setWarrantyState] = useState<WarrantyTicketState | null>(null);
  const journeyDomain = warrantyState?.domain?.trim() || storeDomain;
  const deliveryStoreDomain = warrantyState?.domain?.trim() || storeDomain;
  const [savedFollowup, setSavedFollowup] = useState<WarrantyFollowupAccess | null>(() => readWarrantyFollowup(sessionId));
  const rememberFollowup = useCallback((ticketId: string, token: string, issueType: string) => {
    const value = { sessionId, ticketId, token, issueType };
    setSavedFollowup(value);
    saveWarrantyFollowup(value);
  }, [sessionId]);
  const [terminalEnrichment, setTerminalEnrichment] = useState<TerminalEnrichment | null>(null);
  const [trackingResult, setTrackingResult] = useState<{ sessionId: string; ticketId: string; domain: string; summary: TrackingSummary } | null>(null);
  const trackingSummary = trackingResult?.sessionId === sessionId && trackingResult.ticketId === warrantyState?.ticket_id && trackingResult.domain === deliveryStoreDomain ? trackingResult.summary : null;
  const [handoff, setHandoff] = useState<WarrantyHandoff | null>(null);
  const [optionsUsed, setOptionsUsed] = useState(false);
  const [sessionChecked, setSessionChecked] = useState(false);
  const [contactSubmitted, setContactSubmitted] = useState(false);
  const [contactReceiptSent, setContactReceiptSent] = useState(false);
  const [showCaseLookup, setShowCaseLookup] = useState(() => {
    if (typeof window === "undefined") return false;
    return new URLSearchParams(window.location.search).has("check");
  });
  const [helpConsent, setHelpConsent] = useState<"yes" | "no" | null>(null);
  const [resolutionStage, setResolutionStage] = useState<"review" | "outcome">("review");
  const [deferredPaused, setDeferredPaused] = useState(false);
  const [optionsPanelExpanded, setOptionsPanelExpanded] = useState(true);
  const [issueTypePanelExpanded, setIssueTypePanelExpanded] = useState(true);
  const [emailPanelCollapsed, setEmailPanelCollapsed] = useState(false);
  const [pendingDefectStart, setPendingDefectStart] = useState<string | null>(null);
  const [suggestedIssueType, setSuggestedIssueType] = useState<string | null>(null);
  const [suggestedOption, setSuggestedOption] = useState<AnswerOption | null>(null);
  const [chatConsentAccepted, setChatConsentAccepted] = useState(() => {
    if (typeof window === "undefined") return false;
    return sessionStorage.getItem(CHAT_CONSENT_STORAGE_KEY) === "1";
  });
  const [emailGateDone, setEmailGateDone] = useState(() => {
    if (typeof window === "undefined") return false;
    const status = sessionStorage.getItem(CHAT_EMAIL_GATE_STORAGE_KEY);
    return status === "provided";
  });
  const [intakeContactEmail, setIntakeContactEmail] = useState("");
  const [privacy, setPrivacy] = useState<WarrantyPrivacyState | null>(null);
  const [consentSaving, setConsentSaving] = useState(false);
  const [consentError, setConsentError] = useState<string | null>(null);
  const helpAccessReady = sessionChecked && canAccessWarrantyHelp(chatConsentAccepted, emailGateDone, privacy);
  const journeySeen = useRef(new Set<string>());
  const reportJourney = useCallback((stage: JourneyStage, event: JourneyEvent, instance = "session") => {
    if (!chatConsentAccepted || !sessionChecked) return;
    const key = `${sessionId}:${stage}:${event}:${instance}`;
    if (journeySeen.current.has(key)) return;
    journeySeen.current.add(key);
    reportCustomerJourney({ consentAccepted: true, sessionId, domain: journeyDomain, stage, event });
  }, [chatConsentAccepted, sessionChecked, sessionId, journeyDomain]);

  // Privacy: never keep contact email in sessionStorage (shared devices / XSS surface).
  useEffect(() => {
    if (typeof window === "undefined") return;
    sessionStorage.removeItem(CHAT_CONTACT_EMAIL_STORAGE_KEY);
  }, []);

  const acceptChatConsent = useCallback(async () => {
    if (consentSaving) return;
    setConsentSaving(true);
    setConsentError(null);
    const { storeDomain: policyStore } = resolveStorePolicyUrls(storeDomain);
    try {
      const response = await recordWarrantyChatConsent(sessionId, storeDomain, policyStore);
      if (!response.consent_recorded) throw new Error("Your consent was not saved. Please try again.");
      if (response.privacy) setPrivacy(response.privacy);
      setChatConsentAccepted(true);
      if (typeof window !== "undefined") sessionStorage.setItem(CHAT_CONSENT_STORAGE_KEY, "1");
      reportCustomerJourney({ consentAccepted: true, sessionId, domain: journeyDomain, stage: "consent", event: "completed" });
    } catch (err) {
      setConsentError(err instanceof Error ? err.message : "Could not save consent. Please try again.");
    } finally {
      setConsentSaving(false);
    }
  }, [consentSaving, sessionId, storeDomain, journeyDomain]);

  const completeEmailGate = useCallback(
    async (options: { email: string }) => {
      const email = (options.email || "").trim();
      try {
        const response = await recordWarrantySessionContactEmail(sessionId, {
          customerEmail: email,
          skipped: false,
        });
        if (!response.recorded || response.email_gate_status !== "provided" ||
            !canAccessWarrantyHelp(true, true, response.privacy)) {
          throw new Error("Your email was not saved. Please try again.");
        }
        if (response.privacy) setPrivacy(response.privacy);
      } catch (err) {
        console.warn("warranty session contact email failed", err);
        throw err;
      }
      if (typeof window !== "undefined") {
        sessionStorage.setItem(CHAT_EMAIL_GATE_STORAGE_KEY, "provided");
        sessionStorage.removeItem(CHAT_CONTACT_EMAIL_STORAGE_KEY);
      }
      setIntakeContactEmail(email);
      setEmailGateDone(true);
      reportJourney("email", "completed");
      inputRef.current?.focus();
    },
    [sessionId, reportJourney]
  );

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, helpConsent, resolutionStage]);

  const applySessionResponse = useCallback((resp: WarrantySessionResponse) => {
    if (resp.privacy) {
      setPrivacy(resp.privacy);
      setChatConsentAccepted(resp.privacy.consent_recorded);
      setEmailGateDone(resp.privacy.email_gate_status === "provided");
    }
    const token = resp.followup_token ?? resp.ticket?.followup_token;
    if (resp.ticket?.status === "resolved" && token) {
      rememberFollowup(resp.ticket.ticket_id, token, resp.ticket.issue_type);
    }
    setWarrantyState(resp.ticket ? {
      ...resp.ticket,
      followup_token: resp.followup_token ?? resp.ticket.followup_token,
    } : null);
    setTerminalEnrichment(resp.terminal_enrichment ?? null);
    setHandoff(resp.handoff ?? null);
    setContactSubmitted(Boolean(resp.ticket?.contact_submitted || resp.ticket?.review_requested));
    setContactReceiptSent(Boolean(resp.ticket?.receipt_email_sent));
    setDeferredPaused(resp.ticket?.troubleshooting_outcome === "deferred");
    if (resp.tracking_summary && resp.ticket) {
      setTrackingResult({ sessionId: resp.session_id, ticketId: resp.ticket.ticket_id, domain: resp.ticket.domain?.trim() || storeDomain, summary: resp.tracking_summary });
    } else if (resp.tracking_summary !== undefined || !resp.ticket) {
      setTrackingResult(null);
    }
    if (resp.suggested_option?.answer_key) {
      setSuggestedOption(resp.suggested_option);
    } else if (!resp.side_question) {
      setSuggestedOption(null);
    }
    if (resp.terminal_enrichment?.phase === "awaiting_help_consent") {
      const outcome = resp.ticket?.troubleshooting_outcome;
      if (outcome === "steps_completed") {
        setResolutionStage("outcome");
        setHelpConsent(null);
      } else if (outcome === "unresolved" || outcome === "unable_to_attempt") {
        setResolutionStage("outcome");
        setHelpConsent("yes");
      } else if (outcome === "resolved" || outcome === "delivery_answered") {
        setResolutionStage("outcome");
        setHelpConsent("no");
      } else {
        setResolutionStage("review");
        setHelpConsent(null);
      }
    }
    setOptionsUsed(false);
    return resp;
  }, [rememberFollowup, storeDomain]);

  const refreshWarrantyState = useCallback(async () => {
    const resp = await getWarrantySession(sessionId);
    return applySessionResponse(resp);
  }, [sessionId, applySessionResponse]);

  const restartSession = useCallback(async () => {
    if (loading) return;
    if (typeof window !== "undefined") {
      const ok = window.confirm(
        "Start over? Your current answers will be cleared and a new case will begin."
      );
      if (!ok) return;
    }
    setError(null);
    setLoading(true);
    try {
      try {
        await restartWarrantySession(sessionId, storeDomain);
      } catch {
        // Even if the server call fails (offline / 404 on older backend),
        // we still reset the client so the user is not stuck.
      }
      const newId = uuidv4();
      if (typeof window !== "undefined") {
        sessionStorage.setItem("warranty_session_id", newId);
      }
      setSessionId(newId);
      setMessages([{ role: "assistant", content: WARRANTY_WELCOME_MESSAGE }]);
      setWarrantyState(null);
      setTerminalEnrichment(null);
      setTrackingResult(null);
      setHandoff(null);
      setSuggestedIssueType(null);
      setPendingDefectStart(null);
      setSuggestedOption(null);
      setHelpConsent(null);
      setResolutionStage("review");
      setDeferredPaused(false);
      setContactSubmitted(false);
      setContactReceiptSent(false);
      setOptionsUsed(false);
      setEmailPanelCollapsed(false);
      setInput("");
      if (typeof window !== "undefined") {
        sessionStorage.removeItem(CHAT_CONSENT_STORAGE_KEY);
        sessionStorage.removeItem(CHAT_EMAIL_GATE_STORAGE_KEY);
        sessionStorage.removeItem(CHAT_CONTACT_EMAIL_STORAGE_KEY);
      }
      setChatConsentAccepted(false);
      setEmailGateDone(false);
      setIntakeContactEmail("");
      setPrivacy(null);
      setConsentError(null);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading, sessionId, storeDomain]);

  const goBackOneStep = useCallback(async () => {
    const ticketId = warrantyState?.ticket_id;
    if (!ticketId || loading || !warrantyState?.can_go_back) return;

    setError(null);
    setLoading(true);
    setPendingDefectStart(null);
    setHelpConsent(null);
    setResolutionStage("review");

    try {
      const resp = await goBackWarranty(ticketId);
      applySessionResponse(resp);
      setOptionsUsed(false);
      setContactSubmitted(false);
      setEmailPanelCollapsed(false);

      const trimmed = trimMessagesBeforeLastUser(messages);
      setMessages(trimmed);

      await sleep(THINKING_DELAY_MS);
      const ticket = resp.ticket;
      let content: string | null = null;
      if (ticket?.ready_for_issue_type && ticket.model_name) {
        content =
          `Great — I have **${ticket.model_name}** on file.\n\n` +
          "What type of issue can we help you with? Please choose an option below.";
      } else {
        content = hydrationAssistantContent(ticket, resp);
      }
      if (content) {
        setMessages((prev) => [...prev, assistantMessage(content)]);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Could not go back.";
      setError(msg);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [
    warrantyState?.ticket_id,
    warrantyState?.can_go_back,
    loading,
    applySessionResponse,
    messages,
  ]);

  const goBackFromEmailStep = useCallback(async () => {
    const ticketId = warrantyState?.ticket_id;
    if (!ticketId || loading) return;

    setError(null);
    setLoading(true);
    try {
      const restored = await goBackFromWarrantyContact(ticketId);
      setHelpConsent(null);
      setResolutionStage(restored.stage);
      setEmailPanelCollapsed(false);
      setWarrantyState((previous) =>
        previous
          ? {
              ...previous,
              troubleshooting_outcome: restored.outcome,
            }
          : previous
      );
      setMessages((previous) => trimMessagesBeforeLastUser(previous));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not go back.");
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading, warrantyState?.ticket_id]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const token = params.get("resume");
    if (!token) return;

    let cancelled = false;
    (async () => {
      try {
        const data = await resumeWarrantyFromToken(token);
        if (cancelled) return;
        sessionStorage.setItem("warranty_session_id", data.session_id);
        setIntakeContactEmail("");
        setSessionId(data.session_id);
        setResumeStatus("resumed");
      } catch (err) {
        if (cancelled) return;
        console.warn("warranty resume failed", err);
        setResumeStatus("failed");
      } finally {
        try {
          params.delete("resume");
          const q = params.toString();
          const cleanPath =
            window.location.pathname + (q ? `?${q}` : "") + window.location.hash;
          window.history.replaceState({}, "", cleanPath);
        } catch {
          // ignore — pushing state is best-effort
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    // Wait until any ?resume= handoff finishes so we don't hydrate against
    // the stale local sessionId before the URL token remaps us.
    if (resumeStatus === "resuming") return;

    let cancelled = false;
    (async () => {
      try {
        const resp = await refreshWarrantyState();
        if (cancelled) return;
        const ticket = resp.ticket;
        if (resp.handoff?.requested || (ticket?.current_node?.is_terminal && ticket.current_node.prompt)) {
          const content = hydrationAssistantContent(ticket, resp);
          if (content) {
            setMessages([{ role: "assistant", content }]);
          }
        } else if (ticket?.ready_for_issue_type && ticket.model_name) {
          setMessages([
            { role: "assistant", content: WARRANTY_WELCOME_MESSAGE },
            {
              role: "assistant",
              content: `Great — I have **${ticket.model_name}** on file.\n\nWhat type of issue can we help you with? Please choose an option below.`,
            },
          ]);
        } else if (ticket?.current_node?.prompt && !ticket.current_node.is_terminal) {
          const content = hydrationAssistantContent(ticket, resp);
          if (content) {
            setMessages([{ role: "assistant", content }]);
          }
        } else if (!ticket?.ticket_id) {
          setMessages([{ role: "assistant", content: WARRANTY_WELCOME_MESSAGE }]);
        }
      } catch {
        // Non-fatal
      } finally {
        if (!cancelled) setSessionChecked(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshWarrantyState, resumeStatus]);

  const workflowOptionCount = warrantyState?.current_node?.options?.length ?? 0;
  const workflowNodeId = warrantyState?.current_node?.node_id ?? "";
  const isErrorCodeGateNode = workflowNodeId.startsWith("defect_error_code");

  useEffect(() => {
    const handle = window.setTimeout(() => {
      if (suggestedOption?.answer_key || workflowOptionCount >= 6) {
        setOptionsPanelExpanded(false);
      } else if (workflowOptionCount > 0) {
        setOptionsPanelExpanded(true);
      }
    }, 0);
    return () => window.clearTimeout(handle);
  }, [workflowNodeId, workflowOptionCount, suggestedOption?.answer_key]);

  const appendAssistantFromResponse = useCallback(
    async (
      ticket: WarrantyTicketState | null,
      resp: Pick<WarrantySessionResponse, "assistant_message" | "terminal_enrichment" | "handoff">,
      options?: { skipDelay?: boolean }
    ) => {
      if (!options?.skipDelay) {
        await sleep(THINKING_DELAY_MS);
      }
      const content = assistantContentFromResponse(ticket, resp);
      if (!content) return;
      setMessages((prev) => [...prev, assistantMessage(content)]);
    },
    []
  );

  const promptModelForDefect = useCallback((label: string) => {
    setPendingDefectStart(label);
    setOptionsUsed(false);
    setMessages((prev) => [...prev, assistantMessage(DEFECT_MODEL_PROMPT)]);
  }, []);

  const handleQuickStart = useCallback(
    async (issueType: "installation" | "delivery" | "defect", label: string) => {
      if (loading || !helpAccessReady) return;
      setSuggestedIssueType(null);
      if (issueType !== "defect") setPendingDefectStart(null);
      if (issueType === "defect" && !warrantyState?.model_name?.trim()) {
        setError(null);
        setMessages((prev) => [...prev, { role: "user", content: label }]);
        promptModelForDefect(label);
        return;
      }
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setHelpConsent(null);

      try {
        const resp = await quickStartWarranty(sessionId, issueType, storeDomain);
        applySessionResponse(resp);
        if (!resp.handoff?.requested && resp.ticket?.issue_type === issueType) reportJourney("issue", "completed", issueType);
        setMessages((prev) => [...prev, { role: "user", content: label }]);
        await appendAssistantFromResponse(resp.ticket, resp);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setOptionsUsed(false);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [loading, helpAccessReady, sessionId, storeDomain, applySessionResponse, appendAssistantFromResponse, warrantyState?.model_name, promptModelForDefect, reportJourney]
  );

  const handleDeliveryLookup = useCallback(async (identity: DeliveryLookupIdentity) => {
    if (loading || !helpAccessReady) return;
    setLoading(true);
    setTrackingResult(null);
    reportJourney("delivery", "requested", `lookup-${Date.now()}`);
    try {
      const resp = await lookupDeliveryTracking(sessionId, { ...identity, domain: deliveryStoreDomain });
      applySessionResponse(resp);
      if (!resp.tracking_summary) throw new Error("A current delivery update was not returned. Please try again.");
      reportJourney("delivery", resp.tracking_summary.available ? "completed" : "failed", `lookup-${Date.now()}`);
    } catch (error) {
      reportJourney("delivery", "failed", `lookup-${Date.now()}`);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [loading, helpAccessReady, sessionId, deliveryStoreDomain, applySessionResponse, reportJourney]);

  const requestHumanSupport = useCallback(async () => {
    if (loading || !helpAccessReady) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await requestWarrantyHandoff(sessionId, { domain: storeDomain, reason: "customer_request" });
      applySessionResponse(resp);
      reportJourney("handoff", "requested");
      setMessages((previous) => [...previous, { role: "user", content: "I'd like to speak with a person." }]);
      await appendAssistantFromResponse(resp.ticket, resp, { skipDelay: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save your support request. Please use the support contact details below.");
    } finally {
      setLoading(false);
    }
  }, [loading, helpAccessReady, sessionId, storeDomain, applySessionResponse, appendAssistantFromResponse, reportJourney]);

  const advanceWarranty = useCallback(
    async (answer: string, userLabel: string) => {
      const ticketId = warrantyState?.ticket_id;
      if (!ticketId || loading) return;

      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setHelpConsent(null);
      setMessages((prev) => [...prev, { role: "user", content: userLabel }]);

      try {
        const resp = await submitWarrantyAnswer(ticketId, answer);
        applySessionResponse(resp);
        if (!resp.side_question && !resp.handoff?.requested && warrantyState?.current_node?.node_id === "issue_type" && resp.ticket?.issue_type) reportJourney("issue", "completed", resp.ticket.issue_type);
        if (resp.order_cancel_escalated) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(
              resp.assistant_message ||
                "Your request is saved for team review. You can use the support contact details below for direct assistance."
            ),
          ]);
          setHelpConsent("yes");
          setContactSubmitted(true);
          setOptionsUsed(true);
          return;
        }
        if (resp.side_question && resp.assistant_message) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(resp.assistant_message!),
          ]);
          setOptionsUsed(false);
          return;
        }
        if (resp.tracking_summary?.message) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(resp.tracking_summary!.message),
          ]);
          await appendAssistantFromResponse(resp.ticket, resp, { skipDelay: true });
        } else {
          await appendAssistantFromResponse(resp.ticket, resp);
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setOptionsUsed(false);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [warrantyState, loading, applySessionResponse, appendAssistantFromResponse, reportJourney]
  );

  const startViaSmartIntake = useCallback(
    async (text: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setHelpConsent(null);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setInput("");

      try {
        let resp: WarrantySessionResponse;
        try {
          resp = await smartStartWarranty(sessionId, text, storeDomain);
        } catch (smartErr: unknown) {
          const smartMsg =
            smartErr instanceof Error ? smartErr.message.toLowerCase() : "";
          if (smartMsg.includes("404") || smartMsg.includes("405")) {
            resp = await naturalStartWarranty(sessionId, text, storeDomain);
          } else {
            throw smartErr;
          }
        }

        const smart = resp.smart_start;
        const routingConfirm = smart?.routing_confirmation;
        const needsIntentConfirm = Boolean(
          routingConfirm?.requires_confirmation || routingConfirm?.message
        );
        setSuggestedIssueType(resolveSuggestedIssueType(resp));

        applySessionResponse(resp);

        if (resp.handoff?.requested) {
          await appendAssistantFromResponse(resp.ticket, resp);
          return;
        }

        if (resp.order_cancel_escalated) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(
              resp.assistant_message ||
                "Your request is saved for team review. You can use the support contact details below for direct assistance."
            ),
          ]);
          setHelpConsent("yes");
          setContactSubmitted(true);
          setOptionsUsed(true);
          return;
        }

        if (resp.model_confirmation?.message) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(resp.model_confirmation!.message),
          ]);
          setOptionsUsed(false);
        } else if (needsIntentConfirm && routingConfirm?.message) {
          // Suggest an issue type, but never auto-advance — wait for a tap.
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(routingConfirm.message),
          ]);
          setOptionsUsed(false);
        } else if (
          resp.ticket?.ready_for_issue_type &&
          resp.ticket?.model_name &&
          resp.ticket?.current_node?.node_id === "issue_type"
        ) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(
              `Thanks — I have **${resp.ticket!.model_name}** on file.\n\nWhat type of issue can we help you with? Please choose an option below.`
            ),
          ]);
          setOptionsUsed(false);
        } else {
          await appendAssistantFromResponse(resp.ticket, resp);
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setOptionsUsed(false);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [loading, sessionId, storeDomain, applySessionResponse, appendAssistantFromResponse]
  );

  const confirmInferredModel = useCallback(
    async (correctedModel?: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      const label =
        correctedModel?.trim() ||
        `Yes, ${warrantyState?.model_name ?? "that's my model"}`;
      setMessages((prev) => [...prev, { role: "user", content: label }]);

      try {
        const resp = await confirmWarrantyModel(sessionId, {
          confirmed: !correctedModel?.trim(),
          model: correctedModel?.trim() || undefined,
          domain: storeDomain,
        });
        applySessionResponse(resp);
        if (resp.handoff?.requested) {
          await appendAssistantFromResponse(resp.ticket, resp);
          return;
        }
        if (resp.ticket?.model_confirmed) reportJourney("model", "completed");
        await sleep(THINKING_DELAY_MS);
        setMessages((prev) => [
          ...prev,
          assistantMessage(
            correctedModel?.trim()
              ? `Updated — we'll use **${resp.ticket?.model_name ?? correctedModel}** for this case.`
              : `Great — we'll continue with **${resp.ticket?.model_name ?? "your model"}**.`
          ),
        ]);
        await appendAssistantFromResponse(resp.ticket, resp);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [
      loading,
      sessionId,
      storeDomain,
      warrantyState?.model_name,
      applySessionResponse,
      appendAssistantFromResponse,
      reportJourney,
    ]
  );

  const submitFollowUpNote = useCallback(
    async (text: string) => {
      const ticketId = warrantyState?.ticket_id;
      if (!ticketId || loading) return;
      setError(null);
      setLoading(true);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setInput("");
      try {
        const resp = await submitCustomerNote(ticketId, text);
        if ("session_id" in resp) {
          applySessionResponse(resp);
          await appendAssistantFromResponse(resp.ticket, resp);
          return;
        }
        await sleep(THINKING_DELAY_MS);
        setMessages((prev) => [
          ...prev,
          assistantMessage("Got it — I've added that note to your case for our warranty team."),
        ]);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Could not save your note.";
        setError(msg);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [loading, warrantyState?.ticket_id, applySessionResponse, appendAssistantFromResponse]
  );

  const handleResolutionOutcome = useCallback(
    async (outcome: TroubleshootingOutcome | string, label?: string) => {
      const ticketId = warrantyState?.ticket_id;
      if (!ticketId || loading) return false;

      setError(null);
      setLoading(true);
      try {
        const resp = await submitTroubleshootingOutcome(ticketId, outcome);
        if ("session_id" in resp) {
          applySessionResponse(resp);
          if (label) setMessages((previous) => [...previous, { role: "user", content: label }]);
          await appendAssistantFromResponse(resp.ticket, resp);
          return true;
        }
        const recorded = resp.outcome;
        if (recorded === "resolved" && resp.followup_token) {
          rememberFollowup(ticketId, resp.followup_token, warrantyState?.issue_type || "");
        }
        setWarrantyState((previous) => {
          if (!previous) return previous;
          return applyWarrantyOutcome(previous, resp);
        });

        if (label) {
          setMessages((previous) => [...previous, { role: "user", content: label }]);
        }
        if (recorded === "steps_completed") {
          setResolutionStage("outcome");
          return true;
        }

        reportJourney("outcome", recorded === "deferred" ? "skipped" : "completed", recorded);
        if (recorded === "deferred") {
          setDeferredPaused(true);
          setHelpConsent(null);
          setMessages((previous) => [...previous, assistantMessage("Your progress is saved for later. You can continue here when ready, or keep a resume link below.")]);
          return true;
        }

        if (recorded === "delivery_answered") {
          setHelpConsent("no");
          setMessages((previous) => [...previous, assistantMessage("Thanks — glad the delivery update helped. You can start another request if you need more support.")]);
          return true;
        }

        if (resp.self_service_resolved) {
          setHelpConsent("no");
          setMessages((previous) => [
            ...previous,
            assistantMessage(selfHelpClosing(warrantyState?.issue_type)),
          ]);
          return true;
        }

        setHelpConsent("yes");
        setMessages((previous) => [
          ...previous,
          assistantMessage(
            recorded === "unable_to_attempt"
              ? "Please don’t attempt anything that feels unsafe. You can send a request for team review below without photos or videos."
              : "You can send this case for team review below. Photos and documents are optional."
          ),
        ]);
        return true;
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Could not save your progress.");
        reportJourney("outcome", "failed", `outcome-${Date.now()}`);
        return false;
      } finally {
        setLoading(false);
      }
    },
    [loading, warrantyState, rememberFollowup, applySessionResponse, appendAssistantFromResponse, reportJourney]
  );

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || loading || !helpAccessReady) return;

      const trimmed = text.trim();
      const atIssueTypeNode =
        !warrantyState?.issue_type &&
        warrantyState?.current_node?.node_id === "issue_type";
      const atFirstIntake =
        !warrantyState?.issue_type &&
        !atIssueTypeNode &&
        (!warrantyState?.ticket_id ||
          warrantyState?.current_node?.node_id === "root");
      const needsModelConfirmation = Boolean(warrantyState?.needs_model_confirmation);

      const atTerminal =
        !!warrantyState?.current_node?.is_terminal ||
        warrantyState?.status === "awaiting_admin_review" ||
        warrantyState?.status === "admin_reviewing";

      if (atTerminal && helpConsent === "yes" && !contactSubmitted && warrantyState?.ticket_id) {
        await submitFollowUpNote(trimmed);
        return;
      }

      if (warrantyState?.needs_customer_reply && warrantyState?.ticket_id) {
        await submitFollowUpNote(trimmed);
        await refreshWarrantyState();
        return;
      }

      if (atFirstIntake) {
        await startViaSmartIntake(trimmed);
        return;
      }

      const atIssueTypeWithoutModel =
        atIssueTypeNode && !warrantyState?.model_name?.trim();

      if (needsModelConfirmation) {
        const lower = trimmed.toLowerCase();
        const confirmsModel =
          /^(yes|yeah|yep|y|correct|right|that(?:'s| is) (?:my )?model)\b/.test(
            lower
          );
        if (confirmsModel) {
          await confirmInferredModel();
        } else {
          await confirmInferredModel(trimmed);
        }
        setInput("");
        return;
      }

      if (atIssueTypeWithoutModel || pendingDefectStart) {
        setError(null);
        setLoading(true);
        setInput("");
        try {
          const resp = await registerWarrantyModel(sessionId, trimmed, storeDomain);
          applySessionResponse(resp);
          if (resp.handoff?.requested) {
            setPendingDefectStart(null);
            await appendAssistantFromResponse(resp.ticket, resp);
            return;
          }
          reportJourney("model", "completed");
          const defectLabel = pendingDefectStart;
          setPendingDefectStart(null);
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(
              `Thanks — I have **${resp.ticket?.model_name ?? trimmed}** on file.`
            ),
          ]);
          if (defectLabel && resp.ticket?.ticket_id) {
            const defectResp = await submitWarrantyAnswer(resp.ticket.ticket_id, "defect");
            applySessionResponse(defectResp);
            setMessages((prev) => [...prev, { role: "user", content: defectLabel }]);
            if (defectResp.side_question && defectResp.assistant_message) {
              await sleep(THINKING_DELAY_MS);
              setMessages((prev) => [
                ...prev,
                assistantMessage(defectResp.assistant_message!),
              ]);
              setOptionsUsed(false);
            } else {
              await appendAssistantFromResponse(defectResp.ticket, defectResp);
            }
          }
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : "Something went wrong.";
          setError(msg);
        } finally {
          setLoading(false);
          inputRef.current?.focus();
        }
        return;
      }

      if (atIssueTypeNode && warrantyState?.model_name) {
        await advanceWarranty(trimmed, trimmed);
        setInput("");
        return;
      }

      const optionCount = warrantyState?.current_node?.options?.length ?? 0;
      const nodeType = warrantyState?.current_node?.node_type;
      const isTextCapture =
        nodeType === "question_text" ||
        (optionCount === 0 && !!warrantyState?.ticket_id);
      const gateAllowsTyping = workflowNodeId.startsWith("defect_error_code");

      if (
        warrantyState?.ticket_id &&
        !warrantyState.current_node?.is_terminal &&
        (isTextCapture || gateAllowsTyping || optionCount > 0)
      ) {
        await advanceWarranty(trimmed, trimmed);
        setInput("");
        return;
      }

      if (
        warrantyState?.current_node?.is_terminal &&
        helpConsent === null &&
        !warrantyState?.needs_customer_reply &&
        warrantyState?.ticket_id
      ) {
        const ok = await handleResolutionOutcome(trimmed, trimmed);
        if (ok) setInput("");
        return;
      }
    },
    [
      loading,
      helpAccessReady,
      warrantyState,
      helpConsent,
      contactSubmitted,
      confirmInferredModel,
      startViaSmartIntake,
      advanceWarranty,
      appendAssistantFromResponse,
      submitFollowUpNote,
      refreshWarrantyState,
      pendingDefectStart,
      sessionId,
      storeDomain,
      applySessionResponse,
      workflowNodeId,
      handleResolutionOutcome,
      reportJourney,
    ]
  );

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    sendMessage(input);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  function handleOptionSelect(answerKey: string, label: string) {
    if (!helpAccessReady) return;
    const atIssueTypeNode =
      !warrantyState?.issue_type &&
      warrantyState?.current_node?.node_id === "issue_type";
    if (
      atIssueTypeNode &&
      answerKey === "defect" &&
      !warrantyState?.model_name?.trim()
    ) {
      setMessages((prev) => [...prev, { role: "user", content: label }]);
      promptModelForDefect(label);
      return;
    }
    advanceWarranty(answerKey, label);
  }

  const isAwaitingAdmin =
    warrantyState?.status === "awaiting_admin_review" ||
    warrantyState?.status === "admin_reviewing";
  const needsCustomerReply = Boolean(warrantyState?.needs_customer_reply);
  const customerReplyMessage = needsCustomerReply
    ? (warrantyState?.customer_message ?? null)
    : null;
  const resolvedTeamMessage =
    warrantyState?.status === "resolved"
      ? (warrantyState?.customer_message ?? null)
      : null;
  const caseReference = warrantyState?.case_reference ?? null;
  const isTerminal = warrantyState?.current_node?.is_terminal ?? false;
  const canGoBack = Boolean(helpAccessReady && warrantyState?.can_go_back && !isTerminal && !loading && !handoff?.requested);

  const atIssueTypeNode =
    !warrantyState?.issue_type &&
    warrantyState?.current_node?.node_id === "issue_type";

  const needsFirstIntake =
    sessionChecked &&
    !isTerminal &&
    warrantyState?.status !== "resolved" &&
    !handoff?.requested &&
    !warrantyState?.issue_type &&
    !atIssueTypeNode &&
    (!warrantyState?.ticket_id ||
      warrantyState?.current_node?.node_id === "root");

  const needsModelConfirmation = Boolean(warrantyState?.needs_model_confirmation);

  const showIssueTypeOptions =
    helpAccessReady &&
    sessionChecked &&
    !loading &&
    !isTerminal &&
    !handoff?.requested &&
    !needsModelConfirmation &&
    !warrantyState?.issue_type &&
    (warrantyState?.ready_for_issue_type ||
      (!!warrantyState?.model_name && atIssueTypeNode));

  const hasWorkflowOptions =
    helpAccessReady &&
    warrantyState?.status !== "resolved" &&
    !optionsUsed &&
    !loading &&
    !showIssueTypeOptions &&
    !needsModelConfirmation &&
    !handoff?.requested &&
    (warrantyState?.current_node?.options?.length ?? 0) > 0;

  useEffect(() => {
    if (!showIssueTypeOptions) return;
    const handle = window.setTimeout(() => setIssueTypePanelExpanded(true), 0);
    return () => window.clearTimeout(handle);
  }, [showIssueTypeOptions]);

  const showResolutionGate =
    helpAccessReady &&
    isTerminal &&
    !deferredPaused &&
    !handoff?.requested &&
    terminalEnrichment?.phase === "awaiting_help_consent" &&
    helpConsent === null &&
    !contactSubmitted;

  const inEmailStep = Boolean(
    isTerminal && !deferredPaused && !handoff?.requested && helpConsent === "yes" && !contactSubmitted && warrantyState?.ticket_id
  );

  const showEmailSection = helpAccessReady && inEmailStep;

  const nodeType = warrantyState?.current_node?.node_type;
  const isTextCaptureNode = nodeType === "question_text";
  const atIssueTypeWithoutModel =
    atIssueTypeNode && !warrantyState?.model_name?.trim();
  const showSerialPhotoButton =
    needsFirstIntake ||
    needsModelConfirmation ||
    atIssueTypeWithoutModel ||
    (Boolean(pendingDefectStart) && !warrantyState?.model_name?.trim());

  // Keep the composer visible for the whole claim chat: buttons still work,
  // and free text can map to an option when the meaning is clear.
  const caseClosed = warrantyState?.status === "resolved" && !needsCustomerReply;
  const chatFinished = caseClosed || (contactSubmitted && !needsCustomerReply) || helpConsent === "no" || handoff?.requested;
  const showInputBar =
    helpAccessReady &&
    !chatFinished && !deferredPaused && (!inEmailStep || emailPanelCollapsed);

  const interactionsLocked = !helpAccessReady;
  const canSaveProgress = helpAccessReady && canSaveWarrantyProgress(warrantyState);
  const saveProgressProps = {
    sessionId,
    disabled: loading || interactionsLocked,
    initialCustomerEmail: intakeContactEmail,
    onSaved: () => reportJourney("resume", "requested", `save-${Date.now()}`),
    onError: () => reportJourney("resume", "failed", `save-${Date.now()}`),
    onCustomerEmailConfirmed: (email: string) => { setIntakeContactEmail(email); reportJourney("email", "completed"); },
  };

  const journeyStage: JourneyStage = !helpAccessReady ? "email" : handoff?.requested ? "handoff" : deferredPaused || showResolutionGate ? "outcome" : inEmailStep ? "evidence" : needsModelConfirmation || pendingDefectStart ? "model" : warrantyState?.issue_type === "delivery" ? "delivery" : "issue";

  useEffect(() => {
    if (!helpAccessReady && !chatConsentAccepted) return;
    reportJourney(journeyStage, "shown", `${workflowNodeId}:${resolutionStage}:${deferredPaused ? "paused" : "active"}`);
  }, [helpAccessReady, chatConsentAccepted, reportJourney, journeyStage, workflowNodeId, resolutionStage, deferredPaused]);

  useEffect(() => {
    if (!helpAccessReady) return;
    if (messages.length > 0 && messages[messages.length - 1].role === "assistant") reportJourney("response", "shown", `${workflowNodeId}:${messages.length}`);
    if (trackingSummary?.available || (terminalEnrichment?.interaction_mode === "troubleshooting" && terminalEnrichment.message?.trim())) reportJourney("help", "shown");
  }, [helpAccessReady, messages, workflowNodeId, reportJourney, trackingSummary?.available, terminalEnrichment]);

  useEffect(() => {
    if (resumeStatus === "resumed" || resumeStatus === "failed") reportJourney("resume", resumeStatus === "resumed" ? "resumed" : "failed");
  }, [resumeStatus, reportJourney]);

  useEffect(() => {
    if (error) reportJourney(journeyStage, "failed", `error-${Date.now()}`);
  }, [error, journeyStage, reportJourney]);

  useEffect(() => {
    if (showEmailSection && !/^[\w.+-]+@[\w.-]+\.\w+$/.test(intakeContactEmail.trim())) reportJourney("email", "shown", "review-contact");
  }, [showEmailSection, intakeContactEmail, reportJourney]);

  const inputPlaceholder = needsCustomerReply
    ? "Type your reply to our team here…"
    : needsModelConfirmation
      ? "Type the correct model name (e.g. Hypnos, OS-4000T)…"
    : pendingDefectStart || atIssueTypeWithoutModel
      ? "Enter your chair model (e.g. OS-4000T, 3D LTX)…"
    : needsFirstIntake
    ? "Model + issue (e.g. OS-4000T footrest air not inflating)…"
      : inEmailStep
        ? "Anything else our team should know? Type here…"
            : isErrorCodeGateNode
          ? "Type yes/no, or enter the error code (e.g. C6)…"
          : isTextCaptureNode
            ? "Type your answer…"
            : atIssueTypeNode || hasWorkflowOptions || showResolutionGate
              ? "Or type your answer…"
            : "Enter your chair model…";

  return (
    <div
      className={`mx-auto flex min-h-0 w-full flex-1 flex-col ${
        embed ? "max-w-none" : "max-w-2xl"
      }`}
    >
      <ChatRecordingNoticeBanner
        consentAccepted={chatConsentAccepted}
        onAccept={acceptChatConsent}
        storeDomain={storeDomain}
        disabled={consentSaving || !sessionChecked || resumeStatus === "resuming"}
        pendingLabel={consentSaving ? "Saving consent…" : "Loading chat…"}
        error={consentError}
      />

      {resumeStatus === "failed" && <p role="alert" className="mx-4 mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">That resume link could not be opened. It may have expired. You can check an existing case or start a new request.</p>}

      {warrantyState && !embed && (
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-gray-100 bg-white px-4 py-2">
          <TicketStatusBadge
            status={warrantyState.status}
            ticketId={warrantyState.ticket_id}
            caseReference={warrantyState.case_reference}
          />
          <div className="flex flex-wrap items-center gap-2">
            {warrantyState.model_name && (
              <span className="truncate text-right text-xs text-gray-500">
                {warrantyState.model_name}
              </span>
            )}
            {canSaveProgress && (
              <SaveProgressButton key={sessionId} {...saveProgressProps} />
            )}
            {canGoBack && (
              <button
                type="button"
                onClick={goBackOneStep}
                disabled={loading}
                className="rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
                title="Go back to the previous question"
              >
                ← Back
              </button>
            )}
            <button
              type="button"
              onClick={() => setShowCaseLookup((open) => !open)}
              disabled={loading}
              className="rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
              title="Check an existing warranty case"
            >
              Check case
            </button>
            <button
              type="button"
              onClick={restartSession}
              disabled={loading}
              className="rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
              title="Clear current answers and start a new warranty case"
            >
              Start over
            </button>
          </div>
        </div>
      )}

      {embed && !warrantyState?.ticket_id && (
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-gray-100 bg-white px-3 py-1.5">
          <p className="text-xs text-gray-500">Warranty chat</p>
          <button
            type="button"
            onClick={() => setShowCaseLookup((open) => !open)}
            className="rounded-full border border-gray-200 px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900"
          >
            Check case
          </button>
        </div>
      )}

      {warrantyState?.ticket_id && embed && (
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-gray-100 bg-white px-3 py-1.5">
          <div className="min-w-0 text-left">
            {caseReference ? (
              <p className="truncate text-xs font-medium text-gray-700">
                Case <span className="font-mono">{caseReference}</span>
              </p>
            ) : (
              <p className="text-xs text-gray-500">Warranty chat</p>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
          {canSaveProgress && (
            <SaveProgressButton key={sessionId} {...saveProgressProps} />
          )}
          {canGoBack && (
            <button
              type="button"
              onClick={goBackOneStep}
              disabled={loading}
              className="rounded-full border border-gray-200 px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
              title="Go back to the previous question"
            >
              ← Back
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowCaseLookup((open) => !open)}
            disabled={loading}
            className="rounded-full border border-gray-200 px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
            title="Check an existing warranty case"
          >
            Check case
          </button>
          <button
            type="button"
            onClick={restartSession}
            disabled={loading}
            className="rounded-full border border-gray-200 px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
            title="Clear current answers and start a new warranty case"
          >
            Start over
          </button>
          </div>
        </div>
      )}

      {showCaseLookup && (
        <div className="mx-4 mt-3 shrink-0 rounded-xl border border-gray-200 bg-white px-4 py-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <p className="text-sm font-medium text-gray-900">Check an existing case</p>
            <button
              type="button"
              onClick={() => setShowCaseLookup(false)}
              className="text-xs text-gray-500 hover:text-gray-800"
            >
              Close
            </button>
          </div>
          <WarrantyCaseLookup
            compact
            initialCaseReference={caseReference || ""}
          />
        </div>
      )}

      {isAwaitingAdmin && (contactSubmitted || warrantyState?.review_requested) && !handoff?.requested && (
        <div className="mx-4 mt-3 shrink-0 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-sm font-medium text-amber-800">
            {warrantyState?.issue_type === "installation"
              ? "Install Support Review"
              : warrantyState?.issue_type === "delivery"
                ? "Delivery Case Review"
                : "Under Support Review"}
          </p>
          <p className="mt-0.5 text-xs text-amber-700">
            {warrantyState?.issue_type === "installation"
              ? "We'll help you finish setup."
              : warrantyState?.issue_type === "delivery"
                ? "Your delivery case is with our team."
                : null}{" "}
            {caseReference ? (
              <>
                Your case reference is <strong>{caseReference}</strong>. Save this
                number for follow-up, or contact our support team directly.
              </>
            ) : (
              "Your case is saved for review. You can also contact our support team directly."
            )}
          </p>
        </div>
      )}

      {customerReplyMessage && (
        <div className="mx-4 mt-3 shrink-0 rounded-xl border border-yellow-200 bg-yellow-50 px-4 py-3">
          <p className="text-sm font-medium text-yellow-900">We need a little more information</p>
          <p className="mt-1 whitespace-pre-wrap text-sm text-yellow-800">
            {customerReplyMessage}
          </p>
          <p className="mt-2 text-xs text-yellow-700">
            Reply in the box below — we&apos;ll notify our warranty team.
          </p>
        </div>
      )}

      {resolvedTeamMessage && (
        <div className="mx-4 mt-3 shrink-0 rounded-xl border border-green-200 bg-green-50 px-4 py-3">
          <p className="text-sm font-medium text-green-900">Update from our warranty team</p>
          <p className="mt-1 whitespace-pre-wrap text-sm text-green-800">
            {resolvedTeamMessage}
          </p>
        </div>
      )}

      {savedFollowup?.sessionId === sessionId && ["defect", "installation"].includes(savedFollowup.issueType) && (
          <div className="mx-4 mt-3 shrink-0">
            <OutcomeFollowup key={savedFollowup.ticketId} ticketId={savedFollowup.ticketId} token={savedFollowup.token} />
          </div>
        )}

      <div className="chat-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-4 sm:px-4">
        {!chatConsentAccepted && (
          <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-center text-sm text-amber-900">
            Please review the notice above and tap <strong>I Agree</strong> to
            continue.
          </div>
        )}
        {chatConsentAccepted && sessionChecked && !helpAccessReady && (
          <ChatEmailGate
            disabled={loading}
            onContinue={(email) => completeEmailGate({ email })}
          />
        )}
        {helpAccessReady && <div className="space-y-4">
          {messages.map((msg, i) => (
            <ChatMessageBubble
              key={i}
              message={msg}
              showFeedback={helpAccessReady && msg.role === "assistant"}
              feedbackSessionId={sessionId}
              feedbackTicketId={warrantyState?.ticket_id}
              feedbackDomain={storeDomain}
              feedbackContext="warranty"
            />
          ))}

        </div>}

        {helpAccessReady && handoff?.requested && (
          <div className="mt-4"><HumanHandoffNotice handoff={handoff} caseReference={caseReference} /></div>
        )}

        {helpAccessReady && warrantyState?.issue_type === "delivery" && warrantyState.status !== "resolved" && !deferredPaused && !handoff?.requested && (
          <div>
            <DeliveryTrackingPanel key={`${sessionId}:${deliveryStoreDomain}`} storeDomain={deliveryStoreDomain} initialEmail={intakeContactEmail} summary={trackingSummary} disabled={loading} onLookup={handleDeliveryLookup} />
            {trackingSummary?.available && <button type="button" disabled={loading} onClick={() => handleResolutionOutcome("delivery_answered", "The delivery update answered my question")} className="mt-2 min-h-[44px] w-full rounded-xl border border-sky-300 bg-white px-4 py-2 text-sm font-medium text-sky-900 hover:bg-sky-50 disabled:opacity-50">This delivery update answered my question</button>}
          </div>
        )}

        {helpAccessReady && !chatFinished && !deferredPaused && !handoff?.requested && (
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {sessionChecked && !warrantyState?.issue_type && !showIssueTypeOptions && (
              <button type="button" onClick={() => handleQuickStart("delivery", "Delivery & tracking")} disabled={loading} className="rounded-full border border-sky-300 bg-white px-4 py-2 text-sm font-medium text-sky-900 hover:bg-sky-50 disabled:opacity-50">Delivery & tracking</button>
            )}
            <button type="button" onClick={requestHumanSupport} disabled={loading} className="rounded-full border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50">Ask for a person</button>
          </div>
        )}

        {pendingDefectStart && !handoff?.requested && (
          <div className="mt-3 sm:mt-4 rounded-xl border border-sky-200 bg-sky-50/90 px-4 py-3">
            <p className="text-sm font-medium text-sky-950">Chair model needed</p>
            <p className="mt-1 text-sm text-sky-900">
              Enter your model below to continue with warranty / defect.
            </p>
          </div>
        )}

        {needsModelConfirmation && warrantyState?.model_name && !handoff?.requested && (
          <div className="mt-3 sm:mt-4 rounded-xl border border-violet-200 bg-violet-50/90 px-4 py-3">
            <p className="text-sm font-medium text-violet-950">Confirm your chair model</p>
            <p className="mt-1 text-sm text-violet-900">
              We have <strong>{warrantyState.model_name}</strong> on file. Is that correct?
            </p>
            <button
              type="button"
              disabled={loading || interactionsLocked}
              onClick={() => confirmInferredModel()}
              className="mt-3 rounded-full bg-violet-700 px-4 py-2 text-sm font-medium text-white hover:bg-violet-800 disabled:opacity-50"
            >
              Yes, that&apos;s my model
            </button>
            <p className="mt-2 text-xs text-violet-800">
              Or type the correct model name in the box below.
            </p>
          </div>
        )}

        {showIssueTypeOptions && (
          <div className="mt-3 sm:mt-4">
            <CollapsibleOptionPanel
              title="What can we help you with?"
              hint={
                suggestedIssueType
                  ? "Suggested option is highlighted — tap to confirm"
                  : "Choose one option to continue"
              }
              optionCount={INITIAL_ISSUE_OPTIONS.length}
              expanded={issueTypePanelExpanded}
              onToggle={() => setIssueTypePanelExpanded((open) => !open)}
              disabled={loading || interactionsLocked}
            >
              <AnswerOptions
                options={INITIAL_ISSUE_OPTIONS}
                highlightedKey={suggestedIssueType}
                variant="stack"
                onSelect={(key, label) =>
                  handleQuickStart(
                    key as "installation" | "delivery" | "defect",
                    label
                  )
                }
                disabled={loading || interactionsLocked}
              />
            </CollapsibleOptionPanel>
          </div>
        )}

        {loading && (
          <div className="mt-4 flex items-center gap-2 rounded-xl bg-gray-50 px-3 py-2 text-gray-500">
            <div className="flex gap-1">
              <span className="h-2 w-2 animate-bounce rounded-full bg-brand-400 [animation-delay:0ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-brand-400 [animation-delay:150ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-brand-400 [animation-delay:300ms]" />
            </div>
            <span className="text-xs font-medium">Reviewing your answer…</span>
          </div>
        )}

        {error && (
          <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {hasWorkflowOptions && !isTerminal && (
        <div className="shrink-0 border-t border-gray-100 bg-white px-3 py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))] sm:px-4 sm:py-3 sm:pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          {suggestedOption && (
            <button
              type="button"
              onClick={() =>
                handleOptionSelect(suggestedOption.answer_key, suggestedOption.label)
              }
              disabled={loading || interactionsLocked}
              className="mb-2 min-h-[48px] w-full rounded-xl bg-brand-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
            >
              Yes — {suggestedOption.label}
            </button>
          )}
          <CollapsibleOptionPanel
            title={suggestedOption ? "Other options" : "Choose an option"}
            hint={
              suggestedOption
                ? "Tap Yes above, or choose a different option"
                : "Tap an option, or type your answer"
            }
            optionCount={workflowOptionCount}
            expanded={optionsPanelExpanded}
            onToggle={() => setOptionsPanelExpanded((open) => !open)}
            disabled={loading || interactionsLocked}
          >
            <AnswerOptions
              options={warrantyState!.current_node!.options}
              onSelect={handleOptionSelect}
              disabled={loading || interactionsLocked}
              variant="stack"
              highlightedKey={suggestedOption?.answer_key ?? null}
            />
          </CollapsibleOptionPanel>
        </div>
      )}

      {showResolutionGate && (
        <div className="shrink-0 border-t border-gray-100 bg-white px-3 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-4 sm:py-4 sm:pb-[max(1rem,env(safe-area-inset-bottom))]">
          <TroubleshootingGate
            mode={terminalEnrichment?.interaction_mode ?? "troubleshooting"}
            issueType={
              terminalEnrichment?.issue_type || warrantyState?.issue_type
            }
            stage={resolutionStage}
            stepCount={terminalEnrichment?.diagnosis?.steps?.length ?? 0}
            disabled={loading || interactionsLocked}
            onStepsCompleted={() =>
              handleResolutionOutcome("steps_completed")
            }
            onResolved={() =>
              handleResolutionOutcome(
                "resolved",
                resolutionGateLabels(
                  terminalEnrichment?.issue_type || warrantyState?.issue_type,
                  terminalEnrichment?.interaction_mode ?? "troubleshooting"
                ).resolved
              )
            }
            onUnresolved={() =>
              handleResolutionOutcome(
                "unresolved",
                resolutionGateLabels(
                  terminalEnrichment?.issue_type || warrantyState?.issue_type,
                  terminalEnrichment?.interaction_mode ?? "troubleshooting"
                ).unresolved
              )
            }
            onUnableToAttempt={() =>
              handleResolutionOutcome(
                "unable_to_attempt",
                resolutionGateLabels(
                  terminalEnrichment?.issue_type || warrantyState?.issue_type,
                  terminalEnrichment?.interaction_mode ?? "troubleshooting"
                ).unable
              )
            }
            onDeferred={() => handleResolutionOutcome("deferred", "Save for later")}
          />
        </div>
      )}

      {helpAccessReady && deferredPaused && !handoff?.requested && (
        <div className="shrink-0 border-t border-gray-100 bg-white p-4">
          <p className="mb-3 text-sm text-gray-700">Your case is saved for later. Keep a resume link, or continue now.</p>
          {canSaveProgress && <SaveProgressButton key={sessionId} {...saveProgressProps} />}
          <button type="button" disabled={loading} onClick={() => { setDeferredPaused(false); reportJourney("outcome", "resumed", `continue-${Date.now()}`); }} className="mt-3 min-h-[44px] w-full rounded-xl border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-800 hover:bg-gray-50">Continue this case now</button>
        </div>
      )}

      {showEmailSection && (
        <div className="max-h-[55dvh] shrink-0 overflow-y-auto border-t border-gray-100 bg-white p-3 pb-[max(0.5rem,env(safe-area-inset-bottom))] sm:p-4">
          <EvidenceUploader
            key={warrantyState!.ticket_id}
            ticketId={warrantyState!.ticket_id}
            issueType={warrantyState?.issue_type}
            evidenceRequired={warrantyState!.current_node?.evidence_required}
            initialCustomerEmail={intakeContactEmail}
            collapsed={emailPanelCollapsed}
            onToggleCollapsed={setEmailPanelCollapsed}
            onBack={goBackFromEmailStep}
            backDisabled={loading}
            onCustomerEmailConfirmed={(email) => { setIntakeContactEmail(email); reportJourney("email", "completed"); }}
            onError={() => reportJourney("evidence", "failed", `submit-${Date.now()}`)}
            onContactSuccess={(_email, response) => {
              reportJourney("evidence", "completed");
              setContactSubmitted(true);
              setContactReceiptSent(Boolean(response.receipt_email_sent));
              if (response.case_reference) setWarrantyState((previous) => previous ? { ...previous, case_reference: response.case_reference } : previous);
              setMessages((prev) => [
                ...prev,
                assistantMessage(
                  buildWarrantySubmissionMessage({
                    issueType: warrantyState?.issue_type,
                    caseReference: response.case_reference || warrantyState?.case_reference,
                    receiptEmailSent: Boolean(response.receipt_email_sent),
                  })
                ),
              ]);
            }}
            onUploadSuccess={(filename, response) => {
              reportJourney("evidence", "completed");
              setContactSubmitted(true);
              setContactReceiptSent(Boolean(response.receipt_email_sent));
              if (response.case_reference) setWarrantyState((previous) => previous ? { ...previous, case_reference: response.case_reference } : previous);
              setMessages((prev) => [
                ...prev,
                assistantMessage(
                  `Thank you — "${filename}" has been received. ${buildWarrantySubmissionMessage({ issueType: warrantyState?.issue_type, caseReference: response.case_reference || warrantyState?.case_reference, receiptEmailSent: Boolean(response.receipt_email_sent) })}`
                ),
              ]);
            }}
          />
        </div>
      )}

      {showInputBar && (
        <form
          onSubmit={handleSubmit}
          className="shrink-0 border-t border-gray-200 bg-white px-3 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 sm:px-4"
        >
          {showSerialPhotoButton && (
            <div className="mb-2">
              <SerialPhotoButton
                disabled={loading || interactionsLocked}
                onModelDetected={(name) => {
                  const suffix = input.trim() ? ` ${input.trim()}` : "";
                  setInput(`${name}${suffix}`);
                  inputRef.current?.focus();
                }}
              />
            </div>
          )}
          <div className="flex items-end gap-2 rounded-2xl border border-gray-200 bg-gray-50 p-2 focus-within:border-brand-500 focus-within:ring-1 focus-within:ring-brand-500">
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={inputPlaceholder}
              disabled={loading || interactionsLocked}
              className="min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2 text-base text-gray-900 placeholder-gray-400 focus:outline-none disabled:opacity-60 sm:text-sm"
              style={{ maxHeight: "120px" }}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading || interactionsLocked}
              className={`flex h-11 min-w-[4.5rem] flex-shrink-0 items-center justify-center rounded-xl px-4 text-sm font-semibold transition ${
                !input.trim() || loading
                  ? "cursor-not-allowed bg-gray-200 text-gray-400"
                  : "bg-brand-600 text-white hover:bg-brand-700 active:scale-95"
              }`}
            >
              {loading ? "…" : "Send"}
            </button>
          </div>
          <WarrantyTeamContactFooter compact className="mt-2" />
        </form>
      )}

      {((contactSubmitted && !needsCustomerReply) || helpConsent === "no" || caseClosed) && (
        <div className="shrink-0 border-t border-gray-100 bg-white px-4 py-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-center">
          {contactSubmitted && (
            <div className="space-y-1.5">
              <p className="text-sm font-medium text-gray-800">
                Your case has been submitted.
              </p>
              {caseReference && (
                <p className="text-sm text-gray-700">
                  Case reference:{" "}
                  <span className="font-mono font-semibold">{caseReference}</span>
                </p>
              )}
              <p className="text-sm text-gray-600">
                {contactReceiptSent ? "We emailed a confirmation. Keep your case reference for follow-up." : "Your case is saved for review. For direct assistance, contact our support team below."}
              </p>
              <button
                type="button"
                onClick={() => setShowCaseLookup(true)}
                className="text-sm font-medium text-brand-600 underline hover:text-brand-800"
              >
                Check this case
              </button>
            </div>
          )}
          <WarrantyTeamContactFooter className="mt-4 text-left" />
          <button
            onClick={restartSession}
            disabled={loading}
            className="mt-3 min-h-[44px] text-sm text-brand-600 underline hover:text-brand-800 disabled:opacity-50"
          >
            Start a new case
          </button>
        </div>
      )}
    </div>
  );
}
