"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  getWarrantySession,
  quickStartWarranty,
  naturalStartWarranty,
  restartWarrantySession,
  resumeWarrantyFromToken,
  smartStartWarranty,
  submitCustomerNote,
  submitWarrantyAnswer,
} from "@/lib/api";
import type {
  AnswerOption,
  ChatMessage,
  StepEnrichment,
  TerminalEnrichment,
  WarrantySessionResponse,
  WarrantyTicketState,
} from "@/lib/types";
import {
  assistantContentFromResponse,
  hydrationAssistantContent,
} from "@/lib/warrantyHydration";
import ChatMessageBubble from "./ChatMessageBubble";
import StepEnrichmentPanel from "./StepEnrichmentPanel";
import AnswerOptions from "./AnswerOptions";
import CollapsibleOptionPanel from "./CollapsibleOptionPanel";
import EvidenceUploader from "./EvidenceUploader";
import SaveProgressButton from "./SaveProgressButton";
import SerialPhotoButton from "./SerialPhotoButton";
import TicketStatusBadge from "./TicketStatusBadge";
import { WARRANTY_CONTACT_EMAIL } from "@/lib/evidenceMessage";
import { WARRANTY_WELCOME_MESSAGE } from "@/lib/welcomeMessage";
import WarrantyTeamContactFooter from "./WarrantyTeamContactFooter";

import { resolveWarrantyStoreDomain } from "@/lib/warrantyStoreDomain";

// Short "reviewing your answer…" pause before the assistant bubble appears.
// Kept small because ChatMessageBubble now types the response out on its own,
// giving the actual streamed-response perception.
const THINKING_DELAY_MS = 400;

function assistantMessage(content: string): ChatMessage {
  return { role: "assistant", content, animate: true };
}

const EMAIL_THANK_YOU =
  `Thank you! Our warranty team at ${WARRANTY_CONTACT_EMAIL} will respond within 24 hours.`;

const SELF_HELP_CLOSING =
  "Sounds good — try those steps first. If you still need us, you can start a new case anytime. " +
  "Our warranty team is also available by phone during business hours.";

const INITIAL_ISSUE_OPTIONS: AnswerOption[] = [
  { answer_key: "installation", label: "Setup & installation" },
  { answer_key: "delivery", label: "Delivery & tracking" },
  { answer_key: "defect", label: "Warranty / defect" },
];

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function WarrantyChat({ embed = false }: { embed?: boolean }) {
  const storeDomain = resolveWarrantyStoreDomain();
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
  >("idle");

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warrantyState, setWarrantyState] = useState<WarrantyTicketState | null>(null);
  const [terminalEnrichment, setTerminalEnrichment] = useState<TerminalEnrichment | null>(null);
  const [stepEnrichment, setStepEnrichment] = useState<StepEnrichment | null>(null);
  const [optionsUsed, setOptionsUsed] = useState(false);
  const [sessionChecked, setSessionChecked] = useState(false);
  const [contactSubmitted, setContactSubmitted] = useState(false);
  const [helpConsent, setHelpConsent] = useState<"yes" | "no" | null>(null);
  const [optionsPanelExpanded, setOptionsPanelExpanded] = useState(true);
  const [issueTypePanelExpanded, setIssueTypePanelExpanded] = useState(true);
  const [emailPanelCollapsed, setEmailPanelCollapsed] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, helpConsent, stepEnrichment]);

  const applySessionResponse = useCallback((resp: WarrantySessionResponse) => {
    setWarrantyState(resp.ticket);
    setTerminalEnrichment(resp.terminal_enrichment ?? null);
    setStepEnrichment(resp.step_enrichment ?? null);
    if (resp.terminal_enrichment?.phase === "awaiting_help_consent") {
      setHelpConsent(null);
    }
    setOptionsUsed(false);
    return resp;
  }, []);

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
      setStepEnrichment(null);
      setHelpConsent(null);
      setContactSubmitted(false);
      setOptionsUsed(false);
      setEmailPanelCollapsed(false);
      setInput("");
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading, sessionId, storeDomain]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const token = params.get("resume");
    if (!token) return;

    let cancelled = false;
    setResumeStatus("resuming");
    (async () => {
      try {
        const data = await resumeWarrantyFromToken(token);
        if (cancelled) return;
        sessionStorage.setItem("warranty_session_id", data.session_id);
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
        if (ticket?.current_node?.is_terminal && ticket.current_node.prompt) {
          const content = hydrationAssistantContent(ticket, resp);
          if (content) {
            setMessages([{ role: "assistant", content }]);
          }
        } else if (ticket?.ready_for_issue_type && ticket.model_name) {
          setMessages([
            { role: "assistant", content: WARRANTY_WELCOME_MESSAGE },
            {
              role: "assistant",
              content: `Great — I have **${ticket.model_name}** on file.\n\nWhat type of issue can we help you with? Choose below or describe it in your own words.`,
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

  useEffect(() => {
    if (workflowOptionCount >= 6) {
      setOptionsPanelExpanded(false);
    } else if (workflowOptionCount > 0) {
      setOptionsPanelExpanded(true);
    }
  }, [workflowNodeId, workflowOptionCount]);

  const appendAssistantFromResponse = useCallback(
    async (
      ticket: WarrantyTicketState | null,
      resp: Pick<WarrantySessionResponse, "assistant_message" | "terminal_enrichment">
    ) => {
      await sleep(THINKING_DELAY_MS);
      const content = assistantContentFromResponse(ticket, resp);
      if (!content) return;
      setMessages((prev) => [...prev, assistantMessage(content)]);
    },
    []
  );

  const handleQuickStart = useCallback(
    async (issueType: "installation" | "delivery" | "defect", label: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setHelpConsent(null);

      try {
        const resp = await quickStartWarranty(sessionId, issueType, storeDomain);
        applySessionResponse(resp);
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
    [loading, sessionId, storeDomain, applySessionResponse, appendAssistantFromResponse]
  );

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
        }
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
    [warrantyState?.ticket_id, loading, applySessionResponse, appendAssistantFromResponse]
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
        const jumped =
          routingConfirm?.message ||
          (smart &&
            smart.source === "llm" &&
            smart.applied_keys &&
            smart.applied_keys.length >= 3 &&
            smart.summary);

        applySessionResponse(resp);

        if (routingConfirm?.message) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(routingConfirm.message),
          ]);
        } else if (jumped && smart?.summary) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(
              `Got it — ${smart.summary} I'll skip the extra menu questions and take you straight to the next step.`
            ),
          ]);
        } else if (
          resp.ticket?.ready_for_issue_type &&
          resp.ticket?.model_name &&
          resp.ticket?.current_node?.node_id === "issue_type"
        ) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            assistantMessage(
              `Thanks — I have **${resp.ticket!.model_name}** on file.\n\nWhat type of issue can we help you with? Choose below or describe it in your own words.`
            ),
          ]);
        }

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
    [loading, sessionId, storeDomain, applySessionResponse, appendAssistantFromResponse]
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
        await submitCustomerNote(ticketId, text);
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
    [loading, warrantyState?.ticket_id]
  );

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || loading) return;

      const trimmed = text.trim();
      const atFirstIntake =
        !warrantyState?.issue_type &&
        (!warrantyState?.ticket_id ||
          warrantyState?.current_node?.node_id === "root" ||
          warrantyState?.current_node?.node_id === "issue_type");
      const atIssueType =
        warrantyState?.ready_for_issue_type ||
        warrantyState?.current_node?.node_id === "issue_type";

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

      if (atIssueType && !warrantyState?.issue_type) {
        await startViaSmartIntake(trimmed);
        return;
      }

      if (warrantyState?.ticket_id && !warrantyState.current_node?.is_terminal) {
        await advanceWarranty(trimmed, trimmed);
        setInput("");
        return;
      }

      if (
        warrantyState?.current_node?.is_terminal &&
        helpConsent === null &&
        !warrantyState?.needs_customer_reply
      ) {
        setError("Please tap Yes or No below so we know how to help next.");
        return;
      }
    },
    [
      loading,
      warrantyState,
      helpConsent,
      contactSubmitted,
      startViaSmartIntake,
      advanceWarranty,
      submitFollowUpNote,
      refreshWarrantyState,
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
    advanceWarranty(answerKey, label);
  }

  function handleHelpOffer(key: string, label: string) {
    setMessages((prev) => [...prev, { role: "user", content: label }]);
    if (key === "yes_team_help") {
      setHelpConsent("yes");
      setMessages((prev) => [
        ...prev,
        assistantMessage(
          "No problem — please share your email below so our warranty team can follow up. " +
            "Photos or videos are optional."
        ),
      ]);
    } else {
      setHelpConsent("no");
      setMessages((prev) => [...prev, assistantMessage(SELF_HELP_CLOSING)]);
    }
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

  const needsFirstIntake =
    sessionChecked &&
    !isTerminal &&
    !warrantyState?.issue_type &&
    (!warrantyState?.ticket_id ||
      warrantyState?.current_node?.node_id === "root" ||
      warrantyState?.current_node?.node_id === "issue_type");

  const showIssueTypeOptions =
    sessionChecked &&
    !loading &&
    !isTerminal &&
    !needsFirstIntake &&
    (warrantyState?.ready_for_issue_type ||
      (!!warrantyState?.model_name &&
        warrantyState?.current_node?.node_id === "issue_type" &&
        !warrantyState?.issue_type));

  const hasWorkflowOptions =
    !optionsUsed &&
    !loading &&
    !showIssueTypeOptions &&
    (warrantyState?.current_node?.options?.length ?? 0) > 0;

  useEffect(() => {
    if (showIssueTypeOptions) {
      setIssueTypePanelExpanded(true);
    }
  }, [showIssueTypeOptions]);

  const helpOfferOptions =
    terminalEnrichment?.help_offer_options ?? [];

  const showHelpOffer =
    isTerminal &&
    terminalEnrichment?.phase === "awaiting_help_consent" &&
    helpConsent === null &&
    !contactSubmitted &&
    helpOfferOptions.length > 0;

  const inEmailStep = Boolean(
    isTerminal && helpConsent === "yes" && !contactSubmitted && warrantyState?.ticket_id
  );

  const showEmailSection = inEmailStep;

  const showInputBar =
    needsCustomerReply ||
    !isTerminal ||
    inEmailStep ||
    (isTerminal && helpConsent === null && !contactSubmitted && !showHelpOffer);

  const inputPlaceholder = needsCustomerReply
    ? "Type your reply to our team here…"
    : needsFirstIntake
    ? "Model + issue (e.g. OS-4000T footrest air not inflating)…"
    : showIssueTypeOptions
      ? "Describe your issue (e.g. my chair won't turn on)…"
      : inEmailStep
        ? "Anything else our team should know? Type here…"
        : warrantyState?.ticket_id
          ? "Type your answer…"
          : "Enter your chair model…";

  return (
    <div
      className={`mx-auto flex min-h-0 w-full flex-1 flex-col ${
        embed ? "max-w-none" : "max-w-2xl"
      }`}
    >
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
            {warrantyState.ticket_id && !isTerminal && (
              <SaveProgressButton sessionId={sessionId} disabled={loading} />
            )}
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

      {warrantyState?.ticket_id && embed && (
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2 border-b border-gray-100 bg-white px-3 py-1.5">
          {!isTerminal && (
            <SaveProgressButton sessionId={sessionId} disabled={loading} />
          )}
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
      )}

      {isAwaitingAdmin && (
        <div className="mx-4 mt-3 shrink-0 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-sm font-medium text-amber-800">Under Support Review</p>
          <p className="mt-0.5 text-xs text-amber-700">
            {caseReference ? (
              <>
                Your case reference is <strong>{caseReference}</strong>.
                {" "}Save this number — our team will follow up within 24 hours.
              </>
            ) : (
              "Your case has been prepared for support team review."
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

      <div className="chat-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-4 sm:px-4">
        <div className="space-y-4">
          {messages.map((msg, i) => (
            <ChatMessageBubble
              key={i}
              message={msg}
              showFeedback={msg.role === "assistant"}
              feedbackSessionId={sessionId}
              feedbackTicketId={warrantyState?.ticket_id}
              feedbackDomain={storeDomain}
              feedbackContext="warranty"
            />
          ))}

          {!isTerminal && stepEnrichment && (
            <div className="flex justify-start">
              <div className="max-w-[92%] sm:max-w-[85%]">
                <StepEnrichmentPanel enrichment={stepEnrichment} />
              </div>
            </div>
          )}
        </div>

        {showIssueTypeOptions && (
          <div className="mt-3 sm:mt-4">
            <CollapsibleOptionPanel
              title="What can we help you with?"
              hint="Or type your issue below"
              optionCount={INITIAL_ISSUE_OPTIONS.length}
              expanded={issueTypePanelExpanded}
              onToggle={() => setIssueTypePanelExpanded((open) => !open)}
              disabled={loading}
            >
              <AnswerOptions
                options={INITIAL_ISSUE_OPTIONS}
                variant="stack"
                onSelect={(key, label) =>
                  handleQuickStart(
                    key as "installation" | "delivery" | "defect",
                    label
                  )
                }
                disabled={loading}
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
          <CollapsibleOptionPanel
            title="Choose an option"
            hint="Or type your answer in the box below"
            optionCount={workflowOptionCount}
            expanded={optionsPanelExpanded}
            onToggle={() => setOptionsPanelExpanded((open) => !open)}
            disabled={loading}
          >
            <AnswerOptions
              options={warrantyState!.current_node!.options}
              onSelect={handleOptionSelect}
              disabled={loading}
              variant="stack"
            />
          </CollapsibleOptionPanel>
        </div>
      )}

      {showHelpOffer && (
        <div className="shrink-0 border-t border-gray-100 bg-white px-3 py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))] sm:px-4 sm:py-3 sm:pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          <AnswerOptions
            options={helpOfferOptions}
            onSelect={handleHelpOffer}
            disabled={loading}
            variant="stack"
          />
        </div>
      )}

      {showEmailSection && (
        <div className="shrink-0 border-t border-gray-100 bg-white p-3 pb-[max(0.5rem,env(safe-area-inset-bottom))] sm:p-4">
          <EvidenceUploader
            ticketId={warrantyState!.ticket_id}
            evidenceRequired={warrantyState!.current_node?.evidence_required}
            collapsed={emailPanelCollapsed}
            onToggleCollapsed={setEmailPanelCollapsed}
            onContactSuccess={() => {
              setContactSubmitted(true);
              setMessages((prev) => [
                ...prev,
                assistantMessage(
                  "Thank you — your email has been received. Our warranty team will follow up within 24 hours."
                ),
              ]);
            }}
            onUploadSuccess={(filename) => {
              setContactSubmitted(true);
              setMessages((prev) => [
                ...prev,
                assistantMessage(
                  `Thank you — "${filename}" has been received. Our team will review it shortly.`
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
          {needsFirstIntake && (
            <div className="mb-2">
              <SerialPhotoButton
                disabled={loading}
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
              disabled={loading}
              className="min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2 text-base text-gray-900 placeholder-gray-400 focus:outline-none disabled:opacity-60 sm:text-sm"
              style={{ maxHeight: "120px" }}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
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

      {(contactSubmitted || helpConsent === "no") && (
        <div className="shrink-0 border-t border-gray-100 bg-white px-4 py-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-center">
          {contactSubmitted && (
            <p className="text-sm text-gray-600">
              Your case has been submitted. Our team will be in touch.
            </p>
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
