"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  getWarrantySession,
  quickStartWarranty,
  naturalStartWarranty,
  registerWarrantyModel,
  submitWarrantyAnswer,
} from "@/lib/api";
import type {
  AnswerOption,
  ChatMessage,
  TerminalEnrichment,
  WarrantySessionResponse,
  WarrantyTicketState,
} from "@/lib/types";
import ChatMessageBubble from "./ChatMessageBubble";
import AnswerOptions from "./AnswerOptions";
import EvidenceUploader from "./EvidenceUploader";
import TicketStatusBadge from "./TicketStatusBadge";
import { formatTerminalPrompt, WARRANTY_CONTACT_EMAIL } from "@/lib/evidenceMessage";
import { WARRANTY_WELCOME_MESSAGE } from "@/lib/welcomeMessage";
import WarrantyTeamContactFooter from "./WarrantyTeamContactFooter";

const DOMAIN = "osaki.com";
const THINKING_DELAY_MS = 750;

const EMAIL_THANK_YOU =
  `Thank you! Our warranty team at ${WARRANTY_CONTACT_EMAIL} will respond within 24 hours.`;

const SELF_HELP_CLOSING =
  "Sounds good — try those steps first. If you still need us, you can start a new case anytime. " +
  "Our warranty team is also available by phone during business hours.";

const INITIAL_ISSUE_OPTIONS: AnswerOption[] = [
  { answer_key: "installation", label: "Installation Issue" },
  { answer_key: "delivery", label: "Delivery Issue" },
  { answer_key: "defect", label: "Defect / Malfunction" },
];

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assistantContentFromResponse(
  ticket: WarrantyTicketState | null,
  resp: Pick<WarrantySessionResponse, "assistant_message" | "terminal_enrichment">
): string | null {
  const node = ticket?.current_node;
  if (!node?.prompt && !resp.assistant_message) return null;
  return formatTerminalPrompt(
    node?.prompt ?? "",
    node?.evidence_required,
    node?.evidence_email,
    resp.assistant_message ?? resp.terminal_enrichment?.message
  );
}

export default function WarrantyChat() {
  const [sessionId] = useState<string>(() => {
    if (typeof window === "undefined") return uuidv4();
    const stored = sessionStorage.getItem("warranty_session_id");
    if (stored) return stored;
    const newId = uuidv4();
    sessionStorage.setItem("warranty_session_id", newId);
    return newId;
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warrantyState, setWarrantyState] = useState<WarrantyTicketState | null>(null);
  const [terminalEnrichment, setTerminalEnrichment] = useState<TerminalEnrichment | null>(null);
  const [optionsUsed, setOptionsUsed] = useState(false);
  const [sessionChecked, setSessionChecked] = useState(false);
  const [contactSubmitted, setContactSubmitted] = useState(false);
  const [helpConsent, setHelpConsent] = useState<"yes" | "no" | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, helpConsent]);

  const applySessionResponse = useCallback((resp: WarrantySessionResponse) => {
    setWarrantyState(resp.ticket);
    setTerminalEnrichment(resp.terminal_enrichment ?? null);
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

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await refreshWarrantyState();
        if (cancelled) return;
        const ticket = resp.ticket;
        if (ticket?.current_node?.is_terminal && ticket.current_node.prompt) {
          setMessages([
            {
              role: "assistant",
              content:
                assistantContentFromResponse(ticket, resp) ?? ticket.current_node.prompt,
            },
          ]);
        } else if (ticket?.ready_for_issue_type && ticket.model_name) {
          setMessages([
            { role: "assistant", content: WARRANTY_WELCOME_MESSAGE },
            {
              role: "assistant",
              content: `Great — I have **${ticket.model_name}** on file.\n\nWhat type of issue can we help you with? Choose below or describe it in your own words.`,
            },
          ]);
        } else if (ticket?.current_node?.prompt && !ticket.current_node.is_terminal) {
          setMessages([{ role: "assistant", content: ticket.current_node.prompt }]);
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
  }, [refreshWarrantyState]);

  const appendAssistantFromResponse = useCallback(
    async (
      ticket: WarrantyTicketState | null,
      resp: Pick<WarrantySessionResponse, "assistant_message" | "terminal_enrichment">
    ) => {
      await sleep(THINKING_DELAY_MS);
      const content = assistantContentFromResponse(ticket, resp);
      if (!content) return;
      setMessages((prev) => [...prev, { role: "assistant", content }]);
    },
    []
  );

  const registerModel = useCallback(
    async (text: string) => {
      setError(null);
      setLoading(true);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setInput("");

      try {
        const resp = await registerWarrantyModel(sessionId, text.trim(), DOMAIN);
        applySessionResponse(resp);
        const resolved = resp.resolved_model ?? resp.ticket?.model_name ?? text.trim();
        await sleep(THINKING_DELAY_MS);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Thanks! I have **${resolved}** registered.\n\nWhat type of issue can we help you with? Choose an option below or describe it in your own words.`,
          },
        ]);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [sessionId, applySessionResponse]
  );

  const handleQuickStart = useCallback(
    async (issueType: "installation" | "delivery" | "defect", label: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setHelpConsent(null);

      try {
        const resp = await quickStartWarranty(sessionId, issueType, DOMAIN);
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
    [loading, sessionId, applySessionResponse, appendAssistantFromResponse]
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
        if (resp.tracking_summary?.message) {
          await sleep(THINKING_DELAY_MS);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: resp.tracking_summary!.message },
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

  const startViaNaturalLanguage = useCallback(
    async (text: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setHelpConsent(null);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setInput("");

      try {
        const resp = await naturalStartWarranty(sessionId, text, DOMAIN);
        applySessionResponse(resp);
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
    [loading, sessionId, applySessionResponse, appendAssistantFromResponse]
  );

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || loading) return;

      const trimmed = text.trim();
      const hasModel = Boolean(
        warrantyState?.model_confirmed || warrantyState?.model_name
      );
      const atIssueType =
        warrantyState?.ready_for_issue_type ||
        warrantyState?.current_node?.node_id === "issue_type";

      if (!hasModel) {
        await registerModel(trimmed);
        return;
      }

      if (atIssueType && !warrantyState?.issue_type) {
        await startViaNaturalLanguage(trimmed);
        return;
      }

      if (warrantyState?.ticket_id && !warrantyState.current_node?.is_terminal) {
        await advanceWarranty(trimmed, trimmed);
        setInput("");
        return;
      }

      if (warrantyState?.current_node?.is_terminal && helpConsent === null) {
        setError("Please tap Yes or No below so we know how to help next.");
        return;
      }
    },
    [loading, warrantyState, registerModel, startViaNaturalLanguage, advanceWarranty, helpConsent]
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
        {
          role: "assistant",
          content:
            "No problem — please share your email below so our warranty team can follow up. " +
            "Photos or videos are optional.",
        },
      ]);
    } else {
      setHelpConsent("no");
      setMessages((prev) => [...prev, { role: "assistant", content: SELF_HELP_CLOSING }]);
    }
  }

  const isAwaitingAdmin =
    warrantyState?.status === "awaiting_admin_review" ||
    warrantyState?.status === "admin_reviewing";
  const isTerminal = warrantyState?.current_node?.is_terminal ?? false;

  const needsModelRegistration =
    sessionChecked &&
    !isTerminal &&
    !(warrantyState?.model_confirmed || warrantyState?.model_name);

  const showIssueTypeOptions =
    sessionChecked &&
    !loading &&
    !isTerminal &&
    !needsModelRegistration &&
    (warrantyState?.ready_for_issue_type ||
      (!!warrantyState?.model_name &&
        warrantyState?.current_node?.node_id === "issue_type" &&
        !warrantyState?.issue_type));

  const hasWorkflowOptions =
    !optionsUsed &&
    !loading &&
    !showIssueTypeOptions &&
    (warrantyState?.current_node?.options?.length ?? 0) > 0;

  const helpOfferOptions =
    terminalEnrichment?.help_offer_options ?? [];

  const showHelpOffer =
    isTerminal &&
    terminalEnrichment?.phase === "awaiting_help_consent" &&
    helpConsent === null &&
    !contactSubmitted &&
    helpOfferOptions.length > 0;

  const showEmailSection =
    isTerminal &&
    helpConsent === "yes" &&
    !contactSubmitted &&
    warrantyState?.ticket_id;

  const showInputBar =
    !isTerminal ||
    (isTerminal && helpConsent === null && !contactSubmitted && !showHelpOffer);

  const inputPlaceholder = needsModelRegistration
    ? "Enter your chair model (e.g. OS-4000T, Solo Flex)…"
    : showIssueTypeOptions
      ? "Describe your issue (e.g. my chair won't turn on)…"
      : warrantyState?.ticket_id
        ? "Type your answer…"
        : "Enter your chair model…";

  return (
    <div className="mx-auto flex h-[calc(100dvh-64px)] w-full max-w-2xl flex-col">
      {warrantyState && (
        <div className="flex items-center justify-between border-b border-gray-100 bg-white px-4 py-2">
          <TicketStatusBadge
            status={warrantyState.status}
            ticketId={warrantyState.ticket_id}
          />
          {warrantyState.model_name && (
            <span className="text-xs text-gray-500">{warrantyState.model_name}</span>
          )}
        </div>
      )}

      {isAwaitingAdmin && (
        <div className="mx-4 mt-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-sm font-medium text-amber-800">Under Support Review</p>
          <p className="mt-0.5 text-xs text-amber-700">
            Your case has been prepared for support team review.
          </p>
        </div>
      )}

      <div className="chat-scroll flex-1 overflow-y-auto px-3 py-4 sm:px-4">
        <div className="space-y-4">
          {messages.map((msg, i) => (
            <ChatMessageBubble key={i} message={msg} />
          ))}
        </div>

        {showIssueTypeOptions && (
          <div className="mt-4 rounded-2xl border border-gray-100 bg-white px-3 py-4 shadow-sm sm:px-4">
            <p className="mb-3 text-sm font-medium text-gray-700">
              What type of issue can we help you with?
            </p>
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
        <div className="border-t border-gray-100 bg-white px-3 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-4">
          <p className="mb-2 text-xs font-medium text-gray-600">
            Tap an option or type your answer below
          </p>
          <AnswerOptions
            options={warrantyState!.current_node!.options}
            onSelect={handleOptionSelect}
            disabled={loading}
            variant="stack"
          />
        </div>
      )}

      {showHelpOffer && (
        <div className="border-t border-gray-100 bg-white px-3 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-4">
          <AnswerOptions
            options={helpOfferOptions}
            onSelect={handleHelpOffer}
            disabled={loading}
            variant="stack"
          />
        </div>
      )}

      {showEmailSection && (
        <div className="border-t border-gray-100 bg-white p-3 pb-[max(1rem,env(safe-area-inset-bottom))] sm:p-4">
          <EvidenceUploader
            ticketId={warrantyState!.ticket_id}
            evidenceRequired={warrantyState!.current_node?.evidence_required}
            onContactSuccess={() => {
              setContactSubmitted(true);
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content:
                    "Thank you — your email has been received. Our warranty team will follow up within 24 hours.",
                },
              ]);
            }}
            onUploadSuccess={(filename) => {
              setContactSubmitted(true);
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: `Thank you — "${filename}" has been received. Our team will review it shortly.`,
                },
              ]);
            }}
          />
        </div>
      )}

      {showInputBar && (
        <form
          onSubmit={handleSubmit}
          className="border-t border-gray-200 bg-white px-3 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 sm:px-4"
        >
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
        <div className="border-t border-gray-100 bg-white px-4 py-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-center">
          {contactSubmitted && (
            <p className="text-sm text-gray-600">
              Your case has been submitted. Our team will be in touch.
            </p>
          )}
          <WarrantyTeamContactFooter className="mt-4 text-left" />
          <button
            onClick={() => {
              sessionStorage.removeItem("warranty_session_id");
              window.location.reload();
            }}
            className="mt-3 min-h-[44px] text-sm text-brand-600 underline hover:text-brand-800"
          >
            Start a new case
          </button>
        </div>
      )}
    </div>
  );
}
