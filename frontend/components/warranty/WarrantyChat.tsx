"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  getWarrantySession,
  quickStartWarranty,
  naturalStartWarranty,
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
/** Brief pause so replies feel considered, not instant. */
const THINKING_DELAY_MS = 750;

const EMAIL_THANK_YOU =
  `Thank you! Our warranty team at ${WARRANTY_CONTACT_EMAIL} will respond within 24 hours.`;

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
  if (!node?.prompt) return null;
  return formatTerminalPrompt(
    node.prompt,
    node.evidence_required,
    node.evidence_email,
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
  const [showContactForm, setShowContactForm] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, showContactForm]);

  const applySessionResponse = useCallback((resp: WarrantySessionResponse) => {
    setWarrantyState(resp.ticket);
    setTerminalEnrichment(resp.terminal_enrichment ?? null);
    if (resp.terminal_enrichment?.defer_email) {
      setShowContactForm(false);
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
        const prompt = resp.ticket?.current_node?.prompt;
        if (prompt && !resp.ticket?.current_node?.is_terminal) {
          setMessages([{ role: "assistant", content: prompt }]);
        } else if (prompt && resp.ticket?.current_node?.is_terminal) {
          setMessages([
            {
              role: "assistant",
              content:
                assistantContentFromResponse(resp.ticket, resp) ?? prompt,
            },
          ]);
        } else if (!resp.ticket?.ticket_id) {
          setMessages([{ role: "assistant", content: WARRANTY_WELCOME_MESSAGE }]);
        }
      } catch {
        // Non-fatal — user can still pick an initial option.
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
      const node = ticket?.current_node;
      if (!node?.prompt) return;
      const content = assistantContentFromResponse(ticket, resp);
      if (!content) return;
      setMessages((prev) => [...prev, { role: "assistant", content }]);
    },
    []
  );

  const appendEmailThankYou = useCallback(() => {
    setMessages((prev) => {
      if (prev.some((m) => m.content.includes("will respond within 24 hours"))) {
        return prev;
      }
      return [...prev, { role: "assistant", content: EMAIL_THANK_YOU }];
    });
  }, []);

  const handleQuickStart = useCallback(
    async (issueType: "installation" | "delivery" | "defect", label: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setShowContactForm(false);

      try {
        const resp = await quickStartWarranty(sessionId, issueType, DOMAIN);
        applySessionResponse(resp);
        setMessages([{ role: "user", content: label }]);
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
      setShowContactForm(false);
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
        if (resp.email_notified) {
          appendEmailThankYou();
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
    [
      warrantyState?.ticket_id,
      loading,
      applySessionResponse,
      appendAssistantFromResponse,
      appendEmailThankYou,
    ]
  );

  const startViaNaturalLanguage = useCallback(
    async (text: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setShowContactForm(false);
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

      if (warrantyState?.ticket_id && !warrantyState.current_node?.is_terminal) {
        await advanceWarranty(text.trim(), text.trim());
        setInput("");
        return;
      }

      await startViaNaturalLanguage(text.trim());
    },
    [loading, warrantyState, advanceWarranty, startViaNaturalLanguage]
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

  const isAwaitingAdmin =
    warrantyState?.status === "awaiting_admin_review" ||
    warrantyState?.status === "admin_reviewing";
  const hasWorkflowOptions =
    !optionsUsed &&
    !loading &&
    (warrantyState?.current_node?.options?.length ?? 0) > 0;
  const showInitialOptions =
    sessionChecked &&
    !warrantyState?.ticket_id &&
    !loading;
  const isTerminal = warrantyState?.current_node?.is_terminal ?? false;

  const deferEmail = terminalEnrichment?.defer_email ?? false;
  const wantsContactForm = terminalEnrichment?.show_contact_form !== false;
  const showEmailSection =
    isTerminal &&
    warrantyState?.ticket_id &&
    !contactSubmitted &&
    (!deferEmail || showContactForm) &&
    (wantsContactForm || showContactForm);

  const showStillNeedHelp =
    isTerminal &&
    deferEmail &&
    !showContactForm &&
    !contactSubmitted &&
    warrantyState?.ticket_id;

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
            Your case has been prepared for support team review. Final warranty
            decisions are handled by our support team.
          </p>
        </div>
      )}

      <div className="chat-scroll flex-1 overflow-y-auto px-3 py-4 sm:px-4">
        <div className="space-y-4">
          {messages.map((msg, i) => (
            <ChatMessageBubble key={i} message={msg} />
          ))}
        </div>

        {showInitialOptions && (
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
            <p className="mt-3 text-center text-xs text-gray-400">
              Or describe your issue in the text box below.
            </p>
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

      {showStillNeedHelp && (
        <div className="border-t border-gray-100 bg-white px-3 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-4">
          <button
            type="button"
            onClick={() => setShowContactForm(true)}
            className="flex min-h-[52px] w-full items-center justify-center rounded-2xl bg-brand-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 active:scale-[0.98]"
          >
            I still need help — contact warranty team
          </button>
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

      {!isTerminal && (
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
              placeholder={
                warrantyState?.ticket_id
                  ? "Type your answer…"
                  : "Describe your issue (e.g. my chair won't turn on)…"
              }
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
          <p className="mt-1.5 text-center text-[10px] text-gray-400">
            Warranty decisions are reviewed by our support team.
          </p>
          <WarrantyTeamContactFooter compact className="mt-2" />
        </form>
      )}

      {isTerminal && contactSubmitted && (
        <div className="border-t border-gray-100 bg-white px-4 py-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-center">
          <p className="text-sm text-gray-600">
            Your case has been submitted. Our team will be in touch.
          </p>
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
