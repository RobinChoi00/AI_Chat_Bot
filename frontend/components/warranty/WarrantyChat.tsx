"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  getWarrantySession,
  quickStartWarranty,
  naturalStartWarranty,
  submitWarrantyAnswer,
} from "@/lib/api";
import type { AnswerOption, ChatMessage, WarrantyTicketState } from "@/lib/types";
import ChatMessageBubble from "./ChatMessageBubble";
import AnswerOptions from "./AnswerOptions";
import EvidenceUploader from "./EvidenceUploader";
import TicketStatusBadge from "./TicketStatusBadge";
import { formatTerminalPrompt, WARRANTY_CONTACT_EMAIL } from "@/lib/evidenceMessage";

const DOMAIN = "osaki.com";

const EMAIL_THANK_YOU =
  `Thank you! Our warranty team at ${WARRANTY_CONTACT_EMAIL} will respond within 24 hours.`;

/** Shown immediately on page load — maps to flowchart issue_type answer_keys. */
const INITIAL_ISSUE_OPTIONS: AnswerOption[] = [
  { answer_key: "installation", label: "Installation Issue" },
  { answer_key: "delivery", label: "Delivery Issue" },
  { answer_key: "defect", label: "Defect / Malfunction" },
];

/**
 * WarrantyChat
 * ============
 * Hybrid warranty intake: deterministic flowchart + natural-language answers.
 *
 * Buttons submit answer_keys directly; typed text is mapped server-side via NLP
 * while the workflow engine keeps branching, admin records, and evidence rules.
 */
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
  const [optionsUsed, setOptionsUsed] = useState(false);
  const [sessionChecked, setSessionChecked] = useState(false);
  const [contactSubmitted, setContactSubmitted] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const refreshWarrantyState = useCallback(async () => {
    const resp = await getWarrantySession(sessionId);
    setWarrantyState(resp.ticket);
    setOptionsUsed(false);
    return resp;
  }, [sessionId]);

  // Restore an in-progress ticket when the user refreshes the page.
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
              content: formatTerminalPrompt(
                prompt,
                resp.ticket.current_node.evidence_required,
                resp.ticket.current_node.evidence_email
              ),
            },
          ]);
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

  const appendAssistantPrompt = useCallback((ticket: WarrantyTicketState | null) => {
    const node = ticket?.current_node;
    if (!node?.prompt) return;
    const content = node.is_terminal
      ? formatTerminalPrompt(
          node.prompt,
          node.evidence_required,
          node.evidence_email
        )
      : node.prompt;
    setMessages((prev) => [...prev, { role: "assistant", content }]);
  }, []);

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

      try {
        const resp = await quickStartWarranty(sessionId, issueType, DOMAIN);
        setWarrantyState(resp.ticket);
        setMessages([
          { role: "user", content: label },
        ]);
        appendAssistantPrompt(resp.ticket);
        setOptionsUsed(false);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setOptionsUsed(false);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [loading, sessionId, appendAssistantPrompt]
  );

  const advanceWarranty = useCallback(
    async (answer: string, userLabel: string) => {
      const ticketId = warrantyState?.ticket_id;
      if (!ticketId || loading) return;

      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setMessages((prev) => [...prev, { role: "user", content: userLabel }]);

      try {
        const resp = await submitWarrantyAnswer(ticketId, answer);
        setWarrantyState(resp.ticket);
        if (resp.tracking_summary?.message) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: resp.tracking_summary!.message },
          ]);
        }
        if (resp.email_notified) {
          appendEmailThankYou();
        }
        appendAssistantPrompt(resp.ticket);
        setOptionsUsed(false);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setOptionsUsed(false);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [warrantyState?.ticket_id, loading, appendAssistantPrompt, appendEmailThankYou]
  );

  const startViaNaturalLanguage = useCallback(
    async (text: string) => {
      if (loading) return;
      setError(null);
      setLoading(true);
      setOptionsUsed(true);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setInput("");

      try {
        const resp = await naturalStartWarranty(sessionId, text, DOMAIN);
        setWarrantyState(resp.ticket);
        appendAssistantPrompt(resp.ticket);
        setOptionsUsed(false);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setOptionsUsed(false);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [loading, sessionId, appendAssistantPrompt]
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
    !loading &&
    messages.length === 0;
  const isTerminal = warrantyState?.current_node?.is_terminal ?? false;

  return (
    <div className="mx-auto flex h-[calc(100vh-64px)] w-full max-w-2xl flex-col">
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
          <p className="text-sm font-medium text-amber-800">⏳ Under Support Review</p>
          <p className="mt-0.5 text-xs text-amber-700">
            Your case has been prepared for support team review. Final warranty
            decisions are handled by our support team.
          </p>
        </div>
      )}

      <div className="chat-scroll flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-4 text-5xl">🛡️</div>
            <h2 className="text-lg font-semibold text-gray-800">Warranty Support</h2>
            <p className="mt-1 max-w-xs text-sm text-gray-500">
              What type of issue can we help you with today?
            </p>
            {showInitialOptions && (
              <div className="mt-6 w-full max-w-sm">
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
            <p className="mt-4 text-xs text-gray-400">
              Or describe your issue in the text box below.
            </p>
            <p className="mt-2 text-xs text-gray-400">
              All warranty decisions are reviewed by our support team.
            </p>
          </div>
        )}

        <div className="space-y-3">
          {messages.map((msg, i) => (
            <ChatMessageBubble key={i} message={msg} />
          ))}
        </div>

        {loading && (
          <div className="mt-3 flex items-center gap-2 text-gray-400">
            <div className="flex gap-1">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:0ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:150ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:300ms]" />
            </div>
            <span className="text-xs">Working…</span>
          </div>
        )}

        {error && (
          <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            ⚠️ {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {hasWorkflowOptions && !isTerminal && (
        <div className="border-t border-gray-100 bg-white px-4 py-3">
          <p className="mb-2 text-xs text-gray-500">
            Select an option or type your answer below:
          </p>
          <AnswerOptions
            options={warrantyState!.current_node!.options}
            onSelect={handleOptionSelect}
            disabled={loading}
          />
        </div>
      )}

      {isTerminal && warrantyState?.ticket_id && !contactSubmitted && (
          <div className="border-t border-gray-100 bg-white p-4">
            <EvidenceUploader
              ticketId={warrantyState.ticket_id}
              evidenceRequired={warrantyState.current_node?.evidence_required}
              onContactSuccess={() => {
                setContactSubmitted(true);
                setMessages((prev) => [
                  ...prev,
                  {
                    role: "assistant",
                    content:
                      "✅ Thank you — your email has been received. Our warranty team will follow up within 24 hours.",
                  },
                ]);
              }}
              onUploadSuccess={(filename) => {
                setContactSubmitted(true);
                setMessages((prev) => [
                  ...prev,
                  {
                    role: "assistant",
                    content: `✅ Thank you — "${filename}" has been received. Our team will review it shortly.`,
                  },
                ]);
              }}
            />
          </div>
        )}

      {!isTerminal && (
        <form
          onSubmit={handleSubmit}
          className="border-t border-gray-200 bg-white px-4 pb-4 pt-3"
        >
          <div className="flex items-end gap-2 rounded-xl border border-gray-200 bg-gray-50 p-2 focus-within:border-brand-500 focus-within:ring-1 focus-within:ring-brand-500">
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                warrantyState?.ticket_id
                  ? "Type your answer in your own words…"
                  : "Describe your issue (e.g. my chair won't turn on)…"
              }
              disabled={loading}
              className="flex-1 resize-none bg-transparent px-1 py-1 text-sm text-gray-900 placeholder-gray-400 focus:outline-none disabled:opacity-60"
              style={{ maxHeight: "120px" }}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className={`flex-shrink-0 rounded-lg px-4 py-2 text-sm font-medium transition ${
                !input.trim() || loading
                  ? "cursor-not-allowed bg-gray-200 text-gray-400"
                  : "bg-brand-600 text-white hover:bg-brand-700 active:scale-95"
              }`}
            >
              {loading ? "…" : "Send"}
            </button>
          </div>
          <p className="mt-1.5 text-center text-[10px] text-gray-400">
            Warranty decisions are reviewed by our support team — we never
            promise replacements or repairs automatically.
          </p>
        </form>
      )}

      {isTerminal && contactSubmitted && (
        <div className="border-t border-gray-100 bg-white px-4 py-4 text-center">
          <p className="text-sm text-gray-600">
            Your case has been submitted. Our team will be in touch.
          </p>
          <button
            onClick={() => {
              sessionStorage.removeItem("warranty_session_id");
              window.location.reload();
            }}
            className="mt-2 text-xs text-brand-600 underline hover:text-brand-800"
          >
            Start a new case
          </button>
        </div>
      )}
    </div>
  );
}
