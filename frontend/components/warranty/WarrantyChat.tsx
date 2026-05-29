"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import { streamChat, getWarrantySession } from "@/lib/api";
import type { ChatMessage, WarrantyTicketState } from "@/lib/types";
import ChatMessageBubble from "./ChatMessageBubble";
import AnswerOptions from "./AnswerOptions";
import EvidenceUploader from "./EvidenceUploader";
import TicketStatusBadge from "./TicketStatusBadge";

const DOMAIN = "osaki.com";

/**
 * WarrantyChat
 * ============
 * Full warranty chat widget.
 *
 * Flow:
 * 1. User types a message and submits.
 * 2. Message is streamed to POST /api/v1/chat.
 * 3. After the stream ends, GET /api/v1/warranty/session/{session_id} is
 *    polled to get structured node data (ticket_id, options, status).
 * 4. If options are present, AnswerOptions renders clickable buttons.
 *    Clicking a button sends the answer_key as the next user message.
 * 5. If the ticket status is awaiting_admin_review, a safety banner is shown.
 * 6. If the ticket requires evidence, EvidenceUploader is shown.
 */
export default function WarrantyChat() {
  // Stable session_id for the duration of the browser session
  const [sessionId] = useState<string>(() => {
    if (typeof window === "undefined") return uuidv4();
    const stored = sessionStorage.getItem("warranty_session_id");
    if (stored) return stored;
    const newId = uuidv4();
    sessionStorage.setItem("warranty_session_id", newId);
    return newId;
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingContent, setStreamingContent] = useState<string>("");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warrantyState, setWarrantyState] = useState<WarrantyTicketState | null>(null);
  const [optionsUsed, setOptionsUsed] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when messages or streaming content change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // Fetch warranty session state after each assistant turn
  const refreshWarrantyState = useCallback(async () => {
    try {
      const resp = await getWarrantySession(sessionId);
      setWarrantyState(resp.ticket);
      setOptionsUsed(false);
    } catch {
      // Non-fatal: warranty state is optional UI enrichment
    }
  }, [sessionId]);

  // Send a message (either typed or from a button option)
  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || loading) return;

      const userMsg: ChatMessage = { role: "user", content: text };
      const history = [...messages];
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setStreamingContent("");
      setError(null);
      setLoading(true);
      setOptionsUsed(true);

      let fullResponse = "";
      try {
        const stream = streamChat({
          session_id: sessionId,
          user_query: text,
          chat_history: history,
          current_domain: DOMAIN,
        });

        for await (const chunk of stream) {
          fullResponse += chunk;
          setStreamingContent(fullResponse);
        }

        // Finalize: move streaming content into messages
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: fullResponse },
        ]);
        setStreamingContent("");
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "I'm sorry, I encountered an error. Please try again or contact support.",
          },
        ]);
        setStreamingContent("");
      } finally {
        setLoading(false);
        // Refresh warranty state after assistant responds
        await refreshWarrantyState();
        inputRef.current?.focus();
      }
    },
    [loading, messages, sessionId, refreshWarrantyState]
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

  // Option button clicked: send the answer_key as user message
  function handleOptionSelect(answerKey: string, label: string) {
    // The user sees the human label; the backend receives the answer_key.
    // We send the answer_key as the text so the backend maps it correctly.
    sendMessage(answerKey);
    // Show label as the user's visible message (optimistic update below handles display)
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "user" && last.content === answerKey) {
        return [...prev.slice(0, -1), { role: "user", content: label }];
      }
      return prev;
    });
  }

  // Derived warranty UI state
  const isAwaitingAdmin =
    warrantyState?.status === "awaiting_admin_review" ||
    warrantyState?.status === "admin_reviewing";
  const needsEvidence = warrantyState?.status === "awaiting_evidence";
  const hasOptions =
    !optionsUsed &&
    !loading &&
    (warrantyState?.current_node?.options?.length ?? 0) > 0;
  const isTerminal = warrantyState?.current_node?.is_terminal ?? false;

  return (
    <div className="mx-auto flex h-[calc(100vh-64px)] w-full max-w-2xl flex-col">
      {/* --- Status bar --- */}
      {warrantyState && (
        <div className="flex items-center justify-between border-b border-gray-100 bg-white px-4 py-2">
          <TicketStatusBadge
            status={warrantyState.status}
            ticketId={warrantyState.ticket_id}
          />
          {warrantyState.model_name && (
            <span className="text-xs text-gray-500">
              {warrantyState.model_name}
            </span>
          )}
        </div>
      )}

      {/* --- Admin-review safety banner --- */}
      {isAwaitingAdmin && (
        <div className="mx-4 mt-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-sm font-medium text-amber-800">
            ⏳ Under Support Review
          </p>
          <p className="mt-0.5 text-xs text-amber-700">
            Your case has been prepared for support team review. Final warranty
            decisions are handled by our support team.
          </p>
        </div>
      )}

      {/* --- Message list --- */}
      <div className="chat-scroll flex-1 overflow-y-auto px-4 py-4">
        {/* Empty state */}
        {messages.length === 0 && !streamingContent && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-4 text-5xl">🛡️</div>
            <h2 className="text-lg font-semibold text-gray-800">
              Warranty Support
            </h2>
            <p className="mt-1 max-w-xs text-sm text-gray-500">
              Describe your issue and we&apos;ll guide you through the warranty
              process step by step.
            </p>
            <p className="mt-3 text-xs text-gray-400">
              All warranty decisions are reviewed by our support team.
            </p>
          </div>
        )}

        {/* Message bubbles */}
        <div className="space-y-3">
          {messages.map((msg, i) => (
            <ChatMessageBubble key={i} message={msg} />
          ))}

          {/* Streaming bubble */}
          {streamingContent && (
            <ChatMessageBubble
              message={{ role: "assistant", content: streamingContent }}
              isStreaming
            />
          )}
        </div>

        {/* Loading indicator (before first chunk arrives) */}
        {loading && !streamingContent && (
          <div className="mt-3 flex items-center gap-2 text-gray-400">
            <div className="flex gap-1">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:0ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:150ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:300ms]" />
            </div>
            <span className="text-xs">Thinking…</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            ⚠️ {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* --- Clickable answer options --- */}
      {hasOptions && !isTerminal && (
        <div className="border-t border-gray-100 bg-white px-4 py-3">
          <p className="mb-2 text-xs text-gray-500">Select an option:</p>
          <AnswerOptions
            options={warrantyState!.current_node!.options}
            onSelect={handleOptionSelect}
            disabled={loading}
          />
        </div>
      )}

      {/* --- Evidence uploader (when terminal requires it) --- */}
      {(needsEvidence || (isTerminal && warrantyState?.ticket_id)) &&
        warrantyState?.ticket_id && (
          <div className="border-t border-gray-100 bg-white p-4">
            <EvidenceUploader
              ticketId={warrantyState.ticket_id}
              onUploadSuccess={(filename) => {
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

      {/* --- Text input area --- */}
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
              placeholder="Describe your issue…"
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

      {/* Terminal state: show restart link */}
      {isTerminal && (
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
