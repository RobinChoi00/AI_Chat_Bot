"use client";

import { useState } from "react";
import { submitChatFeedback } from "@/lib/api";

interface Props {
  sessionId: string;
  messageContent: string;
  ticketId?: string;
  domain?: string;
  context?: "warranty" | "chat";
}

type Status = "idle" | "submitting" | "submitted" | "error";

/**
 * Thumbs-up / thumbs-down affordance rendered under each assistant message.
 *
 * On 👍 → immediate submit, buttons collapse to "Thanks!".
 * On 👎 → open a small comment textarea (optional), submit persists the vote
 *          plus any comment. Comment is not required.
 */
export default function FeedbackButtons({
  sessionId,
  messageContent,
  ticketId,
  domain,
  context = "warranty",
}: Props) {
  const [selected, setSelected] = useState<"up" | "down" | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [comment, setComment] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  async function submit(rating: "up" | "down", withComment?: string) {
    setStatus("submitting");
    setErrorMsg("");
    try {
      await submitChatFeedback({
        sessionId,
        rating,
        messageContent,
        comment: withComment,
        ticketId,
        domain,
        context,
      });
      setStatus("submitted");
      setSelected(rating);
    } catch (err) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : "Could not submit feedback.");
      setSelected(null);
    }
  }

  if (status === "submitted") {
    return (
      <span className="text-[11px] text-gray-500">
        Thanks for your feedback{selected === "down" && comment ? " — noted." : "!"}
      </span>
    );
  }

  if (selected === "down") {
    return (
      <div className="flex w-full basis-full flex-col gap-1.5">
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="What went wrong? (optional)"
          maxLength={1000}
          rows={2}
          className="w-full resize-none rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-xs text-gray-800 placeholder-gray-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
        />
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={status === "submitting"}
            onClick={() => submit("down", comment.trim() || undefined)}
            className="rounded-full bg-brand-600 px-3 py-1 text-[11px] font-medium text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {status === "submitting" ? "Sending…" : "Send"}
          </button>
          <button
            type="button"
            disabled={status === "submitting"}
            onClick={() => {
              setSelected(null);
              setStatus("idle");
              setComment("");
            }}
            className="text-[11px] text-gray-500 hover:text-gray-700"
          >
            Cancel
          </button>
          {errorMsg && <span className="text-[11px] text-red-600">{errorMsg}</span>}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={() => submit("up")}
        disabled={status === "submitting"}
        aria-label="Mark answer helpful"
        title="Helpful"
        className="flex h-6 w-6 items-center justify-center rounded-full border border-gray-200 bg-white text-xs text-gray-500 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700 disabled:opacity-60"
      >
        👍
      </button>
      <button
        type="button"
        onClick={() => {
          setSelected("down");
          setStatus("idle");
        }}
        disabled={status === "submitting"}
        aria-label="Mark answer not helpful"
        title="Not helpful"
        className="flex h-6 w-6 items-center justify-center rounded-full border border-gray-200 bg-white text-xs text-gray-500 hover:border-red-300 hover:bg-red-50 hover:text-red-600 disabled:opacity-60"
      >
        👎
      </button>
      {errorMsg && <span className="text-[11px] text-red-600">{errorMsg}</span>}
    </div>
  );
}
