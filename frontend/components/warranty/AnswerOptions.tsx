"use client";

import type { AnswerOption } from "@/lib/types";

interface Props {
  options: AnswerOption[];
  onSelect: (answerKey: string, label: string) => void;
  disabled?: boolean;
  /** inline = pill row (workflow steps); stack = full-width buttons (start screen) */
  variant?: "inline" | "stack";
}

/**
 * Renders the current warranty workflow options as clickable buttons.
 *
 * When the user clicks a button, onSelect is called with the answer_key
 * (sent to the backend via the chat endpoint) and the human-readable label
 * (displayed as the user's message bubble).
 */
export default function AnswerOptions({
  options,
  onSelect,
  disabled,
  variant = "inline",
}: Props) {
  if (!options.length) return null;

  const isStack = variant === "stack";

  return (
    <div
      className={
        isStack
          ? "flex w-full max-w-sm flex-col gap-2.5 px-2 py-1"
          : "flex flex-wrap gap-2 px-2 py-1"
      }
    >
      {options.map((opt) => (
        <button
          key={opt.answer_key}
          onClick={() => onSelect(opt.answer_key, opt.label)}
          disabled={disabled}
          className={`font-medium transition ${
            isStack
              ? `w-full rounded-xl border px-4 py-3 text-sm ${
                  disabled
                    ? "cursor-not-allowed border-gray-200 bg-gray-50 text-gray-400"
                    : "border-brand-500 bg-white text-brand-700 shadow-sm hover:bg-brand-50 active:scale-[0.98]"
                }`
              : `rounded-full border px-4 py-1.5 text-sm ${
                  disabled
                    ? "cursor-not-allowed border-gray-200 bg-gray-50 text-gray-400"
                    : "border-brand-500 bg-white text-brand-700 hover:bg-brand-50 active:scale-95"
                }`
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
