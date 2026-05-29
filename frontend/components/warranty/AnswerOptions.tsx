"use client";

import type { AnswerOption } from "@/lib/types";

interface Props {
  options: AnswerOption[];
  onSelect: (answerKey: string, label: string) => void;
  disabled?: boolean;
}

/**
 * Renders the current warranty workflow options as clickable buttons.
 *
 * When the user clicks a button, onSelect is called with the answer_key
 * (sent to the backend via the chat endpoint) and the human-readable label
 * (displayed as the user's message bubble).
 */
export default function AnswerOptions({ options, onSelect, disabled }: Props) {
  if (!options.length) return null;

  return (
    <div className="flex flex-wrap gap-2 px-2 py-1">
      {options.map((opt) => (
        <button
          key={opt.answer_key}
          onClick={() => onSelect(opt.answer_key, opt.label)}
          disabled={disabled}
          className={`rounded-full border px-4 py-1.5 text-sm font-medium transition
            ${
              disabled
                ? "cursor-not-allowed border-gray-200 bg-gray-50 text-gray-400"
                : "border-brand-500 bg-white text-brand-700 hover:bg-brand-50 active:scale-95"
            }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
