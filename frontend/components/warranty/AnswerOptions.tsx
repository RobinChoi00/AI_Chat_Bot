"use client";

import type { AnswerOption } from "@/lib/types";

interface Props {
  options: AnswerOption[];
  onSelect: (answerKey: string, label: string) => void;
  disabled?: boolean;
  /** inline = compact grid; stack = full-width tap targets (mobile-friendly) */
  variant?: "inline" | "stack";
}

const OPTION_ICONS: Record<string, string> = {
  installation: "🔧",
  delivery: "📦",
  defect: "⚙️",
  warranty: "🛡️",
  sales: "💬",
  power: "🔌",
  remote: "📱",
  air: "💨",
  rolling: "🔄",
  recline: "↕️",
  footrest: "🦶",
  cosmetic: "✨",
  yes: "✓",
  no: "✗",
};

function iconFor(key: string, label: string): string {
  if (OPTION_ICONS[key]) return OPTION_ICONS[key];
  const lower = label.toLowerCase();
  if (lower.startsWith("yes")) return "✓";
  if (lower.startsWith("no")) return "✗";
  return "→";
}

/**
 * Renders warranty workflow options as tap targets.
 * Mobile: compact (44px min height). sm+: roomier buttons for desktop/tablet.
 */
export default function AnswerOptions({
  options,
  onSelect,
  disabled,
  variant = "stack",
}: Props) {
  if (!options.length) return null;

  const isStack = variant === "stack";

  return (
    <div
      className={
        isStack
          ? "flex w-full flex-col gap-2 sm:gap-2.5"
          : "grid grid-cols-1 gap-1.5 sm:grid-cols-2 sm:gap-2"
      }
    >
      {options.map((opt) => (
        <button
          key={opt.answer_key}
          type="button"
          onClick={() => onSelect(opt.answer_key, opt.label)}
          disabled={disabled}
          className={`flex min-h-[44px] w-full items-center gap-2 rounded-xl border px-3 py-2.5 text-left text-[13px] font-medium transition active:scale-[0.98] sm:min-h-[52px] sm:gap-3 sm:rounded-2xl sm:px-4 sm:py-3.5 sm:text-sm ${
            disabled
              ? "cursor-not-allowed border-gray-200 bg-gray-50 text-gray-400"
              : "border-brand-200 bg-white text-gray-800 shadow-sm hover:border-brand-400 hover:bg-brand-50/60"
          }`}
        >
          <span
            className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-brand-50 text-sm sm:h-9 sm:w-9 sm:text-base"
            aria-hidden
          >
            {iconFor(opt.answer_key, opt.label)}
          </span>
          <span className="flex-1 leading-tight sm:leading-snug">{opt.label}</span>
          <span className="flex-shrink-0 text-sm text-brand-500 sm:text-base" aria-hidden>
            ›
          </span>
        </button>
      ))}
    </div>
  );
}
