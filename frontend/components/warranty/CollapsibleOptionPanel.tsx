"use client";

import type { ReactNode } from "react";

interface Props {
  title: string;
  hint?: string;
  optionCount: number;
  expanded: boolean;
  onToggle: () => void;
  children: ReactNode;
  /** When false, only the toggle bar is shown (content hidden). */
  disabled?: boolean;
}

export default function CollapsibleOptionPanel({
  title,
  hint,
  optionCount,
  expanded,
  onToggle,
  children,
  disabled,
}: Props) {
  return (
    <div className="rounded-xl border border-gray-100 bg-white shadow-sm sm:rounded-2xl">
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        aria-expanded={expanded}
        className="flex min-h-[44px] w-full items-center justify-between gap-2 px-3 py-2.5 text-left sm:px-4 sm:py-3 disabled:opacity-60"
      >
        <div className="min-w-0">
          <p className="text-xs font-semibold text-gray-800 sm:text-sm">{title}</p>
          <p className="mt-0.5 text-[11px] text-gray-500 sm:text-xs">
            {expanded
              ? hint ?? "Tap to hide options"
              : `${optionCount} option${optionCount === 1 ? "" : "s"} — tap to show`}
          </p>
        </div>
        <span
          className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gray-100 text-gray-600 transition-transform ${
            expanded ? "rotate-180" : ""
          }`}
          aria-hidden
        >
          ▾
        </span>
      </button>
      {expanded && (
        <div className="border-t border-gray-100 px-3 pb-3 pt-2 sm:px-4 sm:pb-4">{children}</div>
      )}
    </div>
  );
}
