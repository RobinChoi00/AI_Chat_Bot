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
  // Entry / routing
  warranty: "🛡️",
  sales: "💬",
  installation: "🔧",
  delivery: "📦",
  defect: "⚙️",
  other: "📋",

  // Installation
  footrest_or_no_air: "💨",
  general_setup: "📖",

  // Delivery
  no_tracking: "❓",
  has_tracking: "📍",
  yes_box_damage: "📦",
  no_box_damage: "✅",
  signed_cleared: "✅",
  signed_damaged: "⚠️",
  visible_at_unboxing: "📦",
  noticed_later: "🕐",

  // Defect categories
  air: "💨",
  cosmetic: "✨",
  remote: "📱",
  rolling: "🔄",
  power: "🔌",
  recline: "↕️",
  footrest: "🦶",
  voice: "🎙️",
  heat: "🔥",

  // Voice
  voice_no_response: "🔇",
  false_triggers: "📺",
  voice_not_sure: "❓",

  // Air / body areas
  feet_calves: "🦵",
  arms: "💪",
  shoulders_hips: "🫁",
  side_panel: "🧩",
  base: "⬛",

  // Air troubleshooting
  yes_worked: "🔄",
  never_worked: "🚫",
  air_blowing: "💨",
  no_air: "🌬️",
  hose_issue: "🔗",
  hoses_ok: "✅",
  hose_clear: "✅",
  yes_hissing: "💨",
  no_hissing: "🔇",
  pump_running: "⚙️",
  no_sound: "🔇",
  yes_white_glove: "🧤",
  no_white_glove: "📦",

  // Cosmetic
  panels_fixed: "🔧",
  still_damaged: "💥",
  yes_box_damaged: "📦",

  // Remote
  has_power: "🔋",
  no_power: "🪫",
  blank_screen_commands_ok: "📱",
  cable_damaged: "🔌",
  commands_not_responding: "📵",
  fuse_broken: "⚡",
  bad_connection: "🔌",
  intermittent: "〰️",
  all_checked_ok: "🔍",

  // Rolling / mechanism
  noise_up_down: "🔊",
  noise_massaging: "🔊",
  pops: "💥",
  heads_not_moving: "🛑",
  no_movement: "🛑",
  worked_before_stopped: "🔄",
  power_but_no_move: "⚡",

  // Power
  remote_on: "🔋",
  remote_off: "🪫",
  no_response: "📵",
  quick_control_ok: "🎛️",
  back_switch_sound: "🔊",
  recline_not_working: "↕️",
  moves_on_off: "↩️",
  stays_stuck: "🔒",
  powercord_issue: "🔌",
  outlet_no_power: "🪫",
  fuse_blown: "⚡",
  clicking_sound: "🔊",
  no_clicking: "🔇",

  // Recline
  backrest: "🪑",
  zero_gravity: "🌙",
  footrest_recline: "🦶",
  multiple_not_working: "⚠️",
  none_working: "🚫",

  // Footrest defect
  legrest_not_extend: "↔️",
  air_not_inflating: "💨",
  legrest_not_lowering: "↕️",
  foot_rollers: "🦶",
  calf_roller: "🦵",

  // Help offer
  yes_team_help: "🙋",
  no_self_help: "🛠️",

  // Yes / No shortcuts
  yes: "✓",
  no: "✗",
};

const LABEL_ICON_RULES: Array<[RegExp, string]> = [
  [/^yes[,\s]/i, "✓"],
  [/^no[,\s]/i, "✗"],
  [/not sure/i, "❓"],
  [/white glove/i, "🧤"],
  [/tracking number/i, "📍"],
  [/box.*damag/i, "📦"],
  [/signed as/i, "📝"],
  [/unbox/i, "📦"],
  [/noticed later/i, "🕐"],
  [/noise|click|pop|hiss/i, "🔊"],
  [/hose|air blow|inflat/i, "💨"],
  [/fuse|power|outlet|powercord|plug/i, "🔌"],
  [/remote|screen|command/i, "📱"],
  [/voice/i, "🎙️"],
  [/massage|mechanism|roller|head/i, "🔄"],
  [/recline|backrest|zero gravity|footrest|legrest/i, "↕️"],
  [/photo|video|picture/i, "📷"],
  [/worked before|stopped/i, "🔄"],
  [/never worked|not work|no movement|stuck/i, "🚫"],
  [/help me|please help/i, "🙋"],
  [/on my own|try these/i, "🛠️"],
  [/damaged|broken|cut/i, "💥"],
  [/clear|connected|checked|nothing obvious/i, "✅"],
];

function iconFor(key: string, label: string): string {
  if (OPTION_ICONS[key]) return OPTION_ICONS[key];
  for (const [pattern, emoji] of LABEL_ICON_RULES) {
    if (pattern.test(label)) return emoji;
  }
  const lower = label.toLowerCase();
  if (lower.startsWith("yes")) return "✓";
  if (lower.startsWith("no")) return "✗";
  return "📌";
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
