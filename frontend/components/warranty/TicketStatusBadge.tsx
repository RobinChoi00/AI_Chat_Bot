import type { TicketStatus } from "@/lib/types";

interface Props {
  status: TicketStatus | null | undefined;
  ticketId?: string | null;
}

const STATUS_CONFIG: Record<
  TicketStatus,
  { label: string; color: string; icon: string }
> = {
  in_progress:           { label: "In Progress",        color: "bg-blue-100 text-blue-800 border-blue-200",  icon: "🔄" },
  awaiting_admin_review: { label: "Under Review",        color: "bg-amber-100 text-amber-800 border-amber-200", icon: "⏳" },
  awaiting_evidence:     { label: "Evidence Needed",     color: "bg-orange-100 text-orange-800 border-orange-200", icon: "📎" },
  send_info:             { label: "Info Sent",           color: "bg-teal-100 text-teal-800 border-teal-200",  icon: "ℹ️" },
  sales_handoff:         { label: "Sales",               color: "bg-purple-100 text-purple-800 border-purple-200", icon: "💬" },
  admin_reviewing:       { label: "Being Reviewed",      color: "bg-indigo-100 text-indigo-800 border-indigo-200", icon: "👤" },
  need_more_information: { label: "More Info Needed",    color: "bg-yellow-100 text-yellow-800 border-yellow-200", icon: "❓" },
  resolved:              { label: "Resolved",            color: "bg-green-100 text-green-800 border-green-200", icon: "✅" },
};

export default function TicketStatusBadge({ status, ticketId }: Props) {
  if (!status) return null;

  const cfg = STATUS_CONFIG[status] ?? {
    label: status,
    color: "bg-gray-100 text-gray-700 border-gray-200",
    icon: "📋",
  };

  return (
    <div className="flex flex-col gap-1">
      <span
        className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-0.5 text-xs font-medium ${cfg.color}`}
      >
        <span>{cfg.icon}</span>
        {cfg.label}
      </span>
      {ticketId && (
        <span className="pl-1 font-mono text-[10px] text-gray-400">
          #{ticketId.slice(0, 8)}
        </span>
      )}
    </div>
  );
}
