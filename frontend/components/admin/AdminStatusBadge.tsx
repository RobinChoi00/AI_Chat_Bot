import type { AdminTicketStatus } from "@/lib/adminTypes";

interface Props {
  status: AdminTicketStatus | string | null | undefined;
  size?: "sm" | "md";
}

const STATUS_MAP: Record<
  AdminTicketStatus,
  { label: string; cls: string; icon: string }
> = {
  in_progress:           { label: "In Progress",       cls: "bg-blue-100 text-blue-800 border-blue-200",   icon: "🔄" },
  awaiting_admin_review: { label: "Awaiting Review",   cls: "bg-amber-100 text-amber-800 border-amber-200", icon: "⏳" },
  awaiting_evidence:     { label: "Evidence Needed",   cls: "bg-orange-100 text-orange-800 border-orange-200", icon: "📎" },
  send_info:             { label: "Info Sent",          cls: "bg-teal-100 text-teal-800 border-teal-200",   icon: "ℹ️" },
  sales_handoff:         { label: "Sales Handoff",      cls: "bg-purple-100 text-purple-800 border-purple-200", icon: "💬" },
  admin_reviewing:       { label: "Admin Reviewing",    cls: "bg-indigo-100 text-indigo-800 border-indigo-200", icon: "👤" },
  need_more_information: { label: "More Info Needed",   cls: "bg-yellow-100 text-yellow-800 border-yellow-200", icon: "❓" },
  approved:              { label: "Approved",           cls: "bg-green-100 text-green-800 border-green-200", icon: "✅" },
  rejected:              { label: "Rejected",           cls: "bg-red-100 text-red-800 border-red-200",     icon: "❌" },
  resolved:              { label: "Resolved",           cls: "bg-green-100 text-green-800 border-green-200", icon: "✔️" },
  closed:                { label: "Closed",             cls: "bg-gray-100 text-gray-600 border-gray-200",  icon: "🔒" },
};

export default function AdminStatusBadge({ status, size = "sm" }: Props) {
  if (!status) return null;

  const cfg = STATUS_MAP[status as AdminTicketStatus] ?? {
    label: status,
    cls: "bg-gray-100 text-gray-600 border-gray-200",
    icon: "📋",
  };

  const textCls = size === "md" ? "text-sm px-3 py-1" : "text-xs px-2.5 py-0.5";

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border font-medium ${textCls} ${cfg.cls}`}
    >
      <span>{cfg.icon}</span>
      {cfg.label}
    </span>
  );
}
