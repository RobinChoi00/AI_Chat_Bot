"use client";

import { useRouter } from "next/navigation";
import type { AdminTicketStatus } from "@/lib/adminTypes";

interface Props {
  currentStatus: string | undefined;
  currentChannel: string | undefined;
  total: number;
}

const STATUS_OPTIONS: { value: string; label: string }[] = [
  { value: "",                      label: "All statuses" },
  { value: "in_progress",           label: "In Progress" },
  { value: "awaiting_admin_review", label: "Awaiting Review" },
  { value: "awaiting_evidence",     label: "Evidence Needed" },
  { value: "admin_reviewing",       label: "Admin Reviewing" },
  { value: "need_more_information", label: "More Info Needed" },
  { value: "approved",              label: "Approved" },
  { value: "rejected",              label: "Rejected" },
  { value: "closed",                label: "Closed" },
  { value: "resolved",              label: "Resolved" },
];

const CHANNEL_OPTIONS: { value: string; label: string }[] = [
  { value: "",      label: "All channels" },
  { value: "phone", label: "Phone IVR" },
  { value: "web",   label: "Web chat" },
];

export default function AdminQueueFilters({
  currentStatus,
  currentChannel,
  total,
}: Props) {
  const router = useRouter();

  function pushFilters(status: string, channel: string) {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    if (channel) params.set("channel", channel);
    router.push(`/admin/warranty${params.size ? `?${params}` : ""}`);
  }

  function handleStatusChange(val: string) {
    pushFilters(val, currentChannel ?? "");
  }

  function handleChannelChange(val: string) {
    pushFilters(currentStatus ?? "", val);
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="text-sm font-medium text-gray-600">
        Filter by status:
      </label>
      <select
        value={currentStatus ?? ""}
        onChange={(e) =>
          handleStatusChange(e.target.value as AdminTicketStatus | "")
        }
        className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-400"
      >
        {STATUS_OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>

      <label className="text-sm font-medium text-gray-600">
        Channel:
      </label>
      <select
        value={currentChannel ?? ""}
        onChange={(e) => handleChannelChange(e.target.value)}
        className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-400"
      >
        {CHANNEL_OPTIONS.map((o) => (
          <option key={o.value || "all"} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>

      <span className="text-xs text-gray-400">
        {total} ticket{total !== 1 ? "s" : ""}
      </span>

      <button
        onClick={() => router.refresh()}
        className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-50"
      >
        ↻ Refresh
      </button>
    </div>
  );
}
