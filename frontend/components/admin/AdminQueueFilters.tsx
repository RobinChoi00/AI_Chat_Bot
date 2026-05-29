"use client";

import { useRouter } from "next/navigation";
import type { AdminTicketStatus } from "@/lib/adminTypes";

interface Props {
  currentStatus: string | undefined;
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

export default function AdminQueueFilters({ currentStatus, total }: Props) {
  const router = useRouter();

  function handleStatusChange(val: string) {
    const params = new URLSearchParams();
    if (val) params.set("status", val);
    router.push(`/admin/warranty${params.size ? `?${params}` : ""}`);
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
