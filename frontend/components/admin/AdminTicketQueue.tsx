"use client";

import Link from "next/link";
import type { AdminWarrantyTicket } from "@/lib/adminTypes";
import AdminStatusBadge from "./AdminStatusBadge";

interface Props {
  tickets: AdminWarrantyTicket[];
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function AdminTicketQueue({ tickets }: Props) {
  if (!tickets.length) {
    return (
      <div className="rounded-xl border border-dashed border-gray-300 py-16 text-center">
        <p className="text-2xl">📭</p>
        <p className="mt-2 text-sm text-gray-500">No warranty tickets found.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white shadow-sm">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <thead className="bg-gray-50">
          <tr>
            {["Ticket ID", "Status", "Issue / Defect", "Model", "Node", "Created", "Updated"].map(
              (h) => (
                <th
                  key={h}
                  className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500"
                >
                  {h}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {tickets.map((t) => (
            <tr
              key={t.ticket_id}
              className="group cursor-pointer transition hover:bg-indigo-50"
            >
              <td className="px-4 py-3">
                <Link
                  href={`/admin/warranty/${t.ticket_id}`}
                  className="font-mono text-xs text-indigo-600 underline-offset-2 hover:underline"
                >
                  {t.ticket_id.slice(0, 8)}…
                </Link>
              </td>
              <td className="px-4 py-3">
                <AdminStatusBadge status={t.status} />
              </td>
              <td className="px-4 py-3 text-gray-700">
                {t.issue_type || "—"}
                {t.defect_type && (
                  <span className="ml-1 text-gray-400">/ {t.defect_type}</span>
                )}
              </td>
              <td className="px-4 py-3 text-gray-700">
                {t.model_name || <span className="text-gray-400">—</span>}
              </td>
              <td className="max-w-[180px] truncate px-4 py-3 font-mono text-xs text-gray-500">
                {t.current_node_id || "—"}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-xs text-gray-500">
                {formatDate(t.created_at)}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-xs text-gray-500">
                {formatDate(t.updated_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
