"use client";

import type { AdminWarrantyEvidence } from "@/lib/adminTypes";

interface Props {
  evidence: AdminWarrantyEvidence[];
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

const MIME_ICON: Record<string, string> = {
  "image/jpeg": "🖼️",
  "image/png":  "🖼️",
  "application/pdf": "📄",
  "video/mp4":  "🎬",
  "video/quicktime": "🎬",
};

/**
 * MIME types that most browsers can display inline in a new tab.
 * For these we open in a new tab without forcing a download.
 */
function isViewableInBrowser(mime: string | null): boolean {
  if (!mime) return false;
  return (
    mime.startsWith("image/") ||
    mime === "application/pdf"
    // video/* streams well but autoplay policies vary — download is safer
  );
}

function downloadUrl(ticketId: string, evidenceId: number): string {
  return `/api/admin/warranty/tickets/${encodeURIComponent(ticketId)}/evidence/${encodeURIComponent(String(evidenceId))}/download`;
}

export default function AdminEvidenceList({ evidence }: Props) {
  if (!evidence.length) {
    return (
      <p className="text-sm text-gray-400 italic">No evidence uploaded yet.</p>
    );
  }

  return (
    <div className="space-y-2">
      {evidence.map((ev) => {
        const isEmailOnly = ev.evidence_type === "not_available";
        const icon = isEmailOnly ? "✉️" : (ev.mime_type && MIME_ICON[ev.mime_type]) ?? "📎";
        return (
          <div
            key={ev.id}
            className="flex items-start gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3"
          >
            <span className="mt-0.5 text-xl">{icon}</span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-gray-900">
                {isEmailOnly
                  ? "Email only — no photo/video (N/A)"
                  : (ev.original_filename ?? "unnamed file")}
              </p>
              <div className="mt-0.5 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-gray-500">
                <span className="capitalize">
                  {ev.evidence_type.replace(/_/g, " ")}
                </span>
                {!isEmailOnly && <span>{formatBytes(ev.file_size_bytes)}</span>}
                {!isEmailOnly && ev.mime_type && <span>{ev.mime_type}</span>}
                {ev.customer_email && <span>{ev.customer_email}</span>}
                <span>Uploaded: {formatDate(ev.created_at)}</span>
                {ev.emailed ? (
                  <span className="text-teal-600">✉️ Team notified</span>
                ) : (
                  <span className="text-amber-600">⚠️ Email pending</span>
                )}
              </div>

              {/* View / Download action */}
              {!isEmailOnly && (
              <div className="mt-2 flex gap-2">
                {isViewableInBrowser(ev.mime_type) ? (
                  <a
                    href={downloadUrl(ev.ticket_id, ev.id)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 rounded-md border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-[11px] font-medium text-indigo-700 transition hover:bg-indigo-100"
                  >
                    🔍 View in new tab
                  </a>
                ) : null}
                <a
                  href={downloadUrl(ev.ticket_id, ev.id)}
                  download={ev.original_filename ?? true}
                  className="inline-flex items-center gap-1 rounded-md border border-gray-200 bg-white px-2.5 py-1 text-[11px] font-medium text-gray-600 transition hover:bg-gray-50"
                >
                  ⬇ Download
                </a>
              </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
