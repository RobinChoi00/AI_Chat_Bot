import Link from "next/link";
import { historyPagination } from "@/lib/adminHistory";

export default function HistoryPagination({ page, total, href, label }: {
  page: number;
  total: number;
  href: (page: number) => string;
  label: "conversations" | "messages";
}) {
  const paging = historyPagination(page, total);
  return (
    <nav aria-label={`${label} pages`} className="flex flex-wrap items-center justify-between gap-3 py-4 text-sm">
      <p className="text-gray-600">
        Showing {paging.first}–{paging.last} of {total} stored {label}
        {total > 0 && ` · Page ${page} of ${paging.pageCount}`}
      </p>
      <div className="flex gap-2">
        {paging.hasPrevious ? (
          <Link href={href(page - 1)} className="rounded border border-gray-300 bg-white px-3 py-1.5">← Previous</Link>
        ) : <span aria-disabled="true" className="rounded border border-gray-100 px-3 py-1.5 text-gray-400">← Previous</span>}
        {paging.hasNext ? (
          <Link href={href(page + 1)} className="rounded border border-gray-300 bg-white px-3 py-1.5">Next →</Link>
        ) : <span aria-disabled="true" className="rounded border border-gray-100 px-3 py-1.5 text-gray-400">Next →</span>}
      </div>
    </nav>
  );
}
