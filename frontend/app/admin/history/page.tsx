import Link from "next/link";
import { redirect } from "next/navigation";
import HistoryPagination from "@/components/admin/HistoryPagination";
import { fetchHistoryList } from "@/lib/adminHistoryServer";
import { historyDate, historyDetailHref, historyDomain, historyListHref, historyPage, historyPagination, historySource } from "@/lib/adminHistory";
import { HISTORY_SOURCES } from "@/lib/historyTypes";

export const dynamic = "force-dynamic";

export default async function AdminHistoryPage({ searchParams }: {
  searchParams: Promise<{ source?: string; domain?: string; page?: string }>;
}) {
  const params = await searchParams;
  const source = historySource(params.source);
  const domain = historyDomain(params.domain);
  const page = historyPage(params.page);
  const { data, error } = await fetchHistoryList(page, source, domain);
  if (data && page > historyPagination(page, data.total).pageCount) {
    redirect(historyListHref(historyPagination(page, data.total).pageCount, source, domain));
  }
  return (
    <div className="mx-auto max-w-7xl space-y-5 px-6 py-6">
      <div>
        <p className="text-xs font-semibold tracking-wide text-indigo-700">ALL STORED HISTORY</p>
        <h1 className="mt-1 text-xl font-semibold text-gray-900">All conversation history</h1>
        <p className="mt-2 text-sm text-gray-600">
          Warranty, Sales, and general chat records across the full stored period. These are conversation records, not unique people; one conversation may appear in more than one source.
        </p>
        <p className="mt-1 text-xs text-gray-500">Visits without saved messages and deleted records are not included.</p>
      </div>

      <form action="/admin/history" method="get" className="flex flex-wrap items-end gap-3 rounded-xl border border-gray-200 bg-white p-4">
        <label className="text-sm text-gray-700">
          Source
          <select name="source" defaultValue={source ?? ""} className="ml-2 rounded border border-gray-300 px-3 py-2">
            <option value="">All sources</option>
            {HISTORY_SOURCES.map((value) => <option key={value} value={value}>{value === "general" ? "General chat" : value[0].toUpperCase() + value.slice(1)}</option>)}
          </select>
        </label>
        <label className="text-sm text-gray-700">
          Store domain
          <input name="domain" maxLength={200} defaultValue={domain ?? ""} placeholder="e.g. osakiusa.com" className="ml-2 rounded border border-gray-300 px-3 py-2" />
        </label>
        <button type="submit" className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white">Apply filters</button>
        <Link href="/admin/history" className="px-2 py-2 text-sm text-indigo-700 underline">Clear filters</Link>
      </form>

      {error && <p role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}</p>}
      {data && <>
        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
          <span>Warranty: {data.counts.warranty}</span>
          <span>Sales: {data.counts.sales}</span>
          <span>General chat: {data.counts.general}</span>
        </div>
        <HistoryPagination page={page} total={data.total} href={(next) => historyListHref(next, source, domain)} label="conversations" />
        {data.total === 0 ? (
          <p className="rounded-xl border border-dashed border-gray-300 p-10 text-center text-sm text-gray-500">No stored conversations match these filters.</p>
        ) : (
          <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b bg-gray-50 text-xs text-gray-600"><tr>
                {["Source", "Conversation", "Store", "Status", "Messages", "Started", "Updated"].map((title) => <th key={title} className="px-4 py-3">{title}</th>)}
              </tr></thead>
              <tbody className="divide-y divide-gray-100">
                {data.items.map((item) => <tr key={`${item.source}:${item.record_id}`}>
                  <td className="px-4 py-3 capitalize">{item.source}</td>
                  <td className="max-w-sm px-4 py-3">
                    <Link href={historyDetailHref(item.source, item.record_id)} className="break-words font-medium text-indigo-700 underline">{item.title || "Open conversation"}</Link>
                    <p className="mt-1 break-all text-xs text-gray-500">{item.session_id || "No session ID recorded"}</p>
                  </td>
                  <td className="px-4 py-3">{item.domain || "Not recorded"}</td>
                  <td className="px-4 py-3">{item.status || "Not recorded"}</td>
                  <td className="px-4 py-3 tabular-nums">{item.message_count}</td>
                  <td className="whitespace-nowrap px-4 py-3 text-xs">{historyDate(item.created_at)}</td>
                  <td className="whitespace-nowrap px-4 py-3 text-xs">{historyDate(item.updated_at)}</td>
                </tr>)}
              </tbody>
            </table>
          </div>
        )}
        {data.total > 0 && <HistoryPagination page={page} total={data.total} href={(next) => historyListHref(next, source, domain)} label="conversations" />}
      </>}
    </div>
  );
}
