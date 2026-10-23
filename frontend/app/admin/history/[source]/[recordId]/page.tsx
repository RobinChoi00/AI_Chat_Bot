import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import HistoryPagination from "@/components/admin/HistoryPagination";
import { fetchHistoryDetail } from "@/lib/adminHistoryServer";
import { historyDate, historyDetailHref, historyPage, historyPagination, historyRecordId, historySource } from "@/lib/adminHistory";

export const dynamic = "force-dynamic";

export default async function AdminHistoryDetailPage({ params, searchParams }: {
  params: Promise<{ source: string; recordId: string }>;
  searchParams: Promise<{ page?: string }>;
}) {
  const raw = await params;
  const source = historySource(raw.source);
  const recordId = historyRecordId(raw.recordId);
  if (!source || !recordId) notFound();
  const page = historyPage((await searchParams).page);
  const { data, error } = await fetchHistoryDetail(source, recordId, page);
  if (data && page > historyPagination(page, data.total).pageCount) {
    redirect(historyDetailHref(source, recordId, historyPagination(page, data.total).pageCount));
  }
  const ticketId = data?.conversation.source === "warranty" && data.conversation.ticket_id;
  const ticketHref = ticketId && /^[A-Za-z0-9_-]{1,100}$/.test(ticketId) ? `/admin/warranty/${encodeURIComponent(ticketId)}` : null;
  return (
    <div className="mx-auto max-w-5xl space-y-5 px-6 py-6">
      <Link href="/admin/history" className="text-sm text-indigo-700 underline">← All conversation history</Link>
      {error && <p role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}</p>}
      {data && <>
        <header className="rounded-xl border border-gray-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase text-indigo-700">{data.conversation.source} · full stored conversation</p>
          <h1 className="mt-2 break-words text-xl font-semibold">{data.conversation.title || "Conversation"}</h1>
          <dl className="mt-3 grid gap-2 text-sm text-gray-600 sm:grid-cols-2">
            <div><dt className="font-medium">Session</dt><dd className="break-all">{data.conversation.session_id || "Not recorded"}</dd></div>
            <div><dt className="font-medium">Store / status</dt><dd>{data.conversation.domain || "Not recorded"} · {data.conversation.status || "Not recorded"}</dd></div>
            <div><dt className="font-medium">Started</dt><dd>{historyDate(data.conversation.created_at)}</dd></div>
            <div><dt className="font-medium">Updated</dt><dd>{historyDate(data.conversation.updated_at)}</dd></div>
          </dl>
          {ticketHref && <Link href={ticketHref} className="mt-4 inline-block text-sm font-medium text-indigo-700 underline">Open warranty ticket and actions →</Link>}
        </header>
        <p className="text-sm text-gray-600">Full text is displayed as saved. Use message pages to read beyond the first 100 messages; no recent-period cutoff is applied.</p>
        <HistoryPagination page={page} total={data.total} href={(next) => historyDetailHref(source, recordId, next)} label="messages" />
        {data.total === 0 ? <p className="rounded-xl border border-dashed border-gray-300 p-8 text-sm text-gray-500">No messages were saved for this record.</p> : (
          <ol className="space-y-4" start={data.offset + 1}>
            {data.messages.map((message) => <li key={message.id} className={`rounded-xl border p-4 ${message.role === "user" ? "border-gray-200 bg-white" : "border-indigo-100 bg-indigo-50"}`}>
              <div className="mb-2 flex flex-wrap justify-between gap-2 text-xs text-gray-500"><strong>{message.role === "user" ? "Customer" : "Assistant"}</strong><span>{historyDate(message.created_at)}</span></div>
              <p className="whitespace-pre-wrap break-words text-sm text-gray-900">{message.content}</p>
            </li>)}
          </ol>
        )}
        {data.total > 0 && <HistoryPagination page={page} total={data.total} href={(next) => historyDetailHref(source, recordId, next)} label="messages" />}
      </>}
    </div>
  );
}
