import { HISTORY_SOURCES, type HistorySource } from "./historyTypes";

export const HISTORY_PAGE_SIZE = 100;
const MAX_PAGE = Math.floor(Number.MAX_SAFE_INTEGER / HISTORY_PAGE_SIZE);

export function historySource(raw?: string): HistorySource | undefined {
  return HISTORY_SOURCES.find((source) => source === raw);
}

export function historyDomain(raw?: unknown): string | undefined {
  return typeof raw === "string" ? raw.trim().slice(0, 200) || undefined : undefined;
}

export function historyPage(raw?: unknown): number {
  if (typeof raw !== "string" || !/^\d+$/.test(raw)) return 1;
  const value = Number(raw);
  return Number.isSafeInteger(value) && value >= 1 && value <= MAX_PAGE ? value : 1;
}

export function historyRecordId(raw: string): string | null {
  if (!/^[0-9]{1,19}$/.test(raw)) return null;
  const value = BigInt(raw);
  return value > BigInt(0) && value <= BigInt("9223372036854775807") ? value.toString() : null;
}

export function historyListHref(page: number, source?: HistorySource, domain?: string): string {
  const query = new URLSearchParams();
  if (source) query.set("source", source);
  if (domain) query.set("domain", domain);
  if (page > 1) query.set("page", String(page));
  return `/admin/history${query.size ? `?${query}` : ""}`;
}

export function historyDetailHref(source: HistorySource, recordId: string, page = 1): string {
  const id = historyRecordId(recordId);
  if (!id) return "/admin/history";
  return `/admin/history/${encodeURIComponent(source)}/${id}${page > 1 ? `?page=${page}` : ""}`;
}

export function historyPagination(page: number, total: number) {
  const pageCount = Math.max(1, Math.ceil(total / HISTORY_PAGE_SIZE));
  const offset = (page - 1) * HISTORY_PAGE_SIZE;
  return {
    pageCount,
    offset,
    first: total > 0 ? offset + 1 : 0,
    last: Math.min(offset + HISTORY_PAGE_SIZE, total),
    hasPrevious: page > 1,
    hasNext: page < pageCount,
  };
}

export function historyDate(raw: string | null): string {
  if (!raw) return "Not recorded";
  // Show the stored timestamp explicitly; do not reinterpret legacy timezone-less dates.
  return raw.replace("T", " ");
}
