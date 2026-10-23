/** Server-component data access only. Never import this file in a client component. */
import { getBackendUrl } from "./backendUrl";
import { HISTORY_PAGE_SIZE } from "./adminHistory";
import type { HistoryDetailResponse, HistoryListResponse, HistorySource } from "./historyTypes";

type HistoryResult<T> = { data: T; error: null } | { data: null; error: string };

async function fetchHistory<T>(path: string, detail = false): Promise<HistoryResult<T>> {
  const adminKey = process.env.ADMIN_API_KEY;
  if (!adminKey) {
    return { data: null, error: "ADMIN_API_KEY is not configured on the Next.js server. Conversation history could not be loaded." };
  }
  try {
    const response = await fetch(`${getBackendUrl()}${path}`, {
      headers: { "X-Admin-Key": adminKey },
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    if (response.status === 401 || response.status === 403) {
      return { data: null, error: "Admin access was rejected by the backend. Check the server admin key." };
    }
    if (response.status === 404) {
      return { data: null, error: detail ? "This stored conversation was not found." : "The conversation-history backend is not available. Deploy the history API before using this page." };
    }
    if (!response.ok) {
      return { data: null, error: `Conversation history could not be loaded (backend HTTP ${response.status}).` };
    }
    return { data: await response.json() as T, error: null };
  } catch {
    return { data: null, error: "The conversation-history backend could not be reached or returned an unreadable response. Try again after checking the service." };
  }
}

export function fetchHistoryList(page: number, source?: HistorySource, domain?: string) {
  const query = new URLSearchParams({ limit: String(HISTORY_PAGE_SIZE), offset: String((page - 1) * HISTORY_PAGE_SIZE) });
  if (source) query.set("source", source);
  if (domain) query.set("domain", domain);
  return fetchHistory<HistoryListResponse>(`/admin/history?${query}`);
}

export function fetchHistoryDetail(source: HistorySource, recordId: string, page: number) {
  const query = new URLSearchParams({ limit: String(HISTORY_PAGE_SIZE), offset: String((page - 1) * HISTORY_PAGE_SIZE) });
  return fetchHistory<HistoryDetailResponse>(`/admin/history/${encodeURIComponent(source)}/${encodeURIComponent(recordId)}?${query}`, true);
}
