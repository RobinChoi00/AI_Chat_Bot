export const TICKET_PAGE_SIZE = 100;

// Keep every offset exact when serializing it for the backend.
const MAX_PAGE = Math.floor(Number.MAX_SAFE_INTEGER / TICKET_PAGE_SIZE);

export function parseTicketPage(raw: string | undefined): number {
  if (!raw || !/^\d+$/.test(raw)) return 1;
  const page = Number(raw);
  return Number.isSafeInteger(page) && page >= 1 && page <= MAX_PAGE ? page : 1;
}

export function ticketQueueHref(
  page: number,
  status?: string,
  channel?: string
): string {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (channel) params.set("channel", channel);
  if (page > 1) params.set("page", String(page));
  return `/admin/warranty${params.size ? `?${params}` : ""}`;
}

export function ticketPagination(page: number, total: number) {
  const pageCount = Math.max(1, Math.ceil(total / TICKET_PAGE_SIZE));
  const offset = (page - 1) * TICKET_PAGE_SIZE;
  return {
    pageCount,
    offset,
    first: total > 0 ? offset + 1 : 0,
    last: Math.min(offset + TICKET_PAGE_SIZE, total),
    hasPrevious: page > 1,
    hasNext: page < pageCount,
  };
}
