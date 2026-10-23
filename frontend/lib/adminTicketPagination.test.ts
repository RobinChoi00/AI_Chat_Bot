import { describe, expect, it } from "vitest";
import {
  parseTicketPage,
  ticketPagination,
  ticketQueueHref,
  TICKET_PAGE_SIZE,
} from "./adminTicketPagination";

describe("historical ticket navigation", () => {
  it("makes every stored ticket reachable, including a partial final page", () => {
    const total = 205;
    const visited: number[] = [];
    let page = 1;
    while (true) {
      const current = ticketPagination(page, total);
      for (let ticket = current.first; ticket <= current.last; ticket++) visited.push(ticket);
      if (!current.hasNext) break;
      page = parseTicketPage(new URL(ticketQueueHref(page + 1), "https://example.test").searchParams.get("page") ?? undefined);
    }
    expect(page).toBe(3);
    expect(visited).toHaveLength(total);
    expect(new Set(visited).size).toBe(total);
    expect(visited.at(-1)).toBe(total);
    expect(ticketPagination(3, total)).toMatchObject({ offset: 2 * TICKET_PAGE_SIZE, first: 201, last: 205, hasPrevious: true, hasNext: false });
  });

  it("preserves filters when following previous and next links", () => {
    const href = ticketQueueHref(3, "awaiting_admin_review", "phone");
    const url = new URL(href, "https://example.test");
    expect(url.searchParams.get("page")).toBe("3");
    expect(url.searchParams.get("status")).toBe("awaiting_admin_review");
    expect(url.searchParams.get("channel")).toBe("phone");
    expect(ticketQueueHref(1)).toBe("/admin/warranty");
  });

  it.each([undefined, "", "0", "-1", "1.5", "2abc", "Infinity", "9007199254740991"])(
    "uses the first page for unsafe or malformed page %s",
    (raw) => expect(parseTicketPage(raw)).toBe(1)
  );

  it("keeps empty results on one page without navigation", () => {
    expect(ticketPagination(1, 0)).toMatchObject({ pageCount: 1, first: 0, last: 0, hasPrevious: false, hasNext: false });
  });
});
