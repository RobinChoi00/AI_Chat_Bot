import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryConversation, HistoryDetailResponse, HistoryListResponse } from "@/lib/historyTypes";
import { historyDomain, historyPage, historyRecordId } from "@/lib/adminHistory";

vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a>,
}));
vi.mock("next/navigation", () => ({
  redirect: (href: string) => { throw new Error(`REDIRECT:${href}`); },
  notFound: () => { throw new Error("NOT_FOUND"); },
}));

import HistoryPage, { dynamic as listDynamic } from "./page";
import HistoryDetailPage, { dynamic as detailDynamic } from "./[source]/[recordId]/page";

const fetchMock = vi.fn();

function conversation(): HistoryConversation {
  return {
    source: "sales", record_id: "42", session_id: "historical-session", domain: "osakiusa.com",
    status: "closed", title: "Saved consultation from 2020", message_count: 205,
    created_at: "2020-01-02T10:30:00", updated_at: "2020-01-03T11:30:00",
  };
}

function list(): HistoryListResponse {
  return { total: 205, offset: 100, limit: 100, counts: { warranty: 80, sales: 100, general: 25 }, items: [conversation()] };
}

function detail(): HistoryDetailResponse {
  return {
    conversation: conversation(), total: 205, offset: 100, limit: 100,
    messages: [
      { id: "m101", role: "user", content: "Full saved text\n<script>alert('customer')</script>" + "z".repeat(6000), created_at: "2020-01-02T10:31:00" },
      { id: "m102", role: "assistant", content: "Historical reply retained in full", created_at: "2020-01-02T10:32:00" },
    ],
  };
}

beforeEach(() => {
  vi.stubEnv("ADMIN_API_KEY", "history-test-admin-secret");
  vi.stubEnv("BACKEND_INTERNAL_URL", "http://backend.test");
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.resetAllMocks();
});

async function renderList(params: { source?: string; domain?: string; page?: string } = {}) {
  return renderToStaticMarkup(await HistoryPage({ searchParams: Promise.resolve(params) }));
}

async function renderDetail(source = "sales", recordId = "42", page?: string, view?: string) {
  return renderToStaticMarkup(await HistoryDetailPage({
    params: Promise.resolve({ source, recordId }), searchParams: Promise.resolve({ page, view }),
  }));
}

describe("all stored conversation history", () => {
  it("keeps both authenticated pages dynamic", () => {
    expect(listDynamic).toBe("force-dynamic");
    expect(detailDynamic).toBe("force-dynamic");
  });

  it("loads old records with source/domain filters and working second-page links", async () => {
    fetchMock.mockResolvedValue(Response.json(list()));
    const html = await renderList({ page: "2", source: "sales", domain: " osaki " });
    const [rawUrl, options] = fetchMock.mock.calls[0];
    const url = new URL(rawUrl);
    expect(url.pathname).toBe("/admin/history");
    expect(url.searchParams.get("source")).toBe("sales");
    expect(url.searchParams.get("domain")).toBe("osaki");
    expect(url.searchParams.get("limit")).toBe("100");
    expect(url.searchParams.get("offset")).toBe("100");
    expect(url.searchParams.has("days")).toBe(false);
    expect(options).toMatchObject({ cache: "no-store", headers: { "X-Admin-Key": "history-test-admin-secret" } });
    expect(options.signal).toBeInstanceOf(AbortSignal);
    expect(html).toContain("ALL STORED HISTORY");
    expect(html).toContain("2020-01-02");
    expect(html).toContain("Showing 101–200 of 205 stored conversations");
    expect(html).toContain('href="/admin/history?source=sales&amp;domain=osaki&amp;page=3"');
    expect(html).toContain('href="/admin/history/sales/42"');
    expect(html).toContain("not unique people");
    expect(html).not.toContain("history-test-admin-secret");
  });

  it("defaults to all sources and no recent-period cutoff", async () => {
    fetchMock.mockResolvedValue(Response.json({ ...list(), offset: 0 }));
    await renderList();
    expect(fetchMock.mock.calls[0][0]).toBe("http://backend.test/admin/history?limit=100&offset=0");
  });

  it("redirects stale conversation pages while preserving both filters", async () => {
    fetchMock.mockResolvedValue(Response.json(list()));
    await expect(renderList({ page: "9", source: "general", domain: "osakiusa.com" }))
      .rejects.toThrow("REDIRECT:/admin/history?source=general&domain=osakiusa.com&page=3");
  });

  it.each(["sales", "general"] as const)("shows full escaped %s transcript text beyond the first 100 messages", async (source) => {
    const body = detail();
    body.conversation.source = source;
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await renderDetail(source, "42", "2");
    expect(fetchMock.mock.calls[0][0]).toBe(`http://backend.test/admin/history/${source}/42?limit=100&offset=100`);
    expect(html).toContain("Showing 101–200 of 205 stored messages");
    expect(html).toContain(`href="/admin/history/${source}/42?page=3"`);
    expect(html).toContain("z".repeat(6000));
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<script>alert");
    expect(html).toContain("Historical reply retained in full");
    expect(html).not.toContain("history-test-admin-secret");
  });

  it("links warranty actions by ticket ID instead of the numeric history ID", async () => {
    const body = detail();
    body.conversation.source = "warranty";
    body.conversation.ticket_id = "ticket-uuid";
    fetchMock.mockResolvedValue(Response.json(body));
    expect(await renderDetail("warranty")).toContain('href="/admin/warranty/ticket-uuid"');
    body.conversation.ticket_id = "../unsafe";
    fetchMock.mockResolvedValue(Response.json(body));
    expect(await renderDetail("warranty")).not.toContain("Open warranty ticket and actions");
  });

  it("distinguishes recorded replies from workflow prompts and preserves the reference view in pagination", async () => {
    const body = detail();
    body.conversation.source = "warranty";
    body.transcript_mode = "api_recorded";
    body.recorded_since = "2026-10-23T13:00:00";
    body.alternate_view_available = true;
    body.messages = [
      { id: "recorded-1", role: "assistant", content: "Actual API response", created_at: "2026-10-23T13:00:00", origin: "api_response" },
      { id: "action-1", role: "user", content: "Delivery & tracking selected", created_at: null, origin: "user_action" },
    ];
    fetchMock.mockResolvedValue(Response.json(body));
    const recorded = await renderDetail("warranty", "42", "2");
    expect(recorded).toContain("Customer-facing replies");
    expect(recorded).toContain("Recorded customer-facing response");
    expect(recorded).toContain("Recorded customer action");
    expect(recorded).toContain("2026-10-23 13:00:00");
    expect(recorded).toContain("do not confirm that the customer read them");
    expect(recorded).toContain('href="/admin/history/warranty/42?view=workflow"');

    body.transcript_mode = "workflow_reference";
    body.messages = [{ id: "workflow-1", role: "assistant", content: "Internal node prompt", created_at: null, origin: "workflow_prompt" }];
    fetchMock.mockResolvedValue(Response.json(body));
    const workflow = await renderDetail("warranty", "42", "2", "workflow");
    expect(fetchMock.mock.lastCall?.[0]).toBe("http://backend.test/admin/history/warranty/42?limit=100&offset=100&view=workflow");
    expect(workflow).toContain("Internal workflow prompt");
    expect(workflow).toContain("not a verbatim customer transcript");
    expect(workflow).toContain('href="/admin/history/warranty/42?view=workflow&amp;page=3"');
    fetchMock.mockResolvedValue(Response.json(body));
    await expect(renderDetail("warranty", "42", "8", "workflow")).rejects.toThrow("REDIRECT:/admin/history/warranty/42?view=workflow&page=3");
  });

  it("clearly labels old warranty history as reconstructed without implying raw prompts were delivered", async () => {
    const body = detail();
    body.conversation.source = "warranty";
    body.transcript_mode = "legacy_reconstruction";
    body.recorded_since = null;
    body.messages = [
      { id: "prompt", role: "assistant", content: "Raw workflow prompt", created_at: null, origin: "workflow_prompt" },
      { id: "enrichment", role: "assistant", content: "Saved enrichment", created_at: null, origin: "legacy_enrichment" },
    ];
    fetchMock.mockResolvedValue(Response.json(body));
    const html = await renderDetail("warranty");
    expect(html).toContain("Reconstructed history");
    expect(html).toContain("Internal prompts may differ from the reply shown to the customer");
    expect(html).toContain("Legacy reply / enrichment");
    expect(html).not.toContain("Recorded customer-facing response</strong>");
  });

  it("clamps stale transcript pages and rejects malformed source/record paths before fetching", async () => {
    fetchMock.mockResolvedValue(Response.json(detail()));
    await expect(renderDetail("sales", "42", "8")).rejects.toThrow("REDIRECT:/admin/history/sales/42?page=3");
    fetchMock.mockClear();
    await expect(renderDetail("unknown", "42")).rejects.toThrow("NOT_FOUND");
    await expect(renderDetail("sales", "../secrets")).rejects.toThrow("NOT_FOUND");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(historyRecordId("00042")).toBe("42");
    expect(historyRecordId("9223372036854775808")).toBeNull();
    expect(historyPage("2garbage")).toBe(1);
    expect(historyPage("9007199254740991")).toBe(1);
    expect(historyDomain(["osakiusa.com", "titanchair.com"])).toBeUndefined();
    expect(historyPage(["2", "3"])).toBe(1);
  });

  it("separates actual empty history from missing credentials and backend access failures", async () => {
    fetchMock.mockResolvedValue(Response.json({ ...list(), total: 0, offset: 0, counts: { warranty: 0, sales: 0, general: 0 }, items: [] }));
    expect(await renderList()).toContain("No stored conversations match these filters");
    fetchMock.mockClear();
    vi.stubEnv("ADMIN_API_KEY", "");
    const missingKey = await renderList();
    expect(missingKey).toContain("ADMIN_API_KEY is not configured");
    expect(missingKey).not.toContain("No stored conversations");
    expect(fetchMock).not.toHaveBeenCalled();
    vi.stubEnv("ADMIN_API_KEY", "history-test-admin-secret");
    fetchMock.mockResolvedValue(new Response(null, { status: 403 }));
    const rejected = await renderList();
    expect(rejected).toContain("Admin access was rejected");
    expect(rejected).not.toContain("Showing 0–0");
  });

  it("shows missing conversation and backend/network failures without raw server text", async () => {
    fetchMock.mockResolvedValue(new Response("private backend response", { status: 404 }));
    expect(await renderDetail()).toContain("stored conversation was not found");
    fetchMock.mockResolvedValue(new Response("private backend response", { status: 500 }));
    const failed = await renderList();
    expect(failed).toContain("backend HTTP 500");
    expect(failed).not.toContain("private backend response");
    fetchMock.mockRejectedValue(new Error("private transport details"));
    const unreachable = await renderList();
    expect(unreachable).toContain("backend could not be reached");
    expect(unreachable).not.toContain("private transport details");
  });
});
