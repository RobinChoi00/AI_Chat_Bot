import { isValidElement, type ChangeEvent, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { sendWarrantyResumeLink, submitWarrantyContact, uploadEvidence } from "@/lib/api";
import SaveProgressButton from "./SaveProgressButton";
import EvidenceUploader from "./EvidenceUploader";

// Exercise the components' actual event handlers and state transitions without
// a browser dependency. State/ref slots persist between explicit renders.
const hooks = vi.hoisted(() => ({ values: [] as unknown[], cursor: 0 }));
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useState: (initial: unknown) => {
    const slot = hooks.cursor++;
    if (!(slot in hooks.values)) hooks.values[slot] = typeof initial === "function" ? (initial as () => unknown)() : initial;
    return [hooks.values[slot], (next: unknown) => { hooks.values[slot] = typeof next === "function" ? (next as (previous: unknown) => unknown)(hooks.values[slot]) : next; }];
  },
  useRef: (initial: unknown) => {
    const slot = hooks.cursor++;
    if (!(slot in hooks.values)) hooks.values[slot] = { current: initial };
    return hooks.values[slot];
  },
  useMemo: (factory: () => unknown) => factory(),
}));
vi.mock("@/lib/api", () => ({ sendWarrantyResumeLink: vi.fn(), submitWarrantyContact: vi.fn(), uploadEvidence: vi.fn() }));

function render<P>(component: (props: P) => ReactNode, props: P): ReactNode { hooks.cursor = 0; return component(props); }
function elements(node: ReactNode): { type: unknown; props: Record<string, unknown> }[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!isValidElement<Record<string, unknown>>(node)) return [];
  return [{ type: node.type, props: node.props }, ...elements(node.props.children as ReactNode)];
}
function text(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(text).join("");
  return isValidElement<{ children?: ReactNode }>(node) ? text(node.props.children) : "";
}
function button(tree: ReactNode, label: string) {
  const element = elements(tree).find((item) => item.type === "button" && text(item.props.children as ReactNode) === label);
  if (!element) throw new Error(`Missing action: ${label}`);
  return element.props;
}
async function click(props: Record<string, unknown>) { await (props.onClick as () => unknown)(); }

beforeEach(() => {
  hooks.values = []; hooks.cursor = 0; vi.clearAllMocks();
  vi.stubGlobal("sessionStorage", { setItem: vi.fn(() => { throw new Error("Customer contact must stay out of browser storage"); }) });
});
afterEach(() => vi.unstubAllGlobals());

describe("save and resume interactions", () => {
  it("reuses the known email and exposes a copy link when email delivery is unavailable", async () => {
    const resumeUrl = "https://support.example.test/warranty?resume=signed-capability";
    vi.mocked(sendWarrantyResumeLink).mockResolvedValue({ sent: false, queued: false, email_saved: true, customer_email: "b***@example.test", resume_url: resumeUrl, expires_in_days: 7 });
    const saved = vi.fn();
    const props = { sessionId: "session", initialCustomerEmail: "buyer@example.test", onSaved: saved };
    await click(button(render(SaveProgressButton, props), "Email a save link"));
    expect(sendWarrantyResumeLink).toHaveBeenCalledWith("session", "buyer@example.test");
    const next = render(SaveProgressButton, props);
    expect(saved).toHaveBeenCalledWith("saved");
    expect(text(next)).toContain("Progress saved");
    expect(text(next)).not.toContain("Link sent");
    const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) };
    vi.stubGlobal("navigator", { clipboard });
    await click(button(next, "Copy resume link"));
    expect(clipboard.writeText).toHaveBeenCalledWith(resumeUrl);
    expect(text(render(SaveProgressButton, props))).toContain("Link copied");
    expect(sessionStorage.setItem).not.toHaveBeenCalled();
  });

  it("distinguishes queued email from confirmed delivery", async () => {
    vi.mocked(sendWarrantyResumeLink).mockResolvedValue({ sent: false, queued: true, customer_email: "b***@example.test", expires_in_days: 7 });
    const saved = vi.fn();
    const props = { sessionId: "session", initialCustomerEmail: "buyer@example.test", onSaved: saved };
    await click(button(render(SaveProgressButton, props), "Email a save link"));
    expect(saved).toHaveBeenCalledWith("queued");
    expect(text(render(SaveProgressButton, props))).toContain("Email requested");
    expect(text(render(SaveProgressButton, props))).not.toContain("Link sent");
  });

  it.each(["javascript:alert(1)", "http://public.example.test/warranty?resume=capability", "https://secret:password@example.test/warranty"])("rejects an unsafe resume link and keeps the request retryable: %s", async (resume_url) => {
    vi.mocked(sendWarrantyResumeLink).mockResolvedValue({ sent: false, queued: false, customer_email: "", resume_url, email_saved: true, expires_in_days: 7 });
    const saved = vi.fn(), failed = vi.fn();
    const props = { sessionId: "session", initialCustomerEmail: "buyer@example.test", onSaved: saved, onError: failed };
    await click(button(render(SaveProgressButton, props), "Email a save link"));
    const next = render(SaveProgressButton, props);
    expect(saved).not.toHaveBeenCalled();
    expect(failed).toHaveBeenCalledOnce();
    expect(button(next, "Email link").disabled).toBe(false);
    expect(text(next)).toContain("not saved");
    expect(elements(next).some((item) => item.type === "a")).toBe(false);
  });

  it("prevents a rapid duplicate resume email request", async () => {
    let finish!: (value: Awaited<ReturnType<typeof sendWarrantyResumeLink>>) => void;
    vi.mocked(sendWarrantyResumeLink).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const props = { sessionId: "session", initialCustomerEmail: "buyer@example.test" };
    const action = button(render(SaveProgressButton, props), "Email a save link");
    const pending = click(action);
    await click(action);
    expect(sendWarrantyResumeLink).toHaveBeenCalledOnce();
    finish({ sent: true, customer_email: "b***@example.test", expires_in_days: 7 });
    await pending;
  });
});

describe("review request interactions", () => {
  it("allows review without optional evidence and preserves actual response flags", async () => {
    const response = { ticket_id: "case", customer_email: "b***@example.test", evidence_type: "other", evidence_na: true, email_notified: false, receipt_email_sent: false };
    vi.mocked(submitWarrantyContact).mockResolvedValue(response);
    const saved = vi.fn(), emailConfirmed = vi.fn();
    const props = { ticketId: "case", initialCustomerEmail: "buyer@example.test", evidenceRequired: ["damage_photos"], onContactSuccess: saved, onCustomerEmailConfirmed: emailConfirmed };
    const initial = render(EvidenceUploader, props);
    expect(elements(initial).some((item) => item.props.type === "file")).toBe(false);
    expect(button(initial, "Send for review").disabled).toBe(false);
    await click(button(initial, "Send for review"));
    expect(submitWarrantyContact).toHaveBeenCalledWith("case", "buyer@example.test");
    expect(uploadEvidence).not.toHaveBeenCalled();
    expect(saved).toHaveBeenCalledWith("buyer@example.test", response);
    expect(emailConfirmed).toHaveBeenCalledWith("buyer@example.test");
    const next = render(EvidenceUploader, props);
    expect(button(next, "Request saved").disabled).toBe(true);
    expect(text(next)).not.toContain("emailed");
    expect(sessionStorage.setItem).not.toHaveBeenCalled();
  });

  it("supports optional upload and recovery back to sending without an attachment", async () => {
    vi.mocked(submitWarrantyContact).mockResolvedValue({ ticket_id: "case", customer_email: "", evidence_type: "other", evidence_na: true });
    const failed = vi.fn();
    const props = { ticketId: "case", initialCustomerEmail: "buyer@example.test", evidenceRequired: ["damage_photos"], onError: failed };
    await click(button(render(EvidenceUploader, props), "Add photos or documents (optional)"));
    let next = render(EvidenceUploader, props);
    expect(button(next, "Send for review").disabled).toBe(true);
    const fileInput = elements(next).find((item) => item.props.type === "file")!;
    (fileInput.props.onChange as (event: ChangeEvent<HTMLInputElement>) => void)({ target: { files: [{ name: "bad.exe", size: 20 }] } } as unknown as ChangeEvent<HTMLInputElement>);
    expect(failed).toHaveBeenCalledOnce();
    next = render(EvidenceUploader, props);
    expect(text(next)).toContain("not allowed");
    await click(button(next, "Continue without an attachment"));
    next = render(EvidenceUploader, props);
    expect(button(next, "Send for review").disabled).toBe(false);
    await click(button(next, "Send for review"));
    expect(submitWarrantyContact).toHaveBeenCalledOnce();
  });

  it("uploads a selected attachment with the confirmed email and evidence category", async () => {
    const file = { name: "damage.jpg", size: 256 } as File;
    vi.mocked(uploadEvidence).mockResolvedValue({ evidence_id: 1, ticket_id: "case", ticket_status: "awaiting_admin_review", evidence_type: "damage_photos", original_filename: file.name, file_size_bytes: file.size, receipt_email_sent: false });
    const saved = vi.fn();
    const props = { ticketId: "case", initialCustomerEmail: "buyer@example.test", evidenceRequired: ["damage_photos"], onUploadSuccess: saved };
    await click(button(render(EvidenceUploader, props), "Add photos or documents (optional)"));
    const next = render(EvidenceUploader, props);
    const fileInput = elements(next).find((item) => item.props.type === "file")!;
    (fileInput.props.onChange as (event: ChangeEvent<HTMLInputElement>) => void)({ target: { files: [file] } } as unknown as ChangeEvent<HTMLInputElement>);
    await click(button(render(EvidenceUploader, props), "Send for review with attachment"));
    expect(uploadEvidence).toHaveBeenCalledWith("case", "damage_photos", file, "buyer@example.test");
    expect(submitWarrantyContact).not.toHaveBeenCalled();
    expect(saved).toHaveBeenCalledOnce();
  });
});
