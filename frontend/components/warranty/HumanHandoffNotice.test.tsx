import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { requestWarrantyHandoff } from "@/lib/api";
import { assistantContentFromResponse, hydrationAssistantContent } from "@/lib/warrantyHydration";
import type { WarrantyHandoff, WarrantySessionResponse, WarrantyTicketState } from "@/lib/types";
import HumanHandoffNotice from "./HumanHandoffNotice";

const handoff: WarrantyHandoff = {
  requested: true, reason: "callback_request", notification_sent: false,
  phone: "+1-888-848-2630 ext. 3", message: "Your support request and conversation are saved. Please contact our team directly.",
};
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("human support handoff", () => {
  it("gives a real contact action and preserves case context without promising a callback", () => {
    const html = renderToStaticMarkup(<HumanHandoffNotice handoff={handoff} caseReference="WR-123456" />);
    expect(html).toContain("Your request for a person is saved");
    expect(html).toContain("WR-123456");
    expect(html).toContain('href="tel:+18888482630,3"');
    expect(html).toContain('href="mailto:service@osakititan.com"');
    expect(html).toContain("callback or email notification has not been confirmed");
    expect(html).not.toContain("within 24 hours");
    expect(html).not.toContain("callback_request");
  });

  it("uses the saved handoff message instead of resurfacing an internal terminal prompt", () => {
    const ticket: WarrantyTicketState = {
      ticket_id: "case", status: "awaiting_admin_review", issue_type: "defect", model_name: "Chair",
      current_node: { node_id: "internal", node_type: "terminal", is_terminal: true, prompt: "INTERNAL_PARTS_SHIPPING_INSTRUCTION", options: [], evidence_required: ["proof_of_purchase"] },
    };
    const resp: WarrantySessionResponse = { session_id: "session", ticket, handoff };
    expect(assistantContentFromResponse(ticket, resp)).toBe(handoff.message);
    expect(hydrationAssistantContent(ticket, resp)).toBe(handoff.message);
  });

  it("keeps an actual safety escalation focused on stopping use and contacting support", () => {
    const html = renderToStaticMarkup(<HumanHandoffNotice handoff={{ ...handoff, reason: "safety_hazard", message: "Stop using the chair. Do not turn it on again to test it. Contact support." }} />);
    expect(html).toContain("Safety guidance and support");
    expect(html).toContain("Stop using the chair");
    expect(html).toContain("Do not turn it on again");
    expect(html).not.toContain("Your request for a person");
    expect(html).not.toContain("callback or email");
  });

  it("calls the displayed Sales extension for a delivery handoff without inventing a Sales email", () => {
    const html = renderToStaticMarkup(<HumanHandoffNotice handoff={{ ...handoff, department: "sales", phone: "+1-888-848-2630 ext. 2", phone_href: "tel:+18888482630;ext=2", hours: "Sales business hours" }} />);
    expect(html).toContain("Call Sales:");
    expect(html).toContain("ext. 2");
    expect(html).toContain('href="tel:+18888482630;ext=2"');
    expect(html).toContain("Sales business hours");
    expect(html).not.toContain("tel:+18888482630,3");
    expect(html).not.toContain("mailto:");
  });

  it("saves an explicit contact request with the current store and no browser admin credentials", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://backend.test");
    const fetch = vi.fn().mockResolvedValue(Response.json({ session_id: "session", ticket: null, handoff }));
    vi.stubGlobal("fetch", fetch);
    const resp = await requestWarrantyHandoff("session/1", { domain: "titanchair.com", reason: "customer_request" });
    expect(resp.handoff?.requested).toBe(true);
    expect(fetch.mock.calls[0][0]).toBe("http://backend.test/api/v1/warranty/session/session%2F1/handoff");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ domain: "titanchair.com", reason: "customer_request" });
    expect(fetch.mock.calls[0][1].headers).not.toHaveProperty("X-Admin-Key");
  });
});
