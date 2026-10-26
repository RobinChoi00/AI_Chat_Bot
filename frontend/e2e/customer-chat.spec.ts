import { expect, test as base, type Page, type Route } from "@playwright/test";
import type { WarrantyConversationMessage, WarrantySessionResponse, WarrantyTicketState } from "../lib/types";

const test = base.extend<{ pageErrors: string[] }>({
  pageErrors: [async ({ page }, use) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await use(errors);
    expect(errors, "The customer flow must not throw browser or hydration errors").toEqual([]);
  }, { auto: true }],
});

const buyerEmail = "browser.customer@example.test";
const firstQuestion = "My OS-4000T footrest stopped moving.";
const followupQuestion = "Which part needs help?";

/** HTTP contract double only: Next.js, React, events, forms and storage are real. */
class CustomerApi {
  consent = false;
  email = "";
  sessionId = "";
  ticket: WarrantyTicketState | null = null;
  handoff: WarrantySessionResponse["handoff"] = null;
  conversation: WarrantyConversationMessage[] = [];
  failSessionLoads = false;
  failNextEmail = false;
  historyHasMore = false;
  failNextHistory = false;
  terminalReview = false;
  deliveryChecks = 0;
  contactRequests: Record<string, unknown>[] = [];
  unexpected: string[] = [];

  privacy() {
    return {
      consent_required: true,
      consent_recorded: this.consent,
      email_required_before_help: true,
      email_gate_status: this.email ? "provided" : "pending",
      contact_email_available: Boolean(this.email),
    };
  }

  session(): WarrantySessionResponse {
    return {
      session_id: this.sessionId,
      ticket: this.ticket,
      handoff: this.handoff,
      privacy: this.privacy(),
      conversation_messages: this.conversation,
      conversation_has_more: this.historyHasMore,
      conversation_before_id: this.conversation[0]?.id ?? null,
      ...(this.terminalReview ? {
        terminal_enrichment: {
          phase: "awaiting_help_consent" as const,
          interaction_mode: "preparation" as const,
          issue_type: "delivery",
          message: "You can request a review without attaching a photo.",
        },
      } : {}),
    };
  }

  async install(page: Page) {
    await page.route("**/api/v1/**", (route) => this.handle(route));
  }

  async handle(route: Route) {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const sessionMatch = path.match(/^\/api\/v1\/warranty\/session\/([^/]+)(?:\/(.*))?$/);
    const body = method === "POST" ? request.postDataJSON() as Record<string, unknown> : {};
    const json = (value: unknown, status = 200) => route.fulfill({ status, json: value });

    if (path === "/api/v1/journey/events" || path === "/api/v1/chat/feedback") {
      return json({ ok: true });
    }
    if (sessionMatch) {
      this.sessionId = sessionMatch[1];
      const action = sessionMatch[2] || "";
      if (method === "GET" && !action) {
        if (this.failSessionLoads) {
          return json({ detail: "Temporary service interruption" }, 503);
        }
        return json(this.session());
      }
      if (method === "GET" && action === "messages") {
        if (this.failNextHistory) {
          this.failNextHistory = false;
          return json({ detail: "History is temporarily unavailable" }, 503);
        }
        return json({ session_id: this.sessionId, ticket_id: "browser-case", has_more: false, before_id: 1,
          messages: [
            { id: 1, role: "user", content: "My first saved question before these messages." },
            { id: 2, role: "assistant", content: "Earlier support guidance is saved here." },
          ] });
      }
      if (action === "consent") {
        this.consent = true;
        return json({ consent_recorded: true, accepted_at: "2026-10-26T10:00:00-05:00", privacy: this.privacy() });
      }
      if (action === "contact-email") {
        if (this.failNextEmail) {
          this.failNextEmail = false;
          return json({ detail: "Email could not be saved. Please try again." }, 503);
        }
        this.email = String(body.customer_email || "");
        return json({ recorded: true, email_saved: true, email_gate_status: "provided", skipped: false, privacy: this.privacy() });
      }
      if (action === "smart-start") {
        this.ticket = makeTicket("defect", followupQuestion);
        this.conversation = [{ id: 1, role: "user", content: String(body.message) }, { id: 2, role: "assistant", content: followupQuestion }];
        return json({ ...this.session(), assistant_message: followupQuestion });
      }
      if (action === "quick-start") {
        this.ticket = makeTicket("delivery", "What delivery help do you need?", this.terminalReview);
        return json(this.session());
      }
      if (action === "delivery-tracking") {
        this.deliveryChecks += 1;
        if (this.deliveryChecks > 1) return json({ detail: "The carrier is temporarily unavailable." }, 503);
        return json({ ...this.session(), tracking_summary: {
          available: true, status: "In transit", message: "Verified carrier update", carrier: "Test carrier",
          fetched_at: "2026-10-26T15:00:00Z", tracking_url: "https://carrier.example.test/shipment",
        } });
      }
    }
    if (path === "/api/v1/warranty/browser-case/troubleshooting-outcome") {
      return json({ ticket_id: "browser-case", outcome: body.outcome, status: "awaiting_evidence", self_service_resolved: false });
    }
    if (path === "/api/v1/warranty/browser-case/contact") {
      this.contactRequests.push(body);
      return json({ ticket_id: "browser-case", customer_email: "b***r@example.test", evidence_type: "other", evidence_na: true,
        case_reference: "WR-BROWSER", email_notified: false, receipt_email_sent: false });
    }
    this.unexpected.push(`${method} ${path}`);
    return json({ detail: "Unexpected test API request" }, 500);
  }
}

function makeTicket(issue: string, prompt: string, terminal = false): WarrantyTicketState {
  return {
    ticket_id: "browser-case",
    domain: "osakiusa.com",
    status: terminal ? "awaiting_evidence" : "in_progress",
    issue_type: issue,
    model_name: issue === "defect" ? "OS-4000T" : "",
    current_node: {
      node_id: terminal ? "delivery_terminal" : `${issue}_details`,
      node_type: terminal ? "terminal" : "question_text",
      prompt, options: [], is_terminal: terminal,
      ...(terminal ? { evidence_required: ["damage_photos"] } : {}),
    },
  };
}

async function beginChat(page: Page, api: CustomerApi) {
  await api.install(page);
  await page.goto("/warranty");
  await page.getByRole("button", { name: "I Agree", exact: true }).click();
  await page.getByLabel("Email address").fill(buyerEmail);
  await page.getByRole("button", { name: "Start chat", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Message to support" })).toBeVisible();
}

test("requires consent and a saved email before opening the conversation", async ({ page }) => {
  const api = new CustomerApi();
  api.failNextEmail = true;
  await api.install(page);
  await page.goto("/warranty");
  await expect(page.getByRole("textbox", { name: "Message to support" })).toHaveCount(0);
  await page.getByRole("button", { name: "I Agree", exact: true }).click();
  const start = page.getByRole("button", { name: "Start chat", exact: true });
  await expect(start).toBeDisabled();
  await page.getByLabel("Email address").fill("invalid-address");
  await expect(start).toBeDisabled();
  await page.getByLabel("Email address").fill(buyerEmail);
  await start.click();
  await expect(page.getByRole("alert").filter({ hasText: "Email could not be saved" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Message to support" })).toHaveCount(0);
  await start.click();
  await expect(page.getByRole("textbox", { name: "Message to support" })).toBeVisible();
  expect(api.email).toBe(buyerEmail);
  expect(await page.evaluate(() => JSON.stringify({ ...sessionStorage, ...localStorage }))).not.toContain(buyerEmail);
  expect(api.unexpected).toEqual([]);
});

test("restores prior messages and saved email state after a browser reload", async ({ page }) => {
  const api = new CustomerApi();
  await beginChat(page, api);
  await page.getByRole("textbox", { name: "Message to support" }).fill(firstQuestion);
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log", { name: "Conversation" })).toContainText(followupQuestion);
  await page.reload();
  const conversation = page.getByRole("log", { name: "Conversation" });
  await expect(conversation).toContainText(firstQuestion);
  await expect(conversation).toContainText(followupQuestion);
  await expect(page.getByRole("button", { name: "Start chat", exact: true })).toHaveCount(0);
  await expect(page.getByRole("textbox", { name: "Message to support" })).toBeEnabled();
  expect(api.unexpected).toEqual([]);
});

test("keeps a failed session load closed until retry restores the saved conversation", async ({ page }) => {
  const api = new CustomerApi();
  api.consent = true;
  api.email = buyerEmail;
  api.ticket = makeTicket("defect", followupQuestion);
  api.conversation = [{ id: 1, role: "user", content: firstQuestion }, { id: 2, role: "assistant", content: followupQuestion }];
  api.failSessionLoads = true;
  await api.install(page);
  await page.goto("/warranty");
  await expect(page.getByRole("alert").filter({ hasText: "load your saved chat" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Message to support" })).toHaveCount(0);
  await expect(page.getByLabel("Email address")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Loading chat…", exact: true })).toBeDisabled();
  api.failSessionLoads = false;
  await page.getByRole("button", { name: "Retry loading chat" }).click();
  await expect(page.getByRole("log", { name: "Conversation" })).toContainText(firstQuestion);
  await expect(page.getByRole("textbox", { name: "Message to support" })).toBeEnabled();
  await expect(page.getByRole("alert").filter({ hasText: "load your saved chat" })).toHaveCount(0);
  expect(api.unexpected).toEqual([]);
});

test("hides a previous delivery result when refreshing it fails", async ({ page }) => {
  const api = new CustomerApi();
  await beginChat(page, api);
  await page.getByRole("button", { name: "Delivery & tracking", exact: true }).click();
  await page.getByLabel("Order number").fill("12345");
  await expect(page.getByLabel("Email used at checkout")).toHaveValue(buyerEmail);
  await page.getByRole("button", { name: "Check delivery status", exact: true }).click();
  await expect(page.getByText("Latest delivery update", { exact: true })).toBeVisible();
  await expect(page.getByText("In transit", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: "carrier is temporarily unavailable" })).toBeVisible();
  await expect(page.getByText("Latest delivery update", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Open carrier tracking" })).toHaveCount(0);
  expect(api.deliveryChecks).toBe(2);
  expect(api.unexpected).toEqual([]);
});

test("loads earlier conversation messages in order and retries a failed history request", async ({ page }) => {
  const api = new CustomerApi();
  api.consent = true;
  api.email = buyerEmail;
  api.ticket = makeTicket("defect", followupQuestion);
  api.conversation = [{ id: 3, role: "user", content: firstQuestion }, { id: 4, role: "assistant", content: followupQuestion }];
  api.historyHasMore = true;
  api.failNextHistory = true;
  await api.install(page);
  await page.goto("/warranty");
  const load = page.getByRole("button", { name: "Load earlier messages", exact: true });
  await load.click();
  await expect(page.getByRole("alert").filter({ hasText: /earlier|history/i })).toBeVisible();
  await load.click();
  const conversation = page.getByRole("log", { name: "Conversation" });
  await expect(conversation).toContainText("My first saved question before these messages.");
  const text = await conversation.innerText();
  expect(text.indexOf("My first saved question")).toBeLessThan(text.indexOf(firstQuestion));
  expect(text.indexOf(firstQuestion)).toBeLessThan(text.indexOf(followupQuestion));
  await expect(conversation.getByText(followupQuestion, { exact: true })).toHaveCount(1);
  await expect(load).toHaveCount(0);
  await expect(page.getByRole("textbox", { name: "Message to support" })).toBeEnabled();
  expect(api.unexpected).toEqual([]);
});

test("submits a review without an optional attachment and reuses the entered email", async ({ page }) => {
  const api = new CustomerApi();
  api.terminalReview = true;
  await beginChat(page, api);
  await page.getByRole("button", { name: "Delivery & tracking", exact: true }).click();
  await page.getByRole("button", { name: "Continue to delivery review", exact: true }).click();
  await expect(page.getByText("We already have", { exact: false })).toContainText(buyerEmail);
  await expect(page.locator('input[type="file"]')).toHaveCount(0);
  await page.getByRole("button", { name: "Send for review", exact: true }).click();
  await expect(page.getByText("Your case has been submitted.", { exact: true })).toBeVisible();
  expect(api.contactRequests).toEqual([{ customer_email: buyerEmail, evidence_na: true }]);
  expect(api.unexpected).toEqual([]);
});

test("keeps a closed conversation readable after reload without reopening the case", async ({ page }) => {
  const api = new CustomerApi();
  api.consent = true;
  api.email = buyerEmail;
  api.ticket = {
    ...makeTicket("defect", "Your issue is resolved.", true),
    status: "resolved",
    admin_decision: "self_resolved",
    troubleshooting_outcome: "resolved",
  };
  api.conversation = [
    { id: 1, role: "user", content: "The steps fixed my chair." },
    { id: 2, role: "assistant", content: "Your issue is resolved." },
  ];
  await api.install(page);
  await page.goto("/warranty");
  await expect(page.getByRole("log", { name: "Conversation" })).toContainText("The steps fixed my chair.");
  await page.reload();
  const conversation = page.getByRole("log", { name: "Conversation" });
  await expect(conversation).toContainText("The steps fixed my chair.");
  await expect(conversation).toContainText("Your issue is resolved.");
  await expect(page.getByRole("textbox", { name: "Message to support" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Send", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Ask for a person", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Start over", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Start a new case", exact: true })).toBeVisible();
  expect(api.ticket.status).toBe("resolved");
  expect(api.unexpected).toEqual([]);
});

for (const localized of [
  {
    language: "ko" as const,
    heading: "안전 안내 및 지원",
    report: "의자에서 연기가 나요.",
    message: "의자 사용을 즉시 중단하세요. 다시 켜서 시험하지 마세요.",
    call: "보증 지원팀 전화",
    email: "보증 지원팀에 이메일 보내기",
    pending: "지원팀 알림 전송이 아직 확인되지 않았습니다.",
    reference: "접수 번호: WR-SAFETY",
  },
  {
    language: "es" as const,
    heading: "Orientación de seguridad y asistencia",
    report: "Se ven llamas debajo de mi silla.",
    message: "Deje de usar la silla. No vuelva a encenderla para probarla.",
    call: "Llamar a Garantía",
    email: "Enviar correo a Garantía",
    pending: "No se ha confirmado el envío de una notificación al equipo.",
    reference: "Referencia del caso: WR-SAFETY",
  },
]) {
  test(`restores ${localized.language} safety guidance and truthful notification status`, async ({ page }) => {
    const api = new CustomerApi();
    api.consent = true;
    api.email = buyerEmail;
    api.ticket = { ...makeTicket("defect", localized.message), case_reference: "WR-SAFETY" };
    api.handoff = {
      requested: true,
      reason: "safety_hazard",
      language: localized.language,
      notification_sent: false,
      message: localized.message,
      department: "warranty",
      phone: "+1-888-848-2630 ext. 3",
      phone_href: "tel:+18888482630,3",
      email: "service@osakititan.com",
    };
    api.conversation = [
      { id: 1, role: "user", content: localized.report },
      { id: 2, role: "assistant", content: localized.message },
    ];
    await api.install(page);
    await page.goto("/warranty");

    const notice = page.getByRole("region", { name: localized.heading, exact: true });
    await expect(notice).toBeVisible();
    await expect(notice).toHaveAttribute("lang", localized.language);
    await expect(notice).toContainText(localized.message);
    await expect(notice).toContainText(localized.reference);
    await expect(notice).toContainText(localized.pending);
    await expect(notice.getByRole("link", { name: localized.call })).toHaveAttribute("href", "tel:+18888482630,3");
    await expect(notice.getByRole("link", { name: localized.email, exact: true })).toHaveAttribute("href", "mailto:service@osakititan.com");
    await expect(page.getByRole("textbox", { name: "Message to support" })).toHaveCount(0);

    api.handoff.notification_sent = true;
    await page.reload();
    await expect(notice).toBeVisible();
    await expect(notice).toHaveAttribute("lang", localized.language);
    await expect(notice).toContainText(localized.message);
    await expect(notice.getByText(localized.pending, { exact: false })).toHaveCount(0);
    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(localized.report);
    await expect(page.getByRole("textbox", { name: "Message to support" })).toHaveCount(0);
    expect(api.unexpected).toEqual([]);
  });
}
