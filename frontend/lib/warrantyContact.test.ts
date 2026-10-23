import { describe, expect, it } from "vitest";
import { buildWarrantySubmissionMessage, resolveSupportContact } from "./warrantyContact";

describe("case submission acknowledgment", () => {
  it.each(["delivery", "installation", "defect"])("does not claim a %s receipt or response deadline when mail was not confirmed", (issueType) => {
    const text = buildWarrantySubmissionMessage({ issueType, caseReference: "WR-123456", receiptEmailSent: false });
    expect(text).toContain("saved for team review");
    expect(text).toContain("WR-123456");
    expect(text).not.toContain("emailed");
    expect(text).not.toContain("24 hours");
    expect(text).not.toContain("callback");
  });

  it("acknowledges an email only when the receipt API confirms it", () => {
    expect(buildWarrantySubmissionMessage({ receiptEmailSent: true })).toContain("We emailed a confirmation");
    expect(buildWarrantySubmissionMessage({})).not.toContain("emailed");
  });

  it("derives the displayed phone extension safely for older handoff payloads", () => {
    expect(resolveSupportContact({ department: "sales", phone: "+1-888-848-2630 ext. 2" })).toMatchObject({ phoneHref: "tel:+18888482630,2", email: null });
    expect(resolveSupportContact({ phone: "+1-888-848-2630 ext. 3", phone_href: "javascript:alert(1)" }).phoneHref).toBe("tel:+18888482630,3");
    expect(resolveSupportContact({ department: "sales", phone: "+1-888-848-2630 ext. 2", phone_href: "tel:+18888482630;ext=2" }).phoneHref).toBe("tel:+18888482630;ext=2");
  });
});
