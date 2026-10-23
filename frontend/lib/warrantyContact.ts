/** Warranty team contact info shown to customers (keep in sync with config.py). */
import type { SupportContact } from "./types";

export const WARRANTY_SUPPORT_PHONE = "+1-888-848-2630 ext. 3";
export const WARRANTY_SUPPORT_PHONE_HREF = "tel:+18888482630,3";

export const WARRANTY_SUPPORT_EMAIL = "service@osakititan.com";

/** Warranty department hours (not sales). */
export const WARRANTY_SUPPORT_HOURS = "Mon-Fri, 10:00 AM - 6:00 PM CST";

export const WARRANTY_CONTACT_BLURB =
  "Your request is saved for our warranty team to review.";

export function resolveSupportContact(contact?: SupportContact) {
  const department = contact?.department ?? "warranty";
  const phone = contact?.phone?.trim() || (department === "warranty" ? WARRANTY_SUPPORT_PHONE : "");
  const suppliedHref = contact?.phone_href?.trim();
  const parts = phone.match(/^(.*?)(?:\s*(?:ext\.?|extension|x)\s*(\d+))?$/i);
  const number = (parts?.[1] || "").replace(/[^\d+]/g, "");
  const derivedHref = /^\+?\d{7,15}$/.test(number) ? `tel:${number}${parts?.[2] ? `,${parts[2]}` : ""}` : null;
  const phoneHref = suppliedHref && /^tel:\+?[\d(),.\s-]+(?:;ext=\d+)?$/i.test(suppliedHref) ? suppliedHref : derivedHref;
  const candidateEmail = contact?.email?.trim() || (department === "warranty" ? WARRANTY_SUPPORT_EMAIL : "");
  const email = /^[^\s@?&#]+@[^\s@?&#]+\.[^\s@?&#]+$/.test(candidateEmail) ? candidateEmail : null;
  return { department, phone, phoneHref, email, hours: contact?.hours?.trim() || (department === "warranty" ? WARRANTY_SUPPORT_HOURS : "") };
}

/** Only a confirmed receipt result can support claiming that an email was sent. */
export function buildWarrantySubmissionMessage({ issueType, caseReference, receiptEmailSent = false }: {
  issueType?: string | null;
  caseReference?: string | null;
  receiptEmailSent?: boolean;
}): string {
  const ref = caseReference?.trim() ? ` Your case reference is **${caseReference.trim()}**.` : "";
  const receipt = receiptEmailSent ? " We emailed a confirmation." : "";
  const subject = issueType === "installation" ? "installation support request" : issueType === "delivery" ? "delivery case (damage, missing parts, or tracking)" : "product issue";
  return `Thank you — your ${subject} is saved for team review.${ref}${receipt}`;
}
