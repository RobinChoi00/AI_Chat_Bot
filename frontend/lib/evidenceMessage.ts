const DEFAULT_EVIDENCE_EMAIL = "service@osakititan.com";

/** Standard warranty contact footer appended to terminal prompts. */
export function buildWarrantyContactFooter(
  evidenceEmail?: string | null
): string {
  const email = evidenceEmail?.trim() || DEFAULT_EVIDENCE_EMAIL;
  return (
    `For warranty support, contact us at ${email}.\n` +
    `If you leave your email address, our warranty team will respond within 24 hours.`
  );
}

/**
 * Customer-facing evidence request appended to terminal workflow prompts.
 */
export function buildEvidenceRequestMessage(
  evidenceRequired: string[] | undefined,
  evidenceEmail?: string | null
): string | null {
  if (!evidenceRequired?.length) return null;

  const email = evidenceEmail?.trim() || DEFAULT_EVIDENCE_EMAIL;
  const hasVideo = evidenceRequired.includes("video_of_issue");
  const hasDamagePhotos =
    evidenceRequired.includes("damage_photos") ||
    evidenceRequired.includes("box_photos");
  const hasOtherPhotos = evidenceRequired.some(
    (key) => key.startsWith("photo_of_") || key === "signed_delivery_receipt"
  );

  if (hasVideo) {
    if (hasDamagePhotos || hasOtherPhotos) {
      return `Please send photos or a video of the issue to ${email}. You can also upload using the form below.`;
    }
    return `Please send a photo or video of the issue to ${email}. You can also upload using the form below.`;
  }

  if (hasDamagePhotos || hasOtherPhotos) {
    return `Please send the requested photos to ${email}. You can also upload using the form below.`;
  }

  return `Please send the requested files to ${email}. You can also upload using the form below.`;
}

function appendWarrantyFooter(text: string, evidenceEmail?: string | null): string {
  const footer = buildWarrantyContactFooter(evidenceEmail);
  const lower = text.toLowerCase();
  if (
    lower.includes("within 24 hours") &&
    lower.includes(DEFAULT_EVIDENCE_EMAIL)
  ) {
    return text;
  }
  return `${text}\n\n${footer}`;
}

/** Combine the flowchart terminal prompt with a standardized evidence request. */
export function formatTerminalPrompt(
  prompt: string,
  evidenceRequired?: string[],
  evidenceEmail?: string | null
): string {
  const lower = prompt.toLowerCase();
  const emailInPrompt = lower.includes("service@osakititan.com");
  const hasVideoEvidence = evidenceRequired?.includes("video_of_issue");

  if (emailInPrompt && hasVideoEvidence && !lower.includes("video")) {
    return appendWarrantyFooter(
      `${prompt}\n\nPlease include a photo or video of the issue if possible. You can also upload using the form below.`,
      evidenceEmail
    );
  }

  if (emailInPrompt && !hasVideoEvidence) {
    return appendWarrantyFooter(
      `${prompt}\n\nYou can also upload using the form below.`,
      evidenceEmail
    );
  }

  const evidenceNote = buildEvidenceRequestMessage(evidenceRequired, evidenceEmail);
  if (!evidenceNote) {
    return appendWarrantyFooter(prompt, evidenceEmail);
  }
  return appendWarrantyFooter(`${prompt}\n\n${evidenceNote}`, evidenceEmail);
}

export const WARRANTY_CONTACT_EMAIL = DEFAULT_EVIDENCE_EMAIL;

/** Detect the first email address in free text (customer reply). */
export function extractEmailFromText(text: string): string | null {
  const match = text.match(/[\w.+-]+@[\w.-]+\.\w+/);
  return match ? match[0].toLowerCase() : null;
}
