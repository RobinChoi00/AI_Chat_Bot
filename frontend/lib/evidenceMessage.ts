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
      return (
        `If you can, please share photos or a video of the issue using the form below. ` +
        `If not, select N/A and submit your email only — our team will follow up within 24 hours.`
      );
    }
    return (
      `If you can, please share a photo or video using the form below. ` +
      `If not, select N/A and submit your email only — our team will follow up within 24 hours.`
    );
  }

  if (hasDamagePhotos || hasOtherPhotos) {
    return (
      `If you can, please share the requested photos using the form below. ` +
      `If not, select N/A and submit your email only — our team will follow up within 24 hours.`
    );
  }

  return `Please submit your email using the form below so our team can follow up within 24 hours.`;
}

/** Remind customers that email is the final step after the workflow completes. */
export function buildEvidenceEmailRequiredNote(): string {
  return (
    "As a final step, please enter your email address below so our team can follow up within 24 hours. " +
    "Photos or videos are optional — choose N/A if you cannot provide them."
  );
}

function appendEvidenceEmailRequired(text: string): string {
  const note = buildEvidenceEmailRequiredNote();
  const lower = text.toLowerCase();
  if (
    lower.includes("final step") ||
    lower.includes("select n/a") ||
    lower.includes("email address in the upload form")
  ) {
    return text;
  }
  return `${text}\n\n${note}`;
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
    return appendEvidenceEmailRequired(
      appendWarrantyFooter(
        `${prompt}\n\nIf you can, include a photo or video using the form below. Otherwise, select N/A and submit your email only.`,
        evidenceEmail
      )
    );
  }

  if (emailInPrompt && !hasVideoEvidence) {
    return appendEvidenceEmailRequired(
      appendWarrantyFooter(
        `${prompt}\n\nPlease submit your email using the form below as the final step.`,
        evidenceEmail
      )
    );
  }

  const evidenceNote = buildEvidenceRequestMessage(evidenceRequired, evidenceEmail);
  if (!evidenceNote) {
    return appendEvidenceEmailRequired(appendWarrantyFooter(prompt, evidenceEmail));
  }
  return appendEvidenceEmailRequired(
    appendWarrantyFooter(`${prompt}\n\n${evidenceNote}`, evidenceEmail)
  );
}

export const WARRANTY_CONTACT_EMAIL = DEFAULT_EVIDENCE_EMAIL;

/** Detect the first email address in free text (customer reply). */
export function extractEmailFromText(text: string): string | null {
  const match = text.match(/[\w.+-]+@[\w.-]+\.\w+/);
  return match ? match[0].toLowerCase() : null;
}
