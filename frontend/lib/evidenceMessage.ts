const DEFAULT_EVIDENCE_EMAIL = "service@osakititan.com";

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
    return `${prompt}\n\nPlease include a photo or video of the issue if possible. You can also upload using the form below.`;
  }

  if (emailInPrompt && !hasVideoEvidence) {
    return `${prompt}\n\nYou can also upload using the form below.`;
  }

  const evidenceNote = buildEvidenceRequestMessage(evidenceRequired, evidenceEmail);
  if (!evidenceNote) return prompt;
  return `${prompt}\n\n${evidenceNote}`;
}
