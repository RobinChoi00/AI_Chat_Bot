"use client";

import { useMemo, useRef, useState, type ChangeEvent } from "react";
import { submitWarrantyContact, uploadEvidence } from "@/lib/api";
import { WARRANTY_CONTACT_BLURB } from "@/lib/warrantyContact";
import WarrantyTeamContactFooter from "./WarrantyTeamContactFooter";
import type { EvidenceUploadResponse, WarrantyContactResponse } from "@/lib/types";

interface Props {
  ticketId: string;
  issueType?: string;
  evidenceRequired?: string[];
  /** Email already collected at intake — skip re-asking when valid. */
  initialCustomerEmail?: string;
  onContactSuccess?: (customerEmail: string, response: WarrantyContactResponse) => void;
  onUploadSuccess?: (filename: string, response: EvidenceUploadResponse) => void;
  onCustomerEmailConfirmed?: (customerEmail: string) => void;
  onError?: () => void;
  collapsed?: boolean;
  onToggleCollapsed?: (next: boolean) => void;
  onBack?: () => void;
  backDisabled?: boolean;
}

const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".pdf", ".mp4", ".mov"];
const MAX_BYTES = 20 * 1024 * 1024; // 20 MB
const EMAIL_RE = /^[\w.+-]+@[\w.-]+\.\w+$/;

const EVIDENCE_LABELS: Record<string, string> = {
  damage_photos: "Damage Photos",
  box_photos: "Box / Packaging Photos",
  signed_delivery_receipt: "Signed Delivery Receipt",
  video_of_issue: "Photo or Video of Issue",
  proof_of_purchase: "Proof of Purchase",
  photo_of_defect: "Photo of Defect",
  photo_of_chair: "Photo of Chair",
  proof_of_delivery: "Proof of Delivery",
  assembly_photo: "Assembly Photo",
  remote_photo: "Remote Photo",
  photo_of_remote: "Photo of Remote",
  photo_of_cable: "Photo of Cable",
  photo_of_fuse: "Photo of Fuse",
  photo_of_power_area: "Photo of Power Area",
  other: "Other",
};

function isValidEmail(value: string): boolean {
  return EMAIL_RE.test(value.trim());
}

export default function EvidenceUploader({
  ticketId,
  issueType,
  evidenceRequired = [],
  initialCustomerEmail = "",
  onContactSuccess,
  onUploadSuccess,
  onCustomerEmailConfirmed,
  onError,
  collapsed = false,
  onToggleCollapsed,
  onBack,
  backDisabled = false,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const requestInFlight = useRef(false);
  const knownEmail = isValidEmail(initialCustomerEmail)
    ? initialCustomerEmail.trim()
    : "";
  const [editingEmail, setEditingEmail] = useState(false);
  const [customerEmail, setCustomerEmail] = useState(
    () => knownEmail || initialCustomerEmail
  );
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const requiredEvidence = evidenceRequired.length > 0;
  // The server accepts a review request without attachments. Ask for helpful
  // evidence progressively instead of treating missing photos as a dead end.
  const [evidenceNa, setEvidenceNa] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<"success" | "error" | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  const hasKnownEmail = Boolean(knownEmail) && !editingEmail;
  const activeEmail = hasKnownEmail ? knownEmail : customerEmail.trim();

  const evidenceTypeOptions = useMemo(() => {
    if (evidenceRequired.length === 0) {
      return [{ value: "other", label: "Other" }];
    }
    return evidenceRequired.map((key) => ({
      value: key,
      label: EVIDENCE_LABELS[key] ?? key.replace(/_/g, " "),
    }));
  }, [evidenceRequired]);

  const [evidenceType, setEvidenceType] = useState(evidenceTypeOptions[0]?.value ?? "other");

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    setResult(null);
    setErrorMsg("");

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMsg(`File type ${ext} is not allowed. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`);
      onError?.();
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = "";
      return;
    }

    if (file.size > MAX_BYTES) {
      setErrorMsg("File is too large. Maximum size is 20 MB.");
      onError?.();
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = "";
      return;
    }

    setSelectedFile(file);
    setEvidenceNa(false);
  }

  function handleNaChange(checked: boolean) {
    setEvidenceNa(checked);
    setResult(null);
    setErrorMsg("");
    if (checked) {
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  function validateEmail(): string | null {
    if (!activeEmail || !isValidEmail(activeEmail)) {
      setResult("error");
      setErrorMsg("Please enter a valid email address.");
      onError?.();
      return null;
    }
    return activeEmail;
  }

  async function handleSubmit() {
    if (requestInFlight.current || result === "success") return;
    const email = validateEmail();
    if (!email) return;

    const willUpload = !evidenceNa && selectedFile;
    if (!evidenceNa && !selectedFile) {
      setResult("error");
      setErrorMsg("Choose a file, or continue without an attachment.");
      onError?.();
      return;
    }

    setSubmitting(true);
    requestInFlight.current = true;
    setResult(null);
    setErrorMsg("");

    try {
      if (willUpload && selectedFile) {
        const resp = await uploadEvidence(ticketId, evidenceType, selectedFile, email);
        setResult("success");
        onCustomerEmailConfirmed?.(email);
        onUploadSuccess?.(resp.original_filename, resp);
        setSelectedFile(null);
        if (fileRef.current) fileRef.current.value = "";
      } else {
        const resp = await submitWarrantyContact(ticketId, email);
        setResult("success");
        onCustomerEmailConfirmed?.(email);
        onContactSuccess?.(email, resp);
      }
    } catch (err: unknown) {
      setResult("error");
      setErrorMsg(err instanceof Error ? err.message : "Submission failed.");
      onError?.();
    } finally {
      requestInFlight.current = false;
      setSubmitting(false);
    }
  }

  const emailValid = isValidEmail(activeEmail);
  const needsFile = !evidenceNa && !selectedFile;
  const submitDisabled = submitting || result === "success" || !emailValid || needsFile;

  if (collapsed) {
    return (
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-dashed border-gray-300 bg-gray-50 px-3 py-2">
        <p className="text-xs text-gray-600">
          Contact form is hidden — you can keep chatting below.
        </p>
        <div className="flex shrink-0 items-center gap-1.5">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              disabled={backDisabled || submitting}
              className="rounded-full border border-gray-300 bg-white px-3 py-1 text-xs font-medium text-gray-700 hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
              title="Go back to the previous troubleshooting step"
            >
              ← Back
            </button>
          )}
          <button
            type="button"
            onClick={() => onToggleCollapsed?.(false)}
            className="rounded-full border border-brand-200 bg-white px-3 py-1 text-xs font-medium text-brand-700 hover:bg-brand-50"
          >
            Show contact form
          </button>
        </div>
      </div>
    );
  }

  const title =
    issueType === "installation"
      ? "Send this installation case for review"
      : issueType === "delivery"
        ? "Send this delivery case for review"
        : "Send this case for review";

  const reachBlurb =
    issueType === "installation"
      ? "Ready to save this setup case for our team to review?"
      : issueType === "delivery"
        ? "Ready to save this delivery case for review of damage, missing parts, or tracking?"
        : "Ready to save this case for our warranty team to review?";

  const submitLabel = submitting ? "Saving request…" : result === "success" ? "Request saved" : selectedFile && !evidenceNa ? "Send for review with attachment" : "Send for review";

  return (
    <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-4">
      <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
        <p className="text-sm font-medium text-gray-800">{title}</p>
        <div className="flex shrink-0 items-center gap-1.5">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              disabled={backDisabled || submitting}
              className="rounded-full border border-gray-300 bg-white px-2.5 py-0.5 text-[11px] font-medium text-gray-700 hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
              title="Go back to the previous troubleshooting step"
            >
              ← Back
            </button>
          )}
          {onToggleCollapsed && (
            <button
              type="button"
              onClick={() => onToggleCollapsed(true)}
              className="rounded-full border border-gray-200 bg-white px-2.5 py-0.5 text-[11px] font-medium text-gray-600 hover:bg-gray-100"
              title="Hide the contact form so you can keep chatting"
            >
              Add a note instead
            </button>
          )}
        </div>
      </div>

      {hasKnownEmail ? (
        <div className="mb-3 space-y-2">
          <p className="text-xs text-gray-600">
            We already have <strong className="text-gray-900">{knownEmail}</strong> for
            this case. {reachBlurb}{" "}
            {requiredEvidence
              ? "Please attach the requested photos or video if you have them."
              : "Photos and videos are optional."}
          </p>
          <button
            type="button"
            onClick={() => {
              setEditingEmail(true);
              setResult(null);
              setErrorMsg("");
            }}
            className="text-xs font-medium text-brand-700 underline-offset-2 hover:underline"
          >
            Use a different email
          </button>
        </div>
      ) : (
        <>
          <p className="mb-3 text-xs text-gray-500">
            {issueType === "installation"
              ? "Enter your email so our team can follow up with installation support. "
              : issueType === "delivery"
                ? "Enter your email so our team can follow up on this delivery case. "
                : "Enter your email so our warranty team can follow up. "}
            {requiredEvidence
              ? "Please attach the requested photos or video if you have them."
              : "Photos and videos are optional."}
          </p>
          <div className="mb-3">
            <label className="mb-1 block text-xs text-gray-500">
              Your email address <span className="text-red-500">*</span>
            </label>
            <input
              type="email"
              required
              value={customerEmail}
              onChange={(e) => setCustomerEmail(e.target.value)}
              placeholder="you@example.com"
              className="min-h-[48px] w-full rounded-xl border border-gray-200 bg-white px-3 py-2.5 text-base focus:outline-none focus:ring-2 focus:ring-brand-500 sm:text-sm"
            />
            {editingEmail && knownEmail && (
              <button
                type="button"
                onClick={() => {
                  setCustomerEmail(knownEmail);
                  setEditingEmail(false);
                }}
                className="mt-1.5 text-xs font-medium text-gray-600 underline-offset-2 hover:underline"
              >
                Keep {knownEmail}
              </button>
            )}
          </div>
        </>
      )}

      <button
        type="button"
        onClick={() => handleNaChange(!evidenceNa)}
        disabled={submitting || result === "success"}
        aria-expanded={!evidenceNa}
        className="mb-3 min-h-[44px] text-left text-sm font-medium text-brand-700 underline underline-offset-2 disabled:opacity-50"
      >
        {evidenceNa ? "Add photos or documents (optional)" : "Continue without an attachment"}
      </button>

      {!evidenceNa && (
        <div className="mb-3 rounded-lg border border-gray-200 bg-white p-3">
          <p className="mb-2 text-xs font-medium text-gray-600">Attach photo or video</p>

          {evidenceTypeOptions.length > 1 && <div className="mb-3">
            <label className="mb-1 block text-xs text-gray-500">What are you attaching?</label>
            <select
              value={evidenceType}
              onChange={(e) => setEvidenceType(e.target.value)}
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              {evidenceTypeOptions.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>}

          <div>
            <label className="mb-1 block text-xs text-gray-500">
              File{" "}
              <span className="text-gray-400">
                (jpg, jpeg, png, webp, pdf, mp4, mov — max 20 MB)
              </span>
            </label>
            <input
              ref={fileRef}
              type="file"
              accept={ALLOWED_EXTENSIONS.join(",")}
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-600 file:mr-3 file:rounded-full file:border-0 file:bg-brand-50 file:px-3 file:py-1 file:text-xs file:font-medium file:text-brand-700 hover:file:bg-brand-100"
            />
            {selectedFile && (
              <p className="mt-1 text-xs text-gray-500">
                Selected: <strong>{selectedFile.name}</strong>{" "}
                ({(selectedFile.size / 1024).toFixed(1)} KB)
              </p>
            )}
          </div>
        </div>
      )}

      {errorMsg && (
        <p role="alert" className="mb-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">
          ❌ {errorMsg}
        </p>
      )}

      {result === "success" && (
        <p role="status" className="mb-2 rounded-lg bg-green-50 px-3 py-2 text-xs text-green-700">
          ✅ Thank you — {WARRANTY_CONTACT_BLURB}
        </p>
      )}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={submitDisabled}
        className={`min-h-[48px] w-full rounded-xl px-4 py-2.5 text-sm font-semibold transition ${
          submitDisabled
            ? "cursor-not-allowed bg-gray-200 text-gray-400"
            : "bg-brand-600 text-white hover:bg-brand-700 active:scale-[0.98]"
        }`}
      >
        {submitLabel}
      </button>

      <WarrantyTeamContactFooter className="mt-4" />
    </div>
  );
}
