"use client";

import { useMemo, useRef, useState, type ChangeEvent } from "react";
import { submitWarrantyContact, uploadEvidence } from "@/lib/api";
import { WARRANTY_CONTACT_BLURB } from "@/lib/warrantyContact";
import WarrantyTeamContactFooter from "./WarrantyTeamContactFooter";

interface Props {
  ticketId: string;
  evidenceRequired?: string[];
  /** Email already collected at intake — skip re-asking when valid. */
  initialCustomerEmail?: string;
  onContactSuccess?: (customerEmail: string) => void;
  onUploadSuccess?: (filename: string) => void;
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
  evidenceRequired = [],
  initialCustomerEmail = "",
  onContactSuccess,
  onUploadSuccess,
  collapsed = false,
  onToggleCollapsed,
  onBack,
  backDisabled = false,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const knownEmail = isValidEmail(initialCustomerEmail)
    ? initialCustomerEmail.trim()
    : "";
  const [editingEmail, setEditingEmail] = useState(false);
  const [customerEmail, setCustomerEmail] = useState(
    () => knownEmail || initialCustomerEmail
  );
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  /** Default N/A — most customers submit email only without media. */
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
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = "";
      return;
    }

    if (file.size > MAX_BYTES) {
      setErrorMsg("File is too large. Maximum size is 20 MB.");
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
      return null;
    }
    return activeEmail;
  }

  async function handleSubmit() {
    const email = validateEmail();
    if (!email) return;

    const willUpload = !evidenceNa && selectedFile;
    if (!evidenceNa && !selectedFile) {
      setResult("error");
      setErrorMsg("Choose a file, or check N/A to submit without a photo/video.");
      return;
    }

    setSubmitting(true);
    setResult(null);
    setErrorMsg("");

    try {
      if (willUpload && selectedFile) {
        const resp = await uploadEvidence(ticketId, evidenceType, selectedFile, email);
        setResult("success");
        onUploadSuccess?.(resp.original_filename);
        setSelectedFile(null);
        if (fileRef.current) fileRef.current.value = "";
      } else {
        await submitWarrantyContact(ticketId, email);
        setResult("success");
        onContactSuccess?.(email);
      }
    } catch (err: unknown) {
      setResult("error");
      setErrorMsg(err instanceof Error ? err.message : "Submission failed.");
    } finally {
      setSubmitting(false);
    }
  }

  const emailValid = isValidEmail(activeEmail);
  const needsFile = !evidenceNa && !selectedFile;
  const submitDisabled = submitting || !emailValid || needsFile;

  if (collapsed) {
    return (
      <div className="flex items-center justify-between gap-2 rounded-xl border border-dashed border-gray-300 bg-gray-50 px-3 py-2">
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

  const title = hasKnownEmail
    ? "Final step — send to our warranty team?"
    : "Final step — how can we reach you?";

  const submitLabel = submitting
    ? "Submitting…"
    : hasKnownEmail
      ? evidenceNa
        ? "Yes — notify warranty team"
        : selectedFile
          ? "Yes — send with attachment"
          : "Select a file or check N/A"
      : evidenceNa
        ? "Submit email only (N/A)"
        : selectedFile
          ? "Submit with attachment"
          : "Select a file or check N/A";

  return (
    <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-4">
      <div className="mb-1 flex items-start justify-between gap-2">
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
              Hide
            </button>
          )}
        </div>
      </div>

      {hasKnownEmail ? (
        <div className="mb-3 space-y-2">
          <p className="text-xs text-gray-600">
            We already have <strong className="text-gray-900">{knownEmail}</strong> for
            this case. Ready to send it to our warranty team? They&apos;ll follow up
            within 24 hours. Photos and videos are optional.
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
            Enter your email so our warranty team can follow up within 24 hours.{" "}
            Photos and videos are optional.
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

      <label className="mb-3 flex cursor-pointer items-start gap-2 rounded-lg border border-brand-200 bg-brand-50 px-3 py-2.5 text-sm">
        <input
          type="checkbox"
          checked={evidenceNa}
          onChange={(e) => handleNaChange(e.target.checked)}
          className="mt-0.5"
        />
        <span>
          <strong>N/A</strong> — I don&apos;t have photos or videos
          {hasKnownEmail ? " (notify team only)" : " (submit email only)"}
        </span>
      </label>

      {!evidenceNa && (
        <div className="mb-3 rounded-lg border border-gray-200 bg-white p-3">
          <p className="mb-2 text-xs font-medium text-gray-600">Attach photo or video</p>

          <div className="mb-3">
            <label className="mb-1 block text-xs text-gray-500">Evidence type</label>
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
          </div>

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
        <p className="mb-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">
          ❌ {errorMsg}
        </p>
      )}

      {result === "success" && (
        <p className="mb-2 rounded-lg bg-green-50 px-3 py-2 text-xs text-green-700">
          ✅ Thank you — {WARRANTY_CONTACT_BLURB}
        </p>
      )}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={submitDisabled}
        className={`w-full rounded-lg px-4 py-2.5 text-sm font-medium transition ${
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
