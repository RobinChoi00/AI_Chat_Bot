"use client";

import { useMemo, useRef, useState, type ChangeEvent } from "react";
import { submitWarrantyContact, uploadEvidence } from "@/lib/api";

interface Props {
  ticketId: string;
  evidenceRequired?: string[];
  initialCustomerEmail?: string;
  onContactSuccess?: (customerEmail: string) => void;
  onUploadSuccess?: (filename: string) => void;
}

const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".pdf", ".mp4", ".mov"];
const MAX_BYTES = 20 * 1024 * 1024; // 20 MB

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

export default function EvidenceUploader({
  ticketId,
  evidenceRequired = [],
  initialCustomerEmail = "",
  onContactSuccess,
  onUploadSuccess,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customerEmail, setCustomerEmail] = useState(initialCustomerEmail);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<"success" | "error" | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

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
  }

  function validateEmail(): string | null {
    const email = customerEmail.trim();
    if (!email || !/^[\w.+-]+@[\w.-]+\.\w+$/.test(email)) {
      setResult("error");
      setErrorMsg("Please enter a valid email address.");
      return null;
    }
    return email;
  }

  async function handleSubmit() {
    const email = validateEmail();
    if (!email) return;

    setSubmitting(true);
    setResult(null);
    setErrorMsg("");

    try {
      if (selectedFile) {
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

  const emailValid = /^[\w.+-]+@[\w.-]+\.\w+$/.test(customerEmail.trim());
  const submitDisabled = submitting || !emailValid;

  return (
    <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-4">
      <p className="mb-1 text-sm font-medium text-gray-800">
        Final step — how can we reach you?
      </p>
      <p className="mb-3 text-xs text-gray-500">
        Enter your email so our warranty team can follow up within 24 hours.
        Photos and videos are helpful but <strong>not required</strong> — you can
        submit with email only.
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
          className="w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
      </div>

      <div className="mb-3 rounded-lg border border-gray-200 bg-white p-3">
        <p className="mb-2 text-xs font-medium text-gray-600">
          Attach photo or video{" "}
          <span className="font-normal text-gray-400">(optional)</span>
        </p>

        <div className="mb-3">
          <label className="mb-1 block text-xs text-gray-500">Evidence type</label>
          <select
            value={evidenceType}
            onChange={(e) => setEvidenceType(e.target.value)}
            disabled={!selectedFile}
            className="w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:bg-gray-50 disabled:text-gray-400"
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
          {selectedFile ? (
            <p className="mt-1 text-xs text-gray-500">
              Selected: <strong>{selectedFile.name}</strong>{" "}
              ({(selectedFile.size / 1024).toFixed(1)} KB)
            </p>
          ) : (
            <p className="mt-1 text-xs text-gray-400">
              No file selected — that&apos;s OK. Submit your email below.
            </p>
          )}
        </div>
      </div>

      {errorMsg && (
        <p className="mb-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">
          ❌ {errorMsg}
        </p>
      )}

      {result === "success" && (
        <p className="mb-2 rounded-lg bg-green-50 px-3 py-2 text-xs text-green-700">
          ✅ Thank you — our warranty team will follow up at your email within 24 hours.
        </p>
      )}

      <button
        onClick={handleSubmit}
        disabled={submitDisabled}
        className={`w-full rounded-lg px-4 py-2 text-sm font-medium transition ${
          submitDisabled
            ? "cursor-not-allowed bg-gray-200 text-gray-400"
            : "bg-brand-600 text-white hover:bg-brand-700 active:scale-[0.98]"
        }`}
      >
        {submitting
          ? "Submitting…"
          : selectedFile
            ? "Submit with attachment"
            : "Submit email only"}
      </button>

      <p className="mt-2 text-center text-[10px] text-gray-400">
        You can also email us directly at{" "}
        <a
          href="mailto:service@osakititan.com"
          className="text-brand-600 underline hover:text-brand-800"
        >
          service@osakititan.com
        </a>
        .
      </p>
    </div>
  );
}
