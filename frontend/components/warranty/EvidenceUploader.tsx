"use client";

import { useRef, useState, type ChangeEvent } from "react";
import { uploadEvidence } from "@/lib/api";
import type { EvidenceType } from "@/lib/types";

interface Props {
  ticketId: string;
  onUploadSuccess?: (filename: string) => void;
}

const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".pdf", ".mp4", ".mov"];
const MAX_BYTES = 20 * 1024 * 1024; // 20 MB

const EVIDENCE_TYPES: { value: EvidenceType; label: string }[] = [
  { value: "damage_photos",     label: "Damage Photos" },
  { value: "video_of_issue",    label: "Photo or Video of Issue" },
  { value: "proof_of_purchase", label: "Proof of Purchase" },
  { value: "photo_of_defect",   label: "Photo of Defect" },
  { value: "photo_of_chair",    label: "Photo of Chair" },
  { value: "proof_of_delivery", label: "Proof of Delivery" },
  { value: "assembly_photo",    label: "Assembly Photo" },
  { value: "remote_photo",      label: "Remote Photo" },
  { value: "other",             label: "Other" },
];

export default function EvidenceUploader({ ticketId, onUploadSuccess }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [evidenceType, setEvidenceType] = useState<EvidenceType>("damage_photos");
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<"success" | "error" | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    setResult(null);
    setErrorMsg("");

    if (!file) {
      setSelectedFile(null);
      return;
    }

    // Client-side extension check
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMsg(`File type ${ext} is not allowed. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`);
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = "";
      return;
    }

    // Client-side size check
    if (file.size > MAX_BYTES) {
      setErrorMsg("File is too large. Maximum size is 20 MB.");
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = "";
      return;
    }

    setSelectedFile(file);
  }

  async function handleUpload() {
    if (!selectedFile) return;
    setUploading(true);
    setResult(null);
    setErrorMsg("");

    try {
      const resp = await uploadEvidence(ticketId, evidenceType, selectedFile);
      setResult("success");
      onUploadSuccess?.(resp.original_filename);
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = "";
    } catch (err: unknown) {
      setResult("error");
      setErrorMsg(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-4">
      <p className="mb-3 text-sm font-medium text-gray-700">
        📎 Upload Evidence
      </p>

      {/* Evidence type selector */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-gray-500">
          Evidence type
        </label>
        <select
          value={evidenceType}
          onChange={(e) => setEvidenceType(e.target.value as EvidenceType)}
          className="w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
        >
          {EVIDENCE_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
      </div>

      {/* File input */}
      <div className="mb-3">
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

      {/* Error */}
      {errorMsg && (
        <p className="mb-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">
          ❌ {errorMsg}
        </p>
      )}

      {/* Success */}
      {result === "success" && (
        <p className="mb-2 rounded-lg bg-green-50 px-3 py-2 text-xs text-green-700">
          ✅ File uploaded successfully. Our team will review it shortly.
        </p>
      )}

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!selectedFile || uploading}
        className={`w-full rounded-lg px-4 py-2 text-sm font-medium transition ${
          !selectedFile || uploading
            ? "cursor-not-allowed bg-gray-200 text-gray-400"
            : "bg-brand-600 text-white hover:bg-brand-700 active:scale-[0.98]"
        }`}
      >
        {uploading ? "Uploading…" : "Upload File"}
      </button>

      {/* Disclaimer */}
      <p className="mt-2 text-center text-[10px] text-gray-400">
        You can also email photos or videos to{" "}
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
