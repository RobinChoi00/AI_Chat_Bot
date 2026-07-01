"use client";

/**
 * SerialPhotoButton
 *
 * A camera / gallery button rendered above the warranty chat input during the
 * first-intake step. When the customer snaps a photo of the warranty sticker
 * on the base of their chair, we:
 *
 *   1. POST the image to /api/v1/warranty/ocr/serial (see lib/api.ts)
 *   2. Show a small "Detected: OS-4000T (high confidence). Use this?" bar
 *   3. Let the customer confirm (auto-fills the intake textarea) or discard.
 *
 * The photo is never stored on the server — the backend just runs vision OCR
 * and returns the extracted fields.
 */

import { useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { extractSerialFromPhoto, type SerialOcrResponse } from "@/lib/api";

interface Props {
  /** Called with the confirmed model name (already normalized by the backend). */
  onModelDetected: (modelName: string) => void;
  disabled?: boolean;
}

type Status =
  | { kind: "idle" }
  | { kind: "reading" }
  | { kind: "detected"; data: SerialOcrResponse; filename: string }
  | { kind: "not_readable"; filename: string }
  | { kind: "error"; message: string };

const MAX_BYTES = 8 * 1024 * 1024;

export default function SerialPhotoButton({ onModelDetected, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  function openPicker() {
    if (disabled) return;
    inputRef.current?.click();
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    // Reset so picking the same file twice still fires onChange.
    event.target.value = "";
    if (!file) return;

    if (file.size > MAX_BYTES) {
      setStatus({
        kind: "error",
        message: "That photo is over 8 MB. Please retake at a lower resolution.",
      });
      return;
    }

    setStatus({ kind: "reading" });
    try {
      const data = await extractSerialFromPhoto(file);
      if (data.model_name) {
        setStatus({ kind: "detected", data, filename: file.name });
      } else {
        setStatus({ kind: "not_readable", filename: file.name });
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "We could not read that photo.";
      setStatus({ kind: "error", message });
    }
  }

  function confirmDetection() {
    if (status.kind !== "detected") return;
    const name = status.data.model_name;
    if (!name) return;
    onModelDetected(name);
    setStatus({ kind: "idle" });
  }

  function dismiss() {
    setStatus({ kind: "idle" });
  }

  return (
    <div className="flex flex-col gap-2 text-xs">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={openPicker}
          disabled={disabled || status.kind === "reading"}
          className="flex items-center gap-1.5 rounded-full border border-brand-300 bg-brand-50 px-3 py-1.5 text-xs font-medium text-brand-700 hover:bg-brand-100 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <span aria-hidden>📷</span>
          {status.kind === "reading"
            ? "Reading label…"
            : "Snap a photo of the serial sticker"}
        </button>
        <span className="text-gray-500">
          Not sure of your model? We&apos;ll read the sticker on the chair base.
        </span>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFile}
        className="hidden"
      />

      {status.kind === "detected" && status.data.model_name && (
        <div className="rounded-lg border border-emerald-300 bg-emerald-50 px-3 py-2">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="font-medium text-emerald-800">
                Detected: {status.data.model_name}
              </div>
              <div className="text-emerald-700">
                Confidence: {status.data.confidence}
                {status.data.serial_number
                  ? ` · Serial ${status.data.serial_number}`
                  : ""}
              </div>
            </div>
            <div className="flex shrink-0 gap-1.5">
              <button
                type="button"
                onClick={confirmDetection}
                className="rounded-md bg-emerald-600 px-2.5 py-1 text-white hover:bg-emerald-700"
              >
                Use this
              </button>
              <button
                type="button"
                onClick={dismiss}
                className="rounded-md border border-emerald-300 bg-white px-2.5 py-1 text-emerald-700 hover:bg-emerald-100"
              >
                Discard
              </button>
            </div>
          </div>
        </div>
      )}

      {status.kind === "not_readable" && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-amber-800">
          We couldn&apos;t read a model name off that photo. Try again in better
          light, or just type the model below.
        </div>
      )}

      {status.kind === "error" && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700">
          {status.message}
        </div>
      )}
    </div>
  );
}
