"use client";

import { resolveStorePolicyUrls } from "@/lib/welcomeMessage";
import {
  WARRANTY_SUPPORT_PHONE,
  WARRANTY_SUPPORT_PHONE_HREF,
} from "@/lib/warrantyContact";

interface Props {
  className?: string;
  consentAccepted: boolean;
  onAccept: () => void;
}

/** Top-of-chat privacy / recording disclosure with store policy links. */
export default function ChatRecordingNoticeBanner({
  className = "",
  consentAccepted,
  onAccept,
}: Props) {
  const { privacy, terms } = resolveStorePolicyUrls();

  return (
    <div
      className={`shrink-0 border-b border-amber-100 bg-amber-50 px-4 py-2 text-center text-[11px] leading-snug text-amber-800 sm:text-xs ${className}`}
    >
      <p>
        By continuing this chat, you agree to our{" "}
        <a
          href={privacy}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline hover:text-amber-900"
        >
          Privacy Policy
        </a>{" "}
        and{" "}
        <a
          href={terms}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline hover:text-amber-900"
        >
          Terms of Service
        </a>
        . This conversation may be recorded, stored, and reviewed to provide support
        and improve our service.
      </p>
      {!consentAccepted ? (
        <div className="mt-2 space-y-2">
          <p className="text-[10px] text-amber-700 sm:text-[11px]">
            If you do not agree, please close this chat and contact us by phone at{" "}
            <a
              href={WARRANTY_SUPPORT_PHONE_HREF}
              className="font-medium underline hover:text-amber-900"
            >
              {WARRANTY_SUPPORT_PHONE}
            </a>
            .
          </p>
          <button
            type="button"
            onClick={onAccept}
            className="rounded-full bg-amber-700 px-4 py-1.5 text-xs font-semibold text-white hover:bg-amber-800"
          >
            I Agree — start chat
          </button>
        </div>
      ) : (
        <p className="mt-1 text-[10px] text-amber-700 sm:text-[11px]">
          Need help by phone?{" "}
          <a
            href={WARRANTY_SUPPORT_PHONE_HREF}
            className="font-medium underline hover:text-amber-900"
          >
            {WARRANTY_SUPPORT_PHONE}
          </a>
        </p>
      )}
    </div>
  );
}
