"use client";

import { resolveStorePolicyUrls } from "@/lib/welcomeMessage";

interface Props {
  className?: string;
}

/** Top-of-chat privacy / recording disclosure with store policy links. */
export default function ChatRecordingNoticeBanner({ className = "" }: Props) {
  const { privacy, terms } = resolveStorePolicyUrls();

  return (
    <div
      className={`shrink-0 border-b border-amber-100 bg-amber-50 px-4 py-1.5 text-center text-[11px] leading-snug text-amber-800 sm:text-xs ${className}`}
    >
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
    </div>
  );
}
