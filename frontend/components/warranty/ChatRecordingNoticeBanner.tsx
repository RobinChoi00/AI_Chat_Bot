"use client";

import { openExternalLink } from "@/lib/openExternalLink";
import { resolveStorePolicyUrls } from "@/lib/welcomeMessage";
import {
  WARRANTY_SUPPORT_PHONE,
  WARRANTY_SUPPORT_PHONE_HREF,
} from "@/lib/warrantyContact";

interface Props {
  className?: string;
  consentAccepted: boolean;
  onAccept: () => void;
  storeDomain?: string;
  disabled?: boolean;
  error?: string | null;
  pendingLabel?: string;
}

/** Top-of-chat privacy / recording disclosure with store policy links. */
export default function ChatRecordingNoticeBanner({
  className = "",
  consentAccepted,
  onAccept,
  storeDomain,
  disabled = false,
  error,
  pendingLabel = "Saving consent…",
}: Props) {
  const { privacy, terms } = resolveStorePolicyUrls(storeDomain);

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
          onClick={(e) => openExternalLink(privacy, e)}
        >
          Privacy Policy
        </a>{" "}
        and{" "}
        <a
          href={terms}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline hover:text-amber-900"
          onClick={(e) => openExternalLink(terms, e)}
        >
          Terms of Service
        </a>
        . This chat is recorded and stored for support. Our team and service
        providers may review the messages, details, and attachments you share for this case.
      </p>
      <details className="mt-1 text-left">
        <summary className="cursor-pointer text-center underline underline-offset-2">How this chat uses your information</summary>
        <p className="mt-2 text-xs leading-relaxed">
          Messages, order or tracking details, email, and photos or videos you share
          may be reviewed by our support team and processors used for this case
          (including OpenAI to prepare AI-assisted responses, helpdesk, shipping
          lookup, and email delivery). We retain support records and attachments
          under our retention policy, and may preserve them for legal obligations.
        </p>
      </details>
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
            disabled={disabled}
            className="min-h-[44px] rounded-xl bg-amber-700 px-4 py-2 text-sm font-semibold text-white hover:bg-amber-800 disabled:opacity-50"
          >
            {disabled ? pendingLabel : "I Agree"}
          </button>
          {error && <p role="alert" className="text-xs text-red-700">{error}</p>}
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
