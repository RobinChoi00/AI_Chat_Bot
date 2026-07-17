/** Opening assistant message shown when a customer starts a chat session. */

import { resolveWarrantyStoreDomain } from "./warrantyStoreDomain";

/** Prior notice shown at the very top of the chat before the user types. */
export const CHAT_RECORDING_NOTICE =
  "By continuing this chat, you agree to our Privacy Policy and Terms of Service. " +
  "This conversation may be recorded, stored, and reviewed to provide support and improve our service.";

/** Short sticky line (kept for backward-compatible imports). */
export const CHAT_RECORDING_NOTICE_SHORT = CHAT_RECORDING_NOTICE;

/** Shopify policy URLs for the store hosting the chat embed. */
export function resolveStorePolicyUrls(domain?: string): {
  privacy: string;
  terms: string;
} {
  const host = (domain ?? resolveWarrantyStoreDomain())
    .replace(/^https?:\/\//i, "")
    .replace(/\/$/, "")
    .toLowerCase();
  const base = `https://${host}`;
  return {
    privacy: `${base}/policies/privacy-policy`,
    terms: `${base}/policies/terms-of-service`,
  };
}

export const CHAT_WELCOME_MESSAGE =
  "Hello! Welcome to Osaki & Titan support. 👋\n\n" +
  "Which massage chair model do you have? " +
  "You can find it on the serial-number sticker on your chair " +
  "(for example, OS-4000T, Solo Flex, or Hypnos 4D).\n\n" +
  "Tell me your model, or ask about specs, pricing, orders, delivery, warranty, " +
  "or troubleshooting.";

export const WARRANTY_WELCOME_MESSAGE =
  "Hello! 👋\n\n" +
  "This guide helps with **setup, warranty, and delivery** for your Osaki or Titan chair.\n\n" +
  "Tell us your **chair model and what's going wrong** in one message if you can " +
  "(for example: *OS-4000T footrest air not inflating*).\n\n" +
  "If you only know the model for now, type that — we'll ask about the issue next.";
