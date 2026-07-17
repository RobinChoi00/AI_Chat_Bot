/** Opening assistant message shown when a customer starts a chat session. */

import { resolveWarrantyStoreDomain } from "./warrantyStoreDomain";

/** sessionStorage key — set when the customer taps I Agree. */
export const CHAT_CONSENT_STORAGE_KEY = "warranty_chat_consent_accepted";

/** Hostnames that are chat infrastructure, not customer Shopify stores. */
const INTERNAL_POLICY_HOSTS = new Set([
  "help.osakichair.com",
  "admin.osakichair.com",
  "api.osakichair.com",
  "localhost",
  "127.0.0.1",
]);

/** Default policy links when store domain is unknown (chat host, new store, etc.). */
export const DEFAULT_POLICY_URLS = {
  privacy: "https://osakiusa.com/pages/privacy-policy",
  terms: "https://osakiusa.com/pages/terms-of-service",
} as const;

/** @deprecated use DEFAULT_POLICY_URLS — kept for tests/imports */
export const DEFAULT_POLICY_STORE_DOMAIN =
  process.env.NEXT_PUBLIC_DEFAULT_POLICY_STORE?.trim().toLowerCase() ||
  "osakiusa.com";

/**
 * Per-store Privacy / Terms URLs.
 *
 * Shopify themes use either `/policies/...` or `/pages/...` — set the exact URL
 * from each store's live site. Stores without their own page can point at a
 * sibling brand (see osakimassagechair.com → osakiusa.com).
 */
const STORE_POLICY_URLS: Record<
  string,
  { privacy: string; terms: string }
> = {
  "titanchair.com": {
    privacy: "https://titanchair.com/pages/privacy-policy",
    terms: "https://titanchair.com/pages/terms-of-service",
  },
  "www.titanchair.com": {
    privacy: "https://titanchair.com/pages/privacy-policy",
    terms: "https://titanchair.com/pages/terms-of-service",
  },
  "osakiusa.com": {
    privacy: "https://osakiusa.com/pages/privacy-policy",
    terms: "https://osakiusa.com/pages/terms-of-service",
  },
  "www.osakiusa.com": {
    privacy: "https://osakiusa.com/pages/privacy-policy",
    terms: "https://osakiusa.com/pages/terms-of-service",
  },
  /** No dedicated policy pages — use Osaki USA policies (same operator). */
  "osakimassagechair.com": {
    privacy: DEFAULT_POLICY_URLS.privacy,
    terms: DEFAULT_POLICY_URLS.terms,
  },
  "www.osakimassagechair.com": {
    privacy: DEFAULT_POLICY_URLS.privacy,
    terms: DEFAULT_POLICY_URLS.terms,
  },
};

/** Prior notice shown at the very top of the chat before the user types. */
export const CHAT_RECORDING_NOTICE =
  "By continuing this chat, you agree to our Privacy Policy and Terms of Service. " +
  "This conversation may be recorded, stored, and reviewed to provide support and improve our service. " +
  "Messages, email addresses, and photos or videos you share may also be stored for warranty support. " +
  "If you do not agree, please close this chat and contact us by phone instead.";

/** Short sticky line (kept for backward-compatible imports). */
export const CHAT_RECORDING_NOTICE_SHORT = CHAT_RECORDING_NOTICE;

function normalizeHost(domain: string): string {
  return domain.replace(/^https?:\/\//i, "").replace(/\/$/, "").toLowerCase();
}

/** Resolve which Shopify host identifies the customer store. */
export function resolvePolicyStoreDomain(domain?: string): string {
  const host = normalizeHost(domain ?? resolveWarrantyStoreDomain());
  if (INTERNAL_POLICY_HOSTS.has(host) || host.endsWith(".local")) {
    return "osakiusa.com";
  }
  return host;
}

function defaultPolicyUrlsForHost(host: string): { privacy: string; terms: string } {
  const mapped = STORE_POLICY_URLS[host];
  if (mapped) return mapped;

  // Generic Shopify /policies path for stores not yet mapped.
  const base = `https://${host.replace(/^www\./, "")}`;
  return {
    privacy: `${base}/policies/privacy-policy`,
    terms: `${base}/policies/terms-of-service`,
  };
}

/** Policy page URLs for the store hosting the chat embed. */
export function resolveStorePolicyUrls(domain?: string): {
  privacy: string;
  terms: string;
  storeDomain: string;
} {
  const host = resolvePolicyStoreDomain(domain);
  if (INTERNAL_POLICY_HOSTS.has(normalizeHost(domain ?? ""))) {
    return {
      storeDomain: "osakiusa.com",
      ...DEFAULT_POLICY_URLS,
    };
  }

  const urls = STORE_POLICY_URLS[host] ?? defaultPolicyUrlsForHost(host);
  return {
    storeDomain: host,
    privacy: urls.privacy,
    terms: urls.terms,
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
