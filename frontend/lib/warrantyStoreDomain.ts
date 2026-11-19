/**
 * Resolve which Shopify store domain to send to warranty API calls.
 *
 * Priority:
 *  1. ?store= query param (set by warranty-launcher.js from parent hostname)
 *  2. document.referrer host (Shopify parent page when embedded)
 *  3. current iframe hostname
 *
 * Shared help/admin/api hosts map to osakiusa.com so delivery lookups use the
 * OsakiUSA Shopify store instead of Titan.
 */
const HELP_PORTAL_HOSTS = new Set([
  "help.osakichair.com",
  "admin.osakichair.com",
  "api.osakichair.com",
]);

function normalizeStoreHost(raw: string): string {
  return raw
    .replace(/^https?:\/\//i, "")
    .replace(/\/$/, "")
    .toLowerCase()
    .replace(/^www\./, "");
}

function mapHelpPortalHost(host: string): string {
  const normalized = normalizeStoreHost(host);
  if (HELP_PORTAL_HOSTS.has(normalized)) {
    return "osakiusa.com";
  }
  return normalized || "osakiusa.com";
}

export function resolveWarrantyStoreDomain(): string {
  if (typeof window === "undefined") {
    return "osakiusa.com";
  }

  const fromQuery = new URLSearchParams(window.location.search).get("store")?.trim();
  if (fromQuery) {
    return mapHelpPortalHost(fromQuery);
  }

  try {
    if (document.referrer) {
      const refHost = new URL(document.referrer).hostname.toLowerCase();
      const selfHost = window.location.hostname.toLowerCase();
      if (refHost && refHost !== selfHost) {
        return mapHelpPortalHost(refHost);
      }
    }
  } catch {
    /* ignore malformed referrer */
  }

  return mapHelpPortalHost(window.location.hostname || "osakiusa.com");
}
